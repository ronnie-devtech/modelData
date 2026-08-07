#!/usr/bin/env python3
"""开环 ORT 压测：按目标 QPS 投放请求，多 worker 并发跑 ORTSession.run。

延迟统计使用 ORTSession.run() 返回的 run_time（仅 OrtApi::Run 调用本身，
不含输入准备 / 输出转换）。

两种负载模式：
* uniform: 每 1/QPS 秒投放一条请求；
* batch:   每 workers/QPS 秒同时投放 workers 条请求。

用法（在 19923 上）:
  cd /export/chenxuan/ort_test && source env.sh
  python3 pressure_benchmark.py --mode uniform --qps 160 --workers 8 --duration 60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from run_inference import ORTSession, get_model_input_dtypes, ONNX_TO_NP_DTYPE

# ── 路径配置 ──────────────────────────────────────────────
MODEL_PATH = "/export/chenxuan/ort_test/optimized_model.onnx"
REQ_DIR    = "/export/chenxuan/ort_test/req_799_npz"


@dataclass(frozen=True)
class Query:
    query_id: int
    wave_id: int
    sample_index: int
    scheduled_time: float
    release_event: Optional[threading.Event] = None


@dataclass(frozen=True)
class LatencyRecord:
    query_id: int
    wave_id: int
    sample_index: int
    scheduled_time: float
    run_time_ms: float       # ORTSession.run() 返回的 run_time（仅 OrtApi::Run）
    run_end_time: float


@dataclass
class WorkerResult:
    records: list[LatencyRecord] = field(default_factory=list)
    warmup_runs: int = 0


def default_output_dir() -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return Path("pressure_results") / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="开环 ORT 压测，延迟仅统计 OrtApi::Run 耗时。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=Path(MODEL_PATH))
    parser.add_argument("--request-dir", type=Path, default=Path(REQ_DIR))
    parser.add_argument("--mode", choices=("uniform", "batch", "poisson"), default="uniform")
    parser.add_argument("--qps", type=float, default=160.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-seconds", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--progress", type=int, default=100, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.qps <= 0:
        raise SystemExit("--qps must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.warmup_seconds < 0:
        raise SystemExit("--warmup-seconds must be non-negative")
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.progress < 0:
        raise SystemExit("--progress must be non-negative")


def request_sort_key(path: Path) -> tuple[int, object, str]:
    prefix = path.name.split("_", 1)[0]
    if prefix.isdigit():
        return 0, int(prefix), path.name
    return 1, path.name, path.name


def load_requests(
    session: ORTSession, request_dir: Path, limit: Optional[int]
) -> list[dict[str, np.ndarray]]:
    """加载请求 npz，按模型输入 dtype 转换。"""
    paths = sorted(request_dir.glob("*.npz"), key=request_sort_key)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .npz files under {request_dir}")

    input_types = get_model_input_dtypes(session._onnx_path)
    requests: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            feed_dict = {}
            for name in session.input_names:
                if name not in archive:
                    raise FileNotFoundError(
                        f"{path} is missing input '{name}'"
                    )
                arr = archive[name]
                expected = ONNX_TO_NP_DTYPE.get(input_types.get(name, 0))
                if expected and arr.dtype != expected:
                    arr = arr.astype(expected)
                feed_dict[name] = arr
            requests.append(feed_dict)
    return requests


def run_one(session: ORTSession, request: dict[str, np.ndarray]) -> float:
    """跑一条请求，返回 run_time（秒）。释放 output OrtValues。"""
    output_vals, run_time = session.run(request)
    for i in range(len(output_vals)):
        session._ReleaseVal(output_vals[i])
    return run_time


def record_error(
    errors: list[BaseException], error_lock: threading.Lock, error: BaseException
) -> None:
    with error_lock:
        errors.append(error)


def worker_loop(
    worker_id: int,
    worker_count: int,
    session: ORTSession,
    requests: Sequence[dict[str, np.ndarray]],
    work_queue: "queue.Queue[Query]",
    producer_done: threading.Event,
    stop_event: threading.Event,
    warmup_barrier: threading.Barrier,
    warmup_start: threading.Event,
    warmup_start_time: list[float],
    warmup_seconds: float,
    ready_barrier: threading.Barrier,
    start_event: threading.Event,
    result: WorkerResult,
    errors: list[BaseException],
    error_lock: threading.Lock,
    progress_every: int,
) -> None:
    try:
        warmup_barrier.wait()
        warmup_start.wait()
        deadline = warmup_start_time[0] + warmup_seconds
        warmup_index = 0
        while time.perf_counter() < deadline:
            sample_index = (worker_id + warmup_index * worker_count) % len(requests)
            run_one(session, requests[sample_index])
            result.warmup_runs += 1
            warmup_index += 1

        ready_barrier.wait()
        start_event.wait()
        processed = 0
        while not stop_event.is_set():
            try:
                query = work_queue.get(timeout=0.05)
            except queue.Empty:
                if producer_done.is_set():
                    return
                continue
            try:
                if query.release_event is not None:
                    query.release_event.wait()
                run_time = run_one(session, requests[query.sample_index])
                result.records.append(
                    LatencyRecord(
                        query.query_id,
                        query.wave_id,
                        query.sample_index,
                        query.scheduled_time,
                        run_time * 1000.0,
                        time.perf_counter(),
                    )
                )
                processed += 1
                if progress_every and processed % progress_every == 0:
                    print(
                        f"[worker {worker_id}] {processed} done, "
                        f"last run={run_time * 1000.0:.3f}ms",
                        flush=True,
                    )
            finally:
                work_queue.task_done()
    except BaseException as exc:
        stop_event.set()
        for barrier in (warmup_barrier, ready_barrier):
            try:
                barrier.abort()
            except threading.BrokenBarrierError:
                pass
        record_error(errors, error_lock, exc)


def producer_loop(
    mode: str,
    qps: float,
    workers: int,
    request_count: int,
    sample_count: int,
    start_event: threading.Event,
    start_time: list[float],
    work_queue: "queue.Queue[Query]",
    producer_done: threading.Event,
    stop_event: threading.Event,
    max_backlog: list[int],
    backlog_lock: threading.Lock,
    errors: list[BaseException],
    error_lock: threading.Lock,
) -> None:
    try:
        start_event.wait()
        dispatch_size = 1 if (mode == "uniform" or mode == "poisson") else workers
        last_time = start_time[0]
        for first_query_id in range(0, request_count, dispatch_size):
            if stop_event.is_set():
                return
            wave_id = first_query_id // dispatch_size
            if mode == "poisson":
                scheduled_time = last_time + random.expovariate(qps)
            else:
                scheduled_time = last_time + dispatch_size / qps

            last_time = scheduled_time
            delay = scheduled_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)

            release_event = threading.Event() if mode == "batch" else None
            last_query_id = min(first_query_id + dispatch_size, request_count)
            for query_id in range(first_query_id, last_query_id):
                work_queue.put(
                    Query(
                        query_id,
                        wave_id,
                        query_id % sample_count,
                        scheduled_time,
                        release_event,
                    )
                )
            with backlog_lock:
                max_backlog[0] = max(max_backlog[0], work_queue.qsize())
            if release_event is not None:
                release_event.set()
    except BaseException as exc:
        stop_event.set()
        record_error(errors, error_lock, exc)
    finally:
        producer_done.set()


def latency_summary(records: Sequence[LatencyRecord]) -> tuple[float, float, float, float]:
    """返回 (avg, p50, p99, max) ms，基于 OrtApi::Run 耗时。"""
    if not records:
        return float("nan"), float("nan"), float("nan"), float("nan")
    values = np.asarray(
        [record.run_time_ms for record in records], dtype=np.float64
    )
    return (
        float(np.mean(values)),
        float(np.percentile(values, 50)),
        float(np.percentile(values, 99)),
        float(np.max(values)),
    )


def write_results(
    output_dir: Path,
    args: argparse.Namespace,
    request_count: int,
    records: Sequence[LatencyRecord],
    start_time: float,
    end_time: float,
    max_backlog: int,
    errors: Sequence[BaseException],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda record: record.query_id)
    latency_csv = output_dir / "latency.csv"
    with latency_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("query_id", "wave_id", "sample_index", "run_time_ms"))
        for record in records:
            writer.writerow(
                (
                    record.query_id,
                    record.wave_id,
                    record.sample_index,
                    f"{record.run_time_ms:.6f}",
                )
            )

    elapsed = max(0.0, end_time - start_time)
    actual_qps = len(records) / elapsed if elapsed > 0 else 0.0
    drain_time_ms = max(0.0, elapsed - args.duration) * 1000.0
    latency_avg_ms, latency_p50_ms, latency_p99_ms, latency_max_ms = latency_summary(records)
    scheduled_qps = request_count / args.duration
    overload = (
        len(records) != request_count
        or drain_time_ms > max(args.duration * 50.0, 1000.0 / args.qps)
        or max_backlog > args.workers
    )
    summary = {
        "model": str(args.model),
        "request_dir": str(args.request_dir),
        "mode": args.mode,
        "workers": args.workers,
        "target_qps": args.qps,
        "scheduled_qps": scheduled_qps,
        "warmup_seconds": args.warmup_seconds,
        "duration_seconds": args.duration,
        "loaded_samples": len(records),
        "scheduled_queries": request_count,
        "completed_queries": len(records),
        "actual_qps": actual_qps,
        "elapsed_seconds": elapsed,
        "drain_time_ms": drain_time_ms,
        "max_backlog": max_backlog,
        "latency_avg_ms": latency_avg_ms,
        "latency_p50_ms": latency_p50_ms,
        "latency_p99_ms": latency_p99_ms,
        "latency_max_ms": latency_max_ms,
        "overload": overload,
        "runtime_errors": [repr(error) for error in errors],
        "latency_csv": str(latency_csv),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"target_qps={args.qps:.3f}")
    print(f"scheduled_qps={scheduled_qps:.3f}")
    print(f"actual_qps={actual_qps:.3f}")
    print(f"scheduled_queries={request_count}")
    print(f"completed_queries={len(records)}")
    print(f"latency_avg_ms={latency_avg_ms:.3f}")
    print(f"latency_p50_ms={latency_p50_ms:.3f}")
    print(f"latency_p99_ms={latency_p99_ms:.3f}")
    print(f"latency_max_ms={latency_max_ms:.3f}")
    print(f"overload={'true' if overload else 'false'}")
    print(f"summary={summary_path}")
    print(f"latency_csv={latency_csv}")
    return summary


def run(args: argparse.Namespace) -> int:
    args.model = args.model.expanduser().resolve()
    args.request_dir = args.request_dir.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir().resolve()
    )
    if not args.model.is_file():
        raise FileNotFoundError(f"Model does not exist: {args.model}")
    if not args.request_dir.is_dir():
        raise FileNotFoundError(f"Request directory does not exist: {args.request_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")

    print(f"Loading model: {args.model}")
    session = ORTSession(str(args.model), opt_level=2)
    print(f"Session created OK. inputs={len(session.input_names)}, "
          f"outputs={len(session.output_names)}")

    requests = load_requests(session, args.request_dir, args.limit)
    print(f"Loaded {len(requests)} requests")
    # 预热 provider/session 状态（在 warmup / 计量之外）
    run_one(session, requests[0])

    request_count = max(1, int(round(args.qps * args.duration)))
    work_queue: "queue.Queue[Query]" = queue.Queue()
    producer_done = threading.Event()
    stop_event = threading.Event()
    warmup_start = threading.Event()
    start_event = threading.Event()
    warmup_start_time = [0.0]
    start_time = [0.0]
    warmup_barrier = threading.Barrier(args.workers + 1)
    ready_barrier = threading.Barrier(args.workers + 1)
    errors: list[BaseException] = []
    error_lock = threading.Lock()
    max_backlog = [0]
    backlog_lock = threading.Lock()
    results = [WorkerResult() for _ in range(args.workers)]

    workers = [
        threading.Thread(
            target=worker_loop,
            name=f"pressure-worker-{worker_id}",
            args=(
                worker_id,
                args.workers,
                session,
                requests,
                work_queue,
                producer_done,
                stop_event,
                warmup_barrier,
                warmup_start,
                warmup_start_time,
                args.warmup_seconds,
                ready_barrier,
                start_event,
                results[worker_id],
                errors,
                error_lock,
                args.progress,
            ),
        )
        for worker_id in range(args.workers)
    ]
    producer = threading.Thread(
        target=producer_loop,
        name="pressure-producer",
        args=(
            args.mode,
            args.qps,
            args.workers,
            request_count,
            len(requests),
            start_event,
            start_time,
            work_queue,
            producer_done,
            stop_event,
            max_backlog,
            backlog_lock,
            errors,
            error_lock,
        ),
    )

    for worker in workers:
        worker.start()
    warmup_barrier.wait()
    warmup_start_time[0] = time.perf_counter()
    warmup_start.set()
    ready_barrier.wait()
    producer.start()
    start_time[0] = time.perf_counter()
    start_event.set()

    producer.join()
    for worker in workers:
        worker.join()
    end_time = time.perf_counter()

    records = [record for result in results for record in result.records]
    if errors:
        write_results(
            args.output_dir, args, request_count, records,
            start_time[0], end_time, max_backlog[0], errors,
        )
        raise RuntimeError("Benchmark worker failed") from errors[0]

    write_results(
        args.output_dir, args, request_count, records,
        start_time[0], end_time, max_backlog[0], errors,
    )
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())