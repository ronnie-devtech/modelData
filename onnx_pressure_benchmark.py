#!/usr/bin/env python3
"""Small, reproducible open-loop ONNX pressure benchmark.

The benchmark intentionally measures the endpoint path with ordinary
``session.run`` calls.  It supports CUDA, MUSA, and CPU EPs and has two load
patterns:

* uniform: one request every 1 / QPS seconds;
* batch: ``workers`` requests released at the same time every workers / QPS
  seconds.

Examples::

    # CUDA, uniformly spaced requests, total arrival rate 160 QPS.
    CUDA_VISIBLE_DEVICES=0 python onnx_pressure_benchmark.py \\
        --ep cuda --model <model.onnx> --request-dir <request-dir> --mode uniform --qps 160 --workers 8

    # CUDA, eight independent requests released every 50 ms.
    CUDA_VISIBLE_DEVICES=0 python onnx_pressure_benchmark.py \\
        --ep cuda --model <model.onnx> --request-dir <request-dir> --mode batch --qps 160 --workers 8

    # MUSA, with the device selected by the visibility environment.
    MUSA_VISIBLE_DEVICES=0 python onnx_pressure_benchmark.py \\
        --ep musa --model <model.onnx> --request-dir <request-dir> --mode uniform --qps 160 --workers 8

    # CPU baseline for a standard ONNX model (the recommendation model needs
    # a CPU FusedGemm/custom-op implementation and is not a stock CPU EP case).
    taskset -c 0-13 python onnx_pressure_benchmark.py \\
        --ep cpu --model <standard-model.onnx> \\
        --request-dir <standard-request-dir> --mode uniform --qps 10 --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


RUNTIME_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "MUSA_VISIBLE_DEVICES",
    "ORT_MUSA_ENABLE_TF32",
)

HELP_EPILOG = """
Examples:
  CUDA_VISIBLE_DEVICES=0 python onnx_pressure_benchmark.py \\
      --ep cuda --model <model.onnx> --request-dir <request-dir> --mode uniform --qps 160 --workers 8
  CUDA_VISIBLE_DEVICES=0 python onnx_pressure_benchmark.py \\
      --ep cuda --model <model.onnx> --request-dir <request-dir> --mode batch --qps 160 --workers 8
  MUSA_VISIBLE_DEVICES=0 python onnx_pressure_benchmark.py \\
      --ep musa --model <model.onnx> --request-dir <request-dir> --mode uniform --qps 160 --workers 8
  taskset -c 0-13 python onnx_pressure_benchmark.py \\
      --ep cpu --model <standard-model.onnx> \\
      --request-dir <standard-request-dir> --mode uniform --qps 10 --workers 4

The two modes use total request QPS. ``batch`` releases ``workers``
independent requests together; it does not concatenate them into one ONNX
batch.  Both modes use ordinary session.run without I/O binding.  The CPU
example is for a standard ONNX model; the JD recommendation model requires
its CPU FusedGemm/custom-op implementation before CPU EP can load it.
"""

DTYPE_MAP = {
    "tensor(bool)": np.bool_,
    "tensor(double)": np.float64,
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(int8)": np.int8,
    "tensor(string)": np.str_,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(uint8)": np.uint8,
}


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
        description=(
            "Run a reproducible open-loop endpoint benchmark with one shared "
            "ONNX Runtime session."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_EPILOG,
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--ep", choices=("cuda", "musa", "cpu"), default="cuda")
    parser.add_argument("--mode", choices=("uniform", "batch"), default="uniform")
    parser.add_argument("--qps", type=float, default=160.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--inter-op-threads", type=int, default=1)
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
    if args.device_id is not None and args.device_id < 0:
        raise SystemExit("--device-id must be non-negative")
    if args.intra_op_threads < 0 or args.inter_op_threads < 0:
        raise SystemExit("ORT thread counts must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.progress < 0:
        raise SystemExit("--progress must be non-negative")


def configure_device_visibility(ep: str, device_id: Optional[int]) -> Optional[int]:
    # CUDA provider options select the logical device.  MUSA needs visibility
    # configured before importing the provider library; an existing environment
    # variable remains authoritative.
    if ep != "musa":
        return device_id
    if device_id is not None and "MUSA_VISIBLE_DEVICES" not in os.environ:
        os.environ["MUSA_VISIBLE_DEVICES"] = str(device_id)
        return 0
    return 0 if device_id is None else device_id


def import_ort(ep: str):
    import onnxruntime as ort

    if ep != "musa":
        return ort
    if "MUSAExecutionProvider" not in ort.get_available_providers():
        try:
            import onnxruntime_musa as musa_ep
        except ImportError as exc:
            raise RuntimeError(
                "MUSA EP is unavailable and onnxruntime_musa could not be imported"
            ) from exc
        ort.register_execution_provider_library(
            musa_ep.get_ep_name(), musa_ep.get_library_path()
        )
    return ort


def create_session(args: argparse.Namespace):
    musa_logical_id = configure_device_visibility(args.ep, args.device_id)
    ort = import_ort(args.ep)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = args.intra_op_threads
    options.inter_op_num_threads = args.inter_op_threads
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")

    if args.ep == "cpu":
        session = ort.InferenceSession(
            str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
        )
    elif args.ep == "cuda":
        device_id = 0 if args.device_id is None else args.device_id
        session = ort.InferenceSession(
            str(args.model),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"device_id": device_id})],
        )
    else:
        devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "MUSAExecutionProvider"
        ]
        if not devices:
            raise RuntimeError("No MUSA device is available")
        logical_id = 0 if musa_logical_id is None else musa_logical_id
        if logical_id >= len(devices):
            raise RuntimeError(
                f"MUSA logical device {logical_id} is unavailable; "
                f"visible devices={len(devices)}"
            )
        options.add_provider_for_devices([devices[logical_id]], {})

        # The recommendation model may expose the rank-N CPU fallback EP.  It
        # is optional and is not needed for ordinary CUDA/CPU ONNX models.
        rank_devices = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == "MusaRankNGemmCpuExecutionProvider"
        ]
        if rank_devices:
            options.add_provider_for_devices(rank_devices, {})
        session = ort.InferenceSession(str(args.model), sess_options=options)

    expected_provider = {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "musa": "MUSAExecutionProvider",
    }[args.ep]
    if expected_provider not in session.get_providers():
        raise RuntimeError(
            f"{expected_provider} was not selected; providers="
            f"{session.get_providers()}"
        )
    return session


def request_sort_key(path: Path) -> tuple[int, object, str]:
    prefix = path.name.split("_", 1)[0]
    if prefix.isdigit():
        return 0, int(prefix), path.name
    return 1, path.name, path.name


def sample_name_from_request(path: Path) -> str:
    name = path.stem
    for suffix in ("_optimized_static", "_static"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def cast_input_dtype(input_meta: Any, data: np.ndarray) -> np.ndarray:
    expected_dtype = DTYPE_MAP.get(input_meta.type)
    if expected_dtype is None:
        raise NotImplementedError(f"Unsupported input type: {input_meta.type}")
    out = np.asarray(data).astype(expected_dtype, copy=False)
    if input_meta.shape == [] and out.shape == (1,):
        out = np.asarray(out.reshape(-1)[0], dtype=expected_dtype)
    return out


def load_requests(
    session: Any, request_dir: Path, limit: Optional[int]
) -> tuple[list[dict[str, object]], list[str]]:
    paths = sorted(request_dir.glob("*_static.npz"), key=request_sort_key)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No *_static.npz files under {request_dir}")

    input_metas = session.get_inputs()
    requests: list[dict[str, object]] = []
    sample_names: list[str] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            missing = [
                meta.name for meta in input_metas if meta.name not in archive
            ]
            if missing:
                raise FileNotFoundError(
                    f"{path} is missing inputs; first missing={missing[:5]}"
                )
            requests.append(
                {
                    meta.name: cast_input_dtype(meta, archive[meta.name])
                    for meta in input_metas
                }
            )
        sample_names.append(sample_name_from_request(path))
    return requests, sample_names


def run_one(session: Any, request: dict[str, object]) -> None:
    # The returned host arrays are intentionally materialized and discarded:
    # endpoint latency includes the normal output transfer path.
    session.run(None, request)


def record_error(
    errors: list[BaseException], error_lock: threading.Lock, error: BaseException
) -> None:
    with error_lock:
        errors.append(error)


def worker_loop(
    worker_id: int,
    worker_count: int,
    session: Any,
    requests: Sequence[dict[str, object]],
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
                run_start = time.perf_counter()
                run_one(session, requests[query.sample_index])
                run_end = time.perf_counter()
                result.records.append(
                    LatencyRecord(
                        query.query_id,
                        query.wave_id,
                        query.sample_index,
                        query.scheduled_time,
                        run_end,
                    )
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
        dispatch_size = 1 if mode == "uniform" else workers
        interval = dispatch_size / qps
        for first_query_id in range(0, request_count, dispatch_size):
            if stop_event.is_set():
                return
            wave_id = first_query_id // dispatch_size
            scheduled_time = start_time[0] + wave_id * interval
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


def latency_summary(records: Sequence[LatencyRecord]) -> tuple[float, float]:
    if not records:
        return float("nan"), float("nan")
    values = np.asarray(
        [(record.run_end_time - record.scheduled_time) * 1000.0 for record in records],
        dtype=np.float64,
    )
    return float(np.mean(values)), float(np.percentile(values, 99))


def runtime_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key in RUNTIME_ENV_KEYS
        or (
            key.startswith("MUSA_REC_")
            and key != "MUSA_REC_ENABLE_FAST_PATHS"
        )
    }


def write_results(
    output_dir: Path,
    args: argparse.Namespace,
    session: Any,
    request_count: int,
    records: Sequence[LatencyRecord],
    sample_names: Sequence[str],
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
        writer.writerow(("query_id", "wave_id", "sample_index", "latency_ms"))
        for record in records:
            writer.writerow(
                (
                    record.query_id,
                    record.wave_id,
                    record.sample_index,
                    f"{(record.run_end_time - record.scheduled_time) * 1000.0:.6f}",
                )
            )

    elapsed = max(0.0, end_time - start_time)
    actual_qps = len(records) / elapsed if elapsed > 0 else 0.0
    drain_time_ms = max(0.0, elapsed - args.duration) * 1000.0
    latency_avg_ms, latency_p99_ms = latency_summary(records)
    scheduled_qps = request_count / args.duration
    overload = (
        len(records) != request_count
        or drain_time_ms > max(args.duration * 50.0, 1000.0 / args.qps)
        or max_backlog > args.workers
    )
    summary = {
        "model": str(args.model),
        "request_dir": str(args.request_dir),
        "ep": args.ep,
        "providers": session.get_providers(),
        "device_id": args.device_id,
        "runtime_environment": runtime_environment(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "mode": args.mode,
        "workers": args.workers,
        "target_qps": args.qps,
        "scheduled_qps": scheduled_qps,
        "warmup_seconds": args.warmup_seconds,
        "duration_seconds": args.duration,
        "intra_op_threads": args.intra_op_threads,
        "inter_op_threads": args.inter_op_threads,
        "graph_optimization_level": "ORT_DISABLE_ALL",
        "enable_mem_pattern": True,
        "io_binding": False,
        "loaded_samples": len(sample_names),
        "scheduled_queries": request_count,
        "completed_queries": len(records),
        "actual_qps": actual_qps,
        "elapsed_seconds": elapsed,
        "drain_time_ms": drain_time_ms,
        "max_backlog": max_backlog,
        "latency_avg_ms": latency_avg_ms,
        "latency_p99_ms": latency_p99_ms,
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
    print(f"latency_p99_ms={latency_p99_ms:.3f}")
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
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )

    session = create_session(args)
    requests, sample_names = load_requests(session, args.request_dir, args.limit)
    # Populate provider/session state before concurrent workers establish their
    # own stream-local workspaces. This is outside both warmup and measurement.
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

    if errors:
        # Still write the partial record set for debugging, then fail clearly.
        records = [record for result in results for record in result.records]
        write_results(
            args.output_dir,
            args,
            session,
            request_count,
            records,
            sample_names,
            start_time[0],
            end_time,
            max_backlog[0],
            errors,
        )
        raise RuntimeError("Benchmark worker failed") from errors[0]

    records = [record for result in results for record in result.records]
    summary = write_results(
        args.output_dir,
        args,
        session,
        request_count,
        records,
        sample_names,
        start_time[0],
        end_time,
        max_backlog[0],
        errors,
    )
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
