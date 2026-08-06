#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import socket
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class Client:
    pass

class TokenBucket:
    def __init__(self, rate: float):
        self.rate = float(rate)
        self.capacity = float(rate)
        self.tokens = float(rate)
        self.last = time.monotonic()
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def acquire(self):
        with self.cond:
            while True:
                now = time.monotonic()
                elapsed = now - self.last
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                    self.last = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                deficit = 1.0 - self.tokens
                wait = deficit / self.rate
                self.cond.wait(timeout=min(wait, 0.05))


class PoissonPacer:
    def __init__(self, target_qps: float, workers: int):
        self.mean_gap = workers / float(target_qps) if target_qps > 0 else 0.0

    def make_local(self, worker_id: int) -> "_PoissonLocal":
        return _PoissonLocal(self.mean_gap, worker_id)


class _PoissonLocal:
    def __init__(self, mean_gap: float, worker_id: int):
        self.mean_gap = mean_gap
        self.rng = random.Random(0xC0FFEE ^ worker_id)
        self.next_at = time.monotonic()

    def acquire(self):
        now = time.monotonic()
        if now < self.next_at:
            time.sleep(self.next_at - now)
        gap = self.rng.expovariate(1.0 / self.mean_gap) if self.mean_gap > 0 else 0
        self.next_at = max(time.monotonic(), self.next_at + gap)


class _NoPacer:
    def acquire(self):
        return


def perf_worker(
    worker_id: int,
    host: str,
    port: int,
    timeout_ms: int,
    payloads,
    pacer,
    deadline: float,
    target_total: int,
    counter,
    counter_lock: threading.Lock,
    stats,
    stats_lock: threading.Lock,
    args,
    method: str,
):
    client = Client()
    local_lat = []
    local_ok = 0
    local_biz = 0
    local_proto = 0
    local_errs = {}
    rng = random.Random(worker_id)
    try:
        while True:
            if target_total > 0:
                with counter_lock:
                    if counter[0] >= target_total:
                        break
                    counter[0] += 1
            elif time.monotonic() >= deadline:
                break

            pacer.acquire()

            t0 = time.monotonic()
            client.call()
            local_ok += 1
            local_lat.append(int((time.monotonic() - t0) * 1_000_000))
    finally:
        client.close()
        with stats_lock:
            stats.ok += local_ok
            stats.biz_err += local_biz
            stats.proto_err += local_proto
            stats.latencies_us.extend(local_lat)
            for k, v in local_errs.items():
                stats.errors[k] = stats.errors.get(k, 0) + v

def run_perf(args) -> int:
    host, workers, port, payloads, stats, method = None, None, None, None, None, None

    if args.perf_mode == "poisson":
        pp = PoissonPacer(target_qps=args.perf_qps, workers=workers)
        per_worker_pacer = pp.make_local
    elif args.perf_mode == "burst":
        np = _NoPacer()
        per_worker_pacer = lambda i: np
    else:
        raise RuntimeError(f"unknown perf_mode: {args.perf_mode}")

    target_total = args.perf_total
    deadline = time.monotonic() + args.perf_duration if target_total <= 0 else float("inf")
    stats_lock = threading.Lock()
    counter = [0]
    counter_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [
            pool.submit(
                perf_worker,
                i, host, port, args.time_out,
                payloads, per_worker_pacer(i),
                deadline, target_total,
                counter, counter_lock,
                stats, stats_lock,
                args, method,
            )
            for i in range(workers)
        ]
        for f in futs:
            f.result()
