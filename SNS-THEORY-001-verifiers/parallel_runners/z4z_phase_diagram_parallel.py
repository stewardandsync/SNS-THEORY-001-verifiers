#!/usr/bin/env python3
"""Parallel runner for z4z_phase_diagram.py without modifying the verifier."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


ROOT = Path("/home/ahmed/cluster-docs/SNS-THEORY-001-verifiers")
VERIFIER = ROOT / "z4z_phase_diagram.py"
OUT_PATH = ROOT / "Z4Z_phase_diagram_complete.json"
MOD = 4


def _load_verifier():
    spec = importlib.util.spec_from_file_location("z4z_phase_diagram", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _decode_seed(idx: int, k: int) -> tuple[int, ...]:
    seed = []
    for _ in range(k):
        seed.append(idx % MOD)
        idx //= MOD
    return tuple(seed)


def _delta2_weight(x: int) -> int:
    x %= MOD
    if x == 0:
        return 0
    if x in (1, 3):
        return 1
    return 2


def _min_distance_seed(seed: tuple[int, ...], k: int, r: int) -> int:
    """Same code family as z4z_phase_diagram.py, with A built once per seed."""
    best = 10**9
    A = [[seed[(j - i) % k] for j in range(r)] for i in range(k)]
    total_messages = MOD**k
    for msg_idx in range(1, total_messages):
        x = msg_idx
        msg = [0] * k
        msg_w = 0
        for i in range(k):
            v = x % MOD
            x //= MOD
            msg[i] = v
            msg_w += _delta2_weight(v)
            if msg_w >= best:
                break
        if msg_w >= best:
            continue
        total = msg_w
        for j in range(r):
            p = 0
            for i in range(k):
                p += msg[i] * A[i][j]
            total += _delta2_weight(p)
            if total >= best:
                break
        if total < best:
            best = total
            if best == 1:
                return 1
    return best


def _scan_range(args):
    start, end, k, r = args
    max_d = 0
    attainer_count = 0
    best_seed = None
    for idx in range(start, end):
        seed = _decode_seed(idx, k)
        d = _min_distance_seed(seed, k, r)
        if d > max_d:
            max_d = d
            attainer_count = 1
            best_seed = seed
        elif d == max_d:
            attainer_count += 1
    return max_d, attainer_count, best_seed


def _chunks(total: int, n: int):
    step = (total + n - 1) // n
    for start in range(0, total, step):
        yield start, min(start + step, total)


def sweep_cell_parallel(k: int, r: int, workers: int):
    total = MOD**k
    jobs = [(a, b, k, r) for a, b in _chunks(total, workers)]
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        parts = pool.map(_scan_range, jobs)
    max_d = max(part[0] for part in parts)
    count = sum(part[1] for part in parts if part[0] == max_d)
    witness = next(part[2] for part in parts if part[0] == max_d and part[2] is not None)
    return max_d, count, witness, time.time() - t0


def main():
    workers = int(os.environ.get("Z4Z_WORKERS", os.cpu_count() or 1))
    _load_verifier()
    print(f"Parallel Z/4Z phase diagram verifier using {workers} worker processes", flush=True)

    results = {}
    total_start = time.time()
    for k in range(2, 9):
        for r in range(1, 8):
            max_d, count, witness, elapsed = sweep_cell_parallel(k, r, workers)
            key = f"K{k}_R{r}"
            results[key] = {
                "K": k,
                "R": r,
                "max_delta2_distance": max_d,
                "attainer_count": count,
                "witness_seed": list(witness),
                "runtime_seconds": round(elapsed, 2),
            }
            print(
                f"K={k} R={r}  max_d={max_d}  attainers={count}  "
                f"witness={list(witness)}  [{elapsed:.1f}s]",
                flush=True,
            )

    total_elapsed = time.time() - total_start
    assert results["K7_R7"]["max_delta2_distance"] == 8, (
        "FAIL: K=7 R=7 expected max distance 8 (Theorem 20)"
    )
    assert results["K8_R7"]["max_delta2_distance"] == 6, (
        "FAIL: K=8 R=7 expected max distance 6 (Theorem 21)"
    )
    print("\nTheorem 20 check: K=7 R=7 max distance == 8  PASS", flush=True)
    print("Theorem 21 check: K=8 R=7 max distance == 6  PASS", flush=True)

    output = {
        "description": "Z/4Z phase diagram M2(K,R) for K in {2..8}, R in {1..7}",
        "document": "SNS-THEORY-001",
        "theorems": ["Theorem 19", "Theorem 20", "Theorem 21"],
        "provenance": "parallel runner for z4z_phase_diagram.py; verifier file left unchanged",
        "workers": workers,
        "total_runtime_seconds": round(total_elapsed, 1),
        "cells": results,
        "all_assertions_passed": True,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nOutput written to {OUT_PATH.name}", flush=True)
    print(f"Total runtime: {total_elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
