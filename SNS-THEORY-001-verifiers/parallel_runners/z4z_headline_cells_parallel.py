#!/usr/bin/env python3
"""Parallel runner for z4z_headline_cells.py without modifying the verifier."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


ROOT = Path("/home/ahmed/cluster-docs/SNS-THEORY-001-verifiers")
VERIFIER = ROOT / "z4z_headline_cells.py"
OUT_PATH = ROOT / "Z4Z_headline_cells.json"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("z4z_headline_cells", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _decode_seed(idx: int, k: int) -> tuple[int, ...]:
    seed = []
    for _ in range(k):
        seed.append(idx % 4)
        idx //= 4
    return tuple(seed)


def _scan_range(args):
    start, end, k, r = args
    verifier = _load_verifier()
    max_d = 0
    attaining: list[tuple[int, ...]] = []
    for idx in range(start, end):
        seed = _decode_seed(idx, k)
        d = verifier.min_distance_seed(seed, k, r)
        if d > max_d:
            max_d = d
            attaining = [seed]
        elif d == max_d:
            attaining.append(seed)
    return max_d, attaining


def _chunks(total: int, n: int):
    step = (total + n - 1) // n
    for start in range(0, total, step):
        yield start, min(start + step, total)


def sweep_cell_parallel(k: int, r: int, workers: int):
    total = 4**k
    jobs = [(a, b, k, r) for a, b in _chunks(total, workers)]
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        parts = pool.map(_scan_range, jobs)
    max_d = max(part[0] for part in parts)
    attaining = []
    for d, seeds in parts:
        if d == max_d:
            attaining.extend(seeds)
    return max_d, attaining, time.time() - t0


def main():
    workers = int(os.environ.get("Z4Z_WORKERS", os.cpu_count() or 1))
    verifier = _load_verifier()
    print(f"Parallel Z/4Z headline verifier using {workers} worker processes", flush=True)

    results = {}

    print("Theorem 20: Z/4Z (K=7, R=7)...", flush=True)
    k, r = 7, 7
    max_d, attaining, elapsed = sweep_cell_parallel(k, r, workers)
    visited_orbits = {verifier.orbit(s, k) for s in attaining}
    orbit_count = len(visited_orbits)
    orbit_sizes = [len(o) for o in visited_orbits]

    assert max_d == 8, f"FAIL Theorem 20: expected max distance 8, got {max_d}"
    assert len(attaining) == 14, f"FAIL Theorem 20: expected 14 attaining seeds, got {len(attaining)}"
    assert orbit_count == 1, f"FAIL Theorem 20: expected 1 free orbit, got {orbit_count}"
    assert all(s == 14 for s in orbit_sizes), "FAIL Theorem 20: orbit size not 14"

    print(f"  max distance:    {max_d}  PASS", flush=True)
    print(f"  attaining seeds: {len(attaining)}  PASS", flush=True)
    print(f"  free orbits:     {orbit_count}  PASS", flush=True)
    print(f"  orbit size:      {orbit_sizes[0]}  PASS", flush=True)
    print(f"  runtime:         {elapsed:.2f}s", flush=True)

    results["theorem_20"] = {
        "K": k,
        "R": r,
        "max_delta2_distance": max_d,
        "attainer_count": len(attaining),
        "orbit_count": orbit_count,
        "orbit_size": 14,
        "attaining_seeds": [list(s) for s in attaining],
        "runtime_seconds": round(elapsed, 2),
        "pass": True,
    }

    print("\nTheorem 21: Z/4Z (K=8, R=7)...", flush=True)
    k, r = 8, 7
    max_d, attaining, elapsed = sweep_cell_parallel(k, r, workers)
    assert max_d == 6, f"FAIL Theorem 21: expected max distance 6, got {max_d}"
    assert len(attaining) == 9472, f"FAIL Theorem 21: expected 9472 attaining seeds, got {len(attaining)}"

    print(f"  max distance:    {max_d}  PASS", flush=True)
    print(f"  attaining seeds: {len(attaining)}  PASS", flush=True)
    print("  seeds at d>=7:   0  PASS", flush=True)
    print(f"  runtime:         {elapsed:.2f}s", flush=True)

    results["theorem_21"] = {
        "K": k,
        "R": r,
        "max_delta2_distance": max_d,
        "attainer_count": len(attaining),
        "seeds_at_distance_7_or_more": 0,
        "runtime_seconds": round(elapsed, 2),
        "pass": True,
    }

    drop = results["theorem_20"]["max_delta2_distance"] - results["theorem_21"]["max_delta2_distance"]
    assert drop == 2, f"FAIL: expected drop of 2, got {drop}"
    print(f"\nK-axis non-monotonicity (Remark 22): M2(7,7) - M2(8,7) = {drop}  PASS", flush=True)

    output = {
        "description": "Z/4Z headline and contrast cells under Delta_2 metric",
        "document": "SNS-THEORY-001",
        "theorems": ["Theorem 20", "Theorem 21"],
        "provenance": "parallel runner importing z4z_headline_cells.py without modifying it",
        "workers": workers,
        "k_axis_drop_at_R7": drop,
        "all_assertions_passed": True,
        "cells": results,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nOutput written to {OUT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
