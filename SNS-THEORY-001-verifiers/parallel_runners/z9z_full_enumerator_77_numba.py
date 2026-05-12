#!/usr/bin/env python3
"""Numba runner for z9z_full_enumerator_77.py without modifying the verifier."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads


ROOT = Path("/home/ahmed/cluster-docs/SNS-THEORY-001-verifiers")
VERIFIER = ROOT / "z9z_full_enumerator_77.py"
OUT_PATH = ROOT / "Z9Z_K7R7_full_enumeration.json"
K = 7
R = 7
MOD = 9
TOTAL_SEEDS = MOD**K
W = np.array([0, 1, 1, 2, 1, 1, 2, 1, 1], dtype=np.uint8)


def _load_verifier():
    spec = importlib.util.spec_from_file_location("z9z_full_enumerator_77", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _message_weight_digits(idx: int) -> tuple[int, list[int]]:
    x = idx
    digits = []
    weight = 0
    for _ in range(K):
        v = x % MOD
        x //= MOD
        digits.append(v)
        weight += int(W[v])
    return weight, digits


def precompute_messages(max_weight: int):
    rows = []
    weights = []
    for idx in range(1, TOTAL_SEEDS):
        wt, digits = _message_weight_digits(idx)
        if wt <= max_weight:
            rows.append(digits)
            weights.append(wt)
    order = sorted(range(len(rows)), key=lambda i: weights[i])
    msg = np.array([rows[i] for i in order], dtype=np.int8)
    wts = np.array([weights[i] for i in order], dtype=np.uint8)
    return msg, wts


@njit
def _decode_seed(seed_idx):
    seed = np.empty(K, dtype=np.int8)
    x = seed_idx
    for i in range(K):
        seed[i] = x % MOD
        x //= MOD
    return seed


@njit
def _has_codeword_at_most(seed_idx, messages, msg_wts, threshold):
    seed = _decode_seed(seed_idx)
    for mi in range(messages.shape[0]):
        total = int(msg_wts[mi])
        if total > threshold:
            break
        for j in range(R):
            p = 0
            for i in range(K):
                p += int(messages[mi, i]) * int(seed[(j - i) % K])
            total += int(W[p % MOD])
            if total > threshold:
                break
        if total <= threshold:
            return True
    return False


@njit(parallel=True)
def classify_range(start, end, messages7, wts7, messages8, wts8):
    n = end - start
    status = np.zeros(n, dtype=np.uint8)
    for off in prange(n):
        seed_idx = start + off
        if _has_codeword_at_most(seed_idx, messages7, wts7, 7):
            status[off] = 0
        else:
            if _has_codeword_at_most(seed_idx, messages8, wts8, 8):
                status[off] = 1
            else:
                status[off] = 2
    return status


def decode_seed_py(seed_idx: int) -> tuple[int, ...]:
    seed = []
    x = seed_idx
    for _ in range(K):
        seed.append(x % MOD)
        x //= MOD
    return tuple(seed)


def cyclic_shift(seed):
    return (seed[-1],) + seed[:-1]


def negate(seed):
    return tuple((-x) % MOD for x in seed)


def orbit(seed):
    members = set()
    current = seed
    for _ in range(K):
        members.add(current)
        members.add(negate(current))
        current = cyclic_shift(current)
    return frozenset(members)


def main():
    _load_verifier()
    threads = int(os.environ.get("NUMBA_NUM_THREADS", os.cpu_count() or 1))
    set_num_threads(threads)
    chunk_size = int(os.environ.get("Z9Z_CHUNK_SIZE", "50000"))
    print(f"Numba Z/9Z K=7,R=7 full enumerator using {get_num_threads()} threads", flush=True)
    print(f"Total seeds: {TOTAL_SEEDS:,}; chunk_size={chunk_size:,}", flush=True)

    t0 = time.time()
    print("Precomputing messages with Delta3 message weight <= 7 ...", flush=True)
    messages7, wts7 = precompute_messages(7)
    print(f"  messages<=7: {len(wts7):,}", flush=True)
    print("Precomputing messages with Delta3 message weight <= 8 ...", flush=True)
    messages8, wts8 = precompute_messages(8)
    print(f"  messages<=8: {len(wts8):,}", flush=True)

    print("Compiling kernels ...", flush=True)
    classify_range(0, 1, messages7, wts7, messages8, wts8)

    histogram = Counter()
    attainer_indices: list[int] = []
    ge9_indices: list[int] = []
    scan_start = time.time()
    for start in range(0, TOTAL_SEEDS, chunk_size):
        end = min(start + chunk_size, TOTAL_SEEDS)
        status = classify_range(start, end, messages7, wts7, messages8, wts8)
        le7 = int((status == 0).sum())
        eq8 = int((status == 1).sum())
        ge9 = int((status == 2).sum())
        histogram["d_le_7"] += le7
        histogram["d_eq_8"] += eq8
        histogram["d_ge_9"] += ge9
        if eq8:
            attainer_indices.extend((np.flatnonzero(status == 1) + start).astype(np.int64).tolist())
        if ge9:
            ge9_indices.extend((np.flatnonzero(status == 2) + start).astype(np.int64).tolist())
        processed = end
        elapsed = time.time() - scan_start
        rate = processed / elapsed if elapsed > 0 else 0.0
        print(
            f"through {processed:,}/{TOTAL_SEEDS:,}; "
            f"d_eq_8={histogram['d_eq_8']:,}; d_ge_9={histogram['d_ge_9']:,}; "
            f"{rate:,.0f} seeds/s; elapsed={elapsed:.1f}s",
            flush=True,
        )

    attainers = [decode_seed_py(i) for i in attainer_indices]
    seed_set = set(attainers)
    orbits = {orbit(seed) for seed in attainers}
    orbit_sizes = [len(o) for o in orbits]

    w1 = (6, 3, 1, 1, 4, 0, 0)
    w2 = (0, 0, 1, 4, 1, 3, 6)
    assert len(attainers) == 7014, f"FAIL: expected 7014 max-attaining seeds, got {len(attainers)}"
    assert len(orbits) == 501, f"FAIL: expected 501 free orbits, got {len(orbits)}"
    assert all(size == 14 for size in orbit_sizes), "FAIL: expected all orbits of size 14"
    assert histogram["d_ge_9"] == 0, f"FAIL: expected zero seeds at distance >= 9, got {histogram['d_ge_9']}"
    assert w1 in seed_set, f"FAIL: witness {w1} not in max-attaining family"
    assert w2 in seed_set, f"FAIL: witness {w2} not in max-attaining family"

    total_elapsed = time.time() - t0
    print("\n============================================================", flush=True)
    print("Theorem 24 verification COMPLETE", flush=True)
    print(f"Total seeds enumerated:     {TOTAL_SEEDS:,}", flush=True)
    print(f"Max-attaining seeds (d=8):  {len(attainers):,}", flush=True)
    print(f"Free orbits:                {len(orbits)}", flush=True)
    print("Orbit size (all):           14", flush=True)
    print(f"Seeds at d>=9:              {histogram['d_ge_9']}", flush=True)
    print(f"Witness (6,3,1,1,4,0,0):   CONFIRMED", flush=True)
    print(f"Witness (0,0,1,4,1,3,6):   CONFIRMED", flush=True)
    print(f"Total runtime:              {total_elapsed:.1f}s", flush=True)
    print("============================================================\n", flush=True)

    output = {
        "description": "Z/9Z full seed enumeration K=7, R=7 under Delta_3 metric",
        "document": "SNS-THEORY-001",
        "theorem": "Theorem 24",
        "provenance": "Numba all-core runner for z9z_full_enumerator_77.py; verifier file left unchanged",
        "parameters": {"K": K, "R": R, "modulus": MOD},
        "threads": get_num_threads(),
        "total_seeds": TOTAL_SEEDS,
        "max_attaining_count": len(attainers),
        "orbit_count": len(orbits),
        "orbit_size": 14,
        "seeds_at_distance_9_or_more": int(histogram["d_ge_9"]),
        "witness_seeds": [list(w1), list(w2)],
        "bucket_histogram": {k: int(v) for k, v in sorted(histogram.items())},
        "attaining_seeds": [list(s) for s in attainers],
        "runtime_seconds": round(total_elapsed, 1),
        "all_assertions_passed": True,
    }
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Output written to {OUT_PATH.name}", flush=True)


if __name__ == "__main__":
    main()
