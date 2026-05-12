"""
SNS-THEORY-001 Verifier: Z/4Z Phase Diagram (Theorem 19)
=========================================================
Exhaustively enumerates all 4^K seeds for every (K, R) cell in the
49-cell phase diagram over Z/4Z, K in {2..8}, R in {1..7}.

Expected output: Z4Z_phase_diagram_complete.json
Headline cells:
  (K=7, R=7) -> max distance 8  (Theorem 20)
  (K=8, R=7) -> max distance 6  (Theorem 21)

Runtime: ~344 seconds single-threaded (see SNS-THEORY-001 Table 6)
Author: Ahmed M. Mansour, Steward and Sync LLC
Document: SNS-THEORY-001
"""

import json
import itertools
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ring: Z/4Z
# ---------------------------------------------------------------------------
MOD = 4
UNITS = {1, 3}
ZERO_DIVISORS = {2}

def delta2_weight(x):
    x = x % MOD
    if x == 0:
        return 0
    if x in UNITS:
        return 1
    return 2  # nonzero zero-divisor

def delta2_distance(u, v):
    return sum(delta2_weight((a - b) % MOD) for a, b in zip(u, v))

# ---------------------------------------------------------------------------
# Circulant parity matrix A(s): A[i,j] = s[(j-i) mod K]
# ---------------------------------------------------------------------------
def build_parity(seed, K, R):
    return [[seed[(j - i) % K] for j in range(R)] for i in range(K)]

def codeword(msg, seed, K, R):
    A = build_parity(seed, K, R)
    parity = [sum(msg[i] * A[i][j] for i in range(K)) % MOD for j in range(R)]
    return tuple(msg) + tuple(parity)

# ---------------------------------------------------------------------------
# Minimum distance of code C(s) at parameters (K, R)
# ---------------------------------------------------------------------------
def min_distance(seed, K, R):
    best = float('inf')
    for msg in itertools.product(range(MOD), repeat=K):
        if all(m == 0 for m in msg):
            continue
        cw = codeword(msg, seed, K, R)
        zero = (0,) * (K + R)
        d = delta2_distance(cw, zero)
        if d < best:
            best = d
        if best == 1:
            return 1
    return best

# ---------------------------------------------------------------------------
# Phase diagram sweep
# ---------------------------------------------------------------------------
def sweep_cell(K, R):
    max_d = 0
    attainer_count = 0
    best_seed = None
    for seed in itertools.product(range(MOD), repeat=K):
        d = min_distance(seed, K, R)
        if d > max_d:
            max_d = d
            attainer_count = 1
            best_seed = seed
        elif d == max_d:
            attainer_count += 1
    return max_d, attainer_count, best_seed

def main():
    K_range = range(2, 9)
    R_range = range(1, 8)
    results = {}
    total_start = time.time()

    for K in K_range:
        for R in R_range:
            t0 = time.time()
            max_d, count, witness = sweep_cell(K, R)
            elapsed = time.time() - t0
            key = f"K{K}_R{R}"
            results[key] = {
                "K": K, "R": R,
                "max_delta2_distance": max_d,
                "attainer_count": count,
                "witness_seed": list(witness),
                "runtime_seconds": round(elapsed, 2)
            }
            print(f"K={K} R={R}  max_d={max_d}  attainers={count}  "
                  f"witness={list(witness)}  [{elapsed:.1f}s]")

    total_elapsed = time.time() - total_start

    # Spot-check headline cells
    assert results["K7_R7"]["max_delta2_distance"] == 8, \
        "FAIL: K=7 R=7 expected max distance 8 (Theorem 20)"
    assert results["K8_R7"]["max_delta2_distance"] == 6, \
        "FAIL: K=8 R=7 expected max distance 6 (Theorem 21)"
    print("\nTheorem 20 check: K=7 R=7 max distance == 8  PASS")
    print("Theorem 21 check: K=8 R=7 max distance == 6  PASS")

    output = {
        "description": "Z/4Z phase diagram M2(K,R) for K in {2..8}, R in {1..7}",
        "document": "SNS-THEORY-001",
        "theorems": ["Theorem 19", "Theorem 20", "Theorem 21"],
        "total_runtime_seconds": round(total_elapsed, 1),
        "cells": results
    }
    out_path = Path("Z4Z_phase_diagram_complete.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to {out_path}")
    print(f"Total runtime: {total_elapsed:.1f}s")

if __name__ == "__main__":
    main()
