"""
SNS-THEORY-001 Verifier: Z/4Z Headline and Contrast Cells (Theorems 20 & 21)
==============================================================================
Exhaustively verifies:
  Theorem 20: M2(7,7) = 8 over Z/4Z, attained by exactly 14 seeds forming
              one free orbit under <sigma, -id> of order 14
  Theorem 21: M2(8,7) = 6 over Z/4Z, attained by exactly 9,472 seeds,
              zero seeds attain distance >= 7

Output file: Z4Z_headline_cells.json

Runtime: seconds for both cells (see SNS-THEORY-001 Table 6)

Author: Ahmed M. Mansour, Steward and Sync LLC
Document: SNS-THEORY-001, Theorems 20 and 21
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

def delta2_weight(x):
    x = x % MOD
    if x == 0:
        return 0
    if x in UNITS:
        return 1
    return 2

def delta2_vec_weight(vec):
    return sum(delta2_weight(x) for x in vec)

def min_distance_seed(seed, K, R):
    best = float('inf')
    A = [[seed[(j - i) % K] for j in range(R)] for i in range(K)]
    for msg in itertools.product(range(MOD), repeat=K):
        if not any(msg):
            continue
        parity = tuple(sum(msg[i] * A[i][j] for i in range(K)) % MOD for j in range(R))
        cw = msg + parity
        d = delta2_vec_weight(cw)
        if d < best:
            best = d
        if best == 1:
            return 1
    return best

# ---------------------------------------------------------------------------
# Orbit structure under <sigma, -id> on Z/4Z^K
# ---------------------------------------------------------------------------
def cyclic_shift(seed):
    return (seed[-1],) + seed[:-1]

def negate_z4(seed):
    return tuple((-x) % MOD for x in seed)

def orbit(seed, K):
    members = set()
    current = seed
    for _ in range(K):
        members.add(current)
        members.add(negate_z4(current))
        current = cyclic_shift(current)
    return frozenset(members)

# ---------------------------------------------------------------------------
# Sweep one cell
# ---------------------------------------------------------------------------
def sweep_cell(K, R):
    max_d = 0
    attaining = []
    t0 = time.time()
    total = MOD ** K
    for seed in itertools.product(range(MOD), repeat=K):
        d = min_distance_seed(seed, K, R)
        if d > max_d:
            max_d = d
            attaining = [seed]
        elif d == max_d:
            attaining.append(seed)
    elapsed = time.time() - t0
    return max_d, attaining, elapsed

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = {}

    # --- Theorem 20: K=7, R=7 ---
    print("Theorem 20: Z/4Z (K=7, R=7)...")
    K, R = 7, 7
    max_d, attaining, elapsed = sweep_cell(K, R)

    seed_set = set(attaining)
    visited_orbits = set()
    for s in attaining:
        visited_orbits.add(orbit(s, K))
    orbit_count = len(visited_orbits)
    orbit_sizes = [len(o) for o in visited_orbits]

    assert max_d == 8, f"FAIL Theorem 20: expected max distance 8, got {max_d}"
    assert len(attaining) == 14, f"FAIL Theorem 20: expected 14 attaining seeds, got {len(attaining)}"
    assert orbit_count == 1, f"FAIL Theorem 20: expected 1 free orbit, got {orbit_count}"
    assert all(s == 14 for s in orbit_sizes), "FAIL Theorem 20: orbit size not 14"

    print(f"  max distance:    {max_d}  PASS")
    print(f"  attaining seeds: {len(attaining)}  PASS")
    print(f"  free orbits:     {orbit_count}  PASS")
    print(f"  orbit size:      {orbit_sizes[0]}  PASS")
    print(f"  runtime:         {elapsed:.2f}s")

    results["theorem_20"] = {
        "K": K, "R": R,
        "max_delta2_distance": max_d,
        "attainer_count": len(attaining),
        "orbit_count": orbit_count,
        "orbit_size": 14,
        "attaining_seeds": [list(s) for s in attaining],
        "runtime_seconds": round(elapsed, 2),
        "pass": True
    }

    # --- Theorem 21: K=8, R=7 ---
    print("\nTheorem 21: Z/4Z (K=8, R=7)...")
    K, R = 8, 7
    max_d, attaining, elapsed = sweep_cell(K, R)

    # Check no seeds at distance >= 7
    # (verified implicitly: max_d == 6 means no seed reached 7)
    assert max_d == 6, f"FAIL Theorem 21: expected max distance 6, got {max_d}"
    assert len(attaining) == 9472, \
        f"FAIL Theorem 21: expected 9472 attaining seeds, got {len(attaining)}"

    print(f"  max distance:    {max_d}  PASS")
    print(f"  attaining seeds: {len(attaining)}  PASS")
    print(f"  seeds at d>=7:   0  PASS")
    print(f"  runtime:         {elapsed:.2f}s")

    results["theorem_21"] = {
        "K": K, "R": R,
        "max_delta2_distance": max_d,
        "attainer_count": len(attaining),
        "seeds_at_distance_7_or_more": 0,
        "runtime_seconds": round(elapsed, 2),
        "pass": True
    }

    # K-axis non-monotonicity check
    drop = results["theorem_20"]["max_delta2_distance"] - results["theorem_21"]["max_delta2_distance"]
    assert drop == 2, f"FAIL: expected drop of 2, got {drop}"
    print(f"\nK-axis non-monotonicity (Remark 22): M2(7,7) - M2(8,7) = {drop}  PASS")

    output = {
        "description": "Z/4Z headline and contrast cells under Delta_2 metric",
        "document": "SNS-THEORY-001",
        "theorems": ["Theorem 20", "Theorem 21"],
        "k_axis_drop_at_R7": drop,
        "all_assertions_passed": True,
        "cells": results
    }

    out_path = Path("Z4Z_headline_cells.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput written to {out_path}")

if __name__ == "__main__":
    main()
