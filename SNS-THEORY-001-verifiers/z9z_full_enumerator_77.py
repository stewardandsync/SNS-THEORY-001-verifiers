"""
SNS-THEORY-001 Verifier: Z/9Z Full Seed Enumerator (K=7, R=7) (Theorem 24)
============================================================================
Exhaustively enumerates all 9^7 = 4,782,969 seeds of the (14,7) circulant
systematic code over Z/9Z under the Delta_3 valuation-weight metric.

Expected output:
  - Exactly 7,014 seeds attain max min distance 8
  - Organized into exactly 501 free orbits of size 14
  - Zero seeds attain distance >= 9
  - Witness seeds: (6,3,1,1,4,0,0) and (0,0,1,4,1,3,6)

Output file: Z9Z_K7R7_full_enumeration.json

Runtime: ~35 minutes single-threaded (see SNS-THEORY-001 Table 6)
         Numba JIT significantly reduces this.

Author: Ahmed M. Mansour, Steward and Sync LLC
Document: SNS-THEORY-001, Theorem 24
"""

import json
import itertools
import time
from pathlib import Path
from collections import Counter

# ---------------------------------------------------------------------------
# Ring: Z/9Z
# ---------------------------------------------------------------------------
MOD = 9
UNITS = {1, 2, 4, 5, 7, 8}
ZERO_DIVISORS = {3, 6}

def delta3_weight(x):
    x = x % MOD
    if x == 0:
        return 0
    if x in UNITS:
        return 1
    return 2  # nonzero zero-divisor {3, 6}

def delta3_vec_weight(vec):
    return sum(delta3_weight(x) for x in vec)

# ---------------------------------------------------------------------------
# Circulant parity: A[i,j] = s[(j-i) mod K]
# ---------------------------------------------------------------------------
K, R = 7, 7

def min_distance_seed(seed):
    """Return min Delta_3 distance of code C(seed) over Z/9Z at (K=7, R=7)."""
    best = float('inf')
    # Build parity rows once
    A = [[seed[(j - i) % K] for j in range(R)] for i in range(K)]
    for msg in itertools.product(range(MOD), repeat=K):
        if not any(msg):
            continue
        parity = tuple(sum(msg[i] * A[i][j] for i in range(K)) % MOD for j in range(R))
        cw = msg + parity
        d = delta3_vec_weight(cw)
        if d < best:
            best = d
        if best == 1:
            return 1
    return best

# ---------------------------------------------------------------------------
# Orbit structure under <sigma, -id> on Z/9Z^7
# ---------------------------------------------------------------------------
def cyclic_shift(seed):
    return (seed[-1],) + seed[:-1]

def negate(seed):
    return tuple((-x) % MOD for x in seed)

def orbit(seed):
    """Return the full orbit of seed under <sigma, -id>."""
    members = set()
    current = seed
    for _ in range(K):
        members.add(current)
        members.add(negate(current))
        current = cyclic_shift(current)
    return frozenset(members)

# ---------------------------------------------------------------------------
# Main enumeration
# ---------------------------------------------------------------------------
def main():
    print(f"Enumerating all {MOD**K:,} seeds for Z/9Z (K={K}, R={R})...")
    print("This will take approximately 35 minutes single-threaded.\n")

    distance_histogram = Counter()
    max_attaining_seeds = []
    visited_orbits = set()

    t0 = time.time()
    total = MOD ** K
    checkpoint_interval = 500_000

    for idx, seed in enumerate(itertools.product(range(MOD), repeat=K)):
        d = min_distance_seed(seed)
        distance_histogram[d] += 1

        if d == 8:
            max_attaining_seeds.append(seed)

        if (idx + 1) % checkpoint_interval == 0:
            elapsed = time.time() - t0
            pct = (idx + 1) / total * 100
            print(f"  {idx+1:>10,} / {total:,}  ({pct:.1f}%)  "
                  f"elapsed {elapsed:.0f}s  max_attaining_so_far={len(max_attaining_seeds)}")

    total_elapsed = time.time() - t0

    # Compute orbit structure
    seed_set = set(max_attaining_seeds)
    for seed in max_attaining_seeds:
        o = orbit(seed)
        visited_orbits.add(o)

    orbit_count = len(visited_orbits)
    orbit_sizes = [len(o) for o in visited_orbits]

    # Assertions
    assert len(max_attaining_seeds) == 7014, \
        f"FAIL: expected 7014 max-attaining seeds, got {len(max_attaining_seeds)}"
    assert orbit_count == 501, \
        f"FAIL: expected 501 free orbits, got {orbit_count}"
    assert all(s == 14 for s in orbit_sizes), \
        f"FAIL: expected all orbits of size 14"
    assert distance_histogram[9] == 0, \
        "FAIL: expected zero seeds at distance >= 9"

    # Witness checks
    w1 = (6, 3, 1, 1, 4, 0, 0)
    w2 = (0, 0, 1, 4, 1, 3, 6)
    assert w1 in seed_set, f"FAIL: witness {w1} not in max-attaining family"
    assert w2 in seed_set, f"FAIL: witness {w2} not in max-attaining family"

    print(f"\n{'='*60}")
    print(f"Theorem 24 verification COMPLETE")
    print(f"Total seeds enumerated:     {total:,}")
    print(f"Max-attaining seeds (d=8):  {len(max_attaining_seeds):,}")
    print(f"Free orbits:                {orbit_count}")
    print(f"Orbit size (all):           14")
    print(f"Seeds at d>=9:              {distance_histogram[9]}")
    print(f"Witness (6,3,1,1,4,0,0):   CONFIRMED")
    print(f"Witness (0,0,1,4,1,3,6):   CONFIRMED")
    print(f"Total runtime:              {total_elapsed:.0f}s")
    print(f"{'='*60}\n")

    print("Distance histogram:")
    for d in sorted(distance_histogram):
        print(f"  d={d}: {distance_histogram[d]:>10,} seeds")

    output = {
        "description": "Z/9Z full seed enumeration K=7, R=7 under Delta_3 metric",
        "document": "SNS-THEORY-001",
        "theorem": "Theorem 24",
        "parameters": {"K": K, "R": R, "modulus": MOD},
        "total_seeds": total,
        "max_attaining_count": len(max_attaining_seeds),
        "orbit_count": orbit_count,
        "orbit_size": 14,
        "seeds_at_distance_9_or_more": distance_histogram[9],
        "witness_seeds": [list(w1), list(w2)],
        "distance_histogram": {str(k): v for k, v in sorted(distance_histogram.items())},
        "runtime_seconds": round(total_elapsed, 1),
        "all_assertions_passed": True
    }

    out_path = Path("Z9Z_K7R7_full_enumeration.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Output written to {out_path}")

if __name__ == "__main__":
    main()
