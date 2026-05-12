#!/usr/bin/env python3
"""
INDEPENDENT verification of v4.0 §9.0 errata:
the cell (K=4, R=7) over Z/4Z has max d_Δ₂ = 5, NOT 6.

Pure Python, no numpy/numba. Different code path from both:
  - c_param_z4_delta2_exhaust.py (the chain caught with the defect)
  - the third-party Numba verifier that originally flagged it

Construction (matches c_param convention):
  Seed s ∈ (Z/4Z)^K, parity[j] = sum_i m[i] * s[(i - j) mod K]  for j in [0, R)
  Codeword has length K + R.
  A code C(s) achieves d_Δ₂(s) = min over nonzero messages m of Δ₂-weight of codeword.

Δ₂ valuation weight on Z/4Z:  w(0)=0, w(1)=1, w(2)=2, w(3)=1.
"""
import time

W = (0, 1, 2, 1)  # Δ₂ over Z/4Z
MOD = 4

def delta2_weight_vec(vec):
    return sum(W[x % MOD] for x in vec)

def min_distance(seed, K, R):
    """Min Δ₂ distance over all nonzero messages."""
    N = MOD ** K
    min_d = 10**9
    for n in range(1, N):
        # decode message
        m = []
        x = n
        for _ in range(K):
            m.append(x % MOD)
            x //= MOD
        # compute parity
        msg_w = sum(W[v] for v in m)
        parity = [0] * R
        for j in range(R):
            s = 0
            for i in range(K):
                s += m[i] * seed[(i - j) % K]
            parity[j] = s % MOD
        d = msg_w + sum(W[p] for p in parity)
        if d < min_d:
            min_d = d
    return min_d

def sweep(K, R):
    """Find max over seeds of min distance."""
    N_seeds = MOD ** K
    best = 0
    best_seeds = []
    histogram = {}
    for sidx in range(N_seeds):
        s = []
        x = sidx
        for _ in range(K):
            s.append(x % MOD)
            x //= MOD
        d = min_distance(tuple(s), K, R)
        histogram[d] = histogram.get(d, 0) + 1
        if d > best:
            best = d
            best_seeds = [tuple(s)]
        elif d == best:
            best_seeds.append(tuple(s))
    return best, best_seeds, histogram

if __name__ == "__main__":
    print("Independent Z/4Z (K=4, R=7) Δ₂ exhaustive sweep")
    print("=" * 60)
    print(f"Total seeds: {MOD**4}")
    print(f"Messages per seed: {MOD**4 - 1} nonzero")
    t0 = time.time()
    best, best_seeds, hist = sweep(K=4, R=7)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print()
    print(f"max d_Δ₂(K=4, R=7) over Z/4Z = {best}")
    print(f"Histogram of per-seed min-distance: {dict(sorted(hist.items()))}")
    print(f"Number of seeds attaining max: {len(best_seeds)}")
    if len(best_seeds) <= 8:
        print(f"Witness seeds: {best_seeds}")
    print()
    if best == 5:
        print("✓ INDEPENDENT VERIFICATION CONFIRMS v4.0 §9.0 ERRATUM:")
        print("  max = 5, not 6.")
        print("  The c_param chain's prior 6-report was a defect.")
    elif best == 6:
        print("✗ Result disagrees with v4.0 §9.0 erratum (got 6 not 5)")
    else:
        print(f"✗ Unexpected result: {best}")
