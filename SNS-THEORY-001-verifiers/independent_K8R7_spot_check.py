#!/usr/bin/env python3
"""
Independent spot-check of Theorem 12 (SNS-THEORY-001 v4.0 §4A.11)
================================================================
Theorem 12 claims: max d_Δ₃ at (15,8) over Z/9Z = 7, exhaustive over 9^8 seeds,
produced by `c_param_delta3_exhaust.py`.

Why this script exists: the v4.0 draft logs that the analogous Z/4Z verifier
(`c_param_z4_delta2_exhaust.py`) was found by an independent third-party Numba
verifier to overreport d_Δ₂ at the (K=4, R=7) cell -- the local and cluster
sweeps shared a defect. Theorem 12 was produced by the Z/9Z analog of the same
toolchain family, so it deserves an independent spot-check.

This script:
  - uses plain numpy (no numba)
  - builds the generator matrix from scratch each seed (no precomputed support tables)
  - enumerates messages in natural index order (different ordering from
    weight-sorted walk used by the production verifier)
  - early-exits per seed once any message with weight < 8 is found
    (the only operationally relevant question is "does any seed reach d>=8?")

If any sampled seed yields d_Δ₃ >= 8, Theorem 12 is FALSE.
Run: python3 independent_K8R7_spot_check.py
"""
import numpy as np
import time
import random

# Δ₃ weight from SNS-THEORY-001 §2.2
W = np.array([0, 1, 1, 2, 1, 1, 2, 1, 1], dtype=np.uint8)


def build_G(seed, K=8, R=7):
    """G = [I_K | A], A[i,j] = seed[(j-i) mod K] (corrected sign convention)."""
    A = np.zeros((K, R), dtype=np.int64)
    for i in range(K):
        for j in range(R):
            A[i, j] = seed[(j - i) % K]
    return np.hstack([np.eye(K, dtype=np.int64), A])


def find_d_with_early_exit(seed, K=8, threshold=8, batch=2_000_000):
    """Walk the message space in batches; stop as soon as min < threshold.
    Returns (best_so_far, exhausted)."""
    G = build_G(seed, K)
    N = 9 ** K
    best = 9999
    for start in range(0, N, batch):
        end = min(start + batch, N)
        idx = np.arange(start, end, dtype=np.int64)
        M = np.empty((end - start, K), dtype=np.int8)
        for c in range(K):
            M[:, c] = ((idx // (9 ** c)) % 9).astype(np.int8)
        C = (M.astype(np.int64) @ G) % 9
        w = W[C.astype(np.int64)].sum(axis=1)
        if start == 0:
            w[0] = 9999  # exclude zero message
        bm = int(w.min())
        if bm < best:
            best = bm
        if best < threshold:
            return best, False
    return best, True


def main(n_seeds=100, rng_seed=20260507):
    print("Independent spot-check of Theorem 12 (Z/9Z, K=8, R=7)")
    print(f"Sampling {n_seeds} seeds with rng_seed={rng_seed}")
    print()
    random.seed(rng_seed)
    hist = {}
    falsifiers = []
    max_seen = 0
    t0 = time.time()
    for trial in range(1, n_seeds + 1):
        seed = tuple(random.randrange(9) for _ in range(8))
        d, _ = find_d_with_early_exit(seed, threshold=8)
        hist[d] = hist.get(d, 0) + 1
        if d > max_seen:
            max_seen = d
        if d >= 8:
            falsifiers.append((trial, seed, d))
    print(f"Done in {time.time()-t0:.1f}s")
    print()
    print("Distribution of min d_Δ₃:")
    for d in sorted(hist):
        print(f"  d = {d}: {hist[d]} seed(s)")
    print(f"Max d_Δ₃ observed: {max_seen}")
    print()
    if falsifiers:
        print("THEOREM 12 FALSIFIED:")
        for t, s, d in falsifiers:
            print(f"  trial {t}: seed={s}  d_Δ₃ = {d}")
    else:
        print(f"No falsifier in {n_seeds} samples. Theorem 12 survives spot-check.")


if __name__ == "__main__":
    main()
