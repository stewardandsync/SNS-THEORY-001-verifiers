#!/usr/bin/env python3
"""
Independent sandbox verification of Theorem 13 (SNS-THEORY-001 v4.0 §9.8)

Theorem 13 (Z/9Z, K=7, R=7): max d_Δ₃ = 8.
Published witness seed: s = (0, 0, 1, 4, 1, 3, 6).

This script:
  1. Implements the Δ₃ weight function from §2.2.
  2. Builds the (14, 7) circulant systematic code C(s) for the witness seed.
  3. Verifies the worked-example codeword from §9.8.3 line L296.
  4. Exhaustively enumerates all 9^7 - 1 nonzero messages and confirms
     that the minimum d_Δ₃ for this seed is exactly 8.

Notes / errata caught while reproducing:
  - The document's formula  A[i,j] = s[(i-j) mod K]  does NOT reproduce the
    worked example. The convention that matches §9.8.3 is
        A[i,j] = s[(j-i) mod K]
    The theorems are unaffected; only the textual formula needs an erratum.

  - Independently while reproducing this, we noticed that Theorem 12 (§4A.11)
    already settles the (K=8, R=7) cell over Z/9Z (max d_Δ₃ = 7). The body
    text in §9.8.5 calling it "sweep pending" appears to contradict §4A.11.
"""
import numpy as np
import time

# ------------------------------------------------------------------
# Δ₃ weight function (SNS-THEORY-001 §2.2)
# w(0)=0, w(units)=1, w({3,6})=2
# ------------------------------------------------------------------
W = np.array([0, 1, 1, 2, 1, 1, 2, 1, 1], dtype=np.uint8)


def build_circulant_generator(seed, K):
    """G = [I_K | A]  where  A[i, j] = seed[(j - i) mod K]."""
    A = np.zeros((K, K), dtype=np.int64)
    for i in range(K):
        for j in range(K):
            A[i, j] = seed[(j - i) % K]
    return np.hstack([np.eye(K, dtype=np.int64), A])


def min_d_delta3(seed, K=7):
    """Exhaustive min d_Δ₃ over all nonzero messages in (Z/9Z)^K."""
    G = build_circulant_generator(seed, K)
    N = 9 ** K
    idx = np.arange(N, dtype=np.int64)
    M = np.empty((N, K), dtype=np.int64)
    for c in range(K):
        M[:, c] = (idx // (9 ** c)) % 9
    C = (M @ G) % 9
    weights = W[C.astype(np.int64)].sum(axis=1)
    weights[0] = 9999  # exclude the zero message
    return int(weights.min()), int((weights == weights.min()).sum())


def main():
    print("=" * 70)
    print("Theorem 13 witness verification (Z/9Z, K=7, R=7)")
    print("=" * 70)

    # Step 1: spot-check the §9.8.3 worked-example codeword
    cw = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 3, 6])
    w = int(W[cw].sum())
    print(f"\nWorked-example codeword Δ₃-weight: {w}  (document says 8)")
    assert w == 8

    # Step 2: build the witness code and check e_0 -> documented codeword
    seed = (0, 0, 1, 4, 1, 3, 6)
    G = build_circulant_generator(seed, 7)
    e0 = np.array([1, 0, 0, 0, 0, 0, 0])
    cw_e0 = tuple(int(x) for x in (e0 @ G) % 9)
    print(f"e_0 codeword: {cw_e0}")
    assert cw_e0 == (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 3, 6), "codeword mismatch"

    # Step 3: exhaustive min over all 9^7 - 1 messages
    print(f"\nExhaustive search over 9^7 - 1 = {9**7 - 1:,} nonzero messages...")
    t0 = time.time()
    min_d, n_at_min = min_d_delta3(seed, K=7)
    print(f"  done in {time.time() - t0:.2f}s")
    print(f"  min d_Δ₃ for seed {seed} = {min_d}")
    print(f"  messages attaining the min        = {n_at_min:,}")

    print()
    if min_d == 8:
        print("VERIFIED: Theorem 13 witness reproduces in the sandbox.")
    else:
        print(f"MISMATCH: document claims 8, sandbox got {min_d}")


if __name__ == "__main__":
    main()
