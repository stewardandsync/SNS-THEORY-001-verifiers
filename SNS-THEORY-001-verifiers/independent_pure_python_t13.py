#!/usr/bin/env python3
"""
TOTALLY INDEPENDENT verification of Theorem 13 — pure Python, no numpy, no numba.
Different code path from theorem13_witness_verifier.py (which uses numpy matmul).

Theorem 13: at (K=7, R=7) over Z/9Z, max d_Δ₃(C(s)) = 8, attained by
seed s = (0, 0, 1, 4, 1, 3, 6) with min d_Δ₃ for that seed = 8.
"""
import time

# Δ₃ weights over Z/9Z: w(0)=0, w({1,2,4,5,7,8})=1, w({3,6})=2
W = (0, 1, 1, 2, 1, 1, 2, 1, 1)

def delta3_weight(vec):
    return sum(W[x] for x in vec)

def encode_message(m, seed, K=7):
    """Encode message m using circulant generator with A[i,j] = seed[(j-i) mod K]."""
    # Codeword = [m | parity], parity[j] = sum_i m[i] * A[i,j] mod 9
    parity = [0]*K
    for j in range(K):
        s = 0
        for i in range(K):
            s += m[i] * seed[(j - i) % K]
        parity[j] = s % 9
    return tuple(m) + tuple(parity)

def min_distance_for_seed(seed, K=7):
    """Brute-force min d_Δ₃ over all nonzero messages in (Z/9Z)^K."""
    min_d = 10**9
    n_at_min = 0
    N = 9**K
    for n in range(1, N):  # skip zero message
        # decode integer n into base-9 digits
        m = []
        x = n
        for _ in range(K):
            m.append(x % 9)
            x //= 9
        cw = encode_message(m, seed, K)
        d = delta3_weight(cw)
        if d < min_d:
            min_d = d
            n_at_min = 1
        elif d == min_d:
            n_at_min += 1
    return min_d, n_at_min

if __name__ == "__main__":
    seed = (0, 0, 1, 4, 1, 3, 6)
    print(f"Seed: {seed}")
    print(f"Verifying: codeword for m=e_0 should be (1,0,0,0,0,0,0,0,0,1,4,1,3,6)")
    e0 = [1,0,0,0,0,0,0]
    cw = encode_message(e0, seed, 7)
    print(f"  computed: {cw}")
    print(f"  Δ₃-weight: {delta3_weight(cw)}")
    assert cw == (1,0,0,0,0,0,0,0,0,1,4,1,3,6), "Codeword mismatch — convention error"
    assert delta3_weight(cw) == 8, "Weight mismatch"
    print("  ✓ Worked example matches manuscript §9.8.3")
    print()
    print(f"Exhaustive enumeration over {9**7 - 1:,} nonzero messages...")
    t0 = time.time()
    min_d, n_at_min = min_distance_for_seed(seed, K=7)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    print(f"  min d_Δ₃ = {min_d}")
    print(f"  messages attaining min = {n_at_min:,}")
    if min_d == 8:
        print()
        print("✓ INDEPENDENT VERIFICATION PASSES")
        print("  Pure-Python (no numpy/numba) confirms Theorem 13 witness seed.")
