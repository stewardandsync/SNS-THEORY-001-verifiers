"""
c_param_delta3_exhaust.py
Parametric Δ₃-distance exhaustive verifier for circulant systematic codes over Z/9Z.

Generalises c148_delta3_exhaust.py (which was hardcoded to R=6, MAX_SUPPORT=6)
to arbitrary R (gap = number of parity symbols) and MAX_SUPPORT.

For a (K+R, K) systematic circulant code with seed s in (Z/9Z)^K, this script
exhaustively determines the global maximum Δ₃-distance over all 9^K seeds.

Status codes (from check_range, "Step A"):
    0 = seed has a codeword with Δ₃-weight ≤ R-1   (so d_min ≤ R-1; "boring")
    1 = seed has a codeword with Δ₃-weight = R (none below)   (so d_min = R)
    2 = no codeword with Δ₃-weight ≤ R found by fast-pass   (candidate for d ≥ R+1)

Step B then runs full enumeration on status-2 seeds to confirm or refute d ≥ R+1.

To change the sweep, edit the PARAMETERS block below.

Validation note:
    With (K=8, R=6, MAX_SUPPORT=6) this MUST reproduce the c148 result:
    global_max_d_delta3 = 6, no seed reaches d ≥ 7. Use that as a regression test
    before launching the R=7 sweep.
"""

import itertools
import json
import os
import time

import numpy as np
from numba import njit, prange, get_num_threads


# ============================================================================
# PARAMETERS — edit these for different sweeps
# ============================================================================
K = 8
R = 6                  # gap = R; full code length n = K + R
MAX_SUPPORT = R        # patterns up to this support are precomputed
                       # (msg_wt ≥ support, so support > R can never yield wt ≤ R)
MOD = 9
# ============================================================================

TOTAL_SEEDS = MOD ** K
CHUNK_SIZE = 1_000_000
CHECKPOINT_EVERY = 5_000_000
LOG_PATH = f"c_param_R{R}_S{MAX_SUPPORT}_checkpoints.jsonl"
SUMMARY_PATH = f"c_param_R{R}_S{MAX_SUPPORT}_summary.json"


def precompute_messages(max_support, max_msg_weight):
    """Enumerate messages with support in [2, max_support] and Δ₃-weight ≤ max_msg_weight.
    Returns concatenated arrays across all supports (uniform width = max_support, -1 padded).
    Support-1 messages are NOT included here; they are handled by seed_has_support1_word.
    """
    pos_rows = []
    val_rows = []
    msg_wts = []
    hist = {}
    for support in range(2, max_support + 1):
        for pos in itertools.combinations(range(K), support):
            for vals in itertools.product(range(1, MOD), repeat=support):
                msg_wt = sum(2 if v % 3 == 0 else 1 for v in vals)
                if msg_wt > max_msg_weight:
                    continue
                p = [-1] * max_support
                v = [0] * max_support
                for i, (pp, vv) in enumerate(zip(pos, vals)):
                    p[i] = pp
                    v[i] = vv
                pos_rows.append(p)
                val_rows.append(v)
                msg_wts.append(msg_wt)
                hist[msg_wt] = hist.get(msg_wt, 0) + 1

    if pos_rows:
        pos_arr = np.array(pos_rows, dtype=np.int64)
        val_arr = np.array(val_rows, dtype=np.int64)
        wt_arr = np.array(msg_wts, dtype=np.int64)
    else:
        # Edge case: max_support < 2 ⇒ no support≥2 patterns
        width = max(max_support, 1)
        pos_arr = np.full((0, width), -1, dtype=np.int64)
        val_arr = np.zeros((0, width), dtype=np.int64)
        wt_arr = np.zeros((0,), dtype=np.int64)

    # Bookkeeping for support-1 patterns (handled by seed_has_support1_word):
    # K positions × {1,2,4,5,7,8} (six units, weight 1) → K*6 weight-1 patterns
    # K positions × {3,6}        (two non-units, weight 2) → K*2 weight-2 patterns
    hist[1] = hist.get(1, 0) + K * 6
    hist[2] = hist.get(2, 0) + K * 2
    total = len(msg_wts) + K * 8
    return pos_arr, val_arr, wt_arr, hist, total


@njit
def delta_weight_scalar(x):
    x %= MOD
    if x == 0:
        return 0
    if x % 3 == 0:
        return 2
    return 1


@njit
def decode_seed(seed_idx):
    seed = np.empty(K, dtype=np.int64)
    rem = seed_idx
    for i in range(K):
        seed[i] = rem % MOD
        rem //= MOD
    return seed


@njit
def seed_has_support1_word(seed, threshold):
    """Check support-1 messages: msg = u·e_i for u in {1,…,8}.
    Two cases handled: u a unit (weight 1) and u in {3,6} (weight 2)."""
    for i in range(K):
        row_wt = 0
        nonmultiples = 0
        for j in range(R):
            a = seed[(i - j) % K]
            row_wt += delta_weight_scalar(a)
            if a % 3 != 0:
                nonmultiples += 1
        # u a unit (msg_wt=1): codeword weight = 1 + row_wt
        if 1 + row_wt <= threshold:
            return True
        # u = 3 (msg_wt=2): parities = 3·seed[…], delta_weight = 2 if seed%3≠0 else 0
        if 2 + 2 * nonmultiples <= threshold:
            return True
    return False


@njit
def seed_has_patterns(seed, msg_pos, msg_vals, msg_wts, threshold):
    """Check precomputed support≥2 message patterns. Returns True if any yields a
    codeword of Δ₃-weight ≤ threshold."""
    width = msg_pos.shape[1]
    parities = np.zeros(R, dtype=np.int64)
    for mi in range(msg_wts.shape[0]):
        if msg_wts[mi] > threshold:
            continue
        for j in range(R):
            parities[j] = 0
        for t in range(width):
            pos = msg_pos[mi, t]
            if pos < 0:
                break
            val = msg_vals[mi, t]
            for j in range(R):
                parities[j] += val * seed[(pos - j) % K]

        total = msg_wts[mi]
        ok = True
        for j in range(R):
            total += delta_weight_scalar(parities[j])
            if total > threshold:
                ok = False
                break
        if ok:  # total ≤ threshold
            return True
    return False


@njit
def seed_has_small_word_idx(seed_idx, all_pos, all_vals, all_wts, threshold):
    seed = decode_seed(seed_idx)
    if seed_has_support1_word(seed, threshold):
        return True
    if seed_has_patterns(seed, all_pos, all_vals, all_wts, threshold):
        return True
    return False


@njit(parallel=True)
def check_range(start, end, all_pos, all_vals, all_wts):
    n = end - start
    status = np.zeros(n, dtype=np.uint8)
    for off in prange(n):
        seed_idx = start + off
        if seed_has_small_word_idx(seed_idx, all_pos, all_vals, all_wts, R - 1):
            status[off] = 0
        else:
            if seed_has_small_word_idx(seed_idx, all_pos, all_vals, all_wts, R):
                status[off] = 1
            else:
                status[off] = 2
    return status


@njit
def witness_patterns(seed, msg_pos, msg_vals, msg_wts, threshold, witness_msg, witness_parity):
    """Like seed_has_patterns, but writes a witness on success and returns the total weight."""
    width = msg_pos.shape[1]
    parities = np.zeros(R, dtype=np.int64)
    for mi in range(msg_wts.shape[0]):
        if msg_wts[mi] > threshold:
            continue
        for j in range(R):
            parities[j] = 0
        for t in range(width):
            pos = msg_pos[mi, t]
            if pos < 0:
                break
            val = msg_vals[mi, t]
            for j in range(R):
                parities[j] += val * seed[(pos - j) % K]

        for j in range(R):
            parities[j] %= MOD

        total = msg_wts[mi]
        for j in range(R):
            total += delta_weight_scalar(parities[j])

        if total <= threshold:
            for i in range(K):
                witness_msg[i] = 0
            for t in range(width):
                pos = msg_pos[mi, t]
                if pos < 0:
                    break
                witness_msg[pos] = msg_vals[mi, t]
            for j in range(R):
                witness_parity[j] = parities[j]
            return total
    return 99


@njit
def find_atmostR_witness(seed_idx, all_pos, all_vals, all_wts):
    """Find a codeword of Δ₃-weight ≤ R for the seed; returns (d, msg, parity).
    If none exists, returns d=99 with zero msg/parity."""
    seed = decode_seed(seed_idx)
    witness_msg = np.zeros(K, dtype=np.int64)
    witness_parity = np.zeros(R, dtype=np.int64)

    # Support-1 first (matches seed_has_support1_word)
    for i in range(K):
        row_wt = 0
        nonmultiples = 0
        for j in range(R):
            a = seed[(i - j) % K]
            row_wt += delta_weight_scalar(a)
            if a % 3 != 0:
                nonmultiples += 1
        if 1 + row_wt <= R:
            witness_msg[i] = 1
            for j in range(R):
                witness_parity[j] = seed[(i - j) % K] % MOD
            return 1 + row_wt, witness_msg, witness_parity
        if 2 + 2 * nonmultiples <= R:
            witness_msg[i] = 3
            for j in range(R):
                witness_parity[j] = (3 * seed[(i - j) % K]) % MOD
            return 2 + 2 * nonmultiples, witness_msg, witness_parity

    d = witness_patterns(seed, all_pos, all_vals, all_wts, R, witness_msg, witness_parity)
    return d, witness_msg, witness_parity


@njit
def full_min_distance(seed_idx):
    """Full enumeration over all 9^K - 1 nonzero messages.
    Returns (best_d, best_msg, best_parity). Early-exits as soon as best ≤ R."""
    seed = decode_seed(seed_idx)
    best = 99
    best_msg = np.zeros(K, dtype=np.int64)
    best_parity = np.zeros(R, dtype=np.int64)
    msg = np.zeros(K, dtype=np.int64)
    parities = np.zeros(R, dtype=np.int64)

    for msg_idx in range(1, TOTAL_SEEDS):
        rem = msg_idx
        msg_wt = 0
        for i in range(K):
            v = rem % MOD
            rem //= MOD
            msg[i] = v
            msg_wt += delta_weight_scalar(v)
            if msg_wt >= best:
                break
        if msg_wt >= best:
            continue

        for j in range(R):
            parities[j] = 0
        for i in range(K):
            v = msg[i]
            if v == 0:
                continue
            for j in range(R):
                parities[j] += v * seed[(i - j) % K]

        total = msg_wt
        for j in range(R):
            parities[j] %= MOD
            total += delta_weight_scalar(parities[j])

        if total < best:
            best = total
            for i in range(K):
                best_msg[i] = msg[i]
            for j in range(R):
                best_parity[j] = parities[j]
            if best <= R:
                break
    return best, best_msg, best_parity


def format_vec(vec):
    return "[" + ", ".join(str(int(x)) for x in vec) + "]"


def seed_vec_from_idx(seed_idx):
    return [int(x) for x in decode_seed(int(seed_idx))]


def append_log(record):
    with open(LOG_PATH, "a", encoding="ascii") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)
    t_script = time.time()
    print("Parametric Δ₃ exhaustive verifier")
    print(f"  K={K}, R={R} (gap), n={K+R}, MOD={MOD}")
    print(f"  MAX_SUPPORT={MAX_SUPPORT}, max_msg_weight=R={R}")
    print(f"  TOTAL_SEEDS={TOTAL_SEEDS:,}")
    print(f"  Log: {LOG_PATH}")
    print(f"  Summary: {SUMMARY_PATH}")

    print(f"Precomputing messages (support 2..{MAX_SUPPORT}, msg_wt ≤ {R}) ...")
    all_pos, all_vals, all_wts, hist, total_patterns = precompute_messages(MAX_SUPPORT, R)
    arrays = (all_pos, all_vals, all_wts)

    print(f"  Fast-pass patterns (support ≥ 2): {all_wts.shape[0]:,}")
    print(f"  Total patterns including support-1: {total_patterns:,}")
    print(f"  Message-weight histogram: {dict(sorted(hist.items()))}")
    print(f"  Numba threads: {get_num_threads()}")

    print("Compiling kernels ...")
    check_range(0, 1, *arrays)
    find_atmostR_witness(1, *arrays)
    full_min_distance(1)

    processed = 0
    count_ge_R = 0       # status >= 1: d ≥ R
    candidates = []      # status == 2: candidate seeds with potential d ≥ R+1
    first_dR = None
    last_checkpoint = 0
    t0 = time.time()

    for chunk_start in range(0, TOTAL_SEEDS, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, TOTAL_SEEDS)
        status = check_range(chunk_start, chunk_end, *arrays)
        local_ge_R = np.flatnonzero(status >= 1)
        local_ge_Rp1 = np.flatnonzero(status == 2)

        count_ge_R += int(local_ge_R.shape[0])
        if local_ge_Rp1.shape[0]:
            candidates.extend((local_ge_Rp1 + chunk_start).astype(np.int64).tolist())

        if first_dR is None and local_ge_R.shape[0]:
            seed_idx = int(local_ge_R[0] + chunk_start)
            d, msg, parity = find_atmostR_witness(seed_idx, *arrays)
            first_dR = {
                "seed_idx": seed_idx,
                "seed": seed_vec_from_idx(seed_idx),
                "d": int(d),
                "witness_msg": [int(x) for x in msg],
                "witness_parity": [int(x) for x in parity],
                "witness_codeword": [int(x) for x in np.concatenate((msg, parity))],
            }

        processed = chunk_end
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed else 0.0
        print(
            f"Step A through {processed:,}/{TOTAL_SEEDS:,}; "
            f"d>={R}={count_ge_R:,}; candidates(d>={R+1})={len(candidates):,}; "
            f"{rate:,.0f} seeds/s; elapsed={elapsed:.1f}s",
            flush=True,
        )

        if processed - last_checkpoint >= CHECKPOINT_EVERY or processed == TOTAL_SEEDS:
            last_checkpoint = processed
            append_log(
                {
                    "event": "checkpoint",
                    "R": int(R),
                    "MAX_SUPPORT": int(MAX_SUPPORT),
                    "processed": int(processed),
                    "count_ge_R": int(count_ge_R),
                    "candidate_count_ge_Rp1": int(len(candidates)),
                    "candidates_ge_Rp1": [int(x) for x in candidates],
                    "first_dR": first_dR,
                    "elapsed_step_a": elapsed,
                }
            )

    print("Step A complete.")
    print(f"Step B candidates: {len(candidates):,}")

    best_records = []
    count_ge_Rp1 = 0
    global_best = R if count_ge_R else (R - 1)
    fallback_start = time.time()
    for n_idx, seed_idx in enumerate(candidates, start=1):
        d, msg, parity = full_min_distance(seed_idx)
        seed = decode_seed(seed_idx)
        if int(d) >= R + 1:
            count_ge_Rp1 += 1
        if int(d) > global_best:
            global_best = int(d)
            best_records = []
        if int(d) == global_best:
            best_records.append(
                {
                    "seed_idx": int(seed_idx),
                    "seed": [int(x) for x in seed],
                    "d": int(d),
                    "witness_msg": [int(x) for x in msg],
                    "witness_parity": [int(x) for x in parity],
                    "witness_codeword": [int(x) for x in np.concatenate((msg, parity))],
                }
            )
        print(
            f"FULL {n_idx}/{len(candidates)} seed_idx={seed_idx} "
            f"seed={format_vec(seed)} d={int(d)} "
            f"msg={format_vec(msg)} parity={format_vec(parity)}",
            flush=True,
        )

    fallback_elapsed = time.time() - fallback_start
    if global_best == R and not best_records and first_dR is not None:
        best_records = [first_dR]

    summary = {
        "params": {"K": K, "R": R, "MAX_SUPPORT": MAX_SUPPORT, "MOD": MOD},
        "total_seeds_processed": int(processed),
        "step_b_candidates": int(len(candidates)),
        "global_max_d_delta3": int(global_best),
        "count_d_ge_R": int(count_ge_R),
        "count_d_ge_Rp1": int(count_ge_Rp1),
        "best_records": best_records[:20],
        "step_b_runtime_seconds": fallback_elapsed,
        "total_runtime_seconds": time.time() - t_script,
        "message_patterns": {
            "total": int(total_patterns),
            "by_weight": {str(k): int(v) for k, v in sorted(hist.items())},
        },
    }
    with open(SUMMARY_PATH, "w", encoding="ascii") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("SUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
