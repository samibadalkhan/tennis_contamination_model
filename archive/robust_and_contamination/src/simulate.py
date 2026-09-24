"""Point-by-point match simulator — the forecast the spec requires (not iid).

`forecast.p_match` is the analytic iid recursion: correct for reproducing Ingram,
but it "discards burst structure" and cannot carry a contamination rule. This
module simulates a match point by point under the real scoring rules, so a
forecast-time contamination rule (a burst of impaired points) can be injected and
TUNED PER ARM — the mechanism the design hinges on (tennis-contamination.md
"Forecasting"). Within-match dependence also pulls match-win probabilities toward
0.5, which can change calibration on its own (the ablation's top row).

Vectorized across S simulations at once. Contamination: with probability `rate`
one player is impaired over a contiguous run of points (`len_frac` of the match)
at logit severity `sev` — mirroring the estimation-side bursts. rate=0 recovers
the ordinary iid forecast, which the self-tests check against `forecast.p_match`.

Simplification (flagged): tiebreak at 6-6 in every set (no-ad / deciding-set
match-tiebreak variants not yet modeled); first server randomized per sim.

    python -m src.simulate        # self-tests vs the analytic recursion
"""

from __future__ import annotations

import numpy as np

from src import forecast


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def simulate_winprob(pa: float, pb: float, best_of: int, S: int, rng,
                     rate: float = 0.0, sev: float = 0.0, len_frac: float = 0.0) -> float:
    """P(A wins) over S point-by-point simulations. pa/pb = serve-win probs.
    Optional forecast contamination: per sim, prob `rate` one player is impaired
    (serve prob depressed by `sev` in logit) over a contiguous `len_frac` window."""
    need = 3 if best_of == 5 else 2
    approx_pts = int((need + 1.2) * 10 * 6.5)                 # rough match length
    lA, lB = float(_logit(pa)), float(_logit(pb))

    server = (rng.random(S) < 0.5).astype(np.int8)           # 0 = A serves game 1
    pA = np.zeros(S, int); pB = np.zeros(S, int)             # points in current game/TB
    gA = np.zeros(S, int); gB = np.zeros(S, int)             # games in current set
    sA = np.zeros(S, int); sB = np.zeros(S, int)             # sets
    in_tb = np.zeros(S, bool)
    tb_first = np.zeros(S, np.int8)
    pt_in_tb = np.zeros(S, int)
    idx = np.zeros(S, int)                                   # points played (burst clock)
    done = np.zeros(S, bool)

    # contamination assignment
    imp = np.full(S, -1, np.int8)
    if rate > 0:
        hit = rng.random(S) < rate
        imp[hit] = (rng.random(hit.sum()) < 0.5).astype(np.int8)
    L = max(1, int(len_frac * approx_pts))
    b0 = rng.integers(0, max(1, approx_pts - L), S)
    b1 = b0 + L

    for _ in range(approx_pts * 3 + 50):                     # bounded; win-by-2 tails are short
        act = ~done
        if not act.any():
            break
        # effective server this point
        eff = server.copy()
        tb = in_tb & act
        if tb.any():
            eff[tb] = (tb_first[tb] ^ (((pt_in_tb[tb] + 1) // 2) % 2).astype(np.int8))
        base_l = np.where(eff == 0, lA, lB)
        burst = act & (imp == eff) & (idx >= b0) & (idx < b1)
        base_l = np.where(burst, base_l - sev, base_l)
        pwin = 1.0 / (1.0 + np.exp(-base_l))                 # server wins point
        swin = (rng.random(S) < pwin) & act                 # server won this point
        # point goes to the server if swin else to the returner
        a_pt = act & ((eff == 0) & swin | (eff == 1) & ~swin)
        b_pt = act & ~a_pt
        pA += a_pt; pB += b_pt
        idx += act
        pt_in_tb += (in_tb & act)

        # --- resolve completed GAMES (non-tiebreak) ---
        ng = act & ~in_tb
        a_game = ng & (pA >= 4) & (pA - pB >= 2)
        b_game = ng & (pB >= 4) & (pB - pA >= 2)
        gwin = a_game | b_game
        gA += a_game; gB += b_game
        # flip server, reset points on any completed game
        server[gwin] = 1 - server[gwin]
        pA[gwin] = 0; pB[gwin] = 0

        # --- resolve completed TIEBREAKS ---
        a_tb = act & in_tb & (pA >= 7) & (pA - pB >= 2)
        b_tb = act & in_tb & (pB >= 7) & (pB - pA >= 2)
        twin = a_tb | b_tb
        # tiebreak win => that player takes the set 7-6
        sA += a_tb; sB += b_tb
        # next set: first server = receiver of TB's first point = 1 - tb_first
        server[twin] = 1 - tb_first[twin]
        in_tb[twin] = False
        gA[twin] = 0; gB[twin] = 0; pA[twin] = 0; pB[twin] = 0

        # --- resolve completed SETS (by games, not via TB) ---
        aset = gwin & (gA >= 6) & (gA - gB >= 2)
        bset = gwin & (gB >= 6) & (gB - gA >= 2)
        setwin = aset | bset
        sA += aset; sB += bset
        gA[setwin] = 0; gB[setwin] = 0

        # --- enter tiebreak at 6-6 ---
        enter = gwin & (gA == 6) & (gB == 6) & ~setwin
        in_tb[enter] = True
        tb_first[enter] = server[enter]                      # player due to serve starts the TB
        pt_in_tb[enter] = 0
        pA[enter] = 0; pB[enter] = 0

        done = (sA >= need) | (sB >= need)

    return float((sA > sB).mean())


# --- self-tests: no contamination must match the analytic recursion -----------

def _tests():
    rng = np.random.default_rng(0)
    ok = True

    def chk(name, cond, *v):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'ok' if cond else 'FAIL'}] {name}", *v)

    for (pa, pb, bo) in [(0.64, 0.64, 3), (0.68, 0.60, 3), (0.66, 0.62, 5),
                         (0.72, 0.58, 5), (0.60, 0.66, 3)]:
        analytic = forecast.p_match(pa, pb, bo)
        sim = simulate_winprob(pa, pb, bo, S=40000, rng=rng)
        chk(f"sim≈analytic pa={pa} pb={pb} bo{bo}", abs(sim - analytic) < 0.01,
            f"sim={sim:.4f} analytic={analytic:.4f}")
    # contamination on the FAVORITE pulls the match toward 50%
    fav = forecast.p_match(0.70, 0.60, 3)
    con = simulate_winprob(0.70, 0.60, 3, S=40000, rng=rng, rate=0.5, sev=1.5, len_frac=0.3)
    chk("contamination pulls toward 0.5", abs(con - 0.5) < abs(fav - 0.5),
        f"clean={fav:.4f} contaminated={con:.4f}")
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    print("simulate.py self-tests (vs analytic p_match):")
    sys.exit(0 if _tests() else 1)
