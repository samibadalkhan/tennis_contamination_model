"""Match-win probability from serve-win probabilities (Ingram's iid forecast).

The point -> game -> set -> tiebreak -> match recursion (Klaassen & Magnus 2003 /
O'Malley 2008), assuming points are i.i.d. given the server. This is the forecast
used to *reproduce Ingram* and the iid column of the ablation. The robust arm's
tuned burst forecast (a point-by-point simulator with injected contamination)
will live alongside this later; the spec forbids the iid formula there, but for
reproducing Ingram the iid analytic forecast is exactly the target.

Inputs are the two players' probabilities of winning a point ON THEIR OWN SERVE.

    python -m src.forecast        # run the self-tests
"""

from __future__ import annotations

from collections import defaultdict
from math import comb


def p_game(p: float) -> float:
    """P(server wins a game | point-win prob p). First to 4, win by 2."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    q = 1 - p
    win = sum(comb(3 + k, k) * p ** 4 * q ** k for k in range(3))   # 4-0, 4-1, 4-2
    reach_deuce = comb(6, 3) * p ** 3 * q ** 3                       # 3-3
    win += reach_deuce * p ** 2 / (p ** 2 + q ** 2)                 # win from deuce
    return win


def _server_first_pattern_tiebreak(n: int) -> bool:
    """True if player A serves point n (0-indexed) of a tiebreak, A serving first.
    Pattern: A, B,B, A,A, B,B, ... -> A serves when ((n+1)//2) is even."""
    return ((n + 1) // 2) % 2 == 0


def p_tiebreak(pa: float, pb: float, to: int = 7) -> float:
    """P(A wins a tiebreak | A serves first), pa/pb = each player's serve-win prob."""
    dist = {(0, 0): 1.0}
    win_a = 0.0
    for n in range(0, 300):
        nd = defaultdict(float)
        for (a, b), pr in dist.items():
            if (a >= to or b >= to) and abs(a - b) >= 2:
                win_a += pr if a > b else 0.0
                continue
            if _server_first_pattern_tiebreak(n):                   # A serves
                nd[(a + 1, b)] += pr * pa
                nd[(a, b + 1)] += pr * (1 - pa)
            else:                                                   # B serves
                nd[(a, b + 1)] += pr * pb
                nd[(a + 1, b)] += pr * (1 - pb)
        dist = nd
        if not dist:
            break
    return win_a + sum(pr for (a, b), pr in dist.items() if a > b)   # tail (negligible)


def p_set(pa: float, pb: float, a_serves_first: bool) -> float:
    """P(A wins a set | who serves the first game), pa/pb = serve-win probs.
    First to 6, win by 2; tiebreak at 6-6."""
    hold_a, hold_b = p_game(pa), p_game(pb)
    dist = {(0, 0): 1.0}
    win_a = 0.0
    for n in range(0, 40):
        nd = defaultdict(float)
        for (ga, gb), pr in dist.items():
            if (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2:
                win_a += pr if ga > gb else 0.0
                continue
            if ga == 6 and gb == 6:                                 # tiebreak
                # game 13 (index 12) server: A if a_serves_first == (12 even) == a_serves_first
                tb_a_first = a_serves_first                          # index 12 is even -> same as game 0
                win_a += pr * p_tiebreak(pa if tb_a_first else pb,
                                         pb if tb_a_first else pa)
                continue
            a_serves = (n % 2 == 0) == a_serves_first
            hold = hold_a if a_serves else hold_b
            # server holds -> that server's game count +1
            if a_serves:
                nd[(ga + 1, gb)] += pr * hold
                nd[(ga, gb + 1)] += pr * (1 - hold)
            else:
                nd[(ga, gb + 1)] += pr * hold
                nd[(ga + 1, gb)] += pr * (1 - hold)
        dist = nd
        if not dist:
            break
    return win_a + sum(pr for (ga, gb), pr in dist.items() if ga > gb)


def p_match(pa: float, pb: float, best_of: int = 3) -> float:
    """P(A wins the match). pa/pb = each player's point-win prob on their serve.

    Sets are treated as i.i.d. (Ingram's iid forecast); the set-win prob averages
    over which player serves the first game (unknown/alternating)."""
    set_a = 0.5 * (p_set(pa, pb, True) + p_set(pa, pb, False))
    n = 3 if best_of == 5 else 2                                    # sets needed to win
    # P(A reaches n set wins before B) with iid sets
    return sum(comb(n - 1 + l, l) * set_a ** n * (1 - set_a) ** l for l in range(n))


# --- self-tests ---------------------------------------------------------------

def _tests():
    ok = True

    def chk(name, cond, *vals):
        nonlocal ok
        ok = ok and cond
        print(f"  [{'ok' if cond else 'FAIL'}] {name}", *vals)

    # p_game monotone & known value (server winning ~62% of points holds ~80%)
    chk("p_game(0.5)=0.5", abs(p_game(0.5) - 0.5) < 1e-9, round(p_game(0.5), 4))
    # authoritative hold probs (O'Malley 2008): 0.60->0.736, 0.65->0.830
    chk("p_game(0.60)~0.736", abs(p_game(0.60) - 0.736) < 0.002, round(p_game(0.60), 4))
    chk("p_game(0.65)~0.830", abs(p_game(0.65) - 0.830) < 0.002, round(p_game(0.65), 4))
    chk("p_game monotone", p_game(0.55) < p_game(0.65) < p_game(0.75))
    # symmetry: equal servers -> everything 0.5
    chk("p_tiebreak equal=0.5", abs(p_tiebreak(0.64, 0.64) - 0.5) < 1e-6, round(p_tiebreak(0.64, 0.64), 5))
    chk("p_set equal=0.5", abs(p_set(0.64, 0.64, True) - 0.5) < 1e-6, round(p_set(0.64, 0.64, True), 5))
    chk("p_match equal bo3=0.5", abs(p_match(0.64, 0.64, 3) - 0.5) < 1e-6, round(p_match(0.64, 0.64, 3), 5))
    chk("p_match equal bo5=0.5", abs(p_match(0.64, 0.64, 5) - 0.5) < 1e-6, round(p_match(0.64, 0.64, 5), 5))
    # stronger server wins more; best-of-5 amplifies the edge
    m3 = p_match(0.68, 0.62, 3); m5 = p_match(0.68, 0.62, 5)
    chk("stronger A > 0.5", m3 > 0.5, round(m3, 4))
    chk("bo5 amplifies edge", m5 > m3, round(m5, 4), round(m3, 4))
    # a big serve edge -> lopsided
    chk("big edge lopsided", p_match(0.72, 0.58, 5) > 0.9, round(p_match(0.72, 0.58, 5), 4))
    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    print("forecast.py self-tests:")
    sys.exit(0 if _tests() else 1)
