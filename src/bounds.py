"""Limits on the resolution of the list that hold under any vote curve.

(i) Integer arithmetic, no model.  Between positions a and b the vote counts
    are integers v_a >= ... >= v_b, so they can fall by at most v_a - v_b
    steps: at least (b - a) - (v_a - v_b) of the b - a adjacent pairs are
    tied.

(ii) Panel noise, any vote curve.  Voters choose independently, as in Part 2,
    so the count of a restaurant with expected count m is Binomial(N, m / N).
    Expected counts are taken to follow the published order (they do not
    increase down the list), which is what keeping that order means.
    A fresh panel reverses two restaurants with expected counts m_a > m_b
    with probability q = P(X_b > X_a) + P(X_b = X_a) / 2 (ties broken at
    random).  For an adjacent pair to be kept in order with probability at
    least 1 - alpha, m_a must exceed the smallest level g(m_b) with
    q(g, m_b) = alpha.  Since q increases with m_b, r resolved pairs below
    position a need expected counts of at least c_1 < ... < c_r, the chain
    c_0 = 0, c_i = g(c_{i-1}), and the a - 1 restaurants above position a
    need at least c_r each.  All counts add up to at most N L votes, so
    r is at most the largest value with (a - 1) c_r + c_1 + ... + c_r <= N L.
    The bound is generous: it gives every vote to the listed restaurants and
    lets the lowest of them have none.  A fixed-length ballot makes counts
    negatively correlated and adjacent pairs harder, not easier, to resolve.

(iii) Voters grouped by region.  With 40 voters per region, the count of a
    restaurant whose votes come mostly from its own region is less variable
    than a binomial count.  The same chain is computed when a share h of each
    restaurant's expected votes comes from the 40 voters of its region (at
    most one vote from each) and the rest from the other 1,080 voters, each
    part binomial.  This shows how far that grouping can loosen the limit.

Outputs: results/bounds.json, results/bounds_resolution.csv,
report/table_bounds.tex.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

import noise

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_VOTERS = noise.N_VOTERS          # 1,120
BALLOT = noise.BALLOT              # 10
TOTAL_VOTES = N_VOTERS * BALLOT    # 11,200
HOME_VOTERS = 40                   # voters per region
HOME_SHARES = (0.6, 1.0)           # share of expected votes from the home region
ALPHAS = (0.05, 0.10, 0.25)
RANGES = [(1, 10), (11, 25), (26, 50), (51, 100), (1, 100)]


def tie_floor(gap_votes: int, n_pairs: int) -> int:
    """Minimum number of tied adjacent pairs among n_pairs + 1 consecutive
    positions whose top and bottom counts differ by gap_votes."""
    return max(n_pairs - int(gap_votes), 0)


@lru_cache(maxsize=None)
def _pmf(m: float, h: float | None = None) -> np.ndarray:
    """Distribution of the count (0..N) of a restaurant expected to receive m
    votes: binomial over all voters, or, with a home share h, the sum of a
    binomial over the 40 voters of its region and one over the others."""
    m = min(m, float(N_VOTERS))
    if h is None:
        return stats.binom.pmf(np.arange(N_VOTERS + 1), N_VOTERS, m / N_VOTERS)
    rest = N_VOTERS - HOME_VOTERS
    home = max(min(h * m, float(HOME_VOTERS)), m - rest)
    p_home = stats.binom.pmf(np.arange(HOME_VOTERS + 1), HOME_VOTERS, home / HOME_VOTERS)
    p_rest = stats.binom.pmf(np.arange(rest + 1), rest, (m - home) / rest)
    return np.convolve(p_home, p_rest)


def reversal_prob(m_a: float, m_b: float, h: float | None = None) -> float:
    """P(X_b > X_a) + P(X_b = X_a) / 2 for independent counts."""
    pa, pb = _pmf(round(m_a, 9), h), _pmf(round(m_b, 9), h)
    cdf_b = np.cumsum(pb)
    greater = float(np.sum(pa * (1.0 - cdf_b)))        # X_b > X_a
    equal = float(np.sum(pa * pb))
    return greater + 0.5 * equal


def next_level(c: float, alpha: float, h: float | None = None) -> float:
    """Smallest expected count kept above a restaurant with expected count c
    with reversal probability alpha (infinite if no count up to N will do)."""
    f = lambda m: reversal_prob(m, c, h) - alpha
    if c >= N_VOTERS or f(float(N_VOTERS)) > 0:
        return float("inf")
    return float(optimize.brentq(f, c + 1e-9, float(N_VOTERS), xtol=1e-7))


def chain(alpha: float, n: int, h: float | None = None) -> np.ndarray:
    c = [0.0]
    for _ in range(n):
        nxt = next_level(c[-1], alpha, h) if np.isfinite(c[-1]) else float("inf")
        c.append(nxt)
    return np.array(c)


def max_resolved(a: int, b: int, alpha: float, levels: np.ndarray, total: float = TOTAL_VOTES) -> int:
    """Most adjacent pairs among positions a..b that any vote curve can keep
    in order with probability 1 - alpha, given total votes."""
    n_pairs = b - a
    best = 0
    for r in range(min(n_pairs, len(levels) - 1) + 1):   # levels beyond the chain count as unaffordable
        above = (a - 1) * levels[r] if a > 1 else 0.0
        cost = levels[1:r + 1].sum() + above
        if np.isfinite(cost) and cost <= total:
            best = r
    return best


def main():
    RES.mkdir(exist_ok=True)
    rows = []
    levels_out = {}
    for alpha in ALPHAS:
        lv = chain(alpha, 100)
        levels_out[str(alpha)] = [round(float(x), 3) for x in lv[:13]]
        for a, b in RANGES:
            rows.append({"alpha": alpha, "from": a, "to": b, "pairs": b - a,
                         "max_resolved": max_resolved(a, b, alpha, lv)})
    res = pd.DataFrame(rows)
    # voters grouped by region (iii): how far can grouping loosen the limit?
    strat = []
    for h in HOME_SHARES:
        lv = chain(0.05, 100, h)
        for a, b in RANGES:
            strat.append({"alpha": 0.05, "home_share": h, "from": a, "to": b, "pairs": b - a,
                          "max_resolved": max_resolved(a, b, 0.05, lv)})
    strat = pd.DataFrame(strat)
    pd.concat([res.assign(home_share="none"), strat], ignore_index=True).to_csv(
        RES / "bounds_resolution.csv", index=False)

    # expected counts at Nos.51 and 100, and at Nos.26 and 50, in the nine
    # churn-constrained scenarios of Part 2
    gaps = []
    for name, (v100, ratio) in noise.SCENARIOS.items():
        mu = noise.vote_curve(v100, ratio)
        gaps.append({"scenario": name, "v51_minus_v100": float(mu[50] - mu[99]),
                     "v26_minus_v50": float(mu[25] - mu[49]), "v51": float(mu[50]), "v100": float(mu[99])})
    gaps = pd.DataFrame(gaps)
    out = {
        "total_votes": TOTAL_VOTES,
        "max_count_at_51": TOTAL_VOTES / 51,
        "levels": levels_out,
        "resolution": rows,
        "resolution_by_home_share": strat.to_dict("records"),
        "scenario_gaps": gaps.round(3).to_dict("records"),
        "scenario_gap_51_100_range": [float(gaps.v51_minus_v100.min()), float(gaps.v51_minus_v100.max())],
        "scenario_gap_26_50_range": [float(gaps.v26_minus_v50.min()), float(gaps.v26_minus_v50.max())],
        "tie_floor_51_100_at_scenario_gaps": [tie_floor(int(np.ceil(gaps.v51_minus_v100.max())), 49),
                                               tie_floor(int(np.floor(gaps.v51_minus_v100.min())), 49)],
    }
    (RES / "bounds.json").write_text(json.dumps(out, indent=2))
    write_table(res)
    print(res.pivot(index=["from", "to", "pairs"], columns="alpha", values="max_resolved"))
    print(strat.pivot(index=["from", "to", "pairs"], columns="home_share", values="max_resolved"))
    print(gaps.round(1).to_string(index=False))


def write_table(res: pd.DataFrame):
    piv = res.pivot(index=["from", "to", "pairs"], columns="alpha", values="max_resolved").reset_index()
    order = {rng: i for i, rng in enumerate(RANGES)}
    piv = piv.sort_values(by=["from", "to"], key=None)
    piv = piv.iloc[sorted(range(len(piv)), key=lambda i: order[(int(piv.iloc[i]["from"]), int(piv.iloc[i]["to"]))])]
    lines = []
    for _, r in piv.iterrows():
        label = f"{int(r['from'])}--{int(r['to'])}"
        vals = " & ".join(str(int(r[a])) for a in ALPHAS)
        lines.append(f"{label} & {int(r['pairs'])} & {vals} \\\\")
    (ROOT / "report" / "table_bounds.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
