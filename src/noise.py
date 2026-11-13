"""Part 2: how many ranks can ~1,120 ten-vote ballots resolve?

The vote counts are not published, so the expected number of votes by
position is a scenario: a power law mu_r = v1 * r**(-alpha), fixed by the
expected votes of No.100 (v100) and the ratio v1/v100, and continued below
No.100 until all N*L votes are used.  Each voter includes restaurant r with
probability mu_r/N (independently across restaurants; with hundreds of
candidates the fixed ballot length changes nothing at this resolution).

Two questions per scenario:
  (a) resolution of one edition: the spread of each restaurant's rank over
      re-drawn panels of the same size (positions with equal votes are
      ordered at random, which is what a tie-break unrelated to the vote
      count amounts to);
  (b) the year-on-year churn that panel noise alone would create, with no
      change in any restaurant: 'renewal' keeps the ballots of the 75% of
      voters who stay (only the renewed quarter differs), 'redraw' draws all
      ballots afresh.  The two bracket what overlapping 18-month windows
      would give.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import lists
from churn import BANDS, band_of

RES = lists.ROOT / "results"
N_VOTERS = 1120
BALLOT = 10
RENEWAL = 0.25

SCENARIOS = {  # name: (v100, v1/v100)
    "L5": (12, 5), "L8": (12, 8), "L12": (12, 12),
    "M5": (20, 5), "M8": (20, 8), "M12": (20, 12),
    "H5": (28, 5), "H8": (28, 8), "H12": (28, 12),
}


def vote_curve(v100: float, ratio: float, n=N_VOTERS, ballot=BALLOT) -> np.ndarray:
    v1 = v100 * ratio
    alpha = np.log(ratio) / np.log(100.0)
    total = n * ballot
    r = np.arange(1, 200001)
    mu = v1 * r ** (-alpha)
    c = np.cumsum(mu)
    R = int(np.searchsorted(c, total)) + 1
    mu = mu[:R].copy()
    mu[-1] -= c[R - 1] - total  # use exactly n*ballot votes
    return mu


def ranks_with_random_ties(v: np.ndarray, rng) -> np.ndarray:
    """Ranks 1..R with equal counts ordered at random; v is (reps, R)."""
    noise = rng.random(v.shape)
    key = v + noise * 0.5  # integer votes: noise only orders equal counts
    order = np.argsort(-key, axis=1)
    ranks = np.empty_like(order)
    rows = np.arange(v.shape[0])[:, None]
    ranks[rows, order] = np.arange(1, v.shape[1] + 1)[None, :]
    return ranks


def tie_stats(v: np.ndarray, rk: np.ndarray, lo: int, hi: int) -> dict:
    """Share of adjacent positions lo..hi whose vote counts are equal, and the
    number of distinct counts among them (per edition, averaged)."""
    shares, distinct = [], []
    for k in range(v.shape[0]):
        order = np.argsort(rk[k])
        seg = v[k, order[lo - 1:hi]]
        shares.append(np.mean(seg[:-1] == seg[1:]))
        distinct.append(len(np.unique(seg)))
    return {"tied_adjacent_share": float(np.mean(shares)), "distinct_counts": float(np.mean(distinct))}


def resolution(mu: np.ndarray, reps: int, rng) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    p = np.clip(mu / N_VOTERS, 0, 1)
    v = rng.binomial(N_VOTERS, p, size=(reps, len(mu)))
    rk = ranks_with_random_ties(v, rng)
    top = rk[:, :100]
    df = pd.DataFrame({
        "expected_rank": np.arange(1, 101),
        "expected_votes": mu[:100],
        "q05": np.quantile(top, 0.05, axis=0),
        "q50": np.quantile(top, 0.50, axis=0),
        "q95": np.quantile(top, 0.95, axis=0),
        "p_top50": (top <= 50).mean(axis=0),
        "p_top100": (top <= 100).mean(axis=0),
    })
    return df, v, rk


def next_edition(v0: np.ndarray, mu: np.ndarray, mode: str, rng) -> np.ndarray:
    p = np.clip(mu / N_VOTERS, 0, 1)
    if mode == "redraw":
        return rng.binomial(N_VOTERS, p, size=v0.shape)
    if mode == "renewal":
        n_new = int(round(RENEWAL * N_VOTERS))
        leave = rng.binomial(v0, RENEWAL)
        new = rng.binomial(n_new, p, size=v0.shape)
        return v0 - leave + new
    raise ValueError(mode)


def churn_stats(rk0: np.ndarray, rk1: np.ndarray) -> dict:
    out = {}
    d_all, band_all, both_all, rho = [], [], [], []
    for k in range(rk0.shape[0]):
        in0 = rk0[k] <= 100
        r0, r1 = rk0[k][in0], rk1[k][in0]
        both = r1 <= 100
        d_all.append(np.abs(r1 - r0))
        band_all.append(np.array([band_of(int(x)) for x in r0]))
        both_all.append(both)
        rho.append(stats.spearmanr(r0[both], r1[both]).statistic)
    d = np.concatenate(d_all); b = np.concatenate(band_all); bo = np.concatenate(both_all)
    for a, z in BANDS:
        lab = f"{a}-{z}"
        m = b == lab
        out[f"median_abs_{lab}"] = float(np.median(d[m & bo]))
        out[f"exit_{lab}"] = float(1 - bo[m].mean())
    out["spearman_survivors"] = float(np.median(rho))
    return out


def main(reps=2000, churn_reps=400, seed=20261106):
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1)   # second panels, kept apart from the main stream
    RES.mkdir(exist_ok=True)
    rows_tie, rows_churn, rows_curve = [], [], []
    res_tables = {}
    for name, (v100, ratio) in SCENARIOS.items():
        mu = vote_curve(v100, ratio)
        df, v, rk = resolution(mu, reps, rng)
        df.insert(0, "scenario", name)
        res_tables[name] = df
        top100_share = mu[:100].sum() / (N_VOTERS * BALLOT)
        # the published list is itself one panel: compare a panel with the
        # expected order and with a second, independent panel
        v2 = rng2.binomial(N_VOTERS, np.clip(mu / N_VOTERS, 0, 1), size=v.shape)
        rk2 = ranks_with_random_ties(v2, rng2)
        top50_vs_expected = float((rk[:, :50] <= 50).sum(axis=1).mean())
        top50_two_panels = float(((rk <= 50) & (rk2 <= 50)).sum(axis=1).mean())
        del v2, rk2
        rows_curve.append({"scenario": name, "v100": v100, "ratio": ratio, "v1": v100 * ratio,
                           "alpha": np.log(ratio) / np.log(100), "restaurants_with_votes": len(mu),
                           "top100_vote_share": top100_share,
                           "v25": mu[24], "v50": mu[49], "v51": mu[50],
                           "top50_kept_vs_expected": top50_vs_expected,
                           "top50_shared_two_panels": top50_two_panels})
        sub = slice(0, min(reps, 300))
        for lo, hi in [(1, 25), (26, 50), (51, 100)]:
            t = tie_stats(v[sub], rk[sub], lo, hi)
            rows_tie.append({"scenario": name, "positions": f"{lo}-{hi}", **t})
        for mode in ("renewal", "redraw"):
            v0 = v[:churn_reps]
            v1_ = next_edition(v0, mu, mode, rng)
            rk1 = ranks_with_random_ties(v1_, rng)
            rows_churn.append({"scenario": name, "mode": mode, **churn_stats(rk[:churn_reps], rk1)})
    # observed churn with the mechanical Best-of-the-Best shift removed
    obs = pd.read_csv(RES / "part1_bands_bobadj.csv").set_index("band")
    t100 = pd.read_csv(RES / "part1_transitions_top100.csv")
    obs_row = {"scenario": "observed", "mode": "2019-2026"}
    for a, z in BANDS:
        lab = f"{a}-{z}"
        obs_row[f"median_abs_{lab}"] = float(obs.loc[lab, "median_abs_change"])
        obs_row[f"exit_{lab}"] = float(obs.loc[lab, "share_exit_top100"])
    obs_row["spearman_survivors"] = float(t100.spearman_survivors.median())
    rows_churn.append(obs_row)

    pd.DataFrame(rows_curve).to_csv(RES / "part2_curves.csv", index=False)
    pd.DataFrame(rows_tie).to_csv(RES / "part2_ties.csv", index=False)
    pd.DataFrame(rows_churn).to_csv(RES / "part2_churn_vs_observed.csv", index=False)
    pd.concat(res_tables.values()).to_csv(RES / "part2_resolution.csv", index=False)

    print(pd.DataFrame(rows_curve).round(3).to_string())
    print(pd.DataFrame(rows_tie).round(3).to_string())
    print(pd.DataFrame(rows_churn).round(3).to_string())
    for name in ("L8", "M8", "H8", "H12"):
        d = res_tables[name]
        print(name, d[d.expected_rank.isin([1, 2, 5, 10, 25, 50, 75, 100])][["expected_rank", "expected_votes", "q05", "q50", "q95", "p_top50", "p_top100"]].round(3).to_string())


if __name__ == "__main__":
    main()
