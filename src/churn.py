"""Part 1: what the published lists themselves say about rank stability.

Everything here uses only the published ranks (no model).  Outputs go to
results/part1_*.csv and results/part1_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import lists

RES = lists.ROOT / "results"
BANDS = [(1, 10), (11, 25), (26, 50), (51, 75), (76, 100)]
# Best of the Best: from 2019 every former No.1 is ineligible.  The 2019
# edition is the first to which the rule applied.
BOB_FIRST_YEAR = 2019


def band_of(r: int) -> str:
    for a, b in BANDS:
        if a <= r <= b:
            return f"{a}-{b}"
    return ">100"


def consecutive(years):
    return list(zip(years[:-1], years[1:]))


def pairs_top(p: pd.DataFrame, y0: int, y1: int, top: int) -> pd.DataFrame:
    a = p[(p.year == y0) & (p["rank"] <= top)][["rid", "name", "country", "rank"]]
    b = p[(p.year == y1)][["rid", "rank"]].rename(columns={"rank": "rank1"})
    m = a.merge(b, on="rid", how="left").rename(columns={"rank": "rank0"})
    m["y0"], m["y1"] = y0, y1
    m["in_both"] = m["rank1"].notna() & (m["rank1"] <= top)
    return m


def transition_summary(p: pd.DataFrame, years, top: int) -> pd.DataFrame:
    rows = []
    for y0, y1 in consecutive(years):
        m = pairs_top(p, y0, y1, top)
        both = m[m.in_both]
        d = (both.rank1 - both.rank0).abs()
        rho = stats.spearmanr(both.rank0, both.rank1).statistic if len(both) > 2 else np.nan
        top10_0 = set(p[(p.year == y0) & (p["rank"] <= 10)].rid)
        top10_1 = set(p[(p.year == y1) & (p["rank"] <= 10)].rid)
        rows.append({
            "y0": y0, "y1": y1, "top": top, "n_both": len(both),
            "exits": int((~m.in_both).sum()),
            "spearman_survivors": rho,
            "median_abs_change": float(d.median()),
            "mean_abs_change": float(d.mean()),
            "share_moved_10plus": float((d >= 10).mean()),
            "top10_retained": len(top10_0 & top10_1),
            "no1_previous_rank": _prev_rank_of_no1(p, y0, y1),
        })
    return pd.DataFrame(rows)


def _prev_rank_of_no1(p, y0, y1):
    no1 = p[(p.year == y1) & (p["rank"] == 1)].rid.iloc[0]
    r = p[(p.year == y0) & (p.rid == no1)]["rank"]
    return int(r.iloc[0]) if len(r) else None


def band_summary(p: pd.DataFrame, years, bob_adjust: bool = False) -> pd.DataFrame:
    """Rank changes of restaurants present in two consecutive top-100 lists,
    by the band they occupied in the first year (pooled over transitions).

    With bob_adjust, the previous No.1 (ineligible by rule from 2019) is left
    out and every other restaurant is compared with its mechanical position
    one place higher, so that the statistics compare like with like with a
    model in which no restaurant is retired."""
    rows = []
    allm = []
    for y0, y1 in consecutive(years):
        m = pairs_top(p, y0, y1, 100)
        if bob_adjust and y1 >= BOB_FIRST_YEAR:
            m = m[m.rank0 > 1].copy()
            m["rank0"] = m["rank0"] - 1
        m["band"] = m.rank0.astype(int).map(band_of)
        allm.append(m)
    m = pd.concat(allm, ignore_index=True)
    for (a, b) in BANDS:
        lab = f"{a}-{b}"
        g = m[m.band == lab]
        both = g[g.in_both]
        d = (both.rank1 - both.rank0).abs()
        rows.append({
            "band": lab, "n_start": len(g), "n_both": len(both),
            "share_exit_top100": float(1 - g.in_both.mean()),
            "median_abs_change": float(d.median()),
            "q75_abs_change": float(d.quantile(0.75)),
            "q90_abs_change": float(d.quantile(0.90)),
            "share_moved_10plus": float((d >= 10).mean()),
            "share_moved_25plus": float((d >= 25).mean()),
        })
    return pd.DataFrame(rows), m


def half_mobility(m: pd.DataFrame) -> pd.DataFrame:
    """Where the restaurants of each half of the list are one edition later."""
    def where(r):
        if pd.isna(r) or r > 100:
            return "out of top 100"
        return "1-50" if r <= 50 else "51-100"
    m = m.copy()
    m["half0"] = np.where(m.rank0 <= 50, "1-50", "51-100")
    m["dest"] = m.rank1.map(where)
    t = pd.crosstab(m.half0, m.dest, normalize="index")
    return t


def succession(p: pd.DataFrame) -> pd.DataFrame:
    """Each edition's No.1 and where it stood one edition earlier; the 'heir'
    is the highest-ranked restaurant of the previous edition that remained
    eligible (all former No.1s are ineligible from 2019)."""
    years = sorted(p.year.unique())
    rows = []
    for y0, y1 in consecutive(years):
        no1 = p[(p.year == y1) & (p["rank"] == 1)].iloc[0]
        prev = p[(p.year == y0)].sort_values("rank")
        prev_no1 = prev.iloc[0].rid
        if y1 >= BOB_FIRST_YEAR:
            # ineligible: every restaurant that was No.1 in any edition < y1
            past_no1 = set(p[(p.year < y1) & (p["rank"] == 1)].rid)
            eligible = prev[~prev.rid.isin(past_no1)]
        else:
            eligible = prev
        heir = eligible.iloc[0].rid
        r_prev = prev[prev.rid == no1.rid]["rank"]
        rows.append({
            "year": y1, "no1": no1["name"], "prev_rank": int(r_prev.iloc[0]) if len(r_prev) else None,
            "repeat": no1.rid == prev_no1, "heir": eligible.iloc[0]["name"],
            "no1_is_heir": no1.rid == heir, "bob_era": y1 >= BOB_FIRST_YEAR,
        })
    return pd.DataFrame(rows)


def main():
    RES.mkdir(exist_ok=True)
    p = lists.load_panel()
    full = lists.FULL_YEARS
    t100 = transition_summary(p, full, 100)
    all_years = sorted(p.year.unique())
    t50 = transition_summary(p, all_years, 50)
    bands, m = band_summary(p, full)
    bands_adj, _ = band_summary(p, full, bob_adjust=True)
    bands_adj.to_csv(RES / "part1_bands_bobadj.csv", index=False)
    mob = half_mobility(m)
    suc = succession(p)
    t100.to_csv(RES / "part1_transitions_top100.csv", index=False)
    t50.to_csv(RES / "part1_transitions_top50.csv", index=False)
    bands.to_csv(RES / "part1_bands.csv", index=False)
    m.to_csv(RES / "part1_pairs_top100.csv", index=False)
    mob.to_csv(RES / "part1_half_mobility.csv")
    suc.to_csv(RES / "part1_succession.csv", index=False)

    exits50 = m[(m.rank0 <= 50) & ~m.in_both][["y0", "y1", "name", "country", "rank0", "rank1"]]
    exits50.to_csv(RES / "part1_top50_exits_from_top100.csv", index=False)

    bob = suc[suc.bob_era]
    pre = suc[~suc.bob_era & ~suc.repeat]
    late = t50[t50.y0 >= 2010]
    summary = {
        "top100_transitions": len(t100),
        "pairs_pooled": int(len(m)), "pairs_in_both": int(m.in_both.sum()),
        "spearman_survivors_top100_median": float(t100.spearman_survivors.median()),
        "spearman_survivors_top100_range": [float(t100.spearman_survivors.min()), float(t100.spearman_survivors.max())],
        "top50_from_y0_in_y1_51_100_or_out": float(1 - mob.loc["1-50", "1-50"]),
        "half2_to_top50": float(mob.loc["51-100", "1-50"]),
        "half2_out_of_top100": float(mob.loc["51-100", "out of top 100"]),
        "half1_out_of_top100": float(mob.loc["1-50", "out of top 100"]),
        "bob_era_no1_is_heir": f"{int(bob.no1_is_heir.sum())}/{len(bob)}",
        "pre_bob_changes_no1_was_prev2": f"{int((pre.prev_rank == 2).sum())}/{len(pre)}",
        "pre_bob_repeats": f"{int(suc[~suc.bob_era].repeat.sum())}/{int((~suc.bob_era).sum())}",
        "top50_since2010_spearman_median": float(late.spearman_survivors.median()),
        "top50_since2010_survivors_median": float(late.n_both.median()),
    }
    (RES / "part1_summary.json").write_text(json.dumps(summary, indent=2))
    print(t100.round(3).to_string())
    print(bands.round(3).to_string())
    print(mob.round(3))
    print(suc.to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
