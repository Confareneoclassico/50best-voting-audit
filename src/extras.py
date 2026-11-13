"""Numbers quoted in the note that combine the three parts."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import lists
import noise
import gsa
import voting_model as vm

RES = lists.ROOT / "results"


def noise_bands(scenario="M8", reps=600, seed=5):
    """5-95% quantiles of the year-on-year rank change of a restaurant that
    does not change at all and stays in the top 100, by band of its rank in
    year t (both bounds).  Conditioning on staying matches the observed moves,
    which exist only for restaurants in both lists."""
    rng = np.random.default_rng(seed)
    mu = noise.vote_curve(*noise.SCENARIOS[scenario])
    p = np.clip(mu / noise.N_VOTERS, 0, 1)
    v0 = rng.binomial(noise.N_VOTERS, p, size=(reps, len(mu)))
    rk0 = noise.ranks_with_random_ties(v0, rng)
    out = {}
    for mode in ("renewal", "redraw"):
        v1 = noise.next_edition(v0, mu, mode, rng)
        rk1 = noise.ranks_with_random_ties(v1, rng)
        m = (rk0 <= 100) & (rk1 <= 100)
        r0, d = rk0[m], (rk1 - rk0)[m]
        for a, b in [(1, 10), (11, 25), (26, 50), (51, 75), (76, 100)]:
            s = (r0 >= a) & (r0 <= b)
            out[(mode, f"{a}-{b}")] = (float(np.quantile(d[s], 0.05)), float(np.quantile(d[s], 0.95)))
    return out


def classify_moves(bands, y0=2025, y1=2026):
    pairs = pd.read_csv(RES / "part1_pairs_top100.csv")
    m = pairs[(pairs.y0 == y0) & pairs.in_both].copy()
    # remove the mechanical one-place gain from the retired No.1
    m["d"] = m.rank1 - (m.rank0 - 1)
    from churn import band_of
    m["band"] = (m.rank0 - 1).astype(int).clip(lower=1).map(band_of)
    for mode in ("renewal", "redraw"):
        lo = m.band.map(lambda b: bands[(mode, b)][0])
        hi = m.band.map(lambda b: bands[(mode, b)][1])
        m[f"beyond_{mode}"] = (m.d < lo) | (m.d > hi)
    return m


def geography_counts():
    runs = pd.read_csv(RES / "part3_runs.csv")
    ranks = np.load(RES / "part3_ranks.npy")
    ri = pd.read_csv(RES / "part3_rank_intervals.csv")
    off = np.ones(len(runs), bool)
    for f, v in gsa.OFFICIAL_LEVELS.items():
        off &= (runs[f] == v).to_numpy()
    listed = lists.load_panel().query("year == 2026").sort_values("rank").reset_index(drop=True)
    res = {}
    for label, mask in [("Japan", listed.country == "Japan"), ("Thailand", listed.country == "Thailand"),
                        ("United States", listed.country == "United States"), ("Peru", listed.country == "Peru")]:
        cols = np.where(mask.to_numpy())[0]
        n50 = (ranks[off][:, cols] <= 50).sum(axis=1)
        res[label] = {"published_in_top50": int((listed.loc[mask, "rank"] <= 50).sum()),
                      "sim_mean": float(n50.mean()),
                      "sim_dist": {int(k): float(v) for k, v in pd.Series(n50).value_counts(normalize=True).sort_index().items()}}
    return res


def displacement_uncapped():
    """The displacement of the published 51-100 counts positions below No.200
    as No.200.  The same feature without the cap, for comparison."""
    runs = pd.read_csv(RES / "part3_runs.csv")
    ranks = np.load(RES / "part3_ranks.npy").astype(float)
    pub = np.arange(51, 101)
    capped = np.abs(np.minimum(ranks[:, 50:100], vm.DISP_CAP) - pub).mean(axis=1)
    assert np.allclose(capped, runs["disp_lower"].to_numpy()), "ranks and runs disagree"
    runs = runs.assign(disp_uncapped=np.abs(ranks[:, 50:100] - pub).mean(axis=1))
    beyond = ranks[:, 50:100] > vm.DISP_CAP
    out = {"share_beyond_cap": float(beyond.mean()),
           "share_beyond_cap_by_curve": {c: float(beyond[(runs.curve == c).to_numpy()].mean())
                                         for c in gsa.FACTORS["curve"]}}

    def oat(f=None, lev=None):
        sel = np.ones(len(runs), bool)
        for g, v in gsa.OFFICIAL_LEVELS.items():
            sel &= (runs[g] == (lev if g == f else v)).to_numpy()
        return runs[sel]
    off = oat()
    for y in ("disp_lower", "disp_uncapped"):
        out[y] = {"published_rules_mean": float(off[y].mean()),
                  "dowdall_minus_approval": float(oat("aggregation", "dowdall")[y].mean() - off[y].mean()),
                  "ballot7_minus_ballot10": float(oat("ballot", 7)[y].mean() - off[y].mean())}
        fi = gsa.factorial_indices(runs, y, list(gsa.FACTORS), gsa.REPLICATES).set_index("factor")
        out[y]["S"] = fi["S"].round(4).to_dict()
        out[y]["ST"] = fi["ST"].round(4).to_dict()
    return out


def main():
    out = {}
    for sc in ("M8", "H8"):
        b = noise_bands(sc)
        m = classify_moves(b)
        out[f"moves_2025_2026_{sc}"] = {
            "n": int(len(m)),
            "beyond_redraw_band": int(m.beyond_redraw.sum()),
            "beyond_renewal_band": int(m.beyond_renewal.sum()),
            "expected_beyond_by_chance": 0.10 * len(m),
            "names_beyond_redraw": m.loc[m.beyond_redraw, ["name", "rank0", "rank1"]].values.tolist(),
            "bands": {f"{k[0]} {k[1]}": v for k, v in b.items()},
        }
    out["geography_official_rules"] = geography_counts()
    out["displacement_cap"] = displacement_uncapped()
    ri = pd.read_csv(RES / "part3_rank_intervals.csv")
    r2 = pd.read_csv(RES / "part2_resolution.csv")
    sel = [1, 2, 10, 15, 25, 34, 50, 51, 75, 100]
    out["named_intervals"] = []
    for r in sel:
        row = ri[ri.published_rank == r].iloc[0]
        m8 = r2[(r2.scenario == "M8") & (r2.expected_rank == r)].iloc[0]
        h8 = r2[(r2.scenario == "H8") & (r2.expected_rank == r)].iloc[0]
        out["named_intervals"].append({
            "rank": r, "name": row["name"],
            "part3_official": [float(row.off_q05), float(row.off_q95)],
            "part3_all_rules": [float(row.all_q05), float(row.all_q95)],
            "part2_M8": [float(m8.q05), float(m8.q95)], "part2_H8": [float(h8.q05), float(h8.q95)],
            "part3_official_p_top50": float(row.off_p_top50),
        })
    (RES / "extras.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
