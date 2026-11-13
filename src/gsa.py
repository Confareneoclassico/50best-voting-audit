"""Part 3: global sensitivity analysis of the voting rules (full factorial).

Factors (all levels crossed):
  rules  aggregation {approval*, borda, dowdall}, ballot {7, 10*},
         cap_share {0.3, 0.6*, 1.0}, allocation {equal*, supply},
         tiebreak {preference*, random}            (* = published rules)
  world  curve {L8, M8, H8} (Part 2 scenarios), home bias h {3, 5}
  noise  the panel draw, 16 independent replicates per cell

Indices are exact for the factorial: with m(x) the cell mean and s2(x) the
within-cell variance, S_i = Var(E[m|x_i]) / Var(Y) and
S_Ti = E[Var(m | x_~i)] / Var(Y) (corrected for the noise left in cell means),
and S_noise = E[s2] / Var(Y).  The optimal-transport index of Borgonovo,
Figalli, Plischke and Savare is iota_i = E[W2^2(P_Y, P_Y|x_i)] / (2 Var Y),
computed with the exact one-dimensional W2 and split into location, spread
and shape parts (the location part equals S_i / 2).  Intervals come from a
delete-one-panel jackknife (replicate r of every cell left out in turn).
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

import lists
import voting_model as vm

RES = lists.ROOT / "results"
FACTORS = {
    "aggregation": ["approval", "borda", "dowdall"],
    "ballot": [7, 10],
    "cap_share": [0.3, 0.6, 1.0],
    "allocation": ["equal", "supply"],
    "tiebreak": ["preference", "random"],
    "curve": ["L8", "M8", "H8"],
    "home_bias": [3.0, 5.0],
}
OFFICIAL_LEVELS = {"aggregation": "approval", "ballot": 10, "cap_share": 0.6,
                   "allocation": "equal", "tiebreak": "preference"}
OUTPUTS = ["rho100", "overlap50", "top10", "disp_lower", "regions50", "europe50"]
# outputs that are counts: W2 on the integers is at least the probability
# that has to move one unit, so their OT index is also computed after
# spreading each count uniformly over (k - 1/2, k + 1/2)
COUNTS = ["overlap50", "top10", "regions50", "europe50"]
JITTER_SEED = 1
REPLICATES = 16


# ---------------------------------------------------------------- estimators
def w2sq_exact(x: np.ndarray, y: np.ndarray) -> float:
    """Exact squared 2-Wasserstein distance between the empirical
    distributions of two 1-D samples of any sizes: the integral over (0,1) of
    the squared difference of their quantile functions."""
    x = np.sort(np.asarray(x, float)); y = np.sort(np.asarray(y, float))
    n, m = len(x), len(y)
    u = np.union1d(np.arange(1, n + 1) / n, np.arange(1, m + 1) / m)
    lo = np.concatenate([[0.0], u[:-1]])
    mid = 0.5 * (lo + u)
    ix = np.minimum((mid * n).astype(int), n - 1)
    iy = np.minimum((mid * m).astype(int), m - 1)
    return float(np.sum((u - lo) * (x[ix] - y[iy]) ** 2))


def w2sq_parts(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Location, spread and shape parts of the exact squared 2-Wasserstein
    distance between two empirical 1-D distributions,
        W2^2 = (mx - my)^2 + (sx - sy)^2 + 2 sx sy (1 - rho),
    with m and s the means and standard deviations and rho the correlation of
    the two quantile functions over (0, 1).  The parts sum to w2sq_exact."""
    x = np.sort(np.asarray(x, float)); y = np.sort(np.asarray(y, float))
    n, m = len(x), len(y)
    u = np.union1d(np.arange(1, n + 1) / n, np.arange(1, m + 1) / m)
    lo = np.concatenate([[0.0], u[:-1]])
    w = u - lo
    mid = 0.5 * (lo + u)
    qx = x[np.minimum((mid * n).astype(int), n - 1)]
    qy = y[np.minimum((mid * m).astype(int), m - 1)]
    mx, my = float(w @ qx), float(w @ qy)
    sx = float(np.sqrt(w @ (qx - mx) ** 2)); sy = float(np.sqrt(w @ (qy - my) ** 2))
    cov = float(w @ ((qx - mx) * (qy - my)))
    return (mx - my) ** 2, (sx - sy) ** 2, max(2 * (sx * sy - cov), 0.0)


def factorial_indices(df: pd.DataFrame, y: str, factors: list[str], reps: int) -> pd.DataFrame:
    V = df[y].var(ddof=0)
    if V == 0:
        return pd.DataFrame({"factor": factors + ["noise"], "S": 0.0, "ST": 0.0})
    cell = df.groupby(factors)[y]
    m = cell.mean()
    s2 = cell.var(ddof=1).mean()
    rows = []
    for f in factors:
        k = df[f].nunique()
        main = m.groupby(level=f).mean().var(ddof=0)
        others = [g for g in factors if g != f]
        within = m.groupby(level=others).var(ddof=0).mean()
        within_corr = max(within - (k - 1) / k * s2 / reps, 0.0)
        rows.append({"factor": f, "S": main / V, "ST": within_corr / V})
    rows.append({"factor": "noise", "S": s2 / V, "ST": s2 / V})
    return pd.DataFrame(rows)


DUMMY_LEVELS = 3


def add_dummy(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A dummy factor: runs assigned at random to three levels.  Its indices
    are the level that sampling alone produces."""
    df = df.copy()
    df["dummy"] = np.random.default_rng(seed).integers(0, DUMMY_LEVELS, len(df))
    return df


def ot_indices(df: pd.DataFrame, y: str, factors: list[str]) -> pd.DataFrame:
    """OT index per factor and for the dummy, with its location, spread and
    shape parts (each averaged over the levels and divided by 2 Var Y)."""
    Y = df[y].to_numpy(float)
    V = Y.var()
    rows = []
    for f in ["dummy"] + factors:
        if V == 0:
            rows.append({"factor": f, "iota": 0.0, "iota_loc": 0.0, "iota_spread": 0.0, "iota_shape": 0.0})
            continue
        levels = range(DUMMY_LEVELS) if f == "dummy" else FACTORS[f]
        parts = np.array([w2sq_parts(Y, df.loc[df[f] == lev, y].to_numpy(float)) for lev in levels])
        loc, spread, shape = parts.mean(axis=0) / (2 * V)
        rows.append({"factor": f, "iota": loc + spread + shape, "iota_loc": loc,
                     "iota_spread": spread, "iota_shape": shape})
    return pd.DataFrame(rows)


def jackknife_indices(df, y, factors, reps, level=0.95):
    """Delete-one-panel jackknife.  Estimate r leaves out replicate r of every
    cell.  The replicates of a cell are independent draws and cells are
    independent, so the reps groups are independent and identically
    distributed.  Returns t intervals around the full-sample estimates,
    truncated at zero."""
    est = []
    for r in range(reps):
        d = df[df["rep"] != r]
        fi = factorial_indices(d, y, factors, reps - 1).set_index("factor")[["S", "ST"]]
        ot = ot_indices(d, y, factors).set_index("factor")[["iota"]]
        est.append(fi.join(ot, how="outer"))
    index, cols = est[0].index, est[0].columns
    a = np.stack([e.loc[index, cols].to_numpy(float) for e in est])
    se = np.sqrt((reps - 1) / reps * ((a - a.mean(axis=0)) ** 2).sum(axis=0))
    out = pd.DataFrame(se, index=index, columns=[f"{c}_se" for c in cols])
    out.attrs["t"] = float(stats.t.ppf(0.5 + level / 2, reps - 1))
    return out


def analyse(df: pd.DataFrame, reps: int = REPLICATES) -> pd.DataFrame:
    """Sobol' (exact factorial), OT with its parts, and jackknife intervals
    for every output."""
    factors = list(FACTORS)
    df = add_dummy(df)
    all_idx = []
    for y in OUTPUTS:
        fi = factorial_indices(df, y, factors, reps).set_index("factor")
        ot = ot_indices(df, y, factors).set_index("factor")
        jk = jackknife_indices(df, y, factors, reps)
        tab = fi.join(ot, how="outer").join(jk)
        t = jk.attrs["t"]
        for c in ["S", "ST", "iota"]:
            tab[f"{c}_lo"] = (tab[c] - t * tab[f"{c}_se"]).clip(lower=0)
            tab[f"{c}_hi"] = tab[c] + t * tab[f"{c}_se"]
        # the location part of iota is S_i / 2 exactly (balanced design)
        chk = tab.loc[factors]
        assert np.allclose(chk["iota_loc"], chk["S"] / 2, atol=1e-10), y
        if y in COUNTS:
            j = df.copy()
            j[y] = j[y] + np.random.default_rng(JITTER_SEED).uniform(-0.5, 0.5, len(j))
            fj = factorial_indices(j, y, factors, reps).set_index("factor")[["S", "ST"]]
            oj = ot_indices(j, y, factors).set_index("factor")
            jj = jackknife_indices(j, y, factors, reps)
            tab["iota_jit"] = oj["iota"]
            tab["iota_jit_lo"] = (oj["iota"] - t * jj["iota_se"]).clip(lower=0)
            tab["iota_jit_hi"] = oj["iota"] + t * jj["iota_se"]
            tab["iota_jit_shape"] = oj["iota_shape"]
            tab["S_jit"], tab["ST_jit"] = fj["S"], fj["ST"]
        tab.insert(0, "output", y)
        all_idx.append(tab.reset_index())
    idx = pd.concat(all_idx, ignore_index=True)
    idx.to_csv(RES / "part3_indices.csv", index=False)
    print(idx.round(3).to_string())
    return idx


# ---------------------------------------------------------------- simulation
_WORLDS = {}


def _init(worlds):
    global _WORLDS
    _WORLDS = worlds


def _run(task):
    key, rules_kw, rep, seed = task
    world = _WORLDS[key]
    rules = vm.Rules(**rules_kw)
    rng = np.random.default_rng(seed)
    chosen, home = vm.ballots(world, rules, rng)
    s = vm.scores(world, rules, chosen, rng)
    rk = vm.ranks_from_scores(s)
    out = vm.outputs(world, rk)
    out["home_share"] = float(home.mean())
    return key, rules_kw, rep, out, rk[:100].astype(np.int16)


def build_worlds(listed, verbose=True):
    worlds, checks = {}, []
    for curve, h in itertools.product(FACTORS["curve"], FACTORS["home_bias"]):
        w = vm.make_world(curve, h, listed)
        t = time.time()
        hist = vm.calibrate(w, panels=80, iters=40, seed=11)
        chk = vm.check_calibration(w, panels=80, seed=12)
        chk.update({"curve": curve, "home_bias": h, "R": w.R, "iterations": len(hist),
                    "uncapped_home_share": vm.uncapped_home_share(w)})
        checks.append(chk)
        if verbose:
            print(f"calibrated {curve} h={h}: {time.time()-t:.1f}s", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in chk.items()}, flush=True)
        worlds[(curve, h)] = w
    return worlds, pd.DataFrame(checks)


def main(reps=REPLICATES, procs=None):
    RES.mkdir(exist_ok=True)
    listed = vm.load_2026()
    worlds, checks = build_worlds(listed)
    checks.to_csv(RES / "part3_calibration.csv", index=False)
    np.savez_compressed(RES / "part3_worlds.npz", **{f"{c}_{h}": w.theta for (c, h), w in worlds.items()})

    rule_names = ["aggregation", "ballot", "cap_share", "allocation", "tiebreak"]
    rule_grid = list(itertools.product(*[FACTORS[f] for f in rule_names]))
    ss = np.random.SeedSequence(20261106)
    tasks = []
    for key in worlds:
        for combo in rule_grid:
            kw = dict(zip(rule_names, combo))
            for rep in range(reps):
                tasks.append((key, kw, rep, None))
    seeds = ss.generate_state(len(tasks))
    tasks = [(k, kw, r, int(s)) for (k, kw, r, _), s in zip(tasks, seeds)]
    t = time.time()
    procs = procs or os.cpu_count()
    with Pool(procs, initializer=_init, initargs=(worlds,)) as pool:
        results = pool.map(_run, tasks, chunksize=32)
    print(f"{len(tasks)} runs in {time.time()-t:.0f}s", flush=True)

    rows, ranks = [], []
    for (curve, h), kw, rep, out, rk in results:
        rows.append({"curve": curve, "home_bias": h, **kw, "rep": rep, **out})
        ranks.append(rk)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "part3_runs.csv", index=False)
    ranks = np.vstack(ranks)
    np.save(RES / "part3_ranks.npy", ranks)

    analyse(df, reps)

    # rank intervals of the 2026 restaurants: published rules only, and all rules
    off = np.ones(len(df), bool)
    for f, v in OFFICIAL_LEVELS.items():
        off &= (df[f] == v).to_numpy()
    q = lambda a, p: np.quantile(a, p, axis=0)
    ri = pd.DataFrame({
        "published_rank": np.arange(1, 101), "name": listed["name"], "region": listed["region"],
        "off_q05": q(ranks[off], .05), "off_q50": q(ranks[off], .5), "off_q95": q(ranks[off], .95),
        "off_p_top50": (ranks[off] <= 50).mean(0), "off_p_top100": (ranks[off] <= 100).mean(0),
        "all_q05": q(ranks, .05), "all_q50": q(ranks, .5), "all_q95": q(ranks, .95),
        "all_p_top50": (ranks <= 50).mean(0),
    })
    ri.to_csv(RES / "part3_rank_intervals.csv", index=False)

    # effect of each rule level on the list, other rules at published values
    eff = []
    for f in rule_names:
        for lev in FACTORS[f]:
            sel = np.ones(len(df), bool)
            for g, v in OFFICIAL_LEVELS.items():
                if g != f:
                    sel &= (df[g] == v).to_numpy()
            sel &= (df[f] == lev).to_numpy()
            d = df[sel]
            eff.append({"factor": f, "level": lev, "n": int(sel.sum()),
                        **{y: float(d[y].mean()) for y in OUTPUTS + ["home_share"]}})
    pd.DataFrame(eff).to_csv(RES / "part3_rule_effects.csv", index=False)
    print(pd.DataFrame(eff).round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "analyse":
        analyse(pd.read_csv(RES / "part3_runs.csv"))
    else:
        main()
