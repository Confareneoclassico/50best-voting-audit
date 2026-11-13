"""Unit tests for the estimators and the ballot rules."""
import itertools
import sys
from math import lcm
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gsa  # noqa: E402
import voting_model as vm  # noqa: E402


def _w2_brute(x, y):
    n, m = len(x), len(y)
    k = lcm(n, m)
    xs = np.repeat(np.sort(x), k // n)
    ys = np.repeat(np.sort(y), k // m)
    return np.mean((xs - ys) ** 2)


def test_w2_exact_equal_and_unequal_sizes():
    rng = np.random.default_rng(0)
    for n, m in [(5, 5), (6, 4), (7, 3), (12, 18), (1, 9)]:
        x, y = rng.normal(size=n), rng.exponential(size=m)
        assert abs(gsa.w2sq_exact(x, y) - _w2_brute(x, y)) < 1e-12


def test_w2_exact_against_scipy_w1_logic():
    # the same interval construction with |.| instead of (.)^2 gives W1
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=37), rng.normal(1, 2, size=23)
    xs, ys = np.sort(x), np.sort(y)
    u = np.union1d(np.arange(1, 38) / 37, np.arange(1, 24) / 23)
    lo = np.concatenate([[0], u[:-1]]); mid = (lo + u) / 2
    w1 = np.sum((u - lo) * np.abs(xs[(mid * 37).astype(int)] - ys[(mid * 23).astype(int)]))
    assert abs(w1 - stats.wasserstein_distance(x, y)) < 1e-12


def test_w2_parts_sum_and_location_scale():
    rng = np.random.default_rng(5)
    for n, m in [(9, 9), (12, 4), (30, 20)]:
        x, y = rng.normal(size=n), rng.gamma(2.0, size=m)
        parts = gsa.w2sq_parts(x, y)
        assert min(parts) >= 0
        assert abs(sum(parts) - gsa.w2sq_exact(x, y)) < 1e-12
    # y = a + b x: no shape part, location (a + (b - 1) mean x)^2,
    # spread (b - 1)^2 var x
    x = rng.normal(size=25)
    loc, spread, shape = gsa.w2sq_parts(x, 0.7 + 1.8 * x)
    assert abs(loc - (0.7 + 0.8 * x.mean()) ** 2) < 1e-12
    assert abs(spread - 0.8 ** 2 * x.var()) < 1e-12
    assert shape < 1e-12


def test_ot_location_part_is_half_first_order_and_jackknife_runs():
    rng = np.random.default_rng(6)
    reps, rows = 8, []
    for x1, x2 in itertools.product([0, 1, 2], [0, 1]):
        for r in range(reps):
            # x2 changes only the spread, x1 the location
            rows.append({"x1": x1, "x2": x2, "rep": r,
                         "y": x1 + rng.normal(0, 0.5 + 1.5 * x2)})
    df = gsa.add_dummy(pd.DataFrame(rows))
    saved = gsa.FACTORS
    gsa.FACTORS = {"x1": [0, 1, 2], "x2": [0, 1]}
    try:
        fi = gsa.factorial_indices(df, "y", ["x1", "x2"], reps).set_index("factor")
        ot = gsa.ot_indices(df, "y", ["x1", "x2"]).set_index("factor")
        jk = gsa.jackknife_indices(df, "y", ["x1", "x2"], reps)
    finally:
        gsa.FACTORS = saved
    for f in ["x1", "x2"]:
        assert abs(ot.loc[f, "iota_loc"] - fi.loc[f, "S"] / 2) < 1e-12
    assert ot.loc["x2", "iota_spread"] > ot.loc["x2", "iota_loc"]
    assert np.isfinite(jk.loc["x1", "S_se"]) and jk.loc["x1", "S_se"] > 0


def test_factorial_indices_analytic():
    # Y = 2 X1 + X2 + X1 X2 + e, X1 in {-1,1}, X2 in {-1,0,1}, sd(e)=0.5
    rng = np.random.default_rng(2)
    reps = 400
    rows = []
    for x1, x2 in itertools.product([-1, 1], [-1, 0, 1]):
        e = rng.normal(0, 0.5, reps)
        for r in range(reps):
            rows.append({"x1": x1, "x2": x2, "rep": r, "y": 2 * x1 + x2 + x1 * x2 + e[r]})
    df = pd.DataFrame(rows)
    out = gsa.factorial_indices(df, "y", ["x1", "x2"], reps).set_index("factor")
    V = 4 + 2 / 3 + 2 / 3 + 0.25
    assert abs(out.loc["x1", "S"] - 4 / V) < 0.02
    assert abs(out.loc["x2", "S"] - (2 / 3) / V) < 0.02
    assert abs(out.loc["x1", "ST"] - (4 + 2 / 3) / V) < 0.02
    assert abs(out.loc["x2", "ST"] - (4 / 3) / V) < 0.02
    assert abs(out.loc["noise", "S"] - 0.25 / V) < 0.01


def test_ballots_respect_rules():
    listed = vm.load_2026()
    w = vm.make_world("M8", 5.0, listed)
    w.theta = np.log(w.target)
    rng = np.random.default_rng(3)
    for rules in [vm.Rules(), vm.Rules(ballot=7, cap_share=0.3), vm.Rules(cap_share=1.0, allocation="supply")]:
        ch, home = vm.ballots(w, rules, rng)
        assert ch.shape == (1120, rules.ballot)
        assert home.sum(axis=1).max() <= rules.cap
        assert all(len(set(b)) == rules.ballot for b in ch)
    assert vm.voters_per_region(w, vm.Rules(allocation="supply")).sum() == 1120
    assert (vm.voters_per_region(w, vm.Rules()) == 40).all()


def test_preference_tiebreak_only_orders_ties():
    listed = vm.load_2026()
    w = vm.make_world("M8", 3.0, listed)
    w.theta = np.log(w.target)
    rng = np.random.default_rng(4)
    ch, _ = vm.ballots(w, vm.Rules(), rng)
    s = vm.scores(w, vm.Rules(), ch, rng)
    approval = np.bincount(ch.ravel(), minlength=w.R)
    assert np.array_equal(np.floor(s), approval)
