"""Checks of the binomial model, the Best of the Best adjustment, the list
features of the generative model and its tie-break."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import churn  # noqa: E402
import noise  # noqa: E402
import voting_model as vm  # noqa: E402


def test_vote_curve_scenarios():
    for v100, ratio in noise.SCENARIOS.values():
        mu = noise.vote_curve(v100, ratio)
        assert abs(mu.sum() - noise.N_VOTERS * noise.BALLOT) < 1e-6
        assert abs(mu[99] - v100) < 1e-9 and abs(mu[0] - v100 * ratio) < 1e-9
        assert np.all(np.diff(mu[:-1]) < 0) and mu[-1] > 0


def test_random_tie_ranks_respect_counts():
    rng = np.random.default_rng(0)
    v = rng.integers(0, 5, size=(50, 30))
    rk = noise.ranks_with_random_ties(v, rng)
    assert all(sorted(r) == list(range(1, 31)) for r in rk)
    for k in range(50):
        order = np.argsort(rk[k])
        assert np.all(np.diff(v[k, order]) <= 0)
    # equal counts are ordered at random, not by position
    tied = noise.ranks_with_random_ties(np.zeros((400, 2), int), rng)
    assert 0.4 < (tied[:, 0] == 1).mean() < 0.6


def test_best_of_the_best_adjustment():
    """Two editions in which nothing changes except that the No.1 retires:
    with the adjustment every survivor stays put."""
    rows = [{"year": 2024, "rank": r, "rid": f"r{r}", "name": f"r{r}", "country": "X"} for r in range(1, 101)]
    rows += [{"year": 2025, "rank": r - 1, "rid": f"r{r}", "name": f"r{r}", "country": "X"} for r in range(2, 102)]
    p = pd.DataFrame(rows)
    raw, _ = churn.band_summary(p, [2024, 2025])
    adj, _ = churn.band_summary(p, [2024, 2025], bob_adjust=True)
    assert (adj.median_abs_change == 0).all() and (adj.share_exit_top100 == 0).all()
    assert (raw.median_abs_change == 1).all()


def test_list_features_of_an_unchanged_edition():
    listed = vm.load_2026()
    w = vm.make_world("M8", 3.0, listed)
    rk = np.arange(1, w.R + 1)
    out = vm.outputs(w, rk)
    assert abs(out["rho100"] - 1) < 1e-12 and out["overlap50"] == 50 and out["top10"] == 10 and out["disp_lower"] == 0.0
    assert out["regions50"] == listed.loc[:49, "region"].nunique()
    # a restaurant of 51-100 pushed far down counts as No.200 at most
    rk2 = rk.copy()
    rk2[[99, w.R - 1]] = rk2[[w.R - 1, 99]]
    assert abs(vm.outputs(w, rk2)["disp_lower"] - (vm.DISP_CAP - 100) / 50) < 1e-12


def test_preference_tiebreak_breaks_remaining_ties_at_random():
    """Restaurants 0 and 1 get the same votes in the same ballot positions:
    neither the order of preference nor their position may decide."""
    listed = vm.load_2026()
    w = vm.make_world("M8", 3.0, listed)
    chosen = np.array([[0, 2, 3], [1, 2, 3], [4, 5, 6]])
    first = []
    for seed in range(200):
        s = vm.scores(w, vm.Rules(ballot=3), chosen, np.random.default_rng(seed))
        assert np.array_equal(np.floor(s[:7]), [1, 1, 2, 2, 1, 1, 1])
        assert s[2] > s[3]               # same count, better placed on ballots
        first.append(s[0] > s[1])
    assert 0.35 < np.mean(first) < 0.65
