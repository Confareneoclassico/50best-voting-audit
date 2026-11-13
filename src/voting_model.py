"""Part 3: a generative model of the Academy vote, calibrated to the 2026 list.

Each voter belongs to one of the 28 Academy regions and ranks restaurants by
a random utility  u_ij = theta_j + h * [region(j) == region(i)] + e_ij  with
Gumbel e_ij, so that each ballot is a Plackett-Luce draw.  The ballot holds
the best L restaurants subject to at most C in the voter's home region; it is
ordered, so positional rules (Borda, Dowdall) can be applied to it.

The restaurant universe is the 2026 top 100 (in published order) plus the
restaurants below No.100 implied by the vote-curve scenario of Part 2, spread
evenly over the regions.  For each world (vote curve x home bias) the appeals
theta are calibrated so that, under the published rules (approval count of
10-restaurant ballots, at most 6 at home, 40 voters per region), the expected
votes follow the scenario curve: the published order is then the expected
order of the model.  Any departure from the published order in a simulated
edition is due to the panel draw or to a change of rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import lists
from noise import vote_curve, SCENARIOS, N_VOTERS

REGIONS = [
    "Middle East", "Oceania", "Korea", "UK & Ireland", "Italy", "North & West Africa",
    "Spain & Portugal", "South America (South, excl. Brazil)",
    "Austria, Switzerland, Hungary & Slovenia", "Germany", "Japan", "India & Subcontinent",
    "Hong Kong, Macau & Taiwan", "Southeast Asia (North)", "Southeast Asia (South)", "France",
    "South America (North) & the Caribbean", "West USA & West Canada", "Mainland China",
    "Brazil", "Mexico & Central America", "Russia, Central Asia & Eastern Europe",
    "East USA & East Canada", "South & East Africa", "Scandinavia & Baltics",
    "Mid USA & Mid Canada", "Türkiye, Greece & the Balkans", "Benelux",
]
assert len(REGIONS) == 28
EUROPE = lists.EUROPE
DISP_CAP = 200                  # cap on simulated positions in the displacement of 51-100


@dataclass
class World:
    curve: str
    h: float
    names: list
    region: np.ndarray          # region index per restaurant
    target: np.ndarray          # expected votes under the published rules
    theta: np.ndarray = field(default=None)
    n_listed: int = 100

    @property
    def R(self):
        return len(self.target)


@dataclass(frozen=True)
class Rules:
    aggregation: str = "approval"   # approval | borda | dowdall
    ballot: int = 10
    cap_share: float = 0.6          # max share of the ballot in the home region
    allocation: str = "equal"       # equal | supply
    tiebreak: str = "preference"    # preference | random

    @property
    def cap(self):
        return int(np.floor(self.cap_share * self.ballot + 1e-9))


OFFICIAL = Rules()


def load_2026():
    p = lists.load_panel()
    d = p[p.year == 2026].sort_values("rank")
    return d.reset_index(drop=True)


def make_world(curve: str, h: float, listed: pd.DataFrame) -> World:
    v100, ratio = SCENARIOS[curve]
    mu = vote_curve(v100, ratio)
    reg_idx = {r: i for i, r in enumerate(REGIONS)}
    region = np.empty(len(mu), dtype=int)
    region[:100] = [reg_idx[r] for r in listed["region"]]
    # restaurants below No.100: dealt to regions in turn (in order of their
    # expected votes), so every region has the same depth of local candidates
    region[100:] = np.arange(len(mu) - 100) % 28
    names = list(listed["name"]) + [f"tail{k}" for k in range(len(mu) - 100)]
    return World(curve=curve, h=h, names=names, region=region, target=mu.astype(float))


def voters_per_region(world: World, rules: Rules, total=N_VOTERS) -> np.ndarray:
    if rules.allocation == "equal":
        n = np.full(28, total // 28)
    elif rules.allocation == "supply":
        c = np.bincount(world.region[:world.n_listed], minlength=28).astype(float) + 1.0
        n = np.maximum(10, np.floor(total * c / c.sum())).astype(int)
    else:
        raise ValueError(rules.allocation)
    # adjust to the exact total, largest regions first
    diff = total - n.sum()
    order = np.argsort(-n)
    k = 0
    while diff != 0:
        j = order[k % 28]
        step = 1 if diff > 0 else -1
        if n[j] + step >= 10:
            n[j] += step
            diff -= step
        k += 1
    return n


def ballots(world: World, rules: Rules, rng, n_per_region=None):
    """Return (chosen, home_flag): chosen is (N, L) restaurant indices in order
    of preference."""
    if n_per_region is None:
        n_per_region = voters_per_region(world, rules)
    L, C = rules.ballot, rules.cap
    out, homes = [], []
    all_idx = np.arange(world.R)
    for g in range(28):
        n = int(n_per_region[g])
        if n == 0:
            continue
        home = all_idx[world.region == g]
        away = all_idx[world.region != g]
        U = world.theta[None, :] + rng.gumbel(size=(n, world.R))
        if world.h:
            U[:, home] += world.h
        # best C at home and best L away; the capped ballot is the best L of
        # their union (greedy selection down the preference order)
        if C >= len(home):
            top_home = np.broadcast_to(home, (n, len(home)))
        elif C > 0:
            part = np.argpartition(-U[:, home], C - 1, axis=1)[:, :C]
            top_home = home[part]
        else:
            top_home = np.empty((n, 0), dtype=int)
        part = np.argpartition(-U[:, away], L - 1, axis=1)[:, :L]
        top_away = away[part]
        cand = np.concatenate([top_home, top_away], axis=1)
        cu = np.take_along_axis(U, cand, axis=1)
        sel = np.argsort(-cu, axis=1)[:, :L]
        chosen = np.take_along_axis(cand, sel, axis=1)
        out.append(chosen)
        homes.append(world.region[chosen] == g)
    return np.concatenate(out), np.concatenate(homes)


def scores(world: World, rules: Rules, chosen: np.ndarray, rng) -> np.ndarray:
    L = rules.ballot
    pos = np.broadcast_to(np.arange(1, L + 1), chosen.shape)
    approval = np.bincount(chosen.ravel(), minlength=world.R).astype(float)
    borda = np.bincount(chosen.ravel(), weights=(L + 1 - pos).ravel(), minlength=world.R)
    if rules.aggregation == "approval":
        if rules.tiebreak == "preference":
            # order of preference decides only among equal vote counts, and
            # chance only among equal counts with equal preference points
            # (borda is an integer, so the jitter cannot reorder unequal ones)
            s = approval + (borda + 0.5 * rng.random(world.R)) / (L * len(chosen) + 1.0)
        else:
            s = approval + 0.5 * rng.random(world.R)
    elif rules.aggregation == "borda":
        s = borda + 1e-6 * rng.random(world.R)
    elif rules.aggregation == "dowdall":
        s = np.bincount(chosen.ravel(), weights=(1.0 / pos).ravel(), minlength=world.R)
        s = s + 1e-9 * rng.random(world.R)
    else:
        raise ValueError(rules.aggregation)
    return s


def ranks_from_scores(s: np.ndarray) -> np.ndarray:
    order = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=int)
    r[order] = np.arange(1, len(s) + 1)
    return r


def calibrate(world: World, panels=40, iters=30, seed=1, tol=0.01, verbose=False):
    """Fixed-point iteration on theta with common random numbers, so that the
    mean approval votes under the published rules match the target curve."""
    world.theta = np.log(world.target) - np.log(world.target).mean()
    n_reg = voters_per_region(world, OFFICIAL)
    hist = []
    for it in range(iters):
        rng = np.random.default_rng(seed)  # common random numbers
        votes = np.zeros(world.R)
        for _ in range(panels):
            chosen, _h = ballots(world, OFFICIAL, rng, n_reg)
            votes += np.bincount(chosen.ravel(), minlength=world.R)
        votes /= panels
        err = np.log(world.target) - np.log(np.maximum(votes, 1e-3))
        world.theta = world.theta + 0.9 * err
        world.theta -= world.theta.mean()
        rel = np.abs(votes[:world.n_listed] / world.target[:world.n_listed] - 1)
        hist.append((it, float(rel.mean()), float(rel.max())))
        if verbose:
            print(world.curve, world.h, it, round(rel.mean(), 4), round(rel.max(), 4))
        if rel.max() < tol:
            break
    return hist


def check_calibration(world: World, panels=40, seed=99):
    rng = np.random.default_rng(seed)
    n_reg = voters_per_region(world, OFFICIAL)
    votes = np.zeros(world.R)
    home_share = []
    for _ in range(panels):
        chosen, hm = ballots(world, OFFICIAL, rng, n_reg)
        votes += np.bincount(chosen.ravel(), minlength=world.R)
        home_share.append(hm.mean())
    votes /= panels
    rel = votes[:world.n_listed] / world.target[:world.n_listed] - 1
    exp_rank = ranks_from_scores(votes)[:world.n_listed]
    return {
        "mean_abs_rel_err": float(np.abs(rel).mean()), "max_abs_rel_err": float(np.abs(rel).max()),
        "spearman_expected_vs_published": float(pd.Series(exp_rank).corr(pd.Series(np.arange(1, 101)), method="spearman")),
        "home_share_of_votes": float(np.mean(home_share)),
    }


def uncapped_home_share(world: World, panels=10, seed=7):
    """Home share the same voters would choose with no cap (latent home bias)."""
    rules = Rules(cap_share=1.0)
    rng = np.random.default_rng(seed)
    n_reg = voters_per_region(world, rules)
    return float(np.mean([ballots(world, rules, rng, n_reg)[1].mean() for _ in range(panels)]))


def outputs(world: World, rk: np.ndarray, top: int = 50) -> dict:
    """List features of one simulated edition (rk: rank of every restaurant)."""
    pub = np.arange(1, world.n_listed + 1)
    sim = rk[:world.n_listed]
    order = np.argsort(rk)
    top50 = order[:top]
    regs = world.region[top50]
    europe_idx = {i for i, r in enumerate(REGIONS) if r in EUROPE}
    return {
        "rho100": float(pd.Series(sim).corr(pd.Series(pub), method="spearman")),
        "overlap50": int(np.sum(sim[:50] <= 50)),
        "regions50": int(len(np.unique(regs))),
        "europe50": int(np.isin(regs, list(europe_idx)).sum()),
        "top10": int(np.sum(sim[:10] <= 10)),
        # positions below No.200 count as No.200: beyond that the rank depends
        # mostly on the assumed tail of unlisted restaurants
        "disp_lower": float(np.mean(np.abs(np.minimum(sim[50:100], DISP_CAP) - pub[50:100]))),
    }
