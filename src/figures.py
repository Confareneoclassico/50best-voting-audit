"""Figures for the audit note (PDF, sized for a 6.5 in text block)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import lists

RES = lists.ROOT / "results"
FIG = lists.ROOT / "report" / "figures"
# categorical slots 1-3 of the validated reference palette (all-pairs pass)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT = "#9ec5f4"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "lines.linewidth": 1.5,
})


def fig_pairs():
    m = pd.read_csv(RES / "part1_pairs_top100.csv")
    both = m[m.in_both]
    out = m[~m.in_both]
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    ax.plot([1, 100], [1, 100], color=INK2, lw=0.8, zorder=1)
    ax.fill_between([1, 100], [1 - 10, 100 - 10], [1 + 10, 100 + 10], color=GRID, alpha=0.6, lw=0, zorder=0)
    ax.scatter(both.rank0, both.rank1, s=9, color=BLUE, alpha=0.55, lw=0, label="in both top-100 lists")
    jitter = np.random.default_rng(0).uniform(-1.5, 1.5, len(out))
    ax.scatter(out.rank0, 108 + jitter, s=9, color=ORANGE, alpha=0.8, lw=0, label="left the top 100")
    ax.set_xlim(0, 101); ax.set_ylim(0, 112)
    ax.set_yticks([1, 25, 50, 75, 100]); ax.set_xticks([1, 25, 50, 75, 100])
    ax.invert_yaxis()
    ax.set_xlabel("rank in edition t")
    ax.set_ylabel("rank in edition t+1")
    ax.legend(loc="lower left", frameon=False, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_pairs.pdf")
    plt.close(fig)


def fig_churn():
    c = pd.read_csv(RES / "part2_churn_vs_observed.csv")
    bands = ["1-10", "11-25", "26-50", "51-75", "76-100"]
    x = np.arange(len(bands))
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, (key, lab) in zip(axes, [("median_abs", "median |rank change|, survivors"),
                                      ("exit", "share leaving the top 100")]):
        for k, (sc, col, name) in enumerate([("L8", BLUE, "L8"), ("M8", ORANGE, "M8"), ("H8", AQUA, "H8")]):
            lo = c[(c.scenario == sc) & (c["mode"] == "renewal")]
            hi = c[(c.scenario == sc) & (c["mode"] == "redraw")]
            ylo = [float(lo[f"{key}_{b}"].iloc[0]) for b in bands]
            yhi = [float(hi[f"{key}_{b}"].iloc[0]) for b in bands]
            xs = x + (k - 1) * 0.16
            ax.vlines(xs, ylo, yhi, color=col, lw=3.0, label=f"panel noise only, {name}")
        o = c[c.scenario == "observed"].iloc[0]
        ax.scatter(x, [o[f"{key}_{b}"] for b in bands], marker="D", s=22, color=INK, zorder=5,
                   label="observed 2019-2026")
        ax.set_xticks(x, bands)
        ax.set_xlabel("band of rank in edition t")
        ax.set_title(lab, loc="left")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_churn.pdf")
    plt.close(fig)


def fig_intervals():
    ri = pd.read_csv(RES / "part3_rank_intervals.csv")
    r2 = pd.read_csv(RES / "part2_resolution.csv")
    r2 = r2[r2.scenario == "M8"]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = ri.published_rank.to_numpy()
    ax.vlines(x - 0.2, ri.all_q05, ri.all_q95, color="#f6b49a", lw=2.2, label="all rule variants, generative model")
    ax.vlines(x + 0.2, ri.off_q05, ri.off_q95, color=BLUE, lw=1.4, label="published rules, panels drawn again, generative model")
    ax.plot(r2.expected_rank, r2.q05, color=INK2, lw=0.8, ls="--", label="binomial model, scenario M8")
    ax.plot(r2.expected_rank, r2.q95, color=INK2, lw=0.8, ls="--")
    xx = np.arange(1, 101)
    ax.plot(xx, xx, color=INK, lw=0.8, label="published rank")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1000)
    ax.set_yticks([1, 2, 5, 10, 25, 50, 100, 200, 400, 800], ["1", "2", "5", "10", "25", "50", "100", "200", "400", "800"])
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax.set_xlim(0, 101)
    ax.invert_yaxis()
    ax.set_xlabel("published rank, 2026")
    ax.set_ylabel("simulated rank (5th to 95th percentile)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_intervals.pdf")
    plt.close(fig)


LABELS = {"aggregation": "counting rule", "ballot": "ballot length", "cap_share": "home cap",
          "allocation": "voters per region", "tiebreak": "tie-break", "curve": "vote curve",
          "home_bias": "home bias", "noise": "panel draw"}
OUT_LABELS = {"rho100": "(a) rank correlation, top 100", "overlap50": "(b) top 50 kept",
              "top10": "(c) top 10 kept", "disp_lower": "(d) displacement of 51-100",
              "regions50": "(e) regions in the top 50", "europe50": "(f) European places in top 50"}


def fig_indices():
    idx = pd.read_csv(RES / "part3_indices.csv")
    order = list(LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.6), sharey=True)
    for ax, (y, title) in zip(axes.ravel(), OUT_LABELS.items()):
        d = idx[idx.output == y].set_index("factor").loc[order]
        yy = np.arange(len(order))
        ax.barh(yy - 0.19, d["S"], height=0.36, color=BLUE, label="first order $S_i$")
        ax.barh(yy + 0.19, d["ST"], height=0.36, color=BLUE_LIGHT, label="total $S_{Ti}$")
        ax.hlines(yy - 0.19, d["S_lo"], d["S_hi"], color=INK, lw=0.6)
        ax.hlines(yy + 0.19, d["ST_lo"], d["ST_hi"], color=INK, lw=0.6)
        ax.set_yticks(yy, [LABELS[f] for f in order])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_title(title, loc="left", fontsize=8)
    axes[0, 0].invert_yaxis()  # shared y axis: invert once
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig_indices.pdf")
    plt.close(fig)


OT_ORDER = ["aggregation", "ballot", "cap_share", "allocation", "tiebreak", "curve", "home_bias", "dummy"]
OT_LABELS = {**LABELS, "dummy": "dummy (null level)"}


def fig_ot():
    """OT index split into location (= S_i / 2), spread and shape, with the
    total Sobol' index and, for the counts, the index after jittering."""
    idx = pd.read_csv(RES / "part3_indices.csv")
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.9), sharey=True)
    yy = np.arange(len(OT_ORDER))
    for ax, (y, title) in zip(axes.ravel(), OUT_LABELS.items()):
        d = idx[idx.output == y].set_index("factor").loc[OT_ORDER]
        left = np.zeros(len(OT_ORDER))
        for col, color, lab in [("iota_loc", BLUE, "location part ($=S_i/2$)"),
                                ("iota_spread", ORANGE, "spread part"),
                                ("iota_shape", AQUA, "shape part")]:
            v = d[col].to_numpy(float)
            ax.barh(yy, v, left=left, height=0.62, color=color, edgecolor="white", linewidth=0.8, label=lab)
            left += v
        ax.hlines(yy, d["iota_lo"], d["iota_hi"], color=INK, lw=0.6)
        st = d["ST"].to_numpy(float)
        ax.plot(st, yy, ls="none", marker="|", ms=9, mew=1.3, color=INK, label="total Sobol' index $S_{Ti}$")
        if "iota_jit" in d and d["iota_jit"].notna().any():
            ax.plot(d["iota_jit"], yy, ls="none", marker="o", ms=4.2, mfc="white", mec=INK, mew=0.9,
                    label="$\\iota_i$ of the smoothed count")
        ax.set_yticks(yy, [OT_LABELS[f] for f in OT_ORDER])
        ax.set_xlim(0, 0.72)
        ax.set_xticks([0, 0.25, 0.5], ["0", "0.25", "0.5"])
        ax.set_title(title, loc="left", fontsize=8)
    axes[0, 0].invert_yaxis()  # shared y axis: invert once
    handles = {}
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    keys = list(handles)
    loc_, spr, shp = [k for k in keys if "part" in k]
    rest = [k for k in keys if "part" not in k]
    st = [k for k in rest if "Sobol" in k][0]
    jit = [k for k in rest if "Sobol" not in k][0]
    # matplotlib fills legend columns first: this order gives the rows
    # (location, spread, shape) and (S_Ti, jittered index)
    order = [loc_, st, spr, jit, shp]
    fig.legend([handles[k] for k in order], order, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "fig_ot.pdf")
    plt.close(fig)


OT_TABLE_LABELS = {"aggregation": "Counting rule", "ballot": "Ballot length", "cap_share": "Home cap",
                   "allocation": "Voters per region", "tiebreak": "Tie-break", "curve": "Vote curve",
                   "home_bias": "Home bias", "dummy": "Dummy"}
OT_TABLE_OUTPUTS = {
    "rho100": "(a) Rank correlation of the 2026 top 100 with the published order",
    "overlap50": "(b) Published top 50 kept in the top 50",
    "top10": "(c) Published top 10 kept in the top 10",
    "disp_lower": "(d) Mean displacement of the published 51--100, positions below No.200 counted as No.200",
    "regions50": "(e) Academy regions in the top 50",
    "europe50": "(f) European restaurants in the top 50",
}


def write_ot_table():
    """LaTeX rows of the table of OT indices in Appendix C."""
    idx = pd.read_csv(RES / "part3_indices.csv")
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"
    rows = []
    for y, title in OT_TABLE_OUTPUTS.items():
        d = idx[idx.output == y].set_index("factor")
        rows.append(f"\\multicolumn{{9}}{{@{{}}l}}{{{title}}} \\\\")
        for f in OT_ORDER:
            r = d.loc[f]
            jit = f3(r["iota_jit"]) if "iota_jit" in d.columns else "--"
            rows.append(f"\\quad {OT_TABLE_LABELS[f]} & {f3(r['iota'])} & {f3(r['iota_lo'])}--{f3(r['iota_hi'])} & "
                        f"{f3(r['iota_loc'])} & {f3(r['iota_spread'])} & {f3(r['iota_shape'])} & "
                        f"{f3(r['S'])} & {f3(r['ST'])} & {jit} \\\\")
        if y != list(OT_TABLE_OUTPUTS)[-1]:
            rows.append("\\addlinespace")
    (lists.ROOT / "report" / "table_ot.tex").write_text("\n".join(rows) + "\n")


EFFECT_ROWS = [("Counting", "aggregation", [("approval", "approval"), ("borda", "Borda"), ("dowdall", "Dowdall")]),
               ("Ballot", "ballot", [(10, "10"), (7, "7")]),
               ("Home cap", "cap_share", [(0.6, "60\\%"), (0.3, "30\\%"), (1.0, "none")]),
               ("Voters", "allocation", [("equal", "40 per region"), ("supply", "by supply")])]


def write_effects_table():
    """LaTeX rows of the table of rule effects in Section 6."""
    e = pd.read_csv(RES / "part3_rule_effects.csv")
    e["level"] = e["level"].astype(str)
    rows = []
    for label, factor, levels in EFFECT_ROWS:
        for k, (lev, name) in enumerate(levels):
            r = e[(e.factor == factor) & (e.level == str(lev))].iloc[0]
            rows.append(f"{label if k == 0 else ''} & {name} & {r.rho100:.2f} & {r.overlap50:.1f} & {r.top10:.1f} & "
                        f"{r.disp_lower:.1f} & {r.regions50:.1f} & {r.europe50:.1f} \\\\")
    (lists.ROOT / "report" / "table_effects.tex").write_text("\n".join(rows) + "\n")


def write_interval_table():
    """LaTeX rows of Appendix B (published rules, pooled over the worlds)."""
    ri = pd.read_csv(RES / "part3_rank_intervals.csv")
    l = pd.read_csv(lists.RAW / "list_2026_1_100.csv")
    ri = ri.merge(l[["rank", "city"]], left_on="published_rank", right_on="rank")
    rows = [f"{int(r.published_rank)} & {r['name'].replace('&', chr(92) + '&')} & {r.city} & "
            f"{r.off_q05:.0f}--{r.off_q95:.0f} & {r.off_p_top50:.2f} \\\\" for _, r in ri.iterrows()]
    (lists.ROOT / "report" / "table_intervals.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_pairs(); fig_churn()
    if (RES / "part3_indices.csv").exists():
        fig_intervals(); fig_indices(); fig_ot(); write_interval_table(); write_ot_table(); write_effects_table()
    print("figures written to", FIG)
