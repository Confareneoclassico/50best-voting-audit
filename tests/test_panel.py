"""Checks of the assembled panel against independently published figures."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lists  # noqa: E402


def test_ranks_complete_and_unique():
    p = lists.load_panel()
    assert lists.check_panel(p) == []


def test_2023_changes_match_published_column():
    """The 2023 extended list was published with 'change since 2022'; every
    value must follow from our 2022 and 2023 rows (this checks the 2022 list
    and the 2023 extended positions of the archive at once)."""
    p = lists.load_panel()
    v = pd.read_csv(lists.RAW / "validation_2023_changes.csv")
    r22 = p[p.year == 2022].set_index("rid")["rank"]
    r23 = p[p.year == 2023].set_index("rid")["rank"]
    for _, row in v.iterrows():
        rid = lists.canonical(row["name"], 2023)
        if row["change_since_2022"] == "New":
            assert rid not in r22.index, row["name"]
        else:
            expected = int(row["change_since_2022"])
            assert r22[rid] - row["rank_2023"] == expected, row["name"]
        assert r23[rid] == row["rank_2023"], row["name"]


def test_2016_and_2015_match_ajc_list():
    """The AJC printed the full 2016 list with each restaurant's 2015 rank.
    All 50 positions of 2016 must match the corrected panel, and so must every
    2015 rank within the top 50 (including the published tie at No.42).
    A 2015 rank below 50 must not appear in our 2015 top 50."""
    p = lists.load_panel()
    v = pd.read_csv(lists.RAW / "validation_2016_ajc.csv")
    r16 = p[p.year == 2016].set_index("rid")["rank"]
    r15 = p[p.year == 2015].set_index("rid")["rank"]
    assert len(v) == 50 and len(r16) == 50
    for _, row in v.iterrows():
        rid16 = lists.canonical(row["name"], 2016)
        assert r16[rid16] == row["rank_2016"], row["name"]
        rid15 = lists.canonical(row["name"], 2015)
        if row["rank_2015"] <= 50:
            assert r15[rid15] == row["rank_2015"], row["name"]
        else:
            assert rid15 not in r15.index, row["name"]


def test_panel_reproduces_organiser_archive():
    """Every position of the organiser's archive (2002-2025) is in the panel
    with the printed rank, except the four documented corrections, which
    restore the organiser's own announcements."""
    p = lists.load_panel()
    fix = pd.read_csv(lists.RAW / "corrections.csv")
    fixed = {(int(r.year), lists.canonical(r["name"], int(r.year))): int(r.rank_published) for _, r in fix.iterrows()}
    n_checked = 0
    for y in lists.ARCHIVE_YEARS:
        a = lists.read_archive(y)
        got = p[p.year == y].set_index("rid")["rank"]
        for _, row in a.iterrows():
            rid = lists.canonical(row["name"], y)
            assert got[rid] == fixed.get((y, rid), row["rank"]), (y, row["name"])
            n_checked += 1
    assert n_checked == 50 * 20 + 100 * 3


def test_archive_editions_are_complete():
    """Every archive edition has 50 places up to 2022 and 100 from 2023."""
    for y in lists.ARCHIVE_YEARS:
        a = lists.read_archive(y)
        assert len(a) == (100 if y >= 2023 else 50), y


def test_check_panel_rejects_unpublished_ties():
    p = lists.load_panel()
    bad = p.copy()
    i = bad.index[(bad.year == 2016) & (bad["rank"] == 37)][0]
    bad.loc[i, "rank"] = 22          # the error of the organiser's archive
    assert any("2016" in m for m in lists.check_panel(bad))


def test_canonical_aliases():
    c = lists.canonical
    assert c("Restaurant Tim Raue", 2024) == c("Tim Raue", 2026)
    assert c("Steirereck im Stadtpark", 2024) == c("Steirereck", 2026)
    assert c("Gaggan Anand", 2024) == c("Gaggan", 2026)
    assert c("Gaggan", 2019) != c("Gaggan", 2025)
    assert c("Noma", 2016) != c("Noma", 2019)
    assert c("Jordnær", 2023) == c("Jordnaer", 2023)
    assert c("Cracco Peck", 2007) == c("Ristorante Cracco", 2008)
    assert c("Hotel de Paris", 2002) == c("Le Louis XV", 2003)
    assert c("Il Pescatore", 2002) != c("Dal Pescatore", 2003)
