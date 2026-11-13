"""Assemble the published rankings into one panel with stable restaurant ids.

Sources (see data/raw/SOURCES.md):
  * organiser_archive/<year>.txt  the organiser's own archive of past lists
    (the50.com previous-list pages): ranks 1-50 for 2002-2022 and 1-100 for
    2023-2025, as printed, with the four fixes of corrections.csv, which
    restore the organiser's own announcements where the archive departs
    from them
  * list_2019_51_120.csv, list_2021_51_100.csv, list_2022_51_100.csv
    extended lists of those years, from press coverage of the organiser's
    releases (no longer on the organiser's site), each checked against a
    second source
  * list_2026_1_100.csv  the organiser's current list pages
  * restaurant_countries.csv  country of each restaurant
The assembled and corrected panel is written to data/derived/panel.csv.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"

# Editions with a published 51-100 (or 51-120) list that we hold.
FULL_YEARS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]

# Positions that the organiser published as ties (2015: Tickets and Boragó
# both No.42, no No.43).  Any other tie in the panel is an error.
PUBLISHED_TIES = {2015: {42}}

ALIASES = {
    "steirereck im stadtpark": "steirereck",
    "table bruno verjus": "table by bruno verjus",
    "coda dessert dining": "coda",
    "single thread": "singlethread",
    "mil centro": "mil",
    "orfali bros bistro": "orfali bros",
    "favikin": "faviken",
    "maeemo": "maaemo",
    "d o m": "dom",
    "nerua guggenheim bilbao": "nerua",
    "ricard camarena": "ricard camarena",
    "le louis xv alain ducasse": "le louis xv",
    "louis xv": "le louis xv",
    "le louis xv alain ducasse a lhotel de paris": "le louis xv",
    "nihonryori ryugin": "nihonryori ryugin",
    "astrid y gaston": "astrid y gaston",
    "amisfield": "amisfield",
    # spelling variants and other names of the same restaurant within the
    # organiser's archive, 2002-2013 (see Appendix A of the report)
    "le manoir aux quat saison": "le manoir aux quat saisons",
    "gambero roso": "gambero rosso",
    "mathias dhalgren": "mathias dahlgren",
    "de l hotel de ville": "l hotel de ville",
    "l hotel de ville philippe rochat": "l hotel de ville",
    "hotel de paris": "le louis xv",   # 2002, Monaco: the restaurant of the Hotel de Paris
    "cracco peck": "ristorante cracco",  # renamed in 2008
    "troisgros": "maison troisgros",
    "alain ducasse": "alain ducasse au plaza athenee",
    "bras": "michel bras",
    "frantzen lindeberg": "frantzen",
    # spellings of the 2016 list printed by the AJC (validation_2016_ajc.csv)
    "amber at the landmark mandarin oriental": "amber",
    "ultra violet": "ultraviolet by paul pairet",
    "schloss schauenstein in furstenau": "schloss schauenstein",
    "dinner": "dinner by heston blumenthal",
    "quiquedacosta": "quique dacosta",
    "ryugin": "nihonryori ryugin",
    "l arpege": "arpege",
}

US_CA_REGION = {
    "New York": "East USA & East Canada",
    "Pocantico Hills": "East USA & East Canada",
    "Washington": "East USA & East Canada",
    "Montreal": "East USA & East Canada",
    "Chicago": "Mid USA & Mid Canada",
    "Toronto": "Mid USA & Mid Canada",
    "San Francisco": "West USA & West Canada",
    "Healdsburg": "West USA & West Canada",
    "Yountville": "West USA & West Canada",
    "Los Angeles": "West USA & West Canada",
    "Vancouver": "West USA & West Canada",
}

EUROPE = {
    "UK & Ireland", "Italy", "Spain & Portugal", "France", "Germany", "Benelux",
    "Austria, Switzerland, Hungary & Slovenia", "Scandinavia & Baltics",
    "Russia, Central Asia & Eastern Europe", "Türkiye, Greece & the Balkans",
}


def _strip(s: str) -> str:
    s = s.replace("æ", "ae").replace("Æ", "Ae").replace("ø", "o").replace("Ø", "O")
    s = s.replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def canonical(name: str, year: int) -> str:
    """Stable id for a restaurant name as printed in a given edition."""
    s = _strip(str(name)).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[.,'’`\-–—:()/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pre in ("restaurant ", "restaurante "):
        if s.startswith(pre):
            s = s[len(pre):]
    for suf in (" restaurant", " restaurante"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = ALIASES.get(s, s)
    # Two names refer to different restaurants depending on the edition.
    if s in ("gaggan", "gaggan anand"):
        # The original Gaggan closed in 2019; Gaggan Anand (later printed
        # simply as Gaggan) entered as a new restaurant in 2023.
        s = "gaggan anand" if year >= 2023 else "gaggan (original)"
    if s == "noma":
        # Noma reopened in a new location in 2018 and was treated as a new
        # restaurant (eligible again in 2019 and 2021).
        s = "noma 2.0" if year >= 2019 else "noma (original)"
    return s


def region_of(country: str, city: str, regions: dict[str, str]) -> str:
    reg = regions.get(country)
    if reg is None:
        raise KeyError(f"no Academy region for {country!r} ({city})")
    if reg == "USA_CANADA_BY_CITY":
        if city not in US_CA_REGION:
            raise KeyError(f"no US/Canada sub-region for {city!r}")
        return US_CA_REGION[city]
    return reg


ARCHIVE = RAW / "organiser_archive"
ARCHIVE_YEARS = [y for y in range(2002, 2026) if y != 2020]
EXTENDED = {2019: "list_2019_51_120.csv", 2021: "list_2021_51_100.csv",
            2022: "list_2022_51_100.csv", 2026: "list_2026_1_100.csv"}


def read_archive(year: int) -> pd.DataFrame:
    """One edition of the organiser's archive, as printed ('rank|name|city')."""
    rows = [line.split("|") for line in (ARCHIVE / f"{year}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    d = pd.DataFrame(rows, columns=["rank", "name", "city"])
    d["rank"] = d["rank"].str.lstrip("=").astype(int)
    d["year"] = year
    d["source"] = f"organiser_archive/{year}.txt"
    return d


def apply_corrections(arch: pd.DataFrame) -> pd.DataFrame:
    """Apply data/raw/corrections.csv to the archive rows.
    Each correction must find exactly one row with the rank it replaces."""
    arch = arch.copy()
    fix = pd.read_csv(RAW / "corrections.csv")
    arch["_rid"] = [canonical(n, y) for n, y in zip(arch["name"], arch["year"])]
    for _, f in fix.iterrows():
        sel = (arch["year"] == f["year"]) & (arch["_rid"] == canonical(f["name"], f["year"]))
        if sel.sum() != 1 or int(arch.loc[sel, "rank"].iloc[0]) != int(f["rank_in_archive"]):
            raise ValueError(f"correction does not match the archive: {f['year']} {f['name']}")
        arch.loc[sel, "rank"] = int(f["rank_published"])
        arch.loc[sel, "source"] = arch.loc[sel, "source"] + ", corrected"
    return arch.drop(columns="_rid")


def load_panel() -> pd.DataFrame:
    regions = pd.read_csv(RAW / "academy_regions_2026.csv").set_index("country")["region"].to_dict()
    arch = apply_corrections(pd.concat([read_archive(y) for y in ARCHIVE_YEARS], ignore_index=True))
    frames = [arch]
    for y, f in EXTENDED.items():
        d = pd.read_csv(RAW / f)
        d["year"] = y
        d["source"] = f
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p["name"] = p["name"].astype(str).str.strip()
    p["city"] = p["city"].astype(str).str.strip()
    p["rid"] = [canonical(n, y) for n, y in zip(p["name"], p["year"])]
    # the archive prints a city only: the country comes from the restaurant table
    countries = pd.read_csv(RAW / "restaurant_countries.csv").set_index("rid")["country"]
    p["country"] = p["country"].fillna(p["rid"].map(countries))
    p["country"] = p["country"].replace({"Turkey": "Türkiye", "USA": "United States", "UK": "United Kingdom"})
    if p["country"].isna().any():
        raise KeyError(f"no country for {sorted(p.loc[p['country'].isna(), 'rid'].unique())}")
    p["region"] = [region_of(c, ci, regions) if y >= 2019 else None
                   for c, ci, y in zip(p["country"], p["city"], p["year"])]
    p["europe"] = p["region"].isin(EUROPE)
    p = p.sort_values(["year", "rank"]).reset_index(drop=True)
    return p


def check_panel(p: pd.DataFrame) -> list[str]:
    """Integrity checks; returns a list of messages (empty if all pass)."""
    msgs = []
    for y, g in p.groupby("year"):
        top = 120 if y == 2019 else (100 if y in FULL_YEARS else 50)
        ranks = sorted(int(r) for r in g["rank"])
        # standard competition ranking: each position equals one plus the
        # number of restaurants placed above it (1, 2, 2, 4, ...)
        expected = [1 + sum(r2 < r for r2 in ranks) for r in ranks]
        if len(ranks) != top or ranks != expected or ranks[-1] > top:
            msgs.append(f"{y}: ranks are not a complete ranking of {top} places")
        tied = {r for r in ranks if ranks.count(r) > 1}
        if tied != PUBLISHED_TIES.get(y, set()):
            msgs.append(f"{y}: ties at {sorted(tied)} differ from the published ties")
        dup = g["rid"][g["rid"].duplicated()].tolist()
        if dup:
            msgs.append(f"{y}: duplicated restaurant ids {dup}")
    return msgs


if __name__ == "__main__":
    DERIVED.mkdir(parents=True, exist_ok=True)
    p = load_panel()
    msgs = check_panel(p)
    if msgs:
        raise SystemExit("panel checks failed:\n" + "\n".join(msgs))
    print("panel checks passed")
    p.to_csv(DERIVED / "panel.csv", index=False)
    print(p.groupby("year")["rank"].agg(["count", "max"]).T)
