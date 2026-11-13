# A sensitivity audit of The 50 Best Restaurants voting system

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24199990.svg)](https://doi.org/10.5281/zenodo.24199990)

Code, data and working draft for an audit of the voting system behind The 50 Best Restaurants (The World's 50 Best Restaurants until September 2026). The audit separates four levels of evidence: what the published lists 2002 to 2026 show about rank stability (Part 1), limits on resolution that hold under any distribution of votes (`src/bounds.py`), how many positions a vote of about 1,120 ten-restaurant ballots can resolve under scenarios for that distribution (Part 2, binomial model of panel noise), and a global sensitivity analysis of the voting rules with a Plackett-Luce ballot model calibrated to the 2026 list (Part 3).

## Reproduce

Python 3.10 or later with the packages in `requirements.txt`, and a LaTeX installation for the report. The results in `results/` were produced with Python 3.13 and the exact versions in `requirements-lock.txt`. Other versions of numpy may draw different random streams and so give slightly different simulated numbers.

```
make all        # panel, three parts, extra numbers, figures, report (about 5 minutes on 2 cores)
make test       # checks of the panel, the limits, the estimators and the archiving tool
```

`make all` runs, in order:

1. `src/lists.py` assembles and checks the panel of all editions (`data/derived/panel.csv`).
2. `src/churn.py` runs Part 1.
3. `src/bounds.py` computes the limits that hold under any vote curve, with independent binomial counts and, as a check, with voters grouped by region.
4. `src/noise.py` runs Part 2.
5. `src/gsa.py` runs Part 3. It calibrates six worlds, simulates 6,912 editions and computes exact factorial Sobol' indices and optimal-transport indices with the exact one-dimensional W2, split into location, spread and shape parts, with delete-one-panel jackknife intervals.
6. `src/extras.py` computes the numbers quoted in the text.
7. `src/figures.py` draws the figures and writes the generated tables.
8. The paper `report/audit.tex` and its supplementary material `report/supplement.tex` are compiled.

Every random stream is seeded, and results go to `results/`.

## Data

The published lists are in `data/raw/`, with their provenance in `data/raw/SOURCES.md`. Every edition from 2002 to 2025 is in `data/raw/organiser_archive/`, one file per year, as printed in the organiser's own archive of past lists (positions 1-50 to 2022, 1-100 from 2023). The extended lists of 2019, 2021 and 2022, which the archive does not hold, are in the files named by year, and the 2026 list comes from the organiser's current pages. The assembled panel of all editions from 2002 to 2026, with the corrections applied, is `data/derived/panel.csv`.

The archive departs from the organiser's own announcements at four positions (a published tie in 2015, two positions in 2016 and one in 2025). `data/raw/corrections.csv` restores the announced positions, with sources. `tests/test_panel.py` checks that the panel reproduces every position of the archive apart from these four, the published 2023 changes since 2022 for all 50 restaurants of the extended list, and the full 2016 list printed with 2015 ranks.

## Archiving the web sources

Several voting rules are documented only on pages of the former domain theworlds50best.com, which redirect to the50.com since the September 2026 rebrand. `tools/archive_sources.py` (or `make archive`) needs network access to web.archive.org, so run it on your own machine. For each page of the former domain it takes the latest Wayback Machine capture before the rebrand that contains every sentence the report quotes (`data/raw/archive_phrases.csv`), and it asks the Wayback Machine to save every other source in `report/refs.bib` and each page of the organiser's archive. A save that lands on another page after a redirect is recorded but not cited. The results go to `data/raw/archived_sources.csv` and `report/archive_urls.tex`, and recompiling the report adds an "Archived at" link to each reference. A rerun keeps what succeeded and retries the rest, and `python3 tools/archive_sources.py --dry-run` lists the jobs without touching the network.

The run of 13 November 2026 archived 57 of the 61 sources, and the quoted sentences were found in the snapshots of all four quoted pages of the former domain that the Wayback Machine had captured. Not archived: the Asia voting page of the former domain, which the Wayback Machine never captured, and three press pages whose save requests failed (falstaff2022, sezanne2026, restaurantnews2024), which a later run can add. One saved page redirected to an unrelated article, and the report cites another source for its point.

## Model choices to keep in mind

Vote counts are not published. Parts 2 and 3 therefore report results over scenarios for the expected votes by position (a power law fixed by the votes of No.100 and the ratio of No.1 to No.100), constrained only by the observed year-on-year churn. In Part 3 the home unit is the Academy region, whereas the 2026 rule caps votes per country, territory or Special Administrative Region. Restaurants below No.100 are spread evenly over the 28 regions, and the regional normalisation mentioned on the adjudicator's former page is not modelled because it is not defined. The limits of `src/bounds.py` assume independent binomial counts. Grouping the voters by region, as the Academy does, loosens them a little, and `results/bounds_resolution.csv` reports by how much.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The report text, figures and derived results are released under CC BY 4.0. The published lists in `data/raw` record facts published by the organiser and the press, listed with their sources in `data/raw/SOURCES.md`, and are not covered by either licence. To cite the code and data: Lo Piano, S. (2026). A sensitivity audit of The 50 Best Restaurants voting system: code and data (version 1.0.0). Zenodo. https://doi.org/10.5281/zenodo.24199990. The same reference is in `CITATION.cff`.
