"""Archive every web source of the report in the Wayback Machine.

Run locally (it needs network access to web.archive.org):

    python3 tools/archive_sources.py            # look up or create snapshots
    python3 tools/archive_sources.py --dry-run  # list what would be done

For each \\url{...} in report/refs.bib, apart from this repository, which
Zenodo archives, and for each page of the organiser's archive of past lists,
the script

  * finds, for pages of the former domain theworlds50best.com, which now
    redirect, the latest snapshot taken before the September 2026 rebrand
    that contains every sentence the report relies on
    (data/raw/archive_phrases.csv), going back over the last ten captures;
  * asks the Wayback Machine to save the current version of every other page,
    and falls back on a capture made the same day if the request fails.

It writes data/raw/archived_sources.csv and report/archive_urls.tex.  The
report reads the latter, so that recompiling adds 'Archived at ...' to each
reference with a snapshot.  Nothing is changed for a page that fails, and the
script can be run again.
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "report" / "refs.bib"
OUT_CSV = ROOT / "data" / "raw" / "archived_sources.csv"
OUT_TEX = ROOT / "report" / "archive_urls.tex"
PHRASES = ROOT / "data" / "raw" / "archive_phrases.csv"
REBRAND = "20260831"          # last day before the rebrand of September 2026
ARCHIVE_YEARS = [y for y in range(2002, 2026) if y != 2020]
LIST_PAGE = "https://www.the50.com/restaurants/best-in-the-world/previous-list/{}"
UA = {"User-Agent": "w50b-audit-archiver/1.0 (research use)"}
REPO = "https://github.com/Confareneoclassico/50best-voting-audit"   # archived by Zenodo


def bib_urls(text: str) -> list[tuple[str, str]]:
    """(key, url) for every \\url{} in the bibliography."""
    out = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", text, re.S):
        key, body = m.group(1).strip(), m.group(2)
        for u in re.findall(r"\\url\{([^}]*)\}", body):
            out.append((key, u))
    return out


def norm(s: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", " ", s))
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def same_page(a: str, b: str) -> bool:
    """Whether two URLs name the same page, up to scheme, www. and a final slash."""
    norm_url = lambda u: re.sub(r"^https?://(www\.)?", "", u.strip()).rstrip("/").lower()
    return norm_url(a) == norm_url(b)


def snapshots(session, url: str, to: str | None = None, frm: str | None = None, last: int = 10) -> list[str]:
    """Timestamps of the last `last` captures of url with status 200 within
    the dates given (YYYYMMDD), oldest first."""
    params = {"url": url, "output": "json", "filter": "statuscode:200",
              "fl": "timestamp,original", "limit": str(-last)}
    if to:
        params["to"] = to
    if frm:
        params["from"] = frm
    r = session.get("https://web.archive.org/cdx/search/cdx", params=params, timeout=60)
    r.raise_for_status()
    rows = r.json() if r.text.strip() else []
    return [row[0] for row in rows[1:]]       # the first row is the header


def before_rebrand(session, url: str, phrases: list[str]) -> tuple[str | None, str]:
    """(timestamp, phrase_found) of the latest capture before the rebrand that
    contains every quoted phrase, going back over the last ten captures.  If
    none does, the latest capture with phrase_found 'False'.  Without phrases
    to check, the latest capture and ''."""
    ts = snapshots(session, url, to=REBRAND)
    if not ts:
        return None, ""
    if not phrases:
        return ts[-1], ""
    for t in reversed(ts):
        page = norm(session.get(f"https://web.archive.org/web/{t}id_/{url}", timeout=120).text)
        if all(norm(ph) in page for ph in phrases):
            return t, "True"
    return ts[-1], "False"


def save_now(session, url: str):
    """Ask the Wayback Machine to save url; return the snapshot URL or None."""
    for attempt in range(2):
        r = session.get("https://web.archive.org/save/" + url, timeout=180, allow_redirects=True)
        if r.status_code == 429 and attempt == 0:
            time.sleep(90)
            continue
        loc = r.headers.get("Content-Location", "")
        if "/web/" in r.url:
            return r.url
        if loc.startswith("/web/"):
            return "https://web.archive.org" + loc
        return None
    return None


FIELDS = ["key", "url", "kind", "snapshot", "phrase_found", "note"]


def write_csv(rows):
    """Write the table of snapshots (after every job, so that an interrupted
    run keeps what it has done)."""
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pause", type=float, default=15.0, help="seconds between save requests")
    args = ap.parse_args(argv)

    jobs = [(k, u) for k, u in bib_urls(BIB.read_text(encoding="utf-8")) if not u.startswith(REPO)]
    cited = {u for _, u in jobs}
    jobs += [(f"archive{y}", LIST_PAGE.format(y)) for y in ARCHIVE_YEARS if LIST_PAGE.format(y) not in cited]
    phrases: dict[str, list[str]] = {}
    if PHRASES.exists():
        with PHRASES.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                phrases.setdefault(row["url"], []).append(row["phrase"])
    if args.dry_run:
        for key, url in jobs:
            kind = "before rebrand" if "theworlds50best.com" in url else "save now"
            print(f"{kind:15s} {key:22s} {url}")
        return 0

    import requests  # imported here so that --dry-run works without it
    s = requests.Session()
    s.headers.update(UA)
    # earlier successes are kept, by bibliography key and URL; failures and
    # snapshots without the quoted text are tried again
    done = {}
    if OUT_CSV.exists():
        with OUT_CSV.open(encoding="utf-8") as f:
            done = {(row["key"], row["url"]): row for row in csv.DictReader(f)
                    if row["snapshot"] and row["phrase_found"] != "False"}
    rows = []
    for key, url in jobs:
        if (key, url) in done:
            rows.append(done[(key, url)])
            continue
        row = {"key": key, "url": url, "kind": "", "snapshot": "", "phrase_found": "", "note": ""}
        try:
            if "theworlds50best.com" in url:
                row["kind"] = "before rebrand"
                # the snapshot is recorded only once its text has been checked
                ts, found = before_rebrand(s, url, phrases.get(url, []))
                if ts is None:
                    row["note"] = "no capture before the rebrand"
                else:
                    row["snapshot"] = f"https://web.archive.org/web/{ts}/{url}"
                    row["phrase_found"] = found
            else:
                row["kind"] = "saved now"
                snap = save_now(s, url)
                if not snap:      # the capture may exist even if the reply did not say so
                    time.sleep(args.pause)
                    today = snapshots(s, url, frm=time.strftime("%Y%m%d", time.gmtime()), last=1)
                    snap = f"https://web.archive.org/web/{today[-1]}/{url}" if today else None
                row["snapshot"] = snap or ""
                row["note"] = "" if snap else "save request failed, run again later"
                if snap:   # a redirect may land on another page: flag it rather than cite it
                    landed = snap.split("/web/", 1)[1].split("/", 1)[1]
                    if not same_page(landed, url):
                        row["note"] = f"redirected to {landed}, check before citing"
                time.sleep(args.pause)
        except Exception as e:  # network errors are recorded, not fatal
            row["snapshot"], row["phrase_found"] = "", ""
            row["note"] = f"error: {e.__class__.__name__}"
        print(f"{row['kind']:15s} {key:22s} {row['snapshot'] or row['note']}", flush=True)
        rows.append(row)
        write_csv(rows + [r for k, r in done.items() if k not in {(x["key"], x["url"]) for x in rows}])

    write_csv(rows)
    # one snapshot per bibliography key: the first URL of the entry
    lines, seen = [], set()
    for r in rows:
        if r["snapshot"] and r["key"] not in seen and not r["key"].startswith("archive"):
            if r["phrase_found"] == "False" or r["note"].startswith("redirected"):
                continue        # do not cite a snapshot that lacks the quoted text or shows another page
            lines.append(f"\\archiveurl{{{r['key']}}}{{{r['snapshot']}}}")
            seen.add(r["key"])
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [r for r in rows if not r["snapshot"] or r["phrase_found"] == "False"
               or r["note"].startswith("redirected")]
    print(f"\n{len(rows) - len(missing)} of {len(rows)} sources archived. Rerun for the others:")
    for r in missing:
        print(f"  {r['key']}: {r['url']} ({r['note'] or 'quoted text not found in snapshot'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
