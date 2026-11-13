"""Checks of the limits that hold under any vote curve, and of the offline
parts of the archiving tool."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import bounds  # noqa: E402
import archive_sources  # noqa: E402


def test_tie_floor():
    assert bounds.tie_floor(10, 49) == 39
    assert bounds.tie_floor(49, 49) == 0
    assert bounds.tie_floor(80, 49) == 0


def test_reversal_probability():
    assert abs(bounds.reversal_prob(20.0, 20.0) - 0.5) < 1e-12
    assert abs(bounds.reversal_prob(2.0, 0.0) - 0.5 * (1 - 2.0 / bounds.N_VOTERS) ** bounds.N_VOTERS) < 1e-12
    # further apart, fewer reversals; closer from below, more reversals
    assert bounds.reversal_prob(40.0, 20.0) < bounds.reversal_prob(30.0, 20.0)
    assert bounds.reversal_prob(30.0, 25.0) > bounds.reversal_prob(30.0, 20.0)


def test_chain_and_bound():
    lv = bounds.chain(0.05, 30)
    assert lv[0] == 0 and np.all(np.diff(lv) > 0)
    for c0, c1 in zip(lv[:-1], lv[1:]):
        assert abs(bounds.reversal_prob(c1, c0) - 0.05) < 1e-6
    lo = bounds.max_resolved(51, 100, 0.05, bounds.chain(0.05, 50))
    hi = bounds.max_resolved(51, 100, 0.25, bounds.chain(0.25, 50))
    assert 0 < lo < hi <= 49
    # more votes can only resolve more pairs
    assert bounds.max_resolved(51, 100, 0.05, bounds.chain(0.05, 50), total=2 * bounds.TOTAL_VOTES) >= lo


def test_archive_tool_parses_bibliography():
    jobs = archive_sources.bib_urls((ROOT / "report" / "refs.bib").read_text(encoding="utf-8"))
    keys = {k for k, _ in jobs}
    assert {"the50voting2026", "w50bdeloitte", "the50archive"} <= keys
    assert all(u.startswith("http") for _, u in jobs)
    assert archive_sources.norm("<p>Deloitte  will&nbsp;apply</p>") == "deloitte will apply"


def test_grouped_counts_are_less_variable_and_loosen_the_limit():
    k = np.arange(bounds.N_VOTERS + 1)
    for m in (5.0, 30.0, 120.0):
        b, g = bounds._pmf(m), bounds._pmf(m, 0.6)
        assert abs(g.sum() - 1) < 1e-9 and abs(k @ g - m) < 1e-6
        assert k ** 2 @ g - m ** 2 < k ** 2 @ b - m ** 2
    lo = bounds.max_resolved(51, 100, 0.05, bounds.chain(0.05, 50))
    hi = bounds.max_resolved(51, 100, 0.05, bounds.chain(0.05, 50, 1.0))
    assert lo <= hi < 49


class _Resp:
    def __init__(self, text="", data=None):
        self.text, self._data, self.status_code, self.headers, self.url = text, data, 200, {}, ""

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


class _Session:
    """Offline stand-in for the Wayback Machine: two captures before the
    rebrand, of which only the earlier holds the quoted sentence."""
    def __init__(self, captures):
        self.captures = captures

    def get(self, url, params=None, **kw):
        if "cdx" in url:
            rows = [["timestamp", "original"]] + [[t, params["url"]] for t in self.captures]
            return _Resp(text="x" if self.captures else "", data=rows)
        ts = url.split("/web/")[1].split("id_/")[0]
        return _Resp(text="<p>Each region has its own <b>voting panel</b> of 40 members</p>"
                     if ts == "20250301000000" else "<p>Page moved</p>")


def test_archive_tool_walks_back_to_a_capture_with_the_quoted_text():
    phrase = ["each region has its own voting panel of 40 members"]
    s = _Session(["20250301000000", "20260815000000"])
    assert archive_sources.before_rebrand(s, "https://x.org/p", phrase) == ("20250301000000", "True")
    assert archive_sources.before_rebrand(s, "https://x.org/p", []) == ("20260815000000", "")
    s_none = _Session(["20260815000000"])
    assert archive_sources.before_rebrand(s_none, "https://x.org/p", phrase) == ("20260815000000", "False")
    assert archive_sources.before_rebrand(_Session([]), "https://x.org/p", phrase) == (None, "")


def test_archive_tool_end_to_end_offline(tmp_path, monkeypatch):
    """The whole run against a stand-in for the Wayback Machine: a former-domain
    page whose latest capture lacks the quoted text, a page saved at once, and
    a page whose save request fails but whose capture exists, a page that
    refuses the archive's crawler but was captured earlier, and one that was
    never captured.  A second run keeps what succeeded and retries the rest."""
    import types
    old = "https://www.theworlds50best.com/voting/the-voting-system"
    ok, slow, moved = "https://example.org/ok", "https://example.org/slow", "https://example.org/?p=1"
    blocked, never = "https://example.org/blocked", "https://example.org/never"
    bib = tmp_path / "refs.bib"
    bib.write_text("@misc{oldpage,\n  howpublished = {\\url{%s}}\n}\n@misc{okpage,\n  howpublished = {\\url{%s}}\n}\n"
                   "@misc{slowpage,\n  howpublished = {\\url{%s}}\n}\n@misc{movedpage,\n  howpublished = {\\url{%s}}\n}\n"
                   "@misc{blockedpage,\n  howpublished = {\\url{%s}}\n}\n@misc{neverpage,\n  howpublished = {\\url{%s}}\n}\n"
                   % (old, ok, slow, moved, blocked, never))
    phr = tmp_path / "phrases.csv"
    phr.write_text("url,phrase\n%s,each region has its own voting panel of 40 members\n" % old)
    calls = {"save_slow": 0, "save_blocked": 0}

    class Resp(_Resp):
        pass

    class Session:
        headers = {}

        def get(self, url, params=None, **kw):
            if "cdx" in url:
                if params["url"] == old:
                    caps = ["20250301000000", "20260815000000"]
                elif params["url"] == slow and params.get("from"):
                    caps = ["20261113120000"]
                elif params["url"] == blocked and not params.get("from"):
                    caps = ["20240115000000"]
                else:
                    caps = []
                rows = [["timestamp", "original"]] + [[c, params["url"]] for c in caps]
                return Resp(text="x" if caps else "", data=rows)
            if "/save/" in url:
                r = Resp()
                if url.endswith(ok):
                    r.url = "https://web.archive.org/web/20261113110000/" + ok
                elif url.endswith(moved):
                    r.url = "https://web.archive.org/web/20261113110500/https://example.org/another-story/"
                else:
                    calls["save_blocked" if url.endswith(blocked) else "save_slow"] += url.endswith((slow, blocked))
                    r.url = url
                return r
            ts = url.split("/web/")[1].split("id_/")[0]
            return Resp(text="<p>Each region has its own voting panel of 40 members</p>"
                        if ts == "20250301000000" else "<p>moved</p>")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(Session=Session))
    monkeypatch.setattr(archive_sources, "BIB", bib)
    monkeypatch.setattr(archive_sources, "PHRASES", phr)
    monkeypatch.setattr(archive_sources, "OUT_CSV", tmp_path / "archived.csv")
    monkeypatch.setattr(archive_sources, "OUT_TEX", tmp_path / "archive_urls.tex")
    monkeypatch.setattr(archive_sources, "ARCHIVE_YEARS", [])
    monkeypatch.setattr(archive_sources.time, "sleep", lambda s: None)
    assert archive_sources.main(["--pause", "0"]) == 0
    tex = (tmp_path / "archive_urls.tex").read_text()
    assert "\\archiveurl{oldpage}{https://web.archive.org/web/20250301000000/" + old + "}" in tex
    assert "\\archiveurl{okpage}{https://web.archive.org/web/20261113110000/" + ok + "}" in tex
    assert "\\archiveurl{slowpage}{https://web.archive.org/web/20261113120000/" + slow + "}" in tex
    # a page that refuses the crawler gets its latest earlier capture, a page never captured gets nothing
    assert "\\archiveurl{blockedpage}{https://web.archive.org/web/20240115000000/" + blocked + "}" in tex
    assert "neverpage" not in tex
    assert 'neverpage,https://example.org/never,saved now,,,"save request failed' in (tmp_path / "archived.csv").read_text()
    # a save that lands on another page is recorded but not cited
    assert "movedpage" not in tex
    assert "redirected to https://example.org/another-story/" in (tmp_path / "archived.csv").read_text()
    # a second run keeps the successes and asks for nothing again
    assert archive_sources.main(["--pause", "0"]) == 0
    assert calls["save_slow"] == 1 and calls["save_blocked"] == 1
    assert (tmp_path / "archive_urls.tex").read_text() == tex
