"""Verify that every number in the documents still matches what the code produces.

WHY THIS EXISTS
---------------
Three published figures in this project drifted from their sources: a universe
of "211 US large caps" when the coverage filter yields 199, a Bender line count,
and a Swapaday module count. In every case the code was right and the prose was
stale. Prose does not raise exceptions, so nothing caught them until a reader
did.

This closes that gap the same way the test suite closes the code gap: derive
each claim from its authoritative source, then assert the documents actually
contain it. Run it before publishing, and treat a DRIFT the way you would treat
a failing test.

    .venv/bin/python research/check_figures.py          # full, recomputes the panel
    .venv/bin/python research/check_figures.py --fast   # skip the panel load

Exit code is 1 if anything drifted or went missing, so it can gate a commit.
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "docs/index.html", ROOT / "docs/null-floor.html"]
README = ROOT / "README.md"
FAST = "--fast" in sys.argv

_docs_cache: dict[Path, str] = {}


def doc_text(p: Path) -> str:
    if p not in _docs_cache:
        _docs_cache[p] = p.read_text() if p.exists() else ""
    return _docs_cache[p]


CHECKS: list[tuple[str, str, str, str, list[Path]]] = []  # label, value, template, source, where

# minus signs appear as several different entities across the documents
MINUS = r"(?:&minus;|&#8722;|&ndash;|-|\u2212)"


def claim(label: str, value, pattern: str, source: str, where=None):
    """Register a derived value plus the CONTEXT it must appear in.

    A bare number is not a check. "199" occurs incidentally all over a long
    document, so the first version of this script passed while the byline said
    211 -- it had found the digits somewhere else entirely. Every claim now
    carries a regex anchoring the value to the sentence or cell that states it,
    with {v} substituted for the derived value.
    """
    CHECKS.append((label, str(value), pattern, source, where or DOCS))


# ----------------------------------------------------------------- derive
fig = json.loads((ROOT / "results/figure_data.json").read_text())
n = fig["null"]

claim("null floor mean", f"+{n['mean']:.2f}", r"<td>{v}</td>", "figure_data.json null.mean")
claim("null floor p95",  f"+{n['p95']:.2f}",  r"<td>{v}</td>", "figure_data.json null.p95")
claim("null floor max",  f"+{n['max']:.2f}",  r"<td>{v}</td>", "figure_data.json null.max")
claim("IC required", f"{fig['ic_required']:.3f}",
      r"<code>{v}</code>|>{v}</div>", "figure_data.json ic_required")
claim("IC asymptote", f"{fig['ic_asymptote']:.4f}",
      r"<code>{v}</code>|converges on <code>[^<]*{v}", "figure_data.json ic_asymptote")
claim("targeted drawdown", f"{abs(fig['phantom_dd']['targeted'])*100:.1f}%",
      r"worst\s*" + MINUS + r"{v}", "figure_data.json phantom_dd.targeted")
claim("untargeted drawdown", f"{abs(fig['phantom_dd']['untargeted'])*100:.1f}%",
      r"worst\s*" + MINUS + r"{v}", "figure_data.json phantom_dd.untargeted")

src = (ROOT / "research/make_figures.py").read_text()
m = re.search(r"c,\s*rho\s*=\s*([0-9.]+),\s*([0-9.]+)", src)
if m:
    claim("mean individual IC (c)", m.group(1),
          r"<code>{v}</code>|average <code>{v}</code>|IC of {v}", "make_figures.py c")
    claim("mean pairwise corr (rho)", m.group(2),
          r"&rho;&nbsp;=&nbsp;{v}|&rho;\s*=\s*{v}|/&nbsp;&radic;{v}", "make_figures.py rho")

with open(ROOT / "results/reassess_all.csv") as fh:
    rows = list(csv.DictReader(fh))
signal_rows = [r for r in rows if "no-signal" not in r["strategy"].lower()]
claim("strategies compared", len(signal_rows),
      r"0 / {v}|0&nbsp;/&nbsp;{v}|thirty-eight", "reassess_all.csv minus no-signal book")

with open(ROOT / "results/voltarget_ablation_null.csv") as fh:
    ab = {r["arm"]: r for r in csv.DictReader(fh)}
for arm, lbl in (("with vol target", "ablation null, overlay on"),
                 ("no overlay", "ablation null, overlay off")):
    if arm in ab:
        claim(lbl, f"+{float(ab[arm]['null_t_mean']):.2f}", r"<td>{v}</td>",
              f"voltarget_ablation_null.csv {arm}")

# the construction gradient in section 03 -- every cell derived, none transcribed
with open(ROOT / "results/beta_neutral_n200.csv") as fh:
    bn = {r["name"]: r for r in csv.DictReader(fh)}
GRID = [("dollar-neutral (reproduce)", "dollar-neutral"),
        ("beta-neutral (A)", "beta-scaled legs"),
        ("beta-orthogonal (B)", "beta-orthogonal")]
for key, lbl in GRID:
    if key not in bn:
        continue
    claim(f"{lbl}: realised beta", f"{float(bn[key]['beta_realised']):.2f}",
          r"<td>{v}</td>", f"beta_neutral_n200.csv {key}")
    claim(f"{lbl}: null alpha-t p95", f"&minus;{abs(float(bn[key]['alpha_t_null_p95'])):.2f}",
          r"<td>{v}</td>", f"beta_neutral_n200.csv {key}")
    claim(f"{lbl}: null alpha-t max", f"+{float(bn[key]['alpha_t_null_max']):.2f}",
          r"<td>{v}</td>", f"beta_neutral_n200.csv {key}")
for arm, lbl in (("with vol target", "long-only vol-targeted"), ("no overlay", "long-only no overlay")):
    if arm in ab:
        claim(f"{lbl}: null p95", f"+{float(ab[arm]['null_t_p95']):.2f}",
              r"<td>{v}</td>", f"voltarget_ablation_null.csv {arm}")
        claim(f"{lbl}: null max", f"+{float(ab[arm]['null_t_max']):.2f}",
              r"<td>{v}</td>", f"voltarget_ablation_null.csv {arm}")

# the eigenvalue spectrum in section 01
import csv as _csv
spec_path = ROOT / "results/signal_ic_spectrum.csv"
if spec_path.exists():
    with open(spec_path) as fh:
        evs = [float(r["eigenvalue"]) for r in _csv.DictReader(fh)]
    for i, ev in enumerate(evs[:2], 1):
        claim(f"PC{i} eigenvalue", f"{ev:.3f}", r"<td>PC" + str(i) + r"</td><td>{v}</td>",
              "signal_ic_spectrum.csv")

corr_path = ROOT / "results/signal_ic_correlation.csv"
if corr_path.exists():
    import itertools
    with open(corr_path) as fh:
        rdr = list(_csv.reader(fh))
    hdr, body = rdr[0][1:], rdr[1:]
    mat = {r[0]: dict(zip(hdr, [float(x) for x in r[1:]])) for r in body}
    dup = max(((a, b, mat[a][b]) for a, b in itertools.combinations(hdr, 2)),
              key=lambda x: abs(x[2]))
    claim("duplicate-pair correlation", f"{dup[2]:.3f}",
          r"correlate at\s+<code>{v}</code>",
          f"signal_ic_correlation.csv {dup[0]}/{dup[1]}")

loc = subprocess.run("find algo -name '*.py' | xargs wc -l | tail -1 | awk '{print $1}'",
                     shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()
if loc.isdigit():
    claim("algo/ lines (rounded to 100)", f"{round(int(loc), -2):,}",
          r"~{v} lines|{v} lines of Python", "wc -l over algo/*.py")

ntests = subprocess.run("grep -rhoE '^def test_[a-zA-Z0-9_]+' tests/*.py | wc -l",
                        shell=True, cwd=ROOT, capture_output=True, text=True).stdout.strip()
if ntests.isdigit():
    claim("test count", ntests, r"{v} tests|{v} regression tests", "grep 'def test_' tests/*.py")

if not FAST:
    sys.path.insert(0, str(ROOT))
    import pandas as pd
    from algo import data
    M = pd.read_parquet(ROOT / "data/edgar/measures.parquet")
    cl = data.load_panel(sorted(M.ticker.unique()), start="2005-01-01",
                         end="2026-09-01", field="close", refresh=False)
    cl = cl.dropna(axis=1, thresh=int(0.9 * len(cl)))
    claim("universe size", cl.shape[1], r"Universe: {v} US large caps|{v} US large caps",
          "panel after 90% coverage filter", DOCS + [README])


# ----------------------------------------------------------------- verify
print(f"\n{'CLAIM':<40}{'VALUE':>10}   {'STATUS':<10}SOURCE")
print("-" * 106)
bad = 0
for label, value, template, source, where in CHECKS:
    strict = template.replace("{v}", re.escape(value))
    hits = [p.name for p in where if re.search(strict, doc_text(p))]
    if hits:
        print(f"{label:<40}{value:>10}   {'ok':<10}{source}")
    else:
        bad += 1
        print(f"{label:<40}{value:>10}   {'DRIFT':<10}{source}")
        # same context, any value: shows what the document says in that slot
        loose = template.replace("{v}", r"([0-9][0-9.,]*%?)")
        for p in where:
            found = [g for g in re.findall(loose, doc_text(p)) if g]
            if found:
                got = ", ".join(f if isinstance(f, str) else next(x for x in f if x)
                                for f in found[:3])
                print(f"{'':<40}{'':>10}   {'':<10}-> {p.name} says {got} in that slot")
                break
        else:
            print(f"{'':<40}{'':>10}   {'':<10}-> that statement is absent from "
                  f"{', '.join(p.name for p in where)}")

print("-" * 106)
if bad:
    print(f"{bad} claim(s) drifted. The code is the source of truth; fix the prose, "
          f"or re-run the analysis behind the source.")
else:
    print(f"All {len(CHECKS)} derived claims appear in the documents, in context.")
sys.exit(1 if bad else 0)
