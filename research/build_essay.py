"""Build docs/null-floor.html from writing/what-does-nothing-score.md.

The prose lives in markdown so it stays editable and portable; the figures are
generated from results/figure_data.json so they cannot drift from the numbers
they depict. Running this script is the only supported way to produce the page
-- editing the HTML directly means the next build silently discards the edit.

Figures are inserted at anchor sentences rather than at markers in the markdown,
so the markdown stays clean for anywhere it might be republished.
"""
import html
import json
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = json.loads((ROOT / "results/figure_data.json").read_text())


def method_diagram() -> str:
    W, H = 640, 238
    IN_X, IN_W, BH = 6, 132, 46
    PX, PW = 176, 232
    OX, OW = 448, 186
    RA, RB = 72, 160

    def box(x, y, w, h, fill="var(--raised)", stroke="var(--rule-strong)"):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="1"/>')

    def arrow(x1, y, x2, c="var(--rule-strong)"):
        return (f'<line x1="{x1}" y1="{y}" x2="{x2-6}" y2="{y}" stroke="{c}" '
                f'stroke-width="1.2" fill="none"/>'
                f'<polygon points="{x2},{y} {x2-6},{y-3.5} {x2-6},{y+3.5}" fill="{c}"/>')

    steps = ["universe", "top-k selection", "inverse-vol weighting",
             "volatility target", "costs, rebalancing"]
    step_text = "".join(
        f'<text x="{PX+PW/2}" y="{82+18*i}">{s}</text>' for i, s in enumerate(steps))

    return f'''<figure class="fig">
  <svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img"
       aria-label="Two inputs pass through one identical pipeline. A real signal produces one result; forty random signals produce a null distribution. The result is reported as a percentile against that distribution.">
    <text x="0" y="12" font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--faint)" letter-spacing="1.2">HOLD THE PIPELINE CONSTANT, VARY ONLY THE INFORMATION</text>
    {box(PX, 36, PW, 150, "var(--sunk)", "var(--accent)")}
    <text x="{PX+PW/2}" y="58" font-family="IBM Plex Sans, sans-serif" font-size="11.5" font-weight="600" fill="var(--ink)" text-anchor="middle">identical pipeline</text>
    <g font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--muted)" text-anchor="middle">{step_text}</g>
    <text x="{PX+PW/2}" y="174" font-family="IBM Plex Sans, sans-serif" font-size="9.5" fill="var(--faint)" text-anchor="middle" font-style="italic">every overlay inside the control</text>
    {box(IN_X, RA-BH/2, IN_W, BH)}
    <text x="{IN_X+IN_W/2}" y="{RA-3}" font-family="IBM Plex Sans, sans-serif" font-size="11" font-weight="600" fill="var(--ink)" text-anchor="middle">your signal</text>
    <text x="{IN_X+IN_W/2}" y="{RA+13}" font-family="IBM Plex Mono, monospace" font-size="9" fill="var(--muted)" text-anchor="middle">1 candidate</text>
    {box(IN_X, RB-BH/2, IN_W, BH, "var(--raised)", "var(--null)")}
    <text x="{IN_X+IN_W/2}" y="{RB-3}" font-family="IBM Plex Sans, sans-serif" font-size="11" font-weight="600" fill="var(--null)" text-anchor="middle">random numbers</text>
    <text x="{IN_X+IN_W/2}" y="{RB+13}" font-family="IBM Plex Mono, monospace" font-size="9" fill="var(--muted)" text-anchor="middle">&#215;40 draws</text>
    {arrow(IN_X+IN_W, RA, PX)}
    {arrow(IN_X+IN_W, RB, PX, "var(--null)")}
    {arrow(PX+PW, RA, OX)}
    {arrow(PX+PW, RB, OX, "var(--null)")}
    {box(OX, RA-BH/2, OW, BH)}
    <text x="{OX+10}" y="{RA-3}" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--ink)">one result</text>
    <text x="{OX+10}" y="{RA+13}" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--muted)">t = 5.98</text>
    {box(OX, RB-BH/2, OW, BH, "var(--raised)", "var(--null)")}
    <text x="{OX+10}" y="{RB-3}" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--null)">null distribution</text>
    <text x="{OX+10}" y="{RB+13}" font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--muted)">mean t = 2.75</text>
    <line x1="{OX+OW/2}" y1="{RA+BH/2}" x2="{OX+OW/2}" y2="{RB-BH/2}" stroke="var(--rule-strong)" stroke-width="1" stroke-dasharray="3 3" fill="none"/>
    <text x="{OX+OW/2+8}" y="{(RA+RB)/2+4}" font-family="IBM Plex Sans, sans-serif" font-size="10" fill="var(--faint)">compare</text>
    <text x="{W/2}" y="{H-8}" font-family="IBM Plex Sans, sans-serif" font-size="11" fill="var(--ink)" text-anchor="middle">report a <tspan font-weight="600">percentile</tspan>, not a t-statistic</text>
  </svg>
  <figcaption>The control is not a shuffled label or a placebo drawn from a different
    distribution. It is random input through the same universe, selection, weighting,
    volatility target, costs and rebalance schedule &mdash; so the only thing varying is how
    much information the signal carries.</figcaption>
</figure>'''


def drawdown_figure() -> str:
    P = FIG["phantom"]

    def dd(key):
        v = np.array([p[key] for p in P])
        return v / np.maximum.accumulate(v) - 1.0

    tg, ut = dd("targeted"), dd("untargeted")
    W, H, L, R, T, B = 640, 250, 58, 144, 30, 46
    n = len(P)
    lo = min(tg.min(), ut.min()) * 1.06
    xs = lambda i: L + i / (n - 1) * (W - L - R)
    ys = lambda v: T + (0 - v) / (0 - lo) * (H - B - T)
    poly = lambda a: " ".join(f"{xs(i):.1f},{ys(v):.1f}" for i, v in enumerate(a))

    def xat(ym):
        i = next((i for i, p in enumerate(P) if p["d"] == ym), None)
        return xs(i) if i is not None else None

    g0, g1 = xat("2007-10"), xat("2009-03")
    ticks = "".join(
        f'<text x="{xat(f"{y}-01"):.1f}" y="{H-B+18:.1f}" text-anchor="middle">{y}</text>'
        for y in (2008, 2012, 2016, 2020, 2024) if xat(f"{y}-01"))
    dtg = FIG["phantom_dd"]["targeted"] * 100
    dut = FIG["phantom_dd"]["untargeted"] * 100

    return f'''<figure class="fig">
  <svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img"
       aria-label="Drawdown of the same zero-information portfolio with and without volatility targeting. Untargeted falls {abs(dut):.1f} percent at worst; vol-targeted falls {abs(dtg):.1f} percent.">
    <text x="0" y="12" font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--faint)" letter-spacing="1.2">DRAWDOWN &#8212; SAME ZERO-INFORMATION SIGNAL, WITH AND WITHOUT VOL TARGETING</text>
    <rect x="{g0:.1f}" y="{T}" width="{g1-g0:.1f}" height="{H-B-T:.1f}" fill="var(--sunk)"/>
    <text x="{(g0+g1)/2:.1f}" y="{T+12:.1f}" font-family="IBM Plex Mono, monospace" font-size="9" fill="var(--faint)" text-anchor="middle">2008</text>
    <line x1="{L}" y1="{ys(0):.1f}" x2="{W-R+8}" y2="{ys(0):.1f}" stroke="var(--rule-strong)" stroke-width="1" fill="none"/>
    <g font-family="IBM Plex Mono, monospace" font-size="9.5" fill="var(--faint)">
      <text x="{L-8}" y="{ys(0)+3:.1f}" text-anchor="end">0%</text>
      <text x="{L-8}" y="{ys(-0.10)+3:.1f}" text-anchor="end">-10%</text>
      <text x="{L-8}" y="{ys(-0.20)+3:.1f}" text-anchor="end">-20%</text>
      <text x="{L-8}" y="{ys(-0.30)+3:.1f}" text-anchor="end">-30%</text>
      {ticks}
    </g>
    <polyline points="{poly(ut)}" fill="none" stroke="var(--null)" stroke-width="1.6"/>
    <polyline points="{poly(tg)}" fill="none" stroke="var(--accent)" stroke-width="1.8"/>
    <text x="{W-R+14}" y="{ys(dut/100)+4:.1f}" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--null)" font-weight="600">untargeted</text>
    <text x="{W-R+14}" y="{ys(dut/100)+18:.1f}" font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--muted)">worst &#8722;{abs(dut):.1f}%</text>
    <text x="{W-R+14}" y="{ys(dtg/100)-6:.1f}" font-family="IBM Plex Sans, sans-serif" font-size="10.5" fill="var(--accent)" font-weight="600">vol-targeted</text>
    <text x="{W-R+14}" y="{ys(dtg/100)+8:.1f}" font-family="IBM Plex Mono, monospace" font-size="10" fill="var(--muted)">worst &#8722;{abs(dtg):.1f}%</text>
  </svg>
  <figcaption>Neither book forecasts anything &mdash; both hold a randomly chosen 30% of the
    universe. The only difference is the volatility overlay, which cuts exposure as volatility
    rises and so avoids roughly half the 2008 loss. A constant-beta model cannot represent that
    timing and reads the difference as alpha.</figcaption>
</figure>'''


def md_to_html(md: str) -> str:
    def inline(t):
        t = html.escape(t)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", t)
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)

    out, in_code, buf = [], False, []
    in_table, in_list = False, None
    for ln in md.split("\n"):
        if ln.startswith("```"):
            if in_code:
                out.append('<pre class="block">' + html.escape("\n".join(buf)) + "</pre>")
                buf, in_code = [], False
            else:
                in_code = True
            continue
        if in_code:
            buf.append(ln)
            continue
        s = ln.strip()

        # --- tables: | a | b |  with a |---|---| separator row
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") and c for c in cells):
                continue                                   # separator row
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append('<div class="tscroll"><table><tbody>')
                in_table = True
            row = "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells)
            out.append(f"<tr>{row}</tr>")
            continue
        if in_table:
            out.append("</tbody></table></div>")
            in_table = False

        # --- lists
        m_ul = re.match(r"^[-*]\s+(.*)$", s)
        m_ol = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m_ul or m_ol:
            want = "ul" if m_ul else "ol"
            if in_list != want:
                if in_list:
                    out.append(f"</{in_list}>")
                out.append(f"<{want}>")
                in_list = want
            out.append(f"<li>{inline((m_ul or m_ol).group(1 if m_ul else 2))}</li>")
            continue
        if in_list:
            out.append(f"</{in_list}>")
            in_list = None

        if not s:
            continue
        if s == "---":
            out.append("<hr>")
        elif s.startswith("## "):
            out.append(f"<h2>{inline(s[3:])}</h2>")
        elif s.startswith("# "):
            continue
        elif s.startswith("*") and s.endswith("*") and not s.startswith("**"):
            out.append(f'<p class="standfirst">{inline(s[1:-1])}</p>')
        else:
            out.append(f"<p>{inline(s)}</p>")
    if in_table:
        out.append("</tbody></table></div>")
    if in_list:
        out.append(f"</{in_list}>")
    return "\n".join(out)


SHELL = (ROOT / "research/essay_shell.html").read_text()

# Each essay: source markdown, output page, title, and any figures that attach
# to an anchor sentence. Anchors mean the markdown carries no placeholders, and
# a missing anchor is fatal rather than silently dropping the figure.
ESSAYS = {
    "null-floor": {
        "src": "writing/what-does-nothing-score.md",
        "out": "docs/null-floor.html",
        "title": "What Does Nothing Score?",
        "figures": [
            ("<p>A signal containing <strong>nothing</strong> clears the conventional",
             method_diagram),
            ("<p>Half the 2008 loss avoided, by a strategy that knows nothing.",
             drawdown_figure),
        ],
    },
    "method": {
        "src": "writing/plausible-and-wrong.md",
        "out": "docs/method.html",
        "title": "Plausible and Wrong",
        "figures": [],
    },
}

import sys
names = sys.argv[1:] or list(ESSAYS)
for name in names:
    if name not in ESSAYS:
        raise SystemExit(f"unknown essay {name!r}; known: {', '.join(ESSAYS)}")
    cfg = ESSAYS[name]
    src = ROOT / cfg["src"]
    if not src.exists():
        print(f"  skipping {name}: {cfg['src']} not written yet")
        continue
    body = md_to_html(src.read_text())
    for anchor, fig in cfg["figures"]:
        if anchor not in body:
            raise SystemExit(f"anchor not found, figure would be dropped silently:\n  {anchor[:70]}")
        body = body.replace(anchor, fig() + "\n" + anchor)
    out = SHELL.replace("<!--BODY-->", body)
    out = out.replace("<title>What Does Nothing Score?</title>", f"<title>{cfg['title']}</title>")
    out = out.replace("<h1>What Does Nothing Score?</h1>", f"<h1>{cfg['title']}</h1>")
    (ROOT / cfg["out"]).write_text(out)
    print(f"{cfg['out']} rebuilt: {len(out):,} bytes, "
          f"{out.count('<figure')} figures, {out.count('<h2')} sections")
