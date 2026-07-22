#!/usr/bin/env python
"""Build a self-contained HTML review of the correction pilot outputs.

Reads eval/correction_pilot/out/*.corrected.json and emits review.html:
a QA dashboard (summary tiles + findings) over per-file, word-diff-highlighted
Original / Conservative / Aggressive columns so a human can judge fix-vs-fabricate.
"""
from __future__ import annotations

import difflib
import glob
import html
import json
import os
from pathlib import Path

OUT = Path("eval/correction_pilot/out")
DST = Path("eval/correction_pilot/review.html")


def prep(t: str) -> list[str]:
    return t.replace("\n\n", "\n").replace("\n", " ¶ ").split()


def diff_html(orig: str, corr: str) -> tuple[str, int, int]:
    """Return (highlighted_html, n_changed_words, n_deleted_words)."""
    a, b = prep(orig), prep(corr)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    parts: list[str] = []
    changed = deleted = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        seg = html.escape(" ".join(b[j1:j2]))
        if tag == "equal":
            parts.append(seg)
        elif tag == "insert":
            parts.append(f'<mark class="ins">{seg}</mark>')
            changed += (j2 - j1)
        elif tag == "replace":
            parts.append(f'<mark class="chg">{seg}</mark>')
            changed += (j2 - j1)
            deleted += (i2 - i1)
        elif tag == "delete":
            deleted += (i2 - i1)
            parts.append('<span class="del">·</span>')
    out = " ".join(p for p in parts if p)
    return out.replace("¶", "<br>"), changed, deleted


def main() -> int:
    rows = []
    total_cost = 0.0
    for p in sorted(glob.glob(str(OUT / "*.corrected.json")), key=os.path.getsize):
        d = json.load(open(p, encoding="utf-8"))
        orig = d["original_cleaned_text"]
        cons = d.get("conservative_text") or ""
        aggr = d.get("aggressive_text") or ""
        cons_html, cons_chg, cons_del = diff_html(orig, cons)
        aggr_html, aggr_chg, aggr_del = diff_html(orig, aggr)
        n_words = max(1, len(orig.split()))
        loop = len(cons) < 0.7 * max(1, len(orig)) and cons_del > 8
        rows.append({
            "src": d["source_file"], "chars": d["chars_in"],
            "trunc": d["truncated"], "cost": d["usage"]["cost_usd"],
            "orig": html.escape(orig).replace("\n\n", "<br>").replace("\n", "<br>"),
            "cons_html": cons_html, "aggr_html": aggr_html,
            "cons_rate": cons_chg / n_words, "aggr_rate": aggr_chg / n_words,
            "cons_del": cons_del, "aggr_del": aggr_del, "loop": loop,
        })
        total_cost += d["usage"]["cost_usd"]

    n = len(rows)
    n_trunc = sum(1 for r in rows if r["trunc"])
    n_loop = sum(1 for r in rows if r["loop"])
    proj_both = total_cost / n * 8400 if n else 0
    avg_cons = sum(r["cons_rate"] for r in rows) / n if n else 0
    avg_aggr = sum(r["aggr_rate"] for r in rows) / n if n else 0

    index = "\n".join(
        f'<tr><td class="num">{i}</td>'
        f'<td><a href="#f{i}">{html.escape(os.path.basename(r["src"]))}</a>'
        f'<div class="path">{html.escape(os.path.dirname(r["src"]))}</div></td>'
        f'<td class="num">{r["chars"]:,}</td>'
        f'<td class="num">{r["cons_rate"]*100:.0f}%</td>'
        f'<td class="num">{r["aggr_rate"]*100:.0f}%</td>'
        f'<td class="mid">{"·loop" if r["loop"] else ""}'
        f'{" ·trunc" if r["trunc"] else ""}</td>'
        f'<td class="num">${r["cost"]:.4f}</td></tr>'
        for i, r in enumerate(rows, 1)
    )

    cards = []
    for i, r in enumerate(rows, 1):
        chips = [f'<span class="chip">{r["chars"]:,} chars</span>']
        if r["loop"]:
            chips.append('<span class="chip good">loop collapsed</span>')
        if r["trunc"]:
            chips.append('<span class="chip warn">truncated</span>')
        chips.append(f'<span class="chip">cons edit {r["cons_rate"]*100:.0f}%</span>')
        chips.append(f'<span class="chip">aggr edit {r["aggr_rate"]*100:.0f}%</span>')
        cards.append(f'''<section class="card" id="f{i}">
  <header class="card-h">
    <div class="tt"><span class="idx">{i:02d}</span>
      <div><div class="fname">{html.escape(os.path.basename(r["src"]))}</div>
      <div class="path">{html.escape(os.path.dirname(r["src"]))}</div></div></div>
    <div class="chips">{''.join(chips)}</div>
  </header>
  <div class="cols">
    <div class="col"><div class="col-h">Original <span>(local qwen3.5:9b)</span></div>
      <div class="txt deva">{r["orig"]}</div></div>
    <div class="col"><div class="col-h">Conservative <span>word-level fixes</span></div>
      <div class="txt deva">{r["cons_html"]}</div></div>
    <div class="col"><div class="col-h">Aggressive <span>fluent rewrite</span></div>
      <div class="txt deva">{r["aggr_html"]}</div></div>
  </div>
</section>''')

    doc = f'''<title>Transcript correction pilot — review</title>
<style>
:root {{
  --paper:#f5f6f8; --card:#ffffff; --ink:#171a20; --soft:#59616f; --faint:#8b93a1;
  --line:#e3e7ed; --accent:#2f5d8a; --good:#1f8a5b; --warn:#9a6b12; --bad:#c0392b;
  --chg:rgba(47,93,138,.15); --chg-b:#2f5d8a; --ins:rgba(31,138,91,.16); --ins-b:#1f8a5b;
}}
@media (prefers-color-scheme:dark) {{
  :root {{ --paper:#0f1217; --card:#161b22; --ink:#e7ebf1; --soft:#9aa4b2; --faint:#6a7382;
    --line:#252c36; --accent:#7fb0e0; --good:#4cc38a; --warn:#e0b25a; --bad:#e6796b;
    --chg:rgba(127,176,224,.18); --chg-b:#7fb0e0; --ins:rgba(76,195,138,.2); --ins-b:#4cc38a; }}
}}
:root[data-theme="light"] {{
  --paper:#f5f6f8; --card:#ffffff; --ink:#171a20; --soft:#59616f; --faint:#8b93a1;
  --line:#e3e7ed; --accent:#2f5d8a; --good:#1f8a5b; --warn:#9a6b12; --bad:#c0392b;
  --chg:rgba(47,93,138,.15); --chg-b:#2f5d8a; --ins:rgba(31,138,91,.16); --ins-b:#1f8a5b;
}}
:root[data-theme="dark"] {{
  --paper:#0f1217; --card:#161b22; --ink:#e7ebf1; --soft:#9aa4b2; --faint:#6a7382;
  --line:#252c36; --accent:#7fb0e0; --good:#4cc38a; --warn:#e0b25a; --bad:#e6796b;
  --chg:rgba(127,176,224,.18); --chg-b:#7fb0e0; --ins:rgba(76,195,138,.2); --ins-b:#4cc38a;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--paper); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; line-height:1.5; }}
.deva {{ font-family:"Nirmala UI","Noto Sans Devanagari","Mangal",system-ui,sans-serif; }}
.wrap {{ max-width:1440px; margin:0 auto; padding:40px 28px 80px; }}
.eyebrow {{ font-size:12px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); font-weight:600; }}
h1 {{ font-size:30px; margin:.2em 0 .1em; letter-spacing:-.01em; text-wrap:balance; }}
.sub {{ color:var(--soft); max-width:70ch; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:14px; margin:26px 0; }}
.tile {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 18px; }}
.tile .k {{ font-size:12px; color:var(--faint); text-transform:uppercase;
  letter-spacing:.06em; }}
.tile .v {{ font-size:26px; font-weight:650; margin-top:4px;
  font-variant-numeric:tabular-nums; }}
.tile .v small {{ font-size:14px; color:var(--soft); font-weight:500; }}
.findings {{ background:var(--card); border:1px solid var(--line); border-left:3px solid
  var(--accent); border-radius:12px; padding:20px 24px; margin:26px 0; }}
.findings h2 {{ font-size:15px; margin:0 0 12px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--soft); }}
.finding {{ display:flex; gap:12px; padding:10px 0; border-top:1px solid var(--line); }}
.finding:first-of-type {{ border-top:none; }}
.dot {{ flex:0 0 auto; width:9px; height:9px; border-radius:50%; margin-top:7px; }}
.dot.good {{ background:var(--good); }} .dot.bad {{ background:var(--bad); }}
.dot.warn {{ background:var(--warn); }}
.finding b {{ font-weight:650; }}
.finding .ex {{ font-family:"Nirmala UI","Noto Sans Devanagari",sans-serif;
  color:var(--soft); }}
.legend {{ display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin:20px 0;
  font-size:13px; color:var(--soft); }}
mark.chg {{ background:var(--chg); border-bottom:2px solid var(--chg-b);
  border-radius:2px; padding:0 1px; color:inherit; }}
mark.ins {{ background:var(--ins); border-bottom:2px solid var(--ins-b);
  border-radius:2px; padding:0 1px; color:inherit; }}
.del {{ color:var(--bad); opacity:.5; }}
table.idx {{ width:100%; border-collapse:collapse; margin:8px 0 32px; font-size:14px; }}
table.idx th {{ text-align:left; font-size:11px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--faint); padding:8px 10px; border-bottom:1px
  solid var(--line); position:sticky; top:0; background:var(--paper); }}
table.idx td {{ padding:8px 10px; border-bottom:1px solid var(--line);
  vertical-align:top; }}
table.idx td.num {{ text-align:right; font-variant-numeric:tabular-nums;
  white-space:nowrap; }}
table.idx td.mid {{ color:var(--soft); font-size:12px; white-space:nowrap; }}
table.idx a {{ color:var(--accent); text-decoration:none; font-weight:550; }}
table.idx a:hover {{ text-decoration:underline; }}
.path {{ font-size:11px; color:var(--faint); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:20px 22px; margin:18px 0; scroll-margin-top:16px; }}
.card-h {{ display:flex; justify-content:space-between; gap:16px; flex-wrap:wrap;
  align-items:flex-start; margin-bottom:16px; }}
.tt {{ display:flex; gap:12px; align-items:baseline; }}
.idx {{ font-variant-numeric:tabular-nums; font-weight:700; color:var(--accent);
  font-size:15px; }}
.fname {{ font-weight:600; }}
.chips {{ display:flex; flex-wrap:wrap; gap:6px; }}
.chip {{ font-size:12px; padding:3px 9px; border-radius:20px; background:var(--paper);
  border:1px solid var(--line); color:var(--soft); white-space:nowrap; }}
.chip.good {{ color:var(--good); border-color:var(--good); }}
.chip.warn {{ color:var(--warn); border-color:var(--warn); }}
.cols {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
@media (max-width:900px) {{ .cols {{ grid-template-columns:1fr; }} }}
.col {{ min-width:0; }}
.col-h {{ font-size:12px; font-weight:650; text-transform:uppercase;
  letter-spacing:.05em; margin-bottom:8px; color:var(--ink); }}
.col-h span {{ font-weight:400; text-transform:none; letter-spacing:0;
  color:var(--faint); }}
.txt {{ font-size:15px; line-height:1.7; background:var(--paper);
  border:1px solid var(--line); border-radius:8px; padding:12px 14px;
  max-height:360px; overflow-y:auto; word-break:break-word; }}
.foot {{ color:var(--faint); font-size:12px; margin-top:40px;
  border-top:1px solid var(--line); padding-top:16px; }}
</style>
<div class="wrap">
  <div class="eyebrow">Transcript RAG · correction pilot</div>
  <h1>OpenRouter Hindi correction — 30-file review</h1>
  <p class="sub">Each transcript below was already cleaned once locally
  (qwen3.5:9b). This pilot re-corrects it with a cloud model
  (<code>qwen/qwen3.6-27b</code>) in two ways. Highlights mark what the model
  <mark class="chg">changed</mark> or <mark class="ins">added</mark> vs the
  local-cleaned text — every highlight is a decision to audit, good or bad.</p>

  <div class="tiles">
    <div class="tile"><div class="k">Files</div><div class="v">{n}</div></div>
    <div class="tile"><div class="k">Pilot cost</div>
      <div class="v">${total_cost:.2f}</div></div>
    <div class="tile"><div class="k">Corpus projection</div>
      <div class="v">${proj_both:,.0f} <small>both modes · ~8,400</small></div></div>
    <div class="tile"><div class="k">Avg edit rate</div>
      <div class="v">{avg_cons*100:.0f}% <small>cons / {avg_aggr*100:.0f}% aggr</small></div></div>
    <div class="tile"><div class="k">Loops collapsed</div>
      <div class="v">{n_loop} <small>of {n}</small></div></div>
    <div class="tile"><div class="k">Truncated</div>
      <div class="v">{n_trunc} <small>think-leak</small></div></div>
  </div>

  <div class="findings">
    <h2>What the pilot shows</h2>
    <div class="finding"><span class="dot good"></span><div><b>Real wins —
      spelling &amp; loop collapse.</b> Consistent normalization
      (<span class="ex">तु→तू, युरोप→यूरोप, लीजिये→लीजिए</span>) and, in the
      conservative pass, Whisper hallucination loops crushed from 20×→1×
      ({n_loop}/{n} files here). This alone would improve retrieval.</div></div>
    <div class="finding"><span class="dot bad"></span><div><b>Fabrication risk is
      real.</b> On the pravachan the theme word <span class="ex">पोर</span>
      ("poor" — <i>"America is poor within"</i>) became <span class="ex">पूरा</span>
      ("complete") in <b>both</b> modes — the opposite meaning — and unrecoverable
      words like <span class="ex">डकषे</span> were guessed into new garbage despite
      an explicit "leave as-is" instruction.</div></div>
    <div class="finding"><span class="dot warn"></span><div><b>The two modes disagree.</b>
      Conservative collapses loops; aggressive keeps them. Aggressive fixes grammar
      conservative leaves. Neither is uniformly safer — read both columns.</div></div>
  </div>

  <div class="legend">
    <span><mark class="chg">changed</mark> replaced a word</span>
    <span><mark class="ins">added</mark> inserted</span>
    <span><span class="del">·</span> a word was dropped (e.g. loop)</span>
    <span>Edit rate = share of words touched vs the local-cleaned original</span>
  </div>

  <table class="idx">
    <thead><tr><th>#</th><th>File</th><th>Chars</th><th>Cons</th><th>Aggr</th>
      <th>Flags</th><th>Cost</th></tr></thead>
    <tbody>{index}</tbody>
  </table>

  {''.join(cards)}

  <div class="foot">Generated from eval/correction_pilot/out/*.corrected.json ·
    model qwen/qwen3.6-27b · reasoning disabled · originals unmodified.</div>
</div>'''

    DST.write_text(doc, encoding="utf-8")
    print(f"wrote {DST} ({len(doc):,} bytes) · {n} files · ${total_cost:.4f} · "
          f"proj ${proj_both:,.0f} · loops {n_loop} · trunc {n_trunc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
