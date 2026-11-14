"""report.html — one page per run that a person can open and show (T32).

Written by the render step from the content of ``provenance.json``; it adds no result of its own.
The page is a single file with relative links to the images of the same run folder (the photograph is
copied in as ``input.png``), no script and no external resource, so the run folder can be zipped and
opened anywhere. It holds the photograph: do not publish it for a building whose image has no licence.
"""
from __future__ import annotations

import html
import shutil
from typing import TYPE_CHECKING, Any

from i2r.complete import GUESS, REJECTED

if TYPE_CHECKING:
    from i2r.runio import Run

REPORT_FILE = "report.html"
INPUT_COPY = "input.png"

STATUS_TEXT = {
    "completed": ("completed", "ok"),
    GUESS: ("guess, needs expert", "guess"),
    REJECTED: ("rejected by expert", "rejected"),
    "not_completed": ("not completed", "open"),
    "not_rendered": ("not rendered", "open"),
}

CSS = """
body{font:15px/1.45 system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f4f2ee;color:#222}
main{max-width:1180px;margin:0 auto;padding:20px 24px 60px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:34px 0 10px;border-bottom:1px solid #cfc9bd;padding-bottom:4px}
.banner{background:#7a1f1f;color:#fff;padding:10px 24px;font-weight:600}
.tr{background:#fff;border-left:4px solid #b98a2e;padding:10px 14px;margin:14px 0}
.meta{color:#555;font-size:13px}
.views{display:flex;gap:14px;flex-wrap:wrap} .views figure{margin:0;flex:1 1 300px;background:#fff;padding:8px;border:1px solid #d9d4c8}
.views img{width:100%;height:auto;display:block} figcaption{font-size:13px;color:#444;margin-top:6px}
table{border-collapse:collapse;width:100%;background:#fff} th,td{border:1px solid #d9d4c8;padding:6px 8px;text-align:left;vertical-align:top;font-size:14px}
th{background:#ebe7de} td.n{text-align:right;white-space:nowrap}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px;white-space:nowrap}
.ok{background:#d9efd9;color:#14561c}.guess{background:#ffe2bd;color:#7a4300}.rejected{background:#f3c9c9;color:#7a1f1f}.open{background:#e6e6e6;color:#444}
.target{background:#fff;border:1px solid #d9d4c8;margin:12px 0;padding:10px 12px}
.target h3{margin:0 0 6px;font-size:15px} .target img{max-width:100%;height:auto;display:block;margin-top:8px}
code{background:#ebe7de;padding:1px 4px;border-radius:3px;font-size:13px} pre{background:#2b2b2b;color:#eee;padding:10px 12px;overflow:auto;font-size:13px}
ul{margin:6px 0 6px 20px;padding:0}
"""


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f} %" if whole else "-"


def num(value: Any) -> str:
    return "-" if value is None else f"{value:.2f}"


def tag(status: str) -> str:
    text, css = STATUS_TEXT.get(status, (status, "open"))
    return f'<span class="tag {css}">{esc(text)}</span>'


def view_rows(provenance: dict, damage_px: int) -> str:
    rows = []
    for view, info in provenance["views"].items():
        m = info["markings_px"]
        cells = [("completed from a donor, accepted", m["completed_px"]),
                 ("donor placed as a guess", m.get("donor_guess_px", 0)),
                 ("AI guess (generative) in the image", m.get("ai_guess_px", 0)),
                 ("tier A guess withheld", m.get("open_guess_withheld_px", 0)),
                 ("placement rejected by the expert", m.get("open_rejected_px", 0)),
                 ("tier B position implied, open", m["open_tier_b_px"]),
                 ("no evidence, open", m["open_no_evidence_px"]),
                 ("outside the facade limit", m.get("open_outside_facade_px", 0))]
        rows.append(f"<tr><th colspan=3>{esc(view.replace('_', ' '))} view</th></tr>")
        rows += [f"<tr><td>{esc(name)}</td><td class=n>{px:,}</td><td class=n>{pct(px, damage_px)}</td></tr>"
                 for name, px in cells if px]
    return "\n".join(rows)


def target_block(entry: dict) -> str:
    """One placed donor: figures, verdicts in each view, the expert's decision, the comparison image."""
    views = entry["views"]
    view = next((v for v in ("first_construction", "restoration") if v in entry["alignment"]), None)
    al = entry["alignment"].get(view, {}) if view else {}
    acc = entry["acceptance"].get(view) or {}
    place = entry["placement"].get(view) or {}
    donor_px = place.get("donor_px") or {}
    statuses = " &nbsp; ".join(f"{esc(v.replace('_', ' '))}: {tag(s)}" for v, s in views.items())
    reasons = acc.get("reasons") or []
    expert = acc.get("expert")
    expert_text = "none yet" if not expert else esc(expert) + (f" ({esc(acc.get('expert_note'))})" if acc.get("expert_note") else "")
    lines = [
        f"<h3>{esc(entry['target_id'])} &middot; {esc(entry.get('element_id') or entry.get('rule') or entry['kind'])} "
        f"&middot; tier {esc(entry['tier'])}</h3>",
        f"<div>{statuses}</div>",
        f"<div class=meta>donor <code>{esc(entry['donor_id'])}</code>{' (flipped)' if entry.get('flipped') else ''}; "
        f"similarity {num(entry.get('similarity'))}; alignment score {num(al.get('score'))} "
        f"(before the search {num(al.get('score_before'))}); seam agreement {num(al.get('seam_agreement'))}; "
        f"on search bound: {esc(', '.join(al.get('on_bound') or []) or 'none')}; "
        f"donor pixels used {esc(donor_px.get('used_in_damage', '-'))} of {esc(donor_px.get('placed', '-'))}</div>",
        "<div>Why a guess: " + (esc("; ".join(reasons)) if reasons else "&ndash; (passed the automatic tests)") + "</div>",
        f"<div>Expert decision: <b>{expert_text}</b></div>",
    ]
    choice = entry.get("donor_choice")
    if choice:
        lines.insert(3, "<div class=meta>donor chosen by the expert instead of "
                        f"<code>{esc(choice['automatic_donor_id'])}</code>"
                        + (f" ({esc(choice['note'])})" if choice.get("note") else "") + "</div>")
    ext = entry.get("box_extension")
    if ext:
        lines.insert(3, f"<div class=meta>box extended at the {esc(ext['side'])} from {esc(entry.get('element_box'))} to "
                        f"{esc(entry['target_box'])}, the extent of <code>{esc(ext['reference_element'])}</code> "
                        f"(strip in between {num(ext['strip_damage_share'])} damaged)</div>")
    if place.get("panel"):
        lines.append(f'<a href="{esc(place["panel"])}"><img src="{esc(place["panel"])}" '
                     f'alt="photograph, damage mask, donor as placed, result"></a>')
    return '<div class="target">' + "\n".join(lines) + "</div>"


def open_rows(provenance: dict) -> str:
    rows = []
    for e in provenance["targets"]:
        if e["alignment"] or e["tier"] != "C":
            continue
        name = e.get("element_id") or e["kind"].replace("_", " ")
        rows.append(f"<tr><td>{esc(e['target_id'])}</td><td>{esc(name)}</td><td>{esc(e.get('class') or '-')}</td>"
                    f"<td class=n>{esc(e.get('area_px') or '-')}</td><td>{esc(e.get('evidence_reason'))}</td></tr>")
    return "\n".join(rows)


def build(run: "Run", provenance: dict, damage_px: int) -> str:
    sid = provenance["structure_id"]
    run_dir = run.run_dir.as_posix()
    figures = [(INPUT_COPY, "Input photograph")]
    for view, info in provenance["views"].items():
        figures.append((info["image"], f"{view.replace('_', ' ').capitalize()} view"))
        if info.get("ai_guess_image"):
            figures.append((info["ai_guess_image"], "First construction with the AI guess (no evidence)"))
    placed = [e for e in provenance["targets"] if e["alignment"]]
    guesses = [e for e in placed if GUESS in e["views"].values()]
    by_tier = {t: sum(1 for e in provenance["targets"] if e["tier"] == t) for t in ("A", "B", "C")}
    facade = provenance.get("facade") or {}
    figure_html = "\n".join(f'<figure><a href="{esc(f)}"><img src="{esc(f)}" alt="{esc(c)}"></a>'
                            f"<figcaption>{esc(c)}</figcaption></figure>" for f, c in figures)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(sid)} - ImageToRender run report</title><style>{CSS}</style></head><body>
<div class="banner">{esc(provenance['statement'])}</div>
<main>
<h1>{esc(sid)}</h1>
<div class="meta">Run <code>{esc(provenance.get('run_id'))}</code> &middot; code {esc((provenance.get('git_commit') or {}).get('hash', '?'))[:7]}
 &middot; one photograph, one building: every figure below is n = 1</div>
<div class="tr"><b>T&uuml;rk&ccedil;e &ouml;zet.</b> Bu sayfa tek bir ko&#351;unun sonucudur. Beyaz &ccedil;er&ccedil;eveli par&ccedil;alar foto&#287;raftaki
ba&#351;ka bir elemandan kopyalanm&#305;&#351; ve otomatik testleri ge&ccedil;mi&#351;tir; turuncu olanlar "tahmin"dir ve uzman onay&#305; bekler;
taral&#305; alanlar i&ccedil;in foto&#287;rafta kan&#305;t yoktur ve bo&#351; b&#305;rak&#305;lm&#305;&#351;t&#305;r. Bu bir restorasyon karar&#305; de&#287;il, erken a&#351;ama karar deste&#287;idir.</div>

<h2>Views</h2>
<div class="views">{figure_html}</div>

<h2>What was filled, and on what basis</h2>
<p>Damage mask: {damage_px:,} px. Targets: {len(provenance['targets'])}
 (tier A {by_tier['A']}, tier B {by_tier['B']}, tier C {by_tier['C']}).
 Facade limit: {esc(facade.get('source', 'none'))}{f", {facade.get('damage_px_outside', 0):,} damage px outside it" if facade.get('mask_file') else ""}.</p>
<table><tr><th>Area</th><th>Pixels</th><th>Share of the damage</th></tr>
{view_rows(provenance, damage_px)}</table>

<h2>Placed donors ({len(placed)}; {len(guesses)} waiting for the expert)</h2>
<p>Each image: photograph | damage mask | donor as placed | result. The yellow rectangle is the target box.</p>
{''.join(target_block(e) for e in placed) or '<p>None.</p>'}

<h2>Left open: no evidence in the photograph</h2>
<table><tr><th>Target</th><th>What</th><th>Class</th><th>Pixels</th><th>Reason</th></tr>
{open_rows(provenance)}</table>

<h2>What the expert can do</h2>
<ul>
<li>Approve or reject each placed donor:<pre>python tools/target_review.py --run {esc(run_dir)}
python run.py --structure {esc(sid)} --from complete --no-review</pre></li>
<li>Correct the element inventory (boxes, classes, missing elements):<pre>python tools/editor_adapter.py edit --run {esc(run_dir)}</pre></li>
<li>Redraw the damage mask, or draw a facade mask (white = facade):<pre>python tools/mask_editor.py --structure {esc(sid)}
python tools/mask_editor.py --image data/structures/{esc(sid)}/image.png --out data/structures/{esc(sid)}/facade_mask.png</pre></li>
</ul>

<h2>Limits</h2>
<ul>
<li>One building and one photograph; thresholds were chosen on the same building.</li>
<li>A donor is another element of the same photograph, copied box to box in the image plane; curvature and
 perspective are not modelled.</li>
<li>Where nothing in the photograph says what stood there, the area stays open. The generative fill is off by
 default and, when on, is shown apart as an AI guess.</li>
<li>Files: <code>provenance.json</code> (every target, donor, figure and decision), <code>completion.json</code>,
 <code>evidence.json</code>, <code>run.log</code> in this folder.</li>
</ul>
</main></body></html>
"""


def write_report(run: "Run", provenance: dict, damage_px: int) -> None:
    """Write report.html and the copy of the photograph it shows (``paths.image_file`` is a PNG)."""
    shutil.copy2(run.image_path, run.path(INPUT_COPY))
    with run.path(REPORT_FILE).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(build(run, provenance, damage_px))
