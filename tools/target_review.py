#!/usr/bin/env python3
"""Expert review of the placed donors of a run: approve or reject each one.

The pipeline places a donor for every tier A target (and tier B slot) and judges the placement itself
(``i2r/complete.py`` ``judge``). A weak one becomes a "guess, needs expert" and is withheld from the
restoration view. This tool is where the expert decides. Decisions are stored per building in
``data/structures/<id>/target_decisions.json`` and applied by the complete step of the next run:

    approve   the placement is completed in every view whose rule allows it, also when the tests failed
    reject    the placement is withheld in every view, also when the tests passed

A decision is about one placement (donor, flip, scale, shift). If a later run places the donor
differently, the decision is ignored and reported as stale: review that target again.

Usage:
    python tools/target_review.py --run outputs/<id>/<timestamp>                    # window
    python tools/target_review.py --run <run> --list                                # table, no window
    python tools/target_review.py --run <run> --approve t_002 --note "ring continues the fragment"
    python tools/target_review.py --run <run> --reject t_003 t_006
    python tools/target_review.py --run <run> --clear t_003
Then:
    python run.py --structure <id> --from complete --no-review

Window keys: A approve, R reject, C clear, Left / Right previous / next, Esc close. The note field is
saved with the decision. Each image shows: photograph | damage mask | donor as placed | result.
No torch is imported.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from i2r import runio  # noqa: E402
from i2r.complete import (COMPLETION_FILE, DECISION_VALUES, DECISIONS_FILE, placement_signature,  # noqa: E402
                          read_decisions)
from i2r.detect import load_elements  # noqa: E402
from i2r.runio import ContractError, read_json, write_json  # noqa: E402

VIEW_ORDER = ("first_construction", "restoration")     # the first view that has a panel is shown


def structure_folder(run_dir: Path) -> Path:
    config_file = run_dir / "config.yaml"
    if not config_file.is_file():
        raise ContractError(f"{run_dir} is not a run folder (no config.yaml)")
    with config_file.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return runio.structure_dir(cfg, load_elements(run_dir)["structure_id"])


def review_items(run_dir: Path) -> list[dict[str, Any]]:
    """One item per target that has a placed donor in this run, in target order."""
    completion = read_json(run_dir / COMPLETION_FILE)
    items: dict[str, dict[str, Any]] = {}
    for view in VIEW_ORDER:
        for e in completion.get("views", {}).get(view, {}).get("targets", []):
            if "alignment" not in e:
                continue
            if "decision_key" not in e:
                raise ContractError(f"{run_dir / COMPLETION_FILE}: made before the review tool existed; "
                                    "run the complete step again (run.py --from complete)")
            item = items.setdefault(e["target_id"], {
                "target_id": e["target_id"], "key": e["decision_key"], "tier": e["tier"], "donor_id": e["donor_id"],
                "placement": placement_signature({"donor_id": e["donor_id"], "flipped": e["flipped"]}, e["alignment"]),
                "score": e["alignment"].get("score"), "seam": e.get("seam_agreement"),
                "on_bound": e["alignment"].get("on_bound") or [], "reasons": e["acceptance"]["reasons"],
                "panel": e.get("panel_file"), "status": {},
            })
            item["status"][view] = e["status"]
    return sorted(items.values(), key=lambda i: i["target_id"])


def decisions_path(structure_dir: Path) -> Path:
    return structure_dir / DECISIONS_FILE


def set_decision(structure_dir: Path, item: dict[str, Any], decision: str | None, note: str, run_dir: Path) -> None:
    """Write (or with ``decision`` None remove) the decision for ``item`` in the building's file."""
    if decision is not None and decision not in DECISION_VALUES:
        raise ContractError(f"decision must be one of {', '.join(DECISION_VALUES)}, not {decision!r}")
    path = decisions_path(structure_dir)
    decisions = read_decisions(path)
    if decision is None:
        decisions.pop(item["key"], None)
    else:
        decisions[item["key"]] = {
            "decision": decision, "note": note.strip(), "placement": item["placement"],
            "target_id_at_decision": item["target_id"], "run": runio.relative(run_dir),
            "decided_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    if not decisions:
        if path.exists():
            path.unlink()                              # no decision left: no file, so no stale copy in runs
        return
    write_json(path, {"structure_id": structure_dir.name,
                      "about": "Expert decisions per target; written by tools/target_review.py, read by the "
                               "complete step. A decision applies only to the placement it names.",
                      "decisions": decisions})


def current(decisions: dict[str, dict], item: dict[str, Any]) -> str:
    d = decisions.get(item["key"])
    if d is None:
        return "-"
    return d["decision"] if d["placement"] == item["placement"] else f"{d['decision']} (stale: placement changed)"


def describe(item: dict[str, Any], decisions: dict[str, dict]) -> str:
    def num(v: Any) -> str:
        return "-" if v is None else f"{v:.2f}"

    status = ", ".join(f"{v}: {s}" for v, s in item["status"].items())
    p = item["placement"]
    lines = [
        f"{item['target_id']}  tier {item['tier']}  {item['key']}",
        f"donor {p['donor_id']}{' (flipped)' if p['flipped'] else ''}, scale {p['scale']:.2f}, "
        f"shift ({p['dx']}, {p['dy']})",
        f"alignment score {num(item['score'])}   seam agreement {num(item['seam'])}   "
        f"on bound: {', '.join(item['on_bound']) or 'none'}",
        f"automatic result: {status}",
        "why a guess: " + ("; ".join(item["reasons"]) if item["reasons"] else "- (passed the tests)"),
        f"expert decision: {current(decisions, item)}",
    ]
    return "\n".join(lines)


def table(items: list[dict[str, Any]], decisions: dict[str, dict]) -> str:
    rows = ["target  tier  donor                      score  seam   automatic            expert"]
    for i in items:
        auto = i["status"].get("restoration") or next(iter(i["status"].values()))
        score = "-" if i["score"] is None else f"{i['score']:.2f}"
        seam = "-" if i["seam"] is None else f"{i['seam']:.2f}"
        rows.append(f"{i['target_id']:7s} {i['tier']:5s} {str(i['donor_id']):26s} {score:6s} {seam:6s} "
                    f"{auto:20s} {current(decisions, i)}")
    return "\n".join(rows)


def window(run_dir: Path, structure_dir: Path, items: list[dict[str, Any]]) -> None:
    import tkinter as tk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"Target review: {runio.relative(run_dir)}")
    state = {"index": 0, "photo": None}
    head = tk.Label(root, anchor="w", justify="left", font=("Consolas", 10))
    head.pack(fill="x", padx=8, pady=(8, 4))
    picture = tk.Label(root, bg="black")
    picture.pack(padx=8)
    note_row = tk.Frame(root)
    note_row.pack(fill="x", padx=8, pady=4)
    tk.Label(note_row, text="Note:").pack(side="left")
    note = tk.Entry(note_row)
    note.pack(side="left", fill="x", expand=True, padx=(4, 0))
    buttons = tk.Frame(root)
    buttons.pack(pady=(2, 8))

    def show() -> None:
        item = items[state["index"]]
        decisions = read_decisions(decisions_path(structure_dir))
        head.config(text=f"[{state['index'] + 1} / {len(items)}]  " + describe(item, decisions))
        note.delete(0, "end")
        note.insert(0, (decisions.get(item["key"]) or {}).get("note", ""))
        panel = run_dir / item["panel"] if item["panel"] else None
        if panel is not None and panel.is_file():
            image = Image.open(panel).convert("RGB")
            limit = root.winfo_screenwidth() - 80
            if image.width > limit:
                image = image.resize((limit, max(1, round(image.height * limit / image.width))))
            state["photo"] = ImageTk.PhotoImage(image)
            picture.config(image=state["photo"], text="")
        else:
            state["photo"] = None
            picture.config(image="", text="no comparison image for this target", fg="white")

    def decide(value: str | None) -> None:
        set_decision(structure_dir, items[state["index"]], value, note.get(), run_dir)
        show()

    def step(delta: int) -> None:
        state["index"] = (state["index"] + delta) % len(items)
        show()

    for text, command in (("< Previous", lambda: step(-1)), ("Approve (A)", lambda: decide("approve")),
                          ("Reject (R)", lambda: decide("reject")), ("Clear (C)", lambda: decide(None)),
                          ("Next >", lambda: step(1))):
        tk.Button(buttons, text=text, width=13, command=command).pack(side="left", padx=3)

    def key(event: Any) -> None:
        if event.widget is note and event.keysym not in ("Escape",):
            return                                    # typing a note: letters are text, not commands
        action = {"a": lambda: decide("approve"), "r": lambda: decide("reject"), "c": lambda: decide(None),
                  "Left": lambda: step(-1), "Right": lambda: step(1), "Escape": root.destroy}.get(event.keysym)
        if action:
            action()

    root.bind("<Key>", key)
    show()
    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Approve or reject the placed donors of a run.")
    parser.add_argument("--run", required=True, type=Path, help="run folder, e.g. outputs/<id>/<timestamp>")
    parser.add_argument("--list", action="store_true", help="print the targets and decisions, open no window")
    parser.add_argument("--approve", nargs="+", default=[], metavar="TARGET")
    parser.add_argument("--reject", nargs="+", default=[], metavar="TARGET")
    parser.add_argument("--clear", nargs="+", default=[], metavar="TARGET")
    parser.add_argument("--note", default="", help="note stored with --approve / --reject")
    opts = parser.parse_args(argv)
    try:
        run_dir = opts.run.resolve()
        structure_dir = structure_folder(run_dir)
        items = review_items(run_dir)
        if not items:
            print("This run has no placed donor to review.")
            return 0
        by_id = {i["target_id"]: i for i in items}
        changes = [(t, "approve") for t in opts.approve] + [(t, "reject") for t in opts.reject] \
            + [(t, None) for t in opts.clear]
        for target, decision in changes:
            if target not in by_id:
                raise ContractError(f"{target}: no placed donor with this ID in the run. Known: {', '.join(by_id)}")
            set_decision(structure_dir, by_id[target], decision, opts.note, run_dir)
        if not changes and not opts.list:
            window(run_dir, structure_dir, items)
        print(table(items, read_decisions(decisions_path(structure_dir))))
        print(f"\nDecisions: {runio.relative(decisions_path(structure_dir))}")
        print(f"Apply them: python run.py --structure {structure_dir.name} --from complete --no-review")
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
