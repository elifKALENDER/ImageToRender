#!/usr/bin/env python3
"""Expert review of the placed donors of a run: approve or reject each one, or name another donor.

The pipeline places a donor for every tier A target (and tier B slot) and judges the placement itself
(``i2r/complete.py`` ``judge``). A weak one becomes a "guess, needs expert" and is withheld from the
restoration view. This tool is where the expert decides. Decisions are stored per building in
``data/structures/<id>/target_decisions.json`` and applied by the complete step of the next run:

    approve   the placement is completed in every view whose rule allows it, also when the tests failed
    reject    the placement is withheld in every view, also when the tests passed

A decision is about one placement (donor, flip, scale, shift). If a later run places the donor
differently, the decision is ignored and reported as stale: review that target again.

For a tier A target the expert can also name the donor (T34). The choice is stored in the same file and
used by the complete step of the next run instead of the automatic donor; the new placement is then a
guess like any other and is approved or rejected in a second look. Any same-class element that is not
rejected can be named, also one the automatic filters left out (the list says why).

Usage:
    python tools/target_review.py --run outputs/<id>/<timestamp>                    # window
    python tools/target_review.py --run <run> --list                                # table, no window
    python tools/target_review.py --run <run> --approve t_002 --note "ring continues the fragment"
    python tools/target_review.py --run <run> --reject t_003 t_006
    python tools/target_review.py --run <run> --clear t_003
    python tools/target_review.py --run <run> --options t_006                       # donors that can be named
    python tools/target_review.py --run <run> --donor t_006 pilaster_010 [--flip yes|no] [--note "..."]
    python tools/target_review.py --run <run> --auto-donor t_006                    # back to the automatic donor
Then:
    python run.py --structure <id> --from complete --no-review

Window keys: A approve, R reject, C clear, Left / Right previous / next, Esc close. The note field is
saved with the decision. Each image shows: photograph | damage mask | donor as placed | result.
Donor row: pick an element from the list (its crop is shown), "Use this donor" stores it, "Automatic
donor" removes the choice.
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
                          read_decisions, read_donor_choices)
from i2r.detect import load_elements  # noqa: E402
from i2r.runio import ContractError, read_json, write_json  # noqa: E402

VIEW_ORDER = ("first_construction", "restoration")     # the first view that has a panel is shown
DONOR_PREVIEW_PX = 160                                 # height of the donor crop shown in the window


def structure_folder(run_dir: Path) -> Path:
    config_file = run_dir / "config.yaml"
    if not config_file.is_file():
        raise ContractError(f"{run_dir} is not a run folder (no config.yaml)")
    with config_file.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return runio.structure_dir(cfg, load_elements(run_dir)["structure_id"])


def donor_options(record: dict, elements: list[dict]) -> list[dict[str, Any]]:
    """Elements the expert can name as donor of a tier A target: the automatic candidates first (by
    similarity, each in its better orientation), then the other same-class elements with the reason
    the automatic filters gave."""
    if record.get("tier") != "A" or not record.get("element_id"):
        return []
    by_id = {e["id"]: e for e in elements}
    options, seen = [], set()
    for c in record.get("candidates", []):                     # sorted by similarity, both orientations
        if c["donor_id"] in seen or c["donor_id"] not in by_id:
            continue
        seen.add(c["donor_id"])
        options.append({"donor_id": c["donor_id"], "flipped": bool(c["flipped"]),
                        "similarity": c["scores"].get("dinov2_cosine"), "left_out": None,
                        "damage_status": by_id[c["donor_id"]].get("damage_status"), "box": by_id[c["donor_id"]]["box"]})
    reasons = {r["donor_id"]: r["reason"] for r in record.get("rejected_donors", [])}
    for e in elements:
        if e["id"] in seen or e["id"] == record["element_id"] or e["class"] != record["class"] or e.get("rejected"):
            continue
        options.append({"donor_id": e["id"], "flipped": False, "similarity": None,
                        "left_out": reasons.get(e["id"], "not compared"),
                        "damage_status": e.get("damage_status"), "box": e["box"]})
    return options


def option_text(option: dict[str, Any]) -> str:
    sim = "sim  -  " if option["similarity"] is None else f"sim {option['similarity']:.2f}"
    return (f"{option['donor_id']:26s} {sim}  {'flipped' if option['flipped'] else '       '}  "
            f"{option['damage_status'] or '?'}" + (f"  (left out: {option['left_out']})" if option["left_out"] else ""))


def review_items(run_dir: Path) -> list[dict[str, Any]]:
    """One item per target that has a placed donor in this run, in target order."""
    completion = read_json(run_dir / COMPLETION_FILE)
    records = {r["target_id"]: r for r in read_json(run_dir / "evidence.json")["targets"]}
    elements = load_elements(run_dir)["elements"]
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
                "chosen": e.get("donor_choice"),
                "options": donor_options(records.get(e["target_id"], {}), elements),
            })
            item["status"][view] = e["status"]
    return sorted(items.values(), key=lambda i: i["target_id"])


def decisions_path(structure_dir: Path) -> Path:
    return structure_dir / DECISIONS_FILE


def write_file(structure_dir: Path, decisions: dict[str, dict], donors: dict[str, dict]) -> None:
    path = decisions_path(structure_dir)
    if not decisions and not donors:
        if path.exists():
            path.unlink()                              # nothing left: no file, so no stale copy in runs
        return
    doc = {"structure_id": structure_dir.name,
           "about": "Expert decisions per target; written by tools/target_review.py, read by the "
                    "complete step. A decision applies only to the placement it names. 'donors' names the "
                    "donor the expert chose for a target instead of the automatic one.",
           "decisions": decisions}
    if donors:
        doc["donors"] = donors
    write_json(path, doc)


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
    write_file(structure_dir, decisions, read_donor_choices(path))


def set_donor(structure_dir: Path, item: dict[str, Any], donor_id: str | None, flipped: bool | None, note: str,
              run_dir: Path) -> None:
    """Store the expert's donor for ``item`` (``donor_id`` None: back to the automatic donor).
    ``flipped`` None takes the orientation listed for that donor."""
    path = decisions_path(structure_dir)
    donors = read_donor_choices(path)
    if donor_id is None:
        donors.pop(item["key"], None)
    else:
        option = next((o for o in item["options"] if o["donor_id"] == donor_id), None)
        if option is None:
            known = ", ".join(o["donor_id"] for o in item["options"]) or "none (a donor can be named for tier A only)"
            raise ContractError(f"{item['target_id']}: {donor_id!r} cannot be its donor. Possible: {known}")
        donors[item["key"]] = {
            "donor_id": donor_id, "flipped": option["flipped"] if flipped is None else bool(flipped),
            "note": note.strip(), "target_id_at_choice": item["target_id"], "run": runio.relative(run_dir),
            "chosen_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    write_file(structure_dir, read_decisions(path), donors)


def donor_state(donors: dict[str, dict], item: dict[str, Any]) -> str:
    """What the file says about the donor of ``item``, against what this run placed."""
    want = donors.get(item["key"])
    placed = item["placement"]
    if want is None:
        return "chosen by the expert in this run, choice removed since: run the complete step" if item["chosen"] else ""
    if (want["donor_id"], want["flipped"]) == (placed["donor_id"], placed["flipped"]) and item["chosen"]:
        return f"chosen by the expert (automatic: {item['chosen']['automatic_donor_id']})"
    return (f"expert chose {want['donor_id']}{' (flipped)' if want['flipped'] else ''}: "
            "not placed yet, run the complete step")


def current(decisions: dict[str, dict], item: dict[str, Any]) -> str:
    d = decisions.get(item["key"])
    if d is None:
        return "-"
    return d["decision"] if d["placement"] == item["placement"] else f"{d['decision']} (stale: placement changed)"


def describe(item: dict[str, Any], decisions: dict[str, dict], donors: dict[str, dict] | None = None) -> str:
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
    state = donor_state(donors or {}, item)
    if state:
        lines.insert(2, f"donor: {state}")
    return "\n".join(lines)


def table(items: list[dict[str, Any]], decisions: dict[str, dict], donors: dict[str, dict] | None = None) -> str:
    rows = ["target  tier  donor                      score  seam   automatic            expert"]
    for i in items:
        auto = i["status"].get("restoration") or next(iter(i["status"].values()))
        score = "-" if i["score"] is None else f"{i['score']:.2f}"
        seam = "-" if i["seam"] is None else f"{i['seam']:.2f}"
        rows.append(f"{i['target_id']:7s} {i['tier']:5s} {str(i['donor_id']):26s} {score:6s} {seam:6s} "
                    f"{auto:20s} {current(decisions, i)}")
        state = donor_state(donors or {}, i)
        if state:
            rows.append(f"        donor: {state}")
    return "\n".join(rows)


def window(run_dir: Path, structure_dir: Path, items: list[dict[str, Any]]) -> None:
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"Target review: {runio.relative(run_dir)}")
    state = {"index": 0, "photo": None, "donor_photo": None}
    source_path = next((p for p in (run_dir / "input.png",) if p.is_file()), None)
    source = Image.open(source_path).convert("RGB") if source_path else None       # for the donor preview
    head = tk.Label(root, anchor="w", justify="left", font=("Consolas", 10))
    head.pack(fill="x", padx=8, pady=(8, 4))
    picture = tk.Label(root, bg="black")
    picture.pack(padx=8)
    note_row = tk.Frame(root)
    note_row.pack(fill="x", padx=8, pady=4)
    tk.Label(note_row, text="Note:").pack(side="left")
    note = tk.Entry(note_row)
    note.pack(side="left", fill="x", expand=True, padx=(4, 0))
    donor_row = tk.Frame(root)
    donor_row.pack(fill="x", padx=8, pady=4)
    tk.Label(donor_row, text="Donor:").pack(side="left")
    donor_box = ttk.Combobox(donor_row, state="readonly", font=("Consolas", 9), width=90)
    donor_box.pack(side="left", padx=4)
    use_button = tk.Button(donor_row, text="Use this donor", command=lambda: choose(True))
    use_button.pack(side="left", padx=3)
    auto_button = tk.Button(donor_row, text="Automatic donor", command=lambda: choose(False))
    auto_button.pack(side="left", padx=3)
    donor_picture = tk.Label(donor_row, bg="black")
    donor_picture.pack(side="left", padx=8)
    buttons = tk.Frame(root)
    buttons.pack(pady=(2, 8))

    def preview(_event: Any = None) -> None:
        """Crop of the element picked in the donor list, from the run's photograph."""
        options = items[state["index"]]["options"]
        index = donor_box.current()
        if source is None or index < 0 or index >= len(options):
            state["donor_photo"] = None
            donor_picture.config(image="")
            return
        option = options[index]
        crop = source.crop(tuple(option["box"]))
        if option["flipped"]:
            crop = crop.transpose(Image.FLIP_LEFT_RIGHT)
        scale = min(DONOR_PREVIEW_PX / crop.height, DONOR_PREVIEW_PX * 2 / crop.width)
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))))
        state["donor_photo"] = ImageTk.PhotoImage(crop)
        donor_picture.config(image=state["donor_photo"])

    def choose(use: bool) -> None:
        item = items[state["index"]]
        index = donor_box.current()
        if use and 0 <= index < len(item["options"]):
            set_donor(structure_dir, item, item["options"][index]["donor_id"], None, note.get(), run_dir)
        elif not use:
            set_donor(structure_dir, item, None, None, "", run_dir)
        show()

    donor_box.bind("<<ComboboxSelected>>", preview)

    def show() -> None:
        item = items[state["index"]]
        decisions = read_decisions(decisions_path(structure_dir))
        donors = read_donor_choices(decisions_path(structure_dir))
        head.config(text=f"[{state['index'] + 1} / {len(items)}]  " + describe(item, decisions, donors))
        donor_box.config(values=[option_text(o) for o in item["options"]])
        wanted = (donors.get(item["key"]) or {}).get("donor_id", item["placement"]["donor_id"])
        ids = [o["donor_id"] for o in item["options"]]
        if wanted in ids:
            donor_box.current(ids.index(wanted))
        else:
            donor_box.set("")
        for widget in (use_button, auto_button):
            widget.config(state="normal" if item["options"] else "disabled")
        preview()
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
    parser = argparse.ArgumentParser(description="Approve or reject the placed donors of a run, or name another donor.")
    parser.add_argument("--run", required=True, type=Path, help="run folder, e.g. outputs/<id>/<timestamp>")
    parser.add_argument("--list", action="store_true", help="print the targets and decisions, open no window")
    parser.add_argument("--approve", nargs="+", default=[], metavar="TARGET")
    parser.add_argument("--reject", nargs="+", default=[], metavar="TARGET")
    parser.add_argument("--clear", nargs="+", default=[], metavar="TARGET")
    parser.add_argument("--note", default="", help="note stored with --approve / --reject / --donor")
    parser.add_argument("--options", metavar="TARGET", help="list the elements that can be named as its donor")
    parser.add_argument("--donor", nargs=2, metavar=("TARGET", "ELEMENT"),
                        help="use ELEMENT as the donor of TARGET from the next complete step on")
    parser.add_argument("--flip", choices=("auto", "yes", "no"), default="auto",
                        help="with --donor: mirror the donor (auto = the orientation the list shows)")
    parser.add_argument("--auto-donor", nargs="+", default=[], metavar="TARGET",
                        help="remove the expert's donor choice")
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
        donor_changes = ([(opts.donor[0], opts.donor[1])] if opts.donor else []) + [(t, None) for t in opts.auto_donor]
        for target, donor_id in donor_changes + ([(opts.options, None)] if opts.options else []):
            if target not in by_id:
                raise ContractError(f"{target}: no placed donor with this ID in the run. Known: {', '.join(by_id)}")
        for target, donor_id in donor_changes:
            flipped = {"auto": None, "yes": True, "no": False}[opts.flip]
            set_donor(structure_dir, by_id[target], donor_id, flipped, opts.note, run_dir)
        if opts.options:
            options = by_id[opts.options]["options"]
            print(f"Possible donors of {opts.options}:")
            print("\n".join("  " + option_text(o) for o in options) if options else "  none (tier A targets only)")
            print()
        if not changes and not donor_changes and not opts.list and not opts.options:
            window(run_dir, structure_dir, items)
        print(table(items, read_decisions(decisions_path(structure_dir)),
                    read_donor_choices(decisions_path(structure_dir))))
        print(f"\nDecisions: {runio.relative(decisions_path(structure_dir))}")
        print(f"Apply them: python run.py --structure {structure_dir.name} --from complete --no-review")
    except ContractError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
