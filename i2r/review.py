"""Step 2 — expert review of low-confidence detections.

Elements whose detector score is below ``review.confidence_threshold`` are shown on a numbered overlay
(``review_overlay.png``) and as a grid of crops (``review_sheet.png``), and listed in the terminal. The expert can confirm, reclassify or reject them,
several at a time:

    <Enter> or done     confirm every element still open and finish
    k 3 5 7             keep (confirm) elements 3, 5 and 7
    x 3 5 7             reject elements 3, 5 and 7
    c 4 pilaster        change element 4 to class "pilaster" (also confirms it)
    C column pilaster   change every open element of class "column" to "pilaster"
    l                   list the elements still open
    q                   stop; elements still open stay unconfirmed
    ?                   help

Nothing is deleted. Rejected elements stay in ``elements.json`` with ``rejected: true`` so the decision
can be audited; later steps skip them. Every decision is appended to ``review_log.json``.

With ``--no-review`` nothing is asked: every element is kept with ``confirmed: false``.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np
from PIL import Image

from i2r.detect import ELEMENTS_FILE, class_color, load_elements, validate_elements
from i2r.runio import read_json, write_json

if TYPE_CHECKING:
    import argparse

    from i2r.runio import Run

REVIEW_LOG = "review_log.json"
OVERLAY = "review_overlay.png"
SHEET = "review_sheet.png"

HELP = __doc__.split("several at a time:")[1].split("Nothing is deleted")[0].rstrip()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def active(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Elements later steps should use."""
    return [e for e in elements if not e.get("rejected")]


def initialise(elements: list[dict[str, Any]]) -> None:
    """Give every element the review fields, keeping earlier decisions if present."""
    for element in elements:
        element.setdefault("confirmed", False)
        element.setdefault("rejected", False)
        element.setdefault("review_status", "not_reviewed")


class Session:
    """Applies review commands to a list of elements. Pure logic; no file or terminal access."""

    def __init__(self, elements: list[dict[str, Any]], open_ids: list[str], classes: list[str]):
        self.by_id = {e["id"]: e for e in elements}
        self.numbers = {number: eid for number, eid in enumerate(open_ids, start=1)}
        self.open = list(open_ids)
        self.classes = classes
        self.log: list[dict[str, Any]] = []

    def number_of(self, element_id: str) -> int:
        return next(n for n, eid in self.numbers.items() if eid == element_id)

    def _resolve(self, tokens: list[str]) -> tuple[list[str], list[str]]:
        ids, errors = [], []
        for token in tokens:
            if not token.isdigit() or int(token) not in self.numbers:
                errors.append(f"unknown number: {token}")
                continue
            eid = self.numbers[int(token)]
            if eid not in self.open:
                errors.append(f"{token} ({eid}) already decided")
                continue
            ids.append(eid)
        return ids, errors

    def _decide(self, eid: str, action: str, new_class: str | None = None) -> None:
        element = self.by_id[eid]
        before = element["class"]
        if action == "reject":
            element.update(rejected=True, confirmed=False, review_status="rejected")
        elif action == "confirm":
            element.update(rejected=False, confirmed=True, review_status="confirmed")
        elif action == "reclassify":
            element.setdefault("original_class", before)
            element.update(**{"class": new_class}, rejected=False, confirmed=True, review_status="reclassified")
        self.open.remove(eid)
        self.log.append({"time": now(), "element_id": eid, "action": action,
                         "class_before": before, "class_after": element["class"], "score": element["score"]})

    def apply(self, line: str) -> tuple[bool, list[str]]:
        """Apply one command. Returns (finished, messages)."""
        parts = line.strip().split()
        if not parts or parts[0] in ("done", "ok"):
            for eid in list(self.open):
                self._decide(eid, "confirm")
            return True, ["confirmed all remaining"]
        cmd, args = parts[0], parts[1:]
        if cmd == "q":
            return True, [f"stopped; {len(self.open)} element(s) left unconfirmed"]
        if cmd == "?":
            return False, [HELP]
        if cmd == "l":
            return False, [self.describe(eid) for eid in self.open] or ["nothing open"]
        if cmd in ("k", "x"):
            ids, errors = self._resolve(args)
            for eid in ids:
                self._decide(eid, "confirm" if cmd == "k" else "reject")
            done = "kept" if cmd == "k" else "rejected"
            return False, errors + ([f"{done}: {', '.join(ids)}"] if ids else [])
        if cmd == "c":
            if len(args) != 2:
                return False, ["usage: c <number> <class>"]
            if args[1] not in self.classes:
                return False, [f"unknown class {args[1]!r}; choose from: {', '.join(self.classes)}"]
            ids, errors = self._resolve(args[:1])
            for eid in ids:
                self._decide(eid, "reclassify", args[1])
            return False, errors + ([f"{ids[0]} -> {args[1]}"] if ids else [])
        if cmd == "C":
            if len(args) != 2:
                return False, ["usage: C <from_class> <to_class>"]
            if args[1] not in self.classes:
                return False, [f"unknown class {args[1]!r}; choose from: {', '.join(self.classes)}"]
            ids = [eid for eid in self.open if self.by_id[eid]["class"] == args[0]]
            for eid in ids:
                self._decide(eid, "reclassify", args[1])
            return False, [f"{len(ids)} open element(s) of class {args[0]} -> {args[1]}"]
        return False, [f"unknown command {cmd!r}; type ? for help"]

    def describe(self, eid: str) -> str:
        e = self.by_id[eid]
        x1, y1, x2, y2 = e["box"]
        return (f"{self.number_of(eid):>3}  {eid:<14} {e['class']:<10} score {e['score']:.3f}  "
                f"box {x1},{y1}-{x2},{y2} ({x2 - x1}x{y2 - y1})")


def run_session(
    session: Session, ask: Callable[[str], str], say: Callable[[str], None], max_commands: int = 10_000
) -> None:
    say(HELP)
    say("")
    for eid in session.open:
        say(session.describe(eid))
    for _ in range(max_commands):
        if not session.open:
            return
        try:
            line = ask(f"review [{len(session.open)} open]> ")
        except EOFError:
            say("input closed; elements still open stay unconfirmed")
            return
        finished, messages = session.apply(line)
        for message in messages:
            say(message)
        if finished:
            return


def draw_numbers(rgb: np.ndarray, elements: list[dict[str, Any]], session: Session) -> np.ndarray:
    out = rgb.copy()
    for eid in session.open:
        e = session.by_id[eid]
        color = class_color(e["class"])
        x1, y1, x2, y2 = e["box"]
        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), color, 2, cv2.LINE_AA)
    # Labels last so boxes do not cover them.
    for eid in session.open:
        e = session.by_id[eid]
        color = class_color(e["class"])
        x1, y1 = e["box"][:2]
        text = f"{session.number_of(eid)} {e['class']}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1), (min(out.shape[1] - 1, x1 + tw + 4), y1 + th + 6), color, -1)
        cv2.putText(out, text, (x1 + 2, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def contact_sheet(rgb: np.ndarray, session: Session, tile: int = 150, columns: int = 8) -> np.ndarray:
    """Grid of element crops with number and class, readable when the overlay is crowded."""
    label_h = 18
    ids = list(session.open)
    rows = max(1, (len(ids) + columns - 1) // columns)
    sheet = np.full((rows * (tile + label_h), columns * tile, 3), 245, np.uint8)
    for index, eid in enumerate(ids):
        e = session.by_id[eid]
        x1, y1, x2, y2 = e["box"]
        crop = rgb[y1:y2, x1:x2]
        scale = min((tile - 6) / crop.shape[1], (tile - 6) / crop.shape[0])
        size = (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        r, c = divmod(index, columns)
        top, left = r * (tile + label_h), c * tile
        oy, ox = top + label_h + (tile - size[1]) // 2, left + (tile - size[0]) // 2
        sheet[oy:oy + size[1], ox:ox + size[0]] = crop
        color = class_color(e["class"])
        cv2.rectangle(sheet, (left, top), (left + tile - 1, top + label_h - 1), color, -1)
        text = f"{session.number_of(eid)} {e['class']} {e['score']:.3f}"
        cv2.putText(sheet, text, (left + 3, top + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


def open_file(path: Any) -> None:
    """Show the overlay in the default image viewer (Windows); ignored elsewhere."""
    if hasattr(os, "startfile"):
        try:
            os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError:
            pass


def run_step(run: "Run", opts: "argparse.Namespace") -> None:
    log = run.logger
    rcfg = run.config["review"]
    doc = load_elements(run.run_dir)
    elements = doc["elements"]
    initialise(elements)

    threshold = float(rcfg["confidence_threshold"])
    to_review = [e["id"] for e in elements
                 if not e["rejected"] and e["review_status"] == "not_reviewed" and e["score"] < threshold]
    for e in elements:
        if e["review_status"] == "not_reviewed" and e["id"] not in to_review and not e["rejected"]:
            e["review_status"] = "auto_accepted"

    log.info("[review] %d element(s), %d below threshold %.3f", len(elements), len(to_review), threshold)
    session = Session(elements, to_review, list(run.config["classes"]))

    if opts.no_review or not to_review:
        log.info("[review] skipped (%s)", "--no-review" if opts.no_review else "nothing to review")
    else:
        image = np.asarray(Image.open(run.image_path).convert("RGB"))
        Image.fromarray(draw_numbers(image, elements, session)).save(run.path(OVERLAY))
        Image.fromarray(contact_sheet(image, session)).save(run.path(SHEET))
        log.info("[review] numbered overlay: %s", run.path(OVERLAY))
        log.info("[review] element crops:    %s", run.path(SHEET))
        if rcfg.get("open_overlay", True):
            open_file(run.path(OVERLAY))
            open_file(run.path(SHEET))
        run_session(session, ask=input, say=lambda text: print(text, file=sys.stdout, flush=True))

    validate_elements(doc)
    write_json(run.path(ELEMENTS_FILE), doc)

    history = read_json(run.path(REVIEW_LOG)) if run.path(REVIEW_LOG).is_file() else []
    history.extend(session.log)
    write_json(run.path(REVIEW_LOG), history)

    counts: dict[str, int] = {}
    for e in elements:
        counts[e["review_status"]] = counts.get(e["review_status"], 0) + 1
    log.info("[review] result: %s | active elements: %d", counts, len(active(elements)))
    run.meta["steps"].setdefault("review", {})["counts"] = counts
