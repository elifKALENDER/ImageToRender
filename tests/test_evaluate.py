"""Tests for evaluation/evaluate.py (detection matching)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

import evaluate  # noqa: E402


def item(id_: str, cls: str, box: list[float]) -> dict:
    return {"id": id_, "class": cls, "box": box, "score": 0.1, "status": None}


class DetectionMatchTest(unittest.TestCase):
    def test_one_to_one_greedy_by_iou(self) -> None:
        refs = [item("r1", "arch", [0, 0, 10, 10])]
        preds = [item("p1", "arch", [1, 1, 11, 11]), item("p2", "arch", [0, 0, 10, 10])]
        matched = evaluate.match(refs, preds, 0.5)
        self.assertEqual([(ri, pi) for ri, pi, _ in matched], [(0, 1)])

    def test_class_must_match_except_aliases(self) -> None:
        refs = [item("r1", "pilaster", [0, 0, 10, 10]), item("r2", "opening", [20, 0, 30, 10])]
        preds = [item("p1", "column", [0, 0, 10, 10]), item("p2", "window", [20, 0, 30, 10])]
        result = evaluate.evaluate_detection(refs, preds, 0.5)
        self.assertEqual(result["matched"], 1)                  # opening = window counts, column != pilaster
        self.assertEqual(result["recall_class_agnostic"], 1.0)  # both boxes are right when class is ignored

    def test_precision_only_on_classes_the_reference_covers(self) -> None:
        refs = [item("r1", "arch", [0, 0, 10, 10])]
        preds = [item("p1", "arch", [0, 0, 10, 10]), item("p2", "roof", [50, 50, 60, 60])]
        result = evaluate.evaluate_detection(refs, preds, 0.5)
        self.assertEqual(result["precision_on_judged_classes"], 1.0)
        roof = next(r for r in result["per_predicted_class"] if r["class"] == "roof")
        self.assertFalse(roof["judged"])
        self.assertIsNone(roof["precision"])


if __name__ == "__main__":
    unittest.main()
