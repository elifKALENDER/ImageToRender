"""Tests for i2r.review with scripted answers (no terminal, no GPU)."""
from __future__ import annotations

import unittest

from i2r import review

CLASSES = ["arch", "pilaster", "column", "window"]


def elements() -> list[dict]:
    rows = [
        ("arch_01", "arch", 0.03), ("column_01", "column", 0.02), ("column_02", "column", 0.01),
        ("window_01", "window", 0.02), ("arch_02", "arch", 0.40),
    ]
    out = [{"id": i, "class": c, "score": s, "box": [0, 0, 10, 10]} for i, c, s in rows]
    review.initialise(out)
    return out


def scripted(lines: list[str]):
    answers = iter(lines)

    def ask(_prompt: str) -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError

    return ask


class ReviewSessionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.elements = elements()
        self.by_id = {e["id"]: e for e in self.elements}
        # numbers: 1 arch_01, 2 column_01, 3 column_02, 4 window_01
        self.session = review.Session(self.elements, ["arch_01", "column_01", "column_02", "window_01"], CLASSES)
        self.output: list[str] = []

    def run_script(self, lines: list[str]) -> None:
        review.run_session(self.session, scripted(lines), self.output.append)

    def test_reject_reclassify_class_and_confirm_rest(self) -> None:
        self.run_script(["x 4", "C column pilaster", ""])
        self.assertTrue(self.by_id["window_01"]["rejected"])
        self.assertEqual(self.by_id["column_01"]["class"], "pilaster")
        self.assertEqual(self.by_id["column_01"]["original_class"], "column")
        self.assertEqual(self.by_id["column_02"]["review_status"], "reclassified")
        self.assertTrue(self.by_id["arch_01"]["confirmed"])
        self.assertEqual(self.session.open, [])
        self.assertEqual(len(self.session.log), 4)

    def test_single_reclassify_validates_class(self) -> None:
        self.run_script(["c 2 gargoyle", "c 2 pilaster", "q"])
        self.assertTrue(any("unknown class" in line for line in self.output))
        self.assertEqual(self.by_id["column_01"]["class"], "pilaster")
        self.assertFalse(self.by_id["arch_01"]["confirmed"])  # q leaves the rest open

    def test_rejected_elements_are_kept_not_deleted(self) -> None:
        self.run_script(["x 1 2", "q"])
        self.assertEqual(len(self.elements), 5)
        self.assertEqual([e["id"] for e in review.active(self.elements)], ["column_02", "window_01", "arch_02"])

    def test_errors_do_not_change_anything(self) -> None:
        self.run_script(["x 9", "k abc", "zzz", "q"])
        self.assertEqual(self.session.log, [])
        self.assertTrue(any("unknown number: 9" in line for line in self.output))
        self.assertTrue(any("unknown command" in line for line in self.output))

    def test_already_decided_number_is_refused(self) -> None:
        self.run_script(["x 1", "k 1", "q"])
        self.assertTrue(self.by_id["arch_01"]["rejected"])
        self.assertTrue(any("already decided" in line for line in self.output))

    def test_end_of_input_never_blocks(self) -> None:
        self.run_script([])
        self.assertEqual(len(self.session.open), 4)
        self.assertTrue(all(not e["confirmed"] for e in self.elements))


if __name__ == "__main__":
    unittest.main()
