"""Tests for tools/synthetic_damage.py and `evaluate.py completion` scoring (T19). GPU-free."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "evaluation"))

import evaluate  # noqa: E402
import synthetic_damage as sd  # noqa: E402
from i2r.runio import ContractError  # noqa: E402

SCFG = {
    "hidden_fill": [0, 0, 0], "partial_fraction": 0.5, "partial_side": "top", "full_margin_px": 2,
    "background_cases": 2, "background_size_px": [6, 6], "background_clearance_px": 2, "background_grid_px": 2,
    "recovery_iou": 0.5, "background_covered_share": 0.5,
}


def element(id_: str, cls: str, box: list[int], status: str = "intact", rejected: bool = False) -> dict:
    return {"id": id_, "class": cls, "box": box, "mask_file": f"element_masks/{id_}.png",
            "damage_status": status, "rejected": rejected}


def record(tier: str, cls: str | None, box: list[int], kind: str = "fragmented_element", **kw) -> dict:
    return {"target_id": kw.get("target_id", "t_001"), "kind": kind, "tier": tier, "class": cls,
            "target_box": box, "element_id": kw.get("element_id"), "donor_id": kw.get("donor_id"),
            "rule": kw.get("rule"), "similarity": None, "reason": None}


class GeometryTest(unittest.TestCase):
    def test_partial_band_sides(self) -> None:
        box = [10, 20, 30, 60]
        self.assertEqual(sd.partial_band(box, 0.5, "top"), [10, 20, 30, 40])
        self.assertEqual(sd.partial_band(box, 0.5, "bottom"), [10, 40, 30, 60])
        self.assertEqual(sd.partial_band(box, 0.5, "left"), [10, 20, 20, 60])
        self.assertEqual(sd.partial_band(box, 0.5, "right"), [20, 20, 30, 60])
        with self.assertRaises(ContractError):
            sd.partial_band(box, 0.5, "middle")
        with self.assertRaises(ContractError):
            sd.partial_band(box, 1.0, "top")

    def test_expand_box_clipped_to_image(self) -> None:
        self.assertEqual(sd.expand_box([1, 1, 9, 9], 3, 10, 10), [0, 0, 10, 10])

    def test_background_boxes_avoid_elements_and_damage(self) -> None:
        damage = np.zeros((60, 60), np.uint8)
        damage[30:40, 30:40] = 1
        boxes = [[0, 0, 10, 10], [50, 50, 60, 60]]
        rects = sd.background_boxes(boxes, damage, (6, 6), 2, 2, 3)
        self.assertEqual(len(rects), 3)
        blocked = damage.copy()
        for b in boxes:
            blocked[b[1]:b[3], b[0]:b[2]] = 1
        for r in rects:
            self.assertFalse(blocked[max(0, r[1] - 2):r[3] + 2, max(0, r[0] - 2):r[2] + 2].any())
        self.assertEqual(rects, sd.background_boxes(boxes, damage, (6, 6), 2, 2, 3))  # deterministic

    def test_background_boxes_none_when_no_room(self) -> None:
        damage = np.ones((20, 20), np.uint8)
        self.assertEqual(sd.background_boxes([[0, 0, 20, 20]], damage, (6, 6), 2, 2, 2), [])


class PlanTest(unittest.TestCase):
    def doc(self) -> dict:
        return {"structure_id": "s", "image_size": [60, 60], "elements": [
            element("a1", "arch", [0, 0, 10, 10]),
            element("a2", "arch", [50, 50, 60, 60]),
            element("f1", "arch", [30, 0, 40, 10], status="fragmented"),
            element("r1", "arch", [0, 50, 10, 60], rejected=True),
        ]}

    def test_two_cases_per_intact_element_plus_background(self) -> None:
        damage = np.zeros((60, 60), np.uint8)
        damage[30:40, 30:40] = 1
        cases, skipped = sd.plan_cases(self.doc(), damage, SCFG)
        ids = [c["case_id"] for c in cases]
        self.assertEqual(ids[:4], ["partial__a1", "full__a1", "partial__a2", "full__a2"])
        self.assertEqual(sum(c["variant"] == "background" for c in cases), 2)
        self.assertEqual([s["id"] for s in skipped], ["f1"])

    def test_full_case_marks_rejected_and_deletes_nothing(self) -> None:
        doc = self.doc()
        spec = {"variant": "full", "element": doc["elements"][0], "hidden_box": [0, 0, 12, 12]}
        out = sd.case_inventory(doc, spec)
        self.assertEqual(len(out["elements"]), 4)
        self.assertTrue(out["elements"][0]["rejected"])
        self.assertTrue(out["elements"][0]["synthetic_hidden"])
        self.assertFalse(doc["elements"][0]["rejected"])  # input untouched

    def test_write_case_paints_and_adds_to_mask(self) -> None:
        doc = self.doc()
        rgb = np.full((60, 60, 3), 200, np.uint8)
        damage = np.zeros((60, 60), np.uint8)
        damage[0:3, 0:10] = 1  # already damaged part of the band is not "hidden"
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            (run_dir / "element_masks").mkdir(parents=True)
            spec = {"variant": "partial", "element": doc["elements"][0], "hidden_box": [0, 0, 10, 5],
                    "case_id": "partial__a1"}
            gt = sd.write_case(Path(tmp) / "case", spec, rgb, damage, doc, run_dir, "s", [0, 0, 0])
            case = Path(tmp) / "case"
            img = np.asarray(Image.open(case / "image.png"))
            mask = np.asarray(Image.open(case / "damage_mask.png"))
            hidden = np.asarray(Image.open(case / "hidden_mask.png"))
            self.assertEqual(gt["hidden_px"], 20)
            self.assertEqual(gt["expected_tier"], "A")
            self.assertTrue((img[3:5, 0:10] == 0).all())
            self.assertTrue((img[0:3, 0:10] == 200).all())   # original damage left as photographed
            self.assertTrue((img[5:, :] == 200).all())
            self.assertTrue((mask[0:5, 0:10] == 255).all())
            self.assertEqual(int((hidden > 0).sum()), 20)
            seed = json.loads((case / "seed" / "elements.json").read_text(encoding="utf-8"))
            self.assertEqual(len(seed["elements"]), 4)


class ScoringTest(unittest.TestCase):
    GT = {"variant": "full", "class": "arch", "element_box": [0, 0, 10, 10], "hidden_box": [0, 0, 10, 10]}

    def test_recovered_needs_same_class_iou_and_tier_a_or_b(self) -> None:
        recs = [record("B", "arch", [1, 0, 11, 10], kind="implied_slot", rule="B1", donor_id="a2"),
                record("A", "pilaster", [0, 0, 10, 10])]
        a = evaluate.assign_case(self.GT, recs, SCFG)
        self.assertEqual((a["tier"], a["recovered"]), ("B", True))

    def test_tier_c_record_is_not_recovered(self) -> None:
        a = evaluate.assign_case(self.GT, [record("C", "arch", [0, 0, 10, 10])], SCFG)
        self.assertEqual((a["tier"], a["recovered"]), ("C", False))

    def test_low_iou_is_not_recovered(self) -> None:
        a = evaluate.assign_case(self.GT, [record("A", "arch", [6, 0, 16, 10])], SCFG)
        self.assertEqual((a["tier"], a["recovered"]), ("C", False))

    def test_uncovered_region_never_matches(self) -> None:
        a = evaluate.assign_case(self.GT, [record("C", None, [0, 0, 10, 10], kind="uncovered_region")], SCFG)
        self.assertIsNone(a["target"])

    def test_background_tier(self) -> None:
        gt = {"variant": "background", "class": None, "element_box": None, "hidden_box": [0, 0, 10, 10]}
        self.assertEqual(evaluate.assign_case(gt, [record("C", None, [0, 0, 10, 10], kind="uncovered_region")],
                                              SCFG)["tier"], "C")
        self.assertEqual(evaluate.assign_case(gt, [record("A", "arch", [0, 0, 10, 6])], SCFG)["tier"], "A")
        self.assertEqual(evaluate.assign_case(gt, [record("A", "arch", [0, 0, 10, 3])], SCFG)["tier"], "C")

    def test_view_metrics(self) -> None:
        original = np.random.default_rng(0).integers(0, 255, (30, 30, 3), dtype=np.uint8)
        hidden = np.zeros((30, 30), np.uint8)
        hidden[10:20, 10:20] = 1
        case_image = sd.paint(original, hidden, [0, 0, 0])
        same = view_ok = evaluate.view_metrics(original, case_image, case_image, hidden, hidden)
        self.assertEqual((same["filled"], same["preservation"]), (0.0, 0))
        full = evaluate.view_metrics(original, case_image, original, hidden, hidden)
        self.assertEqual(full["filled"], 1.0)
        self.assertAlmostEqual(full["ssim"], 1.0, places=6)
        self.assertLess(view_ok["ssim"], full["ssim"])
        broken = original.copy()
        broken[0, 0] = (broken[0, 0].astype(int) + 7) % 256
        self.assertGreater(evaluate.view_metrics(original, original, broken, hidden, hidden)["preservation"], 0)

    def test_summary_counts(self) -> None:
        base = {"status": "completed", "restoration_preservation": 0, "first_construction_preservation": 0}
        rows = [
            {**base, "variant": "partial", "expected_tier": "A", "tier": "A", "tier_correct": True, "recovered": True},
            {**base, "variant": "partial", "expected_tier": "A", "tier": "C", "tier_correct": False, "recovered": False},
            {**base, "variant": "background", "expected_tier": "C", "tier": "C", "tier_correct": True,
             "recovered": None, "abstained": True},
        ]
        s = {x["variant"]: x for x in evaluate.summarise(rows)}
        self.assertEqual(s["partial"]["recovery"], 0.5)
        self.assertEqual(s["partial"]["tiers"], {"A": 1, "B": 0, "C": 1})
        self.assertEqual(s["background"]["abstention"], 1.0)
        self.assertEqual(s["background"]["preservation_max"], 0)


if __name__ == "__main__":
    unittest.main()


class GridTest(unittest.TestCase):
    def test_parse_grid_and_points(self) -> None:
        grid = evaluate.parse_grid(["evidence.min_similarity=0.6,0.8", "evidence.tier_b.row_tolerance=0.1"])
        self.assertEqual(grid, [("evidence.min_similarity", [0.6, 0.8]), ("evidence.tier_b.row_tolerance", [0.1])])
        points = evaluate.grid_points(grid)
        self.assertEqual(len(points), 2)
        self.assertEqual(points[1], {"evidence.min_similarity": 0.8, "evidence.tier_b.row_tolerance": 0.1})

    def test_parse_grid_rejects_bad_specs_and_model_keys(self) -> None:
        with self.assertRaises(ContractError):
            evaluate.parse_grid(["evidence.min_similarity"])
        with self.assertRaises(ContractError):
            evaluate.parse_grid(["evidence.crop_px=112,224"])

    def test_set_dotted_copies_and_checks_the_key(self) -> None:
        cfg = {"evidence": {"min_similarity": 0.6, "tier_b": {"row_tolerance": 0.15}}}
        out = evaluate.set_dotted(cfg, "evidence.tier_b.row_tolerance", 0.25)
        self.assertEqual(out["evidence"]["tier_b"]["row_tolerance"], 0.25)
        self.assertEqual(cfg["evidence"]["tier_b"]["row_tolerance"], 0.15)
        with self.assertRaises(ContractError):
            evaluate.set_dotted(cfg, "evidence.min_similiarity", 0.7)
        with self.assertRaises(ContractError):
            evaluate.set_dotted(cfg, "nothing.here", 1)

    def test_grid_summary(self) -> None:
        rows = [
            {"variant": "partial", "tier_correct": True, "recovered": True},
            {"variant": "full", "tier_correct": False, "recovered": False},
            {"variant": "background", "tier_correct": True, "recovered": None},
        ]
        s = evaluate.grid_summary(rows, {"by_tier": {"A": 1}})
        self.assertEqual((s["partial_recovery"], s["full_recovery"], s["background_tier_c"]), (1.0, 0.0, 1.0))
        self.assertAlmostEqual(s["tier_accuracy"], 2 / 3)
