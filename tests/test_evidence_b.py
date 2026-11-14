"""Tests for tier B (rows, symmetry) and tier C regions of i2r.evidence on synthetic layouts."""
from __future__ import annotations

import unittest

import numpy as np

from i2r import evidence
from i2r.runio import load_config

W, H = 400, 200


def bcfg(**overrides) -> dict:
    cfg = dict(load_config()["evidence"]["tier_b"])
    cfg.update(overrides)
    return cfg


def el(element_id: str, cls: str, cx: int, cy: int, w: int = 20, h: int = 40,
       status: str = "intact", score: float = 1.0) -> dict:
    return {"id": element_id, "class": cls, "box": [cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
            "score": score, "source": "expert", "damage_status": status}


def damaged(*boxes: list[int]) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 1
    return mask


def row_of_five(skip: int) -> list[dict]:
    """Arches at x = 50, 100, 150, 200, 250 (y = 70); the one at index ``skip`` is missing."""
    return [el(f"arch_{i}", "arch", 50 + 50 * i, 70) for i in range(5) if i != skip]


class TierB1Test(unittest.TestCase):
    def test_gap_in_the_middle_gives_one_target_at_the_third_position(self) -> None:
        damage = damaged([130, 40, 170, 100])
        targets, diag = evidence.find_tier_b(row_of_five(skip=2), damage, bcfg(symmetry_enabled=False))
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertEqual((t["tier"], t["rule"], t["class"]), ("B", "B1", "arch"))
        self.assertEqual(t["target_box"], [140, 50, 160, 90])
        self.assertIn(t["donor_id"], ("arch_1", "arch_3"))       # nearest row member
        self.assertFalse(t["flipped"])
        self.assertAlmostEqual(t["confidence"], 1.0, places=3)   # exact spacing, donor score 1
        self.assertEqual(t["basis"]["between"], ["arch_1", "arch_3"])
        self.assertEqual(diag["B2"]["enabled"], False)

    def test_missing_last_element_is_not_extrapolated(self) -> None:
        damage = damaged([230, 40, 270, 100])
        targets, _ = evidence.find_tier_b(row_of_five(skip=4), damage, bcfg(symmetry_enabled=False))
        self.assertEqual(targets, [])

    def test_gap_outside_the_damage_mask_gives_nothing(self) -> None:
        damage = damaged([300, 150, 340, 190])
        targets, _ = evidence.find_tier_b(row_of_five(skip=2), damage, bcfg(symmetry_enabled=False))
        self.assertEqual(targets, [])

    def test_fragmented_element_in_the_slot_blocks_it(self) -> None:
        elements = row_of_five(skip=-1)
        elements[2]["damage_status"] = "fragmented"               # tier A's job, not B
        damage = damaged([130, 40, 170, 100])
        targets, _ = evidence.find_tier_b(elements, damage, bcfg(symmetry_enabled=False))
        self.assertEqual(targets, [])

    def test_row_too_short(self) -> None:
        elements = [el("arch_0", "arch", 50, 70), el("arch_2", "arch", 150, 70)]
        damage = damaged([80, 40, 120, 100])
        targets, diag = evidence.find_tier_b(elements, damage, bcfg(symmetry_enabled=False))
        self.assertEqual(targets, [])
        self.assertIn("fewer than", diag["B1"]["rows"][0]["skipped"])

    def test_confidence_uses_donor_score(self) -> None:
        elements = row_of_five(skip=2)
        for e in elements:
            e["score"] = 0.5
        targets, _ = evidence.find_tier_b(elements, damaged([130, 40, 170, 100]), bcfg(symmetry_enabled=False))
        self.assertAlmostEqual(targets[0]["confidence"], 0.5, places=3)


def symmetric_layout() -> list[dict]:
    """Facade symmetric about x = 200: window pair (right one missing), arch pair, pilaster pair, a
    door on the axis."""
    return [
        el("window_l", "window", 100, 50),
        el("arch_l", "arch", 60, 140), el("arch_r", "arch", 340, 140),
        el("pilaster_l", "pilaster", 150, 140, w=10), el("pilaster_r", "pilaster", 250, 140, w=10),
        el("door", "door", 200, 140, w=30, h=50),
    ]


class TierB2Test(unittest.TestCase):
    def test_mirrored_pair_with_one_side_removed(self) -> None:
        damage = damaged([280, 20, 320, 80])
        targets, diag = evidence.find_tier_b(symmetric_layout(), damage, bcfg(rows_enabled=False))
        axis = diag["B2"]["axis"]
        self.assertEqual((axis["source"], axis["axis_x"], axis["enabled"]), ("estimated", 200.0, True))
        self.assertEqual(axis["agreement"], 1.0)
        self.assertEqual(len(targets), 1)
        t = targets[0]
        self.assertEqual((t["rule"], t["class"], t["donor_id"], t["flipped"]), ("B2", "window", "window_l", True))
        self.assertEqual(t["target_box"], [290, 30, 310, 70])

    def test_given_axis_is_used(self) -> None:
        damage = damaged([280, 20, 320, 80])
        targets, diag = evidence.find_tier_b(symmetric_layout(), damage,
                                             bcfg(rows_enabled=False, symmetry_axis_x=200))
        self.assertEqual(diag["B2"]["axis"]["source"], "structure.yaml")
        self.assertEqual(len(targets), 1)

    def test_poor_axis_switches_b2_off(self) -> None:
        damage = damaged([280, 20, 320, 80])
        targets, diag = evidence.find_tier_b(symmetric_layout(), damage,
                                             bcfg(rows_enabled=False, symmetry_axis_x=170))
        self.assertEqual(targets, [])
        self.assertFalse(diag["B2"]["axis"]["enabled"])
        self.assertIn("below symmetry_min_agreement", diag["B2"]["axis"]["reason"])

    def test_b1_and_b2_on_the_same_slot_give_one_target(self) -> None:
        # Row of five arches symmetric about x = 150, the third missing: B1 implies it; B2 cannot (the
        # mirror of an element is never the slot itself). Add a right-hand slot that both rules imply.
        elements = [el(f"arch_{i}", "arch", 50 + 50 * i, 70) for i in (0, 1, 2, 3)]  # 50..200, 250 missing
        elements += [el("arch_5", "arch", 300, 70)]
        elements += [el("p_l", "pilaster", 20, 160, w=10), el("p_r", "pilaster", 330, 160, w=10)]
        damage = damaged([230, 40, 270, 100])
        targets, _ = evidence.find_tier_b(elements, damage, bcfg(symmetry_axis_x=175))
        boxes = [t["target_box"] for t in targets]
        self.assertEqual(boxes.count([240, 50, 260, 90]), 1)
        self.assertEqual(len(targets), 1)
        self.assertIn("also_implied_by", targets[0])


class TierCTest(unittest.TestCase):
    def test_uncovered_damage_becomes_regions(self) -> None:
        damage = damaged([10, 10, 50, 50], [100, 100, 120, 110])
        records, masks = evidence.tier_c_regions(damage, [[10, 10, 50, 30]], min_region_px=64)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["target_box"], [10, 30, 50, 50])
        self.assertEqual(records[0]["area_px"], 800)
        self.assertEqual(records[1]["area_px"], 200)
        self.assertTrue(all(r["tier"] == "C" and r["confidence"] == 0.0 for r in records))
        self.assertEqual(int(masks[0].sum()), 800)

    def test_small_leftovers_are_dropped(self) -> None:
        damage = damaged([10, 10, 50, 50])
        records, _ = evidence.tier_c_regions(damage, [[10, 10, 50, 48]], min_region_px=100)
        self.assertEqual(records, [])


class FacadeLimitTest(unittest.TestCase):
    """T27: damage outside the facade limit is its own kind of tier C region."""

    def elements(self) -> list[dict]:
        return [el("a", "arch", 60, 50), el("b", "arch", 200, 50), el("c", "base", 130, 120, w=160, h=20)]

    def test_hull_of_the_element_boxes(self) -> None:
        facade, used = evidence.facade_from_hull(self.elements(), (H, W), 0, 3)
        self.assertEqual(used, 3)
        self.assertEqual(facade[50, 130], 1)            # between the elements
        self.assertEqual(facade[129, 130], 1)           # lowest row of the lowest box
        self.assertEqual(facade[130, 130], 0)           # one row below it
        self.assertEqual(facade[150, 130], 0)           # the ground
        grown, _ = evidence.facade_from_hull(self.elements(), (H, W), 5, 3)
        self.assertEqual(grown[134, 130], 1)
        self.assertEqual(grown[150, 130], 0)

    def test_rejected_elements_do_not_count_and_too_few_give_no_limit(self) -> None:
        elements = self.elements()
        elements[2]["rejected"] = True
        facade, used = evidence.facade_from_hull(elements, (H, W), 0, 3)
        self.assertIsNone(facade)
        self.assertEqual(used, 2)

    def test_mask_reaching_below_the_hull_is_split(self) -> None:
        facade, _ = evidence.facade_from_hull(self.elements(), (H, W), 0, 3)
        damage = damaged([100, 60, 140, 170])           # from the wall down across the base onto the ground
        records, masks = evidence.tier_c_regions(damage, [], min_region_px=64, facade=facade)
        self.assertEqual([r["kind"] for r in records], ["uncovered_region", "outside_facade"])
        inside, outside = records
        self.assertEqual(inside["target_box"], [100, 60, 140, 130])
        self.assertEqual(outside["target_box"], [100, 130, 140, 170])
        self.assertEqual(outside["area_px"], 40 * 40)
        self.assertEqual((outside["tier"], outside["confidence"], outside["donor_id"]), ("C", 0.0, None))
        self.assertIn("outside the facade limit", outside["reason"])
        self.assertFalse((masks[0] & masks[1]).any())
        self.assertEqual(int(masks[0].sum() + masks[1].sum()), int(damage.sum()))

    def test_without_a_limit_nothing_changes(self) -> None:
        damage = damaged([100, 60, 140, 170])
        records, _ = evidence.tier_c_regions(damage, [], min_region_px=64)
        self.assertEqual([r["kind"] for r in records], ["uncovered_region"])


if __name__ == "__main__":
    unittest.main()
