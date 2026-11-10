"""Tests for tier A of i2r.evidence with fake embeddings (no model, no GPU)."""
from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from i2r import evidence
from i2r.runio import REPO_ROOT, ContractError, load_config

W, H = 160, 100
RED, GREEN = (200, 50, 50), (50, 200, 50)


def fake_embed(crops: list[np.ndarray]) -> np.ndarray:
    """Mean colour around mid-grey: same colour -> cosine 1, different colour -> much lower."""
    return np.array([c.reshape(-1, 3).mean(axis=0) - 128.0 for c in crops])


def settings(**overrides) -> dict:
    ecfg = dict(load_config()["evidence"])
    ecfg.update(overrides)
    return ecfg


def element(element_id: str, box: list[int], status: str, cls: str = "arch", **extra) -> dict:
    return {"id": element_id, "class": cls, "box": box, "score": 0.5, "source": "detector",
            "damage_status": status, **extra}


class Scene:
    def __init__(self) -> None:
        self.rgb = np.full((H, W, 3), 128, dtype=np.uint8)
        self.damage = np.zeros((H, W), dtype=np.uint8)
        self.elements: list[dict] = []

    def add(self, el: dict, color: tuple[int, int, int]) -> dict:
        x1, y1, x2, y2 = el["box"]
        self.rgb[y1:y2, x1:x2] = color
        self.elements.append(el)
        return el

    def damage_box(self, box: list[int]) -> None:
        x1, y1, x2, y2 = box
        self.damage[y1:y2, x1:x2] = 1


def standard_scene() -> Scene:
    s = Scene()
    s.add(element("target", [10, 10, 30, 50], "fragmented"), RED)
    s.damage_box([10, 30, 30, 50])                                    # bottom half lost
    s.add(element("same_colour", [40, 10, 60, 50], "intact"), RED)
    s.add(element("other_colour", [70, 10, 90, 52], "intact"), GREEN)
    s.add(element("too_big", [100, 5, 150, 95], "intact"), RED)       # 2.5x wider: size filter
    s.add(element("rejected_one", [40, 55, 60, 95], "intact", rejected=True), RED)
    s.add(element("pilaster_x", [70, 55, 90, 95], "intact", cls="pilaster"), RED)  # other class
    return s


class TierATest(unittest.TestCase):
    def test_ranking_alternatives_and_filters(self) -> None:
        s = standard_scene()
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed)
        self.assertEqual(record["tier"], "A")
        self.assertIsNone(record["reason"])
        self.assertEqual(record["donor_id"], "same_colour")
        self.assertAlmostEqual(record["similarity"], 1.0, places=3)
        self.assertEqual(record["confidence"], record["similarity"])
        self.assertEqual([a["donor_id"] for a in record["alternatives"]], ["other_colour"])
        self.assertLess(record["alternatives"][0]["similarity"], record["similarity"])
        self.assertEqual(record["visible_fraction"], 0.5)
        # both orientations of both size-compatible donors were scored
        self.assertEqual(len(record["candidates"]), 4)
        reasons = {r["donor_id"]: r["reason"] for r in record["rejected_donors"]}
        self.assertIn("size ratio", reasons["too_big"])
        self.assertEqual(reasons["rejected_one"], "rejected")
        self.assertNotIn("pilaster_x", reasons)

    def test_threshold_makes_tier_c(self) -> None:
        s = standard_scene()
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(min_similarity=1.01), fake_embed)
        self.assertEqual(record["tier"], "C")
        self.assertIn("below min_similarity", record["reason"])
        self.assertIsNone(record["donor_id"])
        self.assertEqual(record["alternatives"], [])
        self.assertAlmostEqual(record["similarity"], 1.0, places=3)  # best score kept for diagnosis

    def test_no_donor_of_the_class(self) -> None:
        s = Scene()
        s.add(element("base_1", [10, 10, 30, 50], "fragmented", cls="base"), RED)
        s.damage_box([10, 30, 30, 50])
        s.add(element("arch_1", [40, 10, 60, 50], "intact"), RED)
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed)
        self.assertEqual(record["tier"], "C")
        self.assertIn("no intact element of class", record["reason"])

    def test_fully_damaged_box(self) -> None:
        s = standard_scene()
        s.damage_box([10, 10, 30, 50])
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed)
        self.assertEqual(record["tier"], "C")
        self.assertIn("below min_visible_fraction", record["reason"])
        self.assertEqual(record["candidates"], [])

    def test_mostly_hidden_fragment_is_not_scored(self) -> None:
        s = standard_scene()
        s.damage[:] = 0
        s.damage_box([10, 18, 30, 50])                                    # 80 % hidden, 20 % visible
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(min_visible_fraction=0.25), fake_embed)
        self.assertEqual(record["tier"], "C")
        self.assertIsNone(record["similarity"])
        (record,) = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(min_visible_fraction=0.15), fake_embed)
        self.assertEqual(record["tier"], "A")

    def test_no_donor_reason_names_the_size_filter(self) -> None:
        s = standard_scene()
        for e in s.elements:
            if e["id"] in ("same_colour", "other_colour"):
                e["damage_status"] = "fragmented"
        records = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed)
        first = records[0]
        self.assertEqual(first["tier"], "C")
        self.assertIn("1 size", first["reason"])
        self.assertIn("not intact", first["reason"])

    def test_one_record_per_fragmented_element(self) -> None:
        s = standard_scene()
        s.add(element("target_2", [10, 55, 30, 95], "fragmented"), GREEN)
        s.damage_box([10, 55, 20, 95])
        records = evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), fake_embed)
        self.assertEqual([r["element_id"] for r in records], ["target", "target_2"])
        self.assertEqual(records[1]["donor_id"], "other_colour")
        self.assertEqual([r["target_id"] for r in records], ["t_001", "t_002"])

    def test_wrong_number_of_vectors_fails(self) -> None:
        s = standard_scene()
        with self.assertRaises(ContractError):
            evidence.find_tier_a(s.elements, s.rgb, s.damage, settings(), lambda crops: fake_embed(crops)[:-1])


class CropTest(unittest.TestCase):
    def test_donor_gets_the_target_hole(self) -> None:
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        rgb[0:10, 20:30] = 200      # donor: top half bright, bottom half black
        rgb[10:20, 20:30] = 0
        damage = np.zeros((40, 40), dtype=np.uint8)
        hole = np.zeros((10, 10), dtype=bool)
        hole[5:, :] = True          # target's bottom half is lost
        out = evidence.donor_crop(rgb, damage, [20, 0, 30, 20], [0, 0, 10, 10], hole, flipped=False)
        self.assertEqual(out.shape, (10, 10, 3))
        self.assertTrue((out[5:] == out[0]).all())   # hidden part filled with the visible mean

    def test_flip(self) -> None:
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :5] = 255
        damage = np.zeros((10, 10), dtype=np.uint8)
        out = evidence.donor_crop(rgb, damage, [0, 0, 10, 10], [0, 0, 10, 10], None, flipped=True)
        self.assertEqual(int(out[0, 0, 0]), 0)
        self.assertEqual(int(out[0, 9, 0]), 255)

    def test_module_imports_no_torch(self) -> None:
        code = "import sys; import i2r.evidence; sys.exit(1 if 'torch' in sys.modules else 0)"
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
