"""Tests for i2r.complete: donor transfer, the restoration rule and the preservation check (no GPU)."""
from __future__ import annotations

import copy
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from i2r import complete
from i2r.runio import REPO_ROOT, load_config

W, H = 120, 90
DONOR_BOX = [10, 10, 40, 70]
TARGET_BOX = [60, 10, 90, 70]
SHIFT = (3, 2)                         # the fragment's content sits this far right/down of its box


def config(**restoration) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["restoration"].update(restoration)
    return cfg


def scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Photograph, damage mask, donor element mask, and the donor texture."""
    rng = np.random.default_rng(7)
    texture = rng.integers(0, 256, size=(60, 30, 3), dtype=np.uint8)
    texture = np.asarray(  # smooth a little so correlation has a clear peak but is not trivial
        np.clip(texture.astype(np.float32) * 0.5 + 64, 0, 255), dtype=np.uint8)
    rgb = np.full((H, W, 3), 120, dtype=np.uint8)
    x1, y1, x2, y2 = DONOR_BOX
    rgb[y1:y2, x1:x2] = texture
    tx, ty = TARGET_BOX[0] + SHIFT[0], TARGET_BOX[1] + SHIFT[1]
    rgb[ty:ty + 60, tx:tx + 30] = texture
    damage = np.zeros((H, W), dtype=np.uint8)
    damage[40:72, 60:93] = 1                               # bottom half of the fragment is lost
    original_damaged = rgb.copy()
    original_damaged[damage > 0] = 30                      # what the photograph shows there: rubble
    donor_mask = np.zeros((H, W), dtype=np.uint8)
    donor_mask[y1:y2, x1:x2] = 1
    return original_damaged, damage, donor_mask, texture


def record(tier: str, target_id: str = "t_001", box: list[int] | None = None, donor: str | None = "donor",
           confidence: float = 0.9) -> dict:
    return {"target_id": target_id, "kind": "fragmented_element" if tier == "A" else "implied_slot",
            "tier": tier, "target_box": box or TARGET_BOX, "donor_id": donor,
            "donor_box": DONOR_BOX if donor else None, "flipped": False if donor else None,
            "confidence": confidence}


class TransferTest(unittest.TestCase):
    def test_tier_a_is_aligned_and_filled_only_inside_the_mask(self) -> None:
        rgb, damage, donor_mask, texture = scene()
        out, labels, report = complete.complete_view("restoration", rgb, damage, [record("A")],
                                                     {"donor": donor_mask}, config())
        entry = report["targets"][0]
        self.assertEqual(entry["status"], "completed")
        al = entry["alignment"]
        self.assertEqual((al["dx"], al["dy"], al["scale"]), (SHIFT[0], SHIFT[1], 1.0))
        self.assertGreater(al["score"], 0.99)
        # a pixel well inside the damage gets the donor texture at the aligned position
        y, x = 60, 75
        self.assertTrue(np.array_equal(out[y, x], texture[y - 10 - SHIFT[1], x - 60 - SHIFT[0]]))
        # nothing outside the mask changed, every label lies inside it
        self.assertTrue(np.array_equal(out[damage == 0], rgb[damage == 0]))
        self.assertTrue(np.all(damage[labels > 0] == 1))
        self.assertEqual(entry["pixels_changed"], int((labels == entry["label"]).sum()))
        self.assertEqual(report["completed"], 1)

    def test_feathered_edge(self) -> None:
        rgb, damage, donor_mask, _ = scene()
        cfg = config()
        cfg["complete"]["feather_px"] = 4
        out, _, _ = complete.complete_view("restoration", rgb, damage, [record("A")], {"donor": donor_mask}, cfg)
        # first damaged row: alpha 1/4, so the result lies between rubble (30) and the donor
        first, deep = out[40, 75].astype(int), out[60, 75].astype(int)
        self.assertTrue(np.all(np.abs(first - 30) > 0))
        self.assertTrue(np.all(np.abs(first - 30) < np.abs(deep - 30) + 1))

    def test_donor_damage_is_not_transferred(self) -> None:
        rgb, damage, donor_mask, _ = scene()
        damage[60:70, 10:40] = 1          # the donor's own bottom rows are damaged
        out, labels, report = complete.complete_view("restoration", rgb, damage, [record("A")],
                                                     {"donor": donor_mask}, config())
        self.assertGreater(report["targets"][0]["damage_px_in_box_left_unfilled"], 0)
        self.assertTrue(np.array_equal(out[damage == 0], rgb[damage == 0]))


class RuleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb, self.damage, mask, _ = scene()
        self.masks = {"donor": mask}
        self.records = [record("A"), record("B", "t_002"), record("C", "t_003", donor=None, confidence=0.0)]

    def statuses(self, cfg: dict) -> dict[str, str]:
        _, _, report = complete.complete_view("restoration", self.rgb, self.damage, self.records, self.masks, cfg)
        return {e["tier"]: e["status"] for e in report["targets"]}

    def test_restoration_completes_tier_a_only_by_default(self) -> None:
        self.assertEqual(self.statuses(config()), {"A": "completed", "B": "not_completed", "C": "not_completed"})

    def test_tier_b_switch(self) -> None:
        self.assertEqual(self.statuses(config(include_tier_b=True))["B"], "completed")

    def test_report_keeps_evidence_order(self) -> None:
        _, _, report = complete.complete_view("restoration", self.rgb, self.damage, self.records, self.masks, config())
        self.assertEqual([e["target_id"] for e in report["targets"]], ["t_001", "t_002", "t_003"])


class PreservationTest(unittest.TestCase):
    def test_a_change_outside_the_mask_fails_the_view(self) -> None:
        rgb, damage, donor_mask, _ = scene()
        real_paint = complete.paint

        def leaky_paint(out, labels, window, image, alpha, label):
            n = real_paint(out, labels, window, image, alpha, label)
            out[0, 0] = 255 - out[0, 0]   # deliberately touch a pixel outside the damage mask
            return n

        with mock.patch.object(complete, "paint", leaky_paint):
            with self.assertRaisesRegex(complete.PreservationError, "outside the damage mask"):
                complete.complete_view("restoration", rgb, damage, [record("A")], {"donor": donor_mask}, config())

    def test_module_imports_no_torch(self) -> None:
        code = "import sys; import i2r.complete; sys.exit(1 if 'torch' in sys.modules else 0)"
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
