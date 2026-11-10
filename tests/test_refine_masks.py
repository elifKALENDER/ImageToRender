"""Tests for tools/refine_masks.py without SAM (the masks are given)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from i2r.runio import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tools"))
import refine_masks  # noqa: E402


class RefineMasksTest(unittest.TestCase):
    def test_clip_and_decide(self) -> None:
        mask = np.full((20, 20), 255, np.uint8)
        clipped = refine_masks.clip_to_box(mask, [5, 5, 10, 15])
        self.assertEqual(int((clipped > 0).sum()), 50)
        keep, fill = refine_masks.decide(clipped, [5, 5, 10, 15], 0.2)
        self.assertTrue(keep)
        self.assertEqual(fill, 1.0)
        thin = np.zeros((20, 20), np.uint8)
        thin[5, 5:10] = 255
        keep, fill = refine_masks.decide(thin, [5, 5, 10, 15], 0.2)
        self.assertFalse(keep)
        self.assertAlmostEqual(fill, 0.1)

    def test_refine_updates_only_kept_elements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "element_masks").mkdir()
            a = {"id": "a", "class": "arch", "box": [0, 0, 10, 10], "mask_file": "element_masks/a.png",
                 "mask_kind": "box"}
            b = {"id": "b", "class": "arch", "box": [10, 0, 20, 10], "mask_file": "element_masks/b.png",
                 "mask_kind": "box"}
            good = np.zeros((10, 20), np.uint8)
            good[2:8, 2:8] = 255
            good[0:10, 12:20] = 255              # spills outside a's box: clipped away
            empty = np.zeros((10, 20), np.uint8)
            rows = refine_masks.refine([a, b], [good, empty], [a, b], run_dir, 0.2)
            self.assertEqual([r["kept"] for r in rows], [True, False])
            self.assertEqual(a["mask_kind"], "sam_from_box")
            self.assertEqual(a["previous_mask_file"], "element_masks/a.png")
            saved = np.asarray(Image.open(run_dir / a["mask_file"]))
            self.assertEqual(int((saved > 0).sum()), 36)
            self.assertEqual(b["mask_kind"], "box")
            self.assertIn("below refine_masks.min_fill", rows[1]["reason"])

    def test_copy_upstream_leaves_later_steps_behind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            (src / "element_masks").mkdir(parents=True)
            dst.mkdir()
            for name in ("elements.json", "editor_source_inventory.json", "evidence.json", "damage_mask.png"):
                (src / name).write_text("x")
            copied = refine_masks.copy_upstream(src, dst)
            self.assertEqual(copied, ["editor_source_inventory.json", "element_masks", "elements.json"])
            self.assertFalse((dst / "evidence.json").exists())


if __name__ == "__main__":
    unittest.main()
