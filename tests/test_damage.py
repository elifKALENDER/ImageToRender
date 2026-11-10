"""Tests for i2r.damage and tools/mask_editor.py (synthetic images, no GPU, no window)."""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from i2r import damage
from i2r.runio import REPO_ROOT, ContractError, Run, load_config, read_json, write_json

W, H = 120, 80
CONFIG = load_config()


def save_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((array > 0).astype(np.uint8) * 255, mode="L").save(path)


def rect(box: list[int]) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = box
    out[y1:y2, x1:x2] = 1
    return out


class SyntheticRun:
    """A run folder with an image, two boxed elements (plus optional extras) and a damage mask."""

    def __init__(self, tmp: Path, damage_array: np.ndarray | None, extra: list[dict] | None = None) -> None:
        structure = tmp / "structure"
        run_dir = tmp / "run"
        structure.mkdir()
        run_dir.mkdir()
        Image.new("RGB", (W, H), (180, 170, 150)).save(structure / "image.png")
        if damage_array is not None:
            save_mask(structure / "damage_mask.png", damage_array)
        self.boxes = {"arch_01": [10, 10, 40, 60], "arch_02": [70, 10, 100, 60]}
        elements = []
        for element_id, box in self.boxes.items():
            save_mask(run_dir / "element_masks" / f"{element_id}.png", rect(box))
            elements.append({"id": element_id, "class": "arch", "box": box, "score": 0.5,
                             "mask_file": f"element_masks/{element_id}.png", "source": "detector"})
        for element in extra or []:
            save_mask(run_dir / element["mask_file"], rect(element["box"]))
            elements.append(element)
        write_json(run_dir / "elements.json", {
            "structure_id": "synthetic", "image_size": [W, H],
            "classes": list(CONFIG["classes"]), "elements": elements,
        })
        logger = logging.getLogger(f"test_damage.{id(self)}")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        self.run = Run("synthetic", structure, run_dir, CONFIG, logger, {"steps": {}})

    def elements(self) -> dict[str, dict]:
        return {e["id"]: e for e in read_json(self.run.path("elements.json"))["elements"]}


class DamageStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_half_covered_box_is_the_only_fragmented_one(self) -> None:
        # The mask covers the left half of arch_01 and nothing of arch_02 (TASKS.md T13 acceptance).
        syn = SyntheticRun(self.tmp, rect([10, 10, 25, 60]))
        damage.run_step(syn.run, None)
        elements = syn.elements()
        self.assertEqual(elements["arch_01"]["damage_status"], "fragmented")
        self.assertAlmostEqual(elements["arch_01"]["damage_overlap"], 0.5, places=3)
        self.assertEqual(elements["arch_02"]["damage_status"], "intact")
        self.assertEqual(elements["arch_02"]["damage_overlap"], 0.0)

        regions = read_json(syn.run.path("damage_regions.json"))
        self.assertEqual(regions["summary"]["elements"], {"intact": 1, "fragmented": 1, "fragmented_by_cut": 0,
                                                       "rejected_skipped": 0})
        self.assertEqual(len(regions["regions"]), 1)
        region = regions["regions"][0]
        self.assertEqual(region["box"], [10, 10, 25, 60])
        self.assertEqual(region["elements"][0]["id"], "arch_01")
        self.assertEqual(region["uncovered_px"], 0)
        for name in ("damage_mask.png", "damage_overlay.png", region["mask_file"]):
            self.assertTrue(syn.run.path(name).is_file(), name)
        self.assertIn("damage", syn.run.meta["steps"])

    def test_threshold_boundary(self) -> None:
        # 1 column of 30 = 3.3 % overlap, below intact_max_overlap 0.05 -> intact. The strip runs along
        # the whole left edge (high contact) but the box overlap is also 3.3 %: bordering, not cut.
        syn = SyntheticRun(self.tmp, rect([10, 10, 11, 60]) | rect([0, 70, 30, 80]))
        damage.run_step(syn.run, None)
        self.assertEqual(syn.elements()["arch_01"]["damage_status"], "intact")
        regions = read_json(syn.run.path("damage_regions.json"))
        # the 50-pixel strip is below min_region_px (64): ignored as a region, still counted in overlap
        self.assertEqual(len(regions["regions"]), 1)
        self.assertEqual(regions["summary"]["ignored_small_regions"], {"count": 1, "pixels": 50})

    def test_visible_part_mask_cut_by_damage_is_fragmented(self) -> None:
        # arch_01's mask covers only its surviving top half (as SAM would); the damage takes the bottom
        # half of its box: mask overlap 0, but the element is cut.
        syn = SyntheticRun(self.tmp, rect([10, 35, 40, 60]))
        save_mask(syn.run.path("element_masks/arch_01.png"), rect([10, 10, 40, 35]))
        damage.run_step(syn.run, None)
        element = syn.elements()["arch_01"]
        self.assertEqual(element["damage_overlap"], 0.0)
        self.assertEqual((element["damage_status"], element["damage_reason"]), ("fragmented", "cut"))
        self.assertGreater(element["damage_box_overlap"], 0.4)

    def test_rejected_elements_get_no_status(self) -> None:
        extra = [{"id": "arch_03", "class": "arch", "box": [10, 10, 40, 60], "score": 0.1,
                  "mask_file": "element_masks/arch_03.png", "source": "detector", "rejected": True}]
        syn = SyntheticRun(self.tmp, rect([10, 10, 25, 60]), extra)
        damage.run_step(syn.run, None)
        elements = syn.elements()
        self.assertNotIn("damage_status", elements["arch_03"])
        self.assertEqual(len(elements), 3)  # nothing deleted

    def test_missing_mask_fails(self) -> None:
        syn = SyntheticRun(self.tmp, None)
        with self.assertRaisesRegex(ContractError, "mask_editor"):
            damage.run_step(syn.run, None)

    def test_nearly_empty_mask_fails(self) -> None:
        tiny = np.zeros((H, W), dtype=np.uint8)
        tiny[0, 0] = 1  # 1 / 9600 ~ 0.0001 < min_mask_fraction 0.001
        syn = SyntheticRun(self.tmp, tiny)
        with self.assertRaisesRegex(ContractError, "min_mask_fraction"):
            damage.run_step(syn.run, None)

    def test_mask_of_other_size_fails(self) -> None:
        syn = SyntheticRun(self.tmp, None)
        Image.new("L", (W + 1, H), 255).save(syn.run.damage_mask_path)
        with self.assertRaisesRegex(ContractError, "differs from the image size"):
            damage.run_step(syn.run, None)


class DamageFunctionTest(unittest.TestCase):
    def test_small_regions_are_ignored_and_regions_sorted_by_size(self) -> None:
        mask = rect([0, 0, 10, 10]) | rect([50, 50, 70, 70]) | rect([100, 0, 103, 3])
        records, masks, ignored = damage.split_regions(mask, min_region_px=64)
        self.assertEqual([r["area_px"] for r in records], [400, 100])
        self.assertEqual(ignored, {"count": 1, "pixels": 9})
        self.assertEqual(int(masks[0].sum()), 400)

    def test_uncovered_pixels(self) -> None:
        mask = rect([0, 0, 20, 10])
        records, masks, _ = damage.split_regions(mask, 1)
        elements = [{"id": "a", "class": "arch", "box": [0, 0, 10, 10]}]
        damage.region_contents(records, masks, elements, {"a": rect([0, 0, 10, 10])})
        self.assertEqual(records[0]["uncovered_px"], 100)
        self.assertEqual(records[0]["elements"][0]["share_of_region"], 0.5)

    def test_expert_agreement(self) -> None:
        elements = [
            {"id": "a", "expert_condition": "intact", "damage_status": "intact"},
            {"id": "b", "expert_condition": "fragmented", "damage_status": "intact"},
            {"id": "c", "damage_status": "intact"},
        ]
        result = damage.expert_agreement(elements)
        self.assertEqual((result["compared"], result["agree"]), (2, 1))
        self.assertEqual(result["disagreements"][0]["id"], "b")


class MaskEditorTest(unittest.TestCase):
    def test_imports_no_torch(self) -> None:
        code = ("import sys; sys.path.insert(0, 'tools'); import mask_editor; "
                "bad = [m for m in ('torch', 'diffusers', 'transformers', 'i2r') if m in sys.modules]; "
                "print(bad); sys.exit(1 if bad else 0)")
        result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_proposal_finds_dark_interior_and_writes_files(self) -> None:
        image = np.full((200, 300, 3), 170, dtype=np.uint8)
        image[80:120, 130:170] = 20  # dark square away from the border
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            Image.fromarray(image).save(tmp / "photo.png")
            result = subprocess.run(
                [sys.executable, "tools/mask_editor.py", "--image", str(tmp / "photo.png"),
                 "--out", str(tmp / "mask.png"), "--propose", "--no-gui"],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            proposal = np.asarray(Image.open(tmp / "mask_proposal.png"))
            self.assertTrue(set(np.unique(proposal)) <= {0, 255})
            self.assertTrue(proposal[100, 150] == 255)
            self.assertTrue(proposal[10, 10] == 0)
            self.assertTrue((tmp / "mask_proposal_preview.png").is_file())
            self.assertTrue((tmp / "mask_proposal.json").is_file())
            self.assertFalse((tmp / "mask.png").exists())


if __name__ == "__main__":
    unittest.main()
