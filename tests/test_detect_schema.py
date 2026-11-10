"""Tests for the elements.json contract and the GPU-free parts of i2r.detect."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from i2r import detect
from i2r.runio import ContractError, load_config


def example_document() -> dict:
    return {
        "structure_id": "demo",
        "image_size": [200, 100],
        "classes": ["arch", "window"],
        "elements": [
            {"id": "arch_01", "class": "arch", "box": [10, 10, 50, 60], "score": 0.12,
             "mask_file": "element_masks/arch_01.png", "source": "detector"},
            {"id": "window_01", "class": "window", "box": [120, 20, 160, 90], "score": 0.05,
             "mask_file": "element_masks/window_01.png", "source": "expert"},
        ],
    }


class ElementsContractTest(unittest.TestCase):
    def test_valid_document_passes(self) -> None:
        detect.validate_elements(example_document())

    def assert_invalid(self, mutate, fragment: str) -> None:
        doc = copy.deepcopy(example_document())
        mutate(doc)
        with self.assertRaises(ContractError) as ctx:
            detect.validate_elements(doc)
        self.assertIn(fragment, str(ctx.exception))

    def test_missing_element_key(self) -> None:
        self.assert_invalid(lambda d: d["elements"][0].pop("mask_file"), "mask_file")

    def test_missing_document_key(self) -> None:
        self.assert_invalid(lambda d: d.pop("image_size"), "image_size")

    def test_unknown_class(self) -> None:
        self.assert_invalid(lambda d: d["elements"][0].update({"class": "gargoyle"}), "gargoyle")

    def test_duplicate_id(self) -> None:
        self.assert_invalid(lambda d: d["elements"][1].update({"id": "arch_01"}), "duplicate")

    def test_box_outside_image(self) -> None:
        self.assert_invalid(lambda d: d["elements"][0].update({"box": [10, 10, 250, 60]}), "box")

    def test_inverted_box(self) -> None:
        self.assert_invalid(lambda d: d["elements"][0].update({"box": [50, 10, 10, 60]}), "box")

    def test_bad_source(self) -> None:
        self.assert_invalid(lambda d: d["elements"][0].update({"source": "guess"}), "source")


class HelpersTest(unittest.TestCase):
    def test_prompt_specs_cover_every_configured_class(self) -> None:
        cfg = load_config()
        specs = detect.prompt_specs(cfg["classes"])
        self.assertEqual({s.element_class for s in specs}, set(cfg["classes"]))

    def test_clamp_box_keeps_box_inside_and_non_empty(self) -> None:
        self.assertEqual(detect.clamp_box([-5, 3.4, 300, 3.6], 200, 100), [0, 3, 200, 4])

    def test_views_cover_image_and_include_damage_context(self) -> None:
        cfg = load_config()["detect"]
        damage = np.zeros((904, 628), np.uint8)
        damage[300:500, 250:400] = 255
        views = detect.build_views(628, 904, damage, cfg)
        ids = [v.view_id for v in views]
        self.assertEqual(ids[0], "full")
        self.assertIn("damage_context", ids)
        self.assertLessEqual(len(views), cfg["max_detection_views"])

    def test_views_without_damage_mask(self) -> None:
        cfg = load_config()["detect"]
        ids = [v.view_id for v in detect.build_views(628, 904, None, cfg)]
        self.assertNotIn("damage_context", ids)

    def test_dynamic_threshold_and_fallback(self) -> None:
        cfg = dict(load_config()["detect"])
        specs = [detect.PromptSpec("arch", "arch"), detect.PromptSpec("window", "window")]
        view = detect.View("full", (0, 0, 100, 100), "full_image")
        # prompt 0: best 0.20, ratio 0.58 -> threshold 0.116; keeps 0.20 and 0.15, drops 0.05
        # prompt 1: best 0.005 < floor 0.008 but >= fallback 0.0035 -> one fallback box
        scores = [0.20, 0.15, 0.05, 0.005]
        labels = [0, 0, 0, 1]
        boxes = [[10, 10, 40, 40], [50, 10, 80, 40], [10, 50, 40, 80], [50, 50, 80, 80]]
        selected, _ = detect.select_candidates(scores, labels, boxes, view, specs, cfg)
        arch = [s for s in selected if s["class"] == "arch"]
        window = [s for s in selected if s["class"] == "window"]
        self.assertEqual(sorted(s["score"] for s in arch), [0.15, 0.20])
        self.assertEqual(len(window), 1)
        self.assertEqual(window[0]["quality"], "fallback")

    def test_class_aware_nms_suppresses_only_within_a_class(self) -> None:
        dets = [
            {"class": "arch", "score": 0.9, "box": [0, 0, 10, 10]},
            {"class": "arch", "score": 0.8, "box": [1, 1, 11, 11]},
            {"class": "window", "score": 0.7, "box": [1, 1, 11, 11]},
        ]
        kept = detect.class_aware_nms(dets, 0.42, 10, 10)
        self.assertEqual([(d["class"], d["score"]) for d in kept], [("arch", 0.9), ("window", 0.7)])


class CleanupTest(unittest.TestCase):
    """T12b: class merge and oversized-box removal."""

    def test_class_merge_relabels_and_keeps_detector_label(self) -> None:
        dets = [{"class": "column", "score": 0.2, "box": [0, 0, 10, 10]},
                {"class": "arch", "score": 0.1, "box": [0, 0, 10, 10]}]
        changed = detect.apply_class_merge(dets, {"column": "pilaster"})
        self.assertEqual(changed, 1)
        self.assertEqual([(d["class"], d["detected_class"]) for d in dets],
                         [("pilaster", "column"), ("arch", "arch")])

    def test_merge_before_nms_removes_the_duplicate_box(self) -> None:
        dets = [{"class": "column", "score": 0.2, "box": [0, 0, 10, 40]},
                {"class": "pilaster", "score": 0.1, "box": [1, 0, 11, 40]}]
        detect.apply_class_merge(dets, {"column": "pilaster"})
        kept = detect.class_aware_nms(dets, 0.42, 10, 10)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["detected_class"], "column")

    def test_class_merge_must_use_known_classes_and_no_chains(self) -> None:
        classes = ["column", "pilaster", "arch"]
        detect.check_class_merge({"column": "pilaster"}, classes)
        with self.assertRaises(ContractError):
            detect.check_class_merge({"column": "gargoyle"}, classes)
        with self.assertRaises(ContractError):
            detect.check_class_merge({"column": "pilaster", "pilaster": "arch"}, classes)

    def test_oversized_boxes_are_dropped_with_per_class_override(self) -> None:
        config = {"detect": {"max_element_area_ratio": 0.15},
                  "classes": {"arch": {"prompts": ["arch"]},
                              "roof": {"prompts": ["roof"], "max_area_ratio": 0.5}}}
        dets = [{"class": "arch", "score": 0.3, "box": [0, 0, 50, 50]},    # 25 % of 100x100: dropped
                {"class": "arch", "score": 0.2, "box": [0, 0, 30, 30]},    # 9 %: kept
                {"class": "roof", "score": 0.1, "box": [0, 0, 50, 50]}]    # 25 % but roof allows 50 %
        kept, dropped = detect.split_oversized(dets, 100, 100, config)
        self.assertEqual([(d["class"], d["score"]) for d in kept], [("arch", 0.2), ("roof", 0.1)])
        self.assertEqual(len(dropped), 1)

    def test_main_config_has_no_class_merge(self) -> None:
        self.assertEqual(load_config()["detect"]["class_merge"], {})


if __name__ == "__main__":
    unittest.main()
