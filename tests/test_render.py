"""Tests for i2r.render: marking stays inside the mask, every changed pixel has a source, provenance keys.

Runs complete.run_step and render.run_step on a synthetic run folder (no GPU)."""
from __future__ import annotations

import copy
import logging
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml
from PIL import Image

from i2r import complete, render
from i2r.runio import REPO_ROOT, ContractError, Run, load_config, read_json, write_json
from tests.test_complete import DONOR_BOX, TARGET_BOX, scene

PROVENANCE_KEYS = {"target_id", "class", "tier", "target_box", "donor_id", "donor_box", "similarity",
                   "confidence", "alternatives", "method", "flipped", "views", "pixels_changed"}


REFERENCE_RUN = REPO_ROOT / "outputs" / "s01_test_church" / "20261112_141348"     # local, fill off
REFERENCE_STRUCTURE = REPO_ROOT / "data" / "structures" / "s01_test_church"
OPEN_AREA = (slice(75, 88), slice(5, 35))        # damage with no target: tier C, what the generative pass fills


def fake_generative_fill(image: np.ndarray, area: np.ndarray, config: dict, log) -> tuple[np.ndarray, dict]:
    """Stands in for complete.generative_fill (no torch): every pixel grey 200."""
    return np.full_like(image, 200), {"method": "generative", "seconds": 0.0, "peak_vram_gb": None,
                                      "attention": "none", "debug": {}}


def build_run(tmp: Path, include_tier_b: bool = False, generative: bool = False, display: str = "separate",
              open_area: bool = False) -> Run:
    tmp.mkdir(parents=True, exist_ok=True)
    rgb, damage, donor_mask, _ = scene()
    if open_area:
        damage[OPEN_AREA] = 1
    h, w = damage.shape
    structure, run_dir = tmp / "structure", tmp / "run"
    structure.mkdir()
    (run_dir / "element_masks").mkdir(parents=True)
    Image.fromarray(rgb).save(structure / "image.png")
    Image.fromarray(damage * 255).save(run_dir / "damage_mask.png")
    Image.fromarray(donor_mask * 255).save(run_dir / "element_masks" / "donor.png")
    target_mask = np.zeros_like(damage)
    x1, y1, x2, y2 = TARGET_BOX
    target_mask[y1:y2, x1:x2] = 1
    Image.fromarray(target_mask * 255).save(run_dir / "element_masks" / "target.png")
    cfg = copy.deepcopy(load_config())
    cfg["restoration"]["include_tier_b"] = include_tier_b
    cfg["first_construction"]["generative"] = generative
    cfg["first_construction"]["generative_display"] = display
    write_json(run_dir / "elements.json", {
        "structure_id": "synthetic", "image_size": [w, h], "classes": list(cfg["classes"]),
        "elements": [
            {"id": "donor", "class": "pilaster", "box": DONOR_BOX, "score": 1.0, "mask_file": "element_masks/donor.png",
             "source": "expert", "damage_status": "intact"},
            {"id": "target", "class": "pilaster", "box": TARGET_BOX, "score": 1.0,
             "mask_file": "element_masks/target.png", "source": "expert", "damage_status": "fragmented"},
        ]})
    base = {"alternatives": [], "similarity": None, "reason": None, "rule": None, "element_id": None}
    write_json(run_dir / "evidence.json", {"targets": [
        {**base, "target_id": "t_001", "kind": "fragmented_element", "element_id": "target", "class": "pilaster",
         "tier": "A", "target_box": TARGET_BOX, "donor_id": "donor", "donor_box": DONOR_BOX, "flipped": False,
         "similarity": 0.9, "confidence": 0.9},
        {**base, "target_id": "t_002", "kind": "implied_slot", "rule": "B1", "class": "pilaster", "tier": "B",
         "target_box": [95, 40, 115, 80], "donor_id": "donor", "donor_box": DONOR_BOX, "flipped": False,
         "confidence": 0.5},
        {**base, "target_id": "t_003", "kind": "uncovered_region", "class": None, "tier": "C",
         "target_box": [95, 40, 115, 80], "donor_id": None, "donor_box": None, "flipped": None,
         "confidence": 0.0, "reason": "no element evidence for these damage pixels"},
    ]})
    # a second damaged area for the B slot
    damage[40:80, 95:115] = 1
    Image.fromarray(damage * 255).save(run_dir / "damage_mask.png")
    logger = logging.getLogger(f"test_render.{id(tmp)}")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return Run("synthetic", structure, run_dir, cfg, logger, {"steps": {}, "run_id": "synthetic/test"})


def legend_texts(rcfg: dict, counts: dict) -> list[str]:
    return [text for _, text in render.view_legend(rcfg, counts)]


class RenderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def render(self, include_tier_b: bool = False, views: tuple[str, ...] = ("restoration",)) -> Run:
        run = build_run(self.tmp, include_tier_b)
        opts = types.SimpleNamespace(views=list(views))
        complete.run_step(run, opts)
        render.run_step(run, opts)
        return run

    def test_files_and_provenance(self) -> None:
        run = self.render()
        for name in ("restoration.png", "restoration_source_map.png", "provenance.json", "targets_overlay.png",
                     "legend.png", "restoration_completed.png", "restoration_source.png", "completion.json"):
            self.assertTrue(run.path(name).is_file(), name)
        prov = read_json(run.path("provenance.json"))
        self.assertEqual(prov["statement"], "AI proposal, not a restoration decision")
        self.assertEqual([e["target_id"] for e in prov["targets"]], ["t_001", "t_002", "t_003"])
        for entry in prov["targets"]:
            self.assertTrue(PROVENANCE_KEYS <= set(entry), PROVENANCE_KEYS - set(entry))
        by_id = {e["target_id"]: e for e in prov["targets"]}
        self.assertEqual(by_id["t_001"]["views"], {"restoration": "completed"})
        self.assertEqual(by_id["t_001"]["method"], "donor_transfer")
        self.assertEqual(by_id["t_002"]["views"], {"restoration": "not_completed"})
        self.assertEqual(by_id["t_003"]["method"], "none")

    def test_report_page(self) -> None:
        run = self.render(views=("restoration", "first_construction"))
        page = run.path("report.html").read_text(encoding="utf-8")
        self.assertTrue(run.path("input.png").is_file())
        self.assertIn("AI proposal, not a restoration decision", page)
        for text in ("t_001", "t_002", 'src="restoration.png"', 'src="first_construction.png"', 'src="input.png"',
                     "targets/t_001_first_construction.png", "tools/target_review.py", "guess, needs expert"):
            self.assertIn(text, page, text)
        self.assertNotIn("http://", page)                           # self-contained: no external resource
        self.assertNotIn("https://", page)
        self.assertNotIn("<script", page)

    def test_every_changed_pixel_has_a_completed_source(self) -> None:
        run = self.render()
        photo = np.asarray(Image.open(run.image_path).convert("RGB"))
        completed = np.asarray(Image.open(run.path("restoration_completed.png")))
        labels = np.asarray(Image.open(run.path("restoration_source.png")))
        changed = np.any(completed != photo, axis=-1)
        self.assertTrue(changed.any())
        self.assertTrue(np.all(labels[changed] > 0))
        prov = read_json(run.path("provenance.json"))
        names = prov["views"]["restoration"]["labels"]
        status = {e["target_id"]: e["views"]["restoration"] for e in prov["targets"]}
        for value in np.unique(labels[labels > 0]):
            self.assertEqual(status[names[str(int(value))]], "completed")

    def test_marking_stays_inside_the_damage_mask(self) -> None:
        run = self.render()
        photo = np.asarray(Image.open(run.image_path).convert("RGB"))
        damage = np.asarray(Image.open(run.path("damage_mask.png"))) > 0
        marked = np.asarray(Image.open(run.path("restoration.png")).convert("RGB"))
        h = photo.shape[0]
        self.assertGreater(marked.shape[0], h)                       # label bar added below
        self.assertTrue(np.array_equal(marked[:h][~damage], photo[~damage]))
        self.assertFalse(np.array_equal(marked[:h][damage], photo[damage]))

    def test_open_tier_b_is_hatched_blue_and_included_b_is_not(self) -> None:
        run = self.render()
        marked = np.asarray(Image.open(run.path("restoration.png")).convert("RGB"))
        blue = tuple(run.config["restoration"]["hatch_color_b"])
        region = marked[40:80, 95:115].reshape(-1, 3)
        self.assertTrue(any(tuple(p) == blue for p in region))

    def test_both_views(self) -> None:
        run = self.render(views=("restoration", "first_construction"))
        for name in ("first_construction.png", "first_construction_source_map.png",
                     "first_construction_conjecture_overlay.png"):
            self.assertTrue(run.path(name).is_file(), name)
        prov = read_json(run.path("provenance.json"))
        by_id = {e["target_id"]: e for e in prov["targets"]}
        self.assertEqual(by_id["t_002"]["views"], {"restoration": "not_completed",
                                                   "first_construction": "guess_needs_expert"})
        self.assertGreater(prov["views"]["first_construction"]["markings_px"]["donor_guess_px"], 0)
        self.assertEqual(by_id["t_003"]["views"], {"restoration": "not_completed",
                                                   "first_construction": "not_completed"})
        self.assertEqual(prov["views"]["first_construction"]["markings_px"]["generative_px"], 0)
        # the marked images differ only inside the tier B target
        a = np.asarray(Image.open(run.path("restoration.png")).convert("RGB"))
        b = np.asarray(Image.open(run.path("first_construction.png")).convert("RGB"))
        h = np.asarray(Image.open(run.image_path)).shape[0]
        differ = np.any(a[:h] != b[:h], axis=-1)
        inside_b = np.zeros_like(differ)
        inside_b[40:80, 95:115] = True
        self.assertTrue(differ.any())
        self.assertFalse((differ & ~inside_b).any())

    def test_tier_b_switch_completes_and_records_it(self) -> None:
        run = self.render(include_tier_b=True)
        prov = read_json(run.path("provenance.json"))
        by_id = {e["target_id"]: e for e in prov["targets"]}
        self.assertEqual(by_id["t_002"]["views"]["restoration"], "completed")


class GenerativeMarkingTest(unittest.TestCase):
    """T26: generated pixels are an AI guess, never shown or legended as donor pixels."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def both_views(self, name: str, generative: bool, display: str = "separate") -> Run:
        run = build_run(self.tmp / name, generative=generative, display=display, open_area=True)
        opts = types.SimpleNamespace(views=["restoration", "first_construction"])
        with mock.patch.object(complete, "generative_fill", side_effect=fake_generative_fill) as fill:
            complete.run_step(run, opts)
        self.assertEqual(fill.call_count, 1 if generative else 0)
        render.run_step(run, opts)
        return run

    def test_mark_view_gives_generated_pixels_no_donor_outline_or_tone_shift(self) -> None:
        cfg = load_config()
        rcfg, gcfg = cfg["restoration"], cfg["first_construction"]
        completed = np.full((40, 60, 3), 100, np.uint8)
        labels = np.zeros((40, 60), np.uint16)
        labels[5:25, 5:25] = 1                                     # donor target
        labels[5:25, 30:50] = 2                                    # generative
        damage = (labels > 0).astype(np.uint8)                     # nothing left open
        marked, counts = render.mark_view(completed, damage, labels, [], {}, rcfg, generative_label=2, gcfg=gcfg)
        donor, guess = marked[labels == 1], marked[labels == 2]
        white = np.all(donor == rcfg["outline_color"], axis=-1)
        shifted = int(round(100 + 155 * rcfg["tone_shift"]))
        self.assertTrue(white.any())                               # donor: outlined ...
        self.assertTrue(np.all(donor[~white] == shifted))          # ... and tone-shifted
        self.assertFalse(np.any(np.all(guess == rcfg["outline_color"], axis=-1)))
        self.assertFalse(np.any(np.all(guess == shifted, axis=-1)))
        untouched = np.all(guess == 100, axis=-1)                  # between the stripes: the generated pixel itself
        self.assertTrue(untouched.any() and not untouched.all())
        self.assertEqual(counts["completed_px"], 400)
        self.assertEqual(counts["ai_guess_px"], 400)
        self.assertEqual(counts["ai_guess_marks_px"], int((~untouched).sum()))
        # legend: an AI-guess item; no "hatched" line because no pixel is hatched
        texts = legend_texts(rcfg, counts)
        self.assertIn(render.AI_GUESS_LEGEND, texts)
        self.assertFalse(any("hatched" in t for t in texts))

    def test_open_damage_outside_the_facade_has_its_own_hatch_and_legend(self) -> None:
        rcfg = load_config()["restoration"]
        completed = np.full((40, 60, 3), 100, np.uint8)
        labels = np.zeros((40, 60), np.uint16)
        damage = np.zeros((40, 60), np.uint8)
        damage[5:35, 5:55] = 1
        facade = np.ones((40, 60), np.uint8)
        facade[20:, :] = 0                                         # rows 20 and below: the ground
        marked, counts = render.mark_view(completed, damage, labels, [], {}, rcfg, facade=facade)
        self.assertEqual(counts["open_no_evidence_px"], 15 * 50)
        self.assertEqual(counts["open_outside_facade_px"], 15 * 50)
        red, grey = rcfg["hatch_color"], rcfg["hatch_color_outside"]
        top, bottom = marked[5:20, 5:55].reshape(-1, 3), marked[20:35, 5:55].reshape(-1, 3)
        self.assertTrue(np.any(np.all(top == red, axis=-1)) and not np.any(np.all(top == grey, axis=-1)))
        self.assertTrue(np.any(np.all(bottom == grey, axis=-1)) and not np.any(np.all(bottom == red, axis=-1)))
        # the hatch lines are where they were without a limit; only their colour differs outside it
        plain, _ = render.mark_view(completed, damage, labels, [], {}, rcfg)
        self.assertTrue(np.array_equal(np.any(marked != completed, axis=-1), np.any(plain != completed, axis=-1)))
        texts = legend_texts(rcfg, counts)
        self.assertIn(render.OUTSIDE_FACADE_LEGEND, texts)
        self.assertIn("hatched: no evidence", texts)
        self.assertNotIn(render.OUTSIDE_FACADE_LEGEND, legend_texts(rcfg, {**counts, "open_outside_facade_px": 0}))

    def test_a_donor_placed_as_a_guess_is_striped_not_outlined(self) -> None:
        cfg = load_config()
        rcfg, gcfg = cfg["restoration"], cfg["first_construction"]
        completed = np.full((40, 60, 3), 100, np.uint8)
        labels = np.zeros((40, 60), np.uint16)
        labels[5:25, 5:25] = 1                                     # accepted donor
        labels[5:25, 30:50] = 2                                    # donor placed as a guess
        damage = (labels > 0).astype(np.uint8)
        marked, counts = render.mark_view(completed, damage, labels, [], {}, rcfg, gcfg=gcfg,
                                          donor_guess={"labels": {2}, "withheld": set()})
        guess = marked[labels == 2]
        shifted = int(round(100 + 155 * rcfg["tone_shift"]))
        self.assertFalse(np.any(np.all(guess == rcfg["outline_color"], axis=-1)))
        self.assertFalse(np.any(np.all(guess == shifted, axis=-1)))
        self.assertTrue(np.any(np.all(guess == rcfg["guess_color"], axis=-1)))         # dashed outline
        self.assertTrue(np.any(np.all(marked[labels == 1] == rcfg["outline_color"], axis=-1)))
        self.assertEqual((counts["completed_px"], counts["donor_guess_px"]), (400, 400))
        self.assertIn(render.DONOR_GUESS_LEGEND, legend_texts(rcfg, counts))

    def test_a_withheld_guess_is_hatched_in_the_guess_colour(self) -> None:
        rcfg = load_config()["restoration"]
        completed = np.full((40, 60, 3), 100, np.uint8)
        labels = np.zeros((40, 60), np.uint16)
        damage = np.zeros((40, 60), np.uint8)
        damage[5:35, 5:55] = 1
        records = [{"target_id": "t_001", "tier": "A", "kind": "fragmented_element", "target_box": [5, 5, 30, 35]}]
        marked, counts = render.mark_view(completed, damage, labels, records, {"t_001": "guess_needs_expert"}, rcfg,
                                          donor_guess={"labels": set(), "withheld": {"t_001"}})
        self.assertEqual(counts["open_guess_withheld_px"], 25 * 30)
        self.assertEqual(counts["open_no_evidence_px"], 25 * 30)
        inside, outside = marked[5:35, 5:30].reshape(-1, 3), marked[5:35, 30:55].reshape(-1, 3)
        self.assertTrue(np.any(np.all(inside == rcfg["guess_color"], axis=-1)))
        self.assertFalse(np.any(np.all(outside == rcfg["guess_color"], axis=-1)))
        self.assertIn(render.WITHHELD_GUESS_LEGEND, legend_texts(rcfg, counts))

    def test_a_rejected_placement_is_cross_hatched_in_its_own_colour(self) -> None:
        rcfg = load_config()["restoration"]
        completed = np.full((40, 60, 3), 100, np.uint8)
        labels = np.zeros((40, 60), np.uint16)
        damage = np.zeros((40, 60), np.uint8)
        damage[5:35, 5:55] = 1
        records = [{"target_id": "t_001", "tier": "A", "kind": "fragmented_element", "target_box": [5, 5, 30, 35]}]
        marked, counts = render.mark_view(completed, damage, labels, records, {"t_001": "rejected_by_expert"}, rcfg,
                                          donor_guess={"labels": set(), "withheld": set(), "rejected": {"t_001"}})
        self.assertEqual(counts["open_rejected_px"], 25 * 30)
        self.assertEqual(counts["open_no_evidence_px"], 25 * 30)          # the rest of the damage, not the box
        inside, outside = marked[5:35, 5:30].reshape(-1, 3), marked[5:35, 30:55].reshape(-1, 3)
        self.assertTrue(np.any(np.all(inside == rcfg["rejected_color"], axis=-1)))
        self.assertFalse(np.any(np.all(inside == rcfg["hatch_color"], axis=-1)))
        self.assertFalse(np.any(np.all(outside == rcfg["rejected_color"], axis=-1)))
        self.assertIn(render.REJECTED_LEGEND, legend_texts(rcfg, counts))
        self.assertNotIn(render.REJECTED_LEGEND, legend_texts(rcfg, {**counts, "open_rejected_px": 0}))

    def test_legend_lists_only_what_occurs(self) -> None:
        rcfg = load_config()["restoration"]
        zero = {"completed_px": 0, "ai_guess_px": 0, "open_tier_b_px": 0, "open_no_evidence_px": 0}
        self.assertEqual(legend_texts(rcfg, zero), [])
        self.assertEqual(legend_texts(rcfg, {**zero, "open_no_evidence_px": 3}), ["hatched: no evidence"])
        self.assertEqual(legend_texts(rcfg, {**zero, "completed_px": 3}), ["completed from a donor (outlined)"])
        render.bar(200, rcfg, "restoration", [])                    # an empty legend is not replaced by the default

    def test_separate_keeps_the_view_identical_to_the_run_without_the_fill(self) -> None:
        off = self.both_views("off", generative=False)
        on = self.both_views("on", generative=True, display="separate")
        for name in ("first_construction.png", "first_construction_source_map.png", "restoration.png"):
            self.assertEqual(on.path(name).read_bytes(), off.path(name).read_bytes(), name)
        self.assertTrue(on.path("first_construction_ai_guess.png").is_file())
        self.assertFalse(off.path("first_construction_ai_guess.png").exists())
        view = read_json(on.path("provenance.json"))["views"]["first_construction"]
        self.assertEqual(view["generative_display"], "separate")
        self.assertEqual(view["ai_guess_image"], "first_construction_ai_guess.png")
        self.assertEqual(view["markings_px"]["ai_guess_px"], 0)
        self.assertGreater(view["markings_px"]["open_no_evidence_px"], 0)
        self.assertGreater(view["ai_guess_markings_px"]["ai_guess_px"], 0)
        self.assertEqual(view["ai_guess_markings_px"]["open_no_evidence_px"], 0)
        # the guess image: banner above, then the photograph, unchanged outside the damage mask
        photo = np.asarray(Image.open(on.image_path).convert("RGB"))
        damage = np.asarray(Image.open(on.path("damage_mask.png"))) > 0
        guess = np.asarray(Image.open(on.path("first_construction_ai_guess.png")).convert("RGB"))
        top = render.banner(photo.shape[1], render.AI_GUESS_BANNER, 1.0).shape[0]
        body = guess[top:top + photo.shape[0]]
        self.assertTrue(np.all(guess[0, 0] == render.CONJECTURE_COLOUR))
        self.assertTrue(np.array_equal(body[~damage], photo[~damage]))
        # run_meta says whether the pass ran
        self.assertEqual(on.meta["steps"]["complete"]["generative_fill"]["ran"], True)
        self.assertEqual(off.meta["steps"]["complete"]["generative_fill"],
                         {"enabled": False, "ran": False, "pixels_changed": {}})

    def test_inline_marks_the_guess_in_the_view_itself(self) -> None:
        run = self.both_views("inline", generative=True, display="inline")
        self.assertFalse(run.path("first_construction_ai_guess.png").exists())
        view = read_json(run.path("provenance.json"))["views"]["first_construction"]
        counts = view["markings_px"]
        self.assertGreater(counts["ai_guess_px"], 0)
        self.assertEqual(counts["open_no_evidence_px"], 0)
        texts = legend_texts(run.config["restoration"], counts)
        self.assertIn(render.AI_GUESS_LEGEND, texts)
        self.assertNotIn("hatched: no evidence", texts)
        # in the open area the generated grey (200) shows between the stripes, never the donor outline or tone
        marked = np.asarray(Image.open(run.path("first_construction.png")).convert("RGB"))[OPEN_AREA]
        flat = marked.reshape(-1, 3)
        self.assertTrue(np.any(np.all(flat == 200, axis=-1)))
        self.assertFalse(np.any(np.all(flat == 255, axis=-1)))
        self.assertFalse(np.any(np.all(flat == int(round(200 + 55 * run.config["restoration"]["tone_shift"])), axis=-1)))

    def test_unknown_display_is_refused(self) -> None:
        run = build_run(self.tmp / "bad", display="panels")
        opts = types.SimpleNamespace(views=["restoration"])
        complete.run_step(run, opts)
        with self.assertRaises(ContractError):
            render.run_step(run, opts)

    @unittest.skipUnless(REFERENCE_RUN.is_dir() and (REFERENCE_STRUCTURE / "image.png").is_file(),
                         "local s01 reference run not present")
    def test_reference_run_views_are_byte_identical(self) -> None:
        """Re-rendering a copy of the s01 reference run (fill off) reproduces both marked views."""
        copy_dir = self.tmp / "reference"
        shutil.copytree(REFERENCE_RUN, copy_dir)
        with (copy_dir / "config.yaml").open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cfg["first_construction"] = {**load_config()["first_construction"], **cfg["first_construction"]}
        logger = logging.getLogger("test_render.reference")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        run = Run("s01_test_church", REFERENCE_STRUCTURE, copy_dir, cfg, logger, {"steps": {}, "run_id": "copy"})
        render.run_step(run, types.SimpleNamespace(views=["restoration", "first_construction"]))
        for name in ("restoration.png", "first_construction.png"):
            self.assertEqual(run.path(name).read_bytes(), (REFERENCE_RUN / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
