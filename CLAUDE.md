# ImageToRender — instructions for Claude

Research/portfolio repo. Input: a photograph of a damaged historical building. Output: a Venice-Charter
"restoration view" and a "first-construction view" with a provenance record. Use: early-stage decision
support, never a restoration decision. The owner is an architect and writes in Turkish.

## Read first
- `docs/TASKS.md`: ordered task list; its "Before you start" section names the next task (T19).
- `docs/ARCHITECTURE.md`: design (6 modules detect, review, damage, evidence, complete, render; evidence tiers A, B, C).
- `docs/EXPERIMENTS.md`: what the old experiments were. `archive/README.md`: which run each archived script produced.

## Environment (fixed)
- Windows, RTX 2060 6 GB, conda env `ImageToRender_GPU`, Python 3.10.
- torch 2.4.0+cu121, xformers 0.0.27.post2, diffusers 0.20.2, transformers 4.33.2, huggingface-hub 0.16.4,
  accelerate 0.22.0, numpy 1.26.4, opencv-python 4.8.1.78. Packages are installed with `--no-deps`.
- Never install, upgrade or remove a package. If a task seems to need one, stop and ask, naming its compatibility with these versions.
- Keep `setup/` as is.
- Tests: `python -m unittest discover tests` (GPU-free). Run them before every commit.

## Rules
- Work on branch `refactor/structure` (the planned `feat/pipeline` branch was never created). Never commit to `main`. Commit and push only when asked or when the task list says so.
- Commit message style: `feat(T12): ...` / `fix(T12b): ...`; end with the attribution line given by the session.
- `archive/` is never edited and never imported; copy the function you need into `i2r/` or `tools/`.
- No bare `except`. No hard-coded thresholds in modules; they go in `config.yaml`. A building-specific setting goes in `data/structures/<id>/structure.yaml` with the reason in a comment.
- `tools/inventory_editor.py` is not modified (its SHA-256 matches the script that produced the locked inventory). Use `tools/editor_adapter.py`.
- Each step reads and writes one run folder `outputs/<structure>/<timestamp>/`; steps never search for "the newest run" (only `run.py --from` does). File formats are checked with `ContractError`/`require_keys`.
- Nothing is deleted from `elements.json`; rejected elements stay with `rejected: true`.

## Honesty and documents
- Report results as they are: n = 1 structure stays "n = 1". Mark open research questions `RND-xxx` / `R#`. Do not inflate status.
- Docs are in English with a 3-4 line Turkish summary at the top. Talk to the owner in Turkish.
- Test photographs (`s01_test_church`, `s02_test_arch`) have unknown source and licence: images stay out of git (`data/README.md`).
- Decide on sensible defaults and say what you chose. Ask only for what is the author's to decide (building identity, licence, merging into main, damage mask for s02).

## Layout
`run.py` entry point; `config.yaml`; `i2r/` modules (`runio`, `detect`, `review`, `damage`, `evidence`, `complete`, `render`); `tools/` (`check_env`, `inventory_editor`, `editor_adapter`, `mask_editor`, `refine_masks`, `synthetic_damage`); `evaluation/evaluate.py`; `tests/`; `data/structures/<id>/`; `docs/`; `archive/`; `outputs/` and `assets/` (local, untracked).

## Current state
- Done: T1 to T21 plus T16b (detect OWL-ViT + SAM, review, editor adapter, mask editor, damage, evidence tiers A/B/C, donor transfer, restoration view, first-construction view, `tools/refine_masks.py`, synthetic damage `tools/synthetic_damage.py` + `evaluation/evaluate.py completion`). All six `run.py` steps are real. 130 tests pass.
- s01 (n = 1): damage mask version 3 (provisional, see `data/structures/s01_test_church/SOURCE.md`); damage labels agree with the expert on 23 of 23 elements after two mask corrections made where they disagreed (19 of 23 with the author's first mask); 19 targets (5 A, 2 B, 12 C). Restoration view completes 5 targets (4,971 px); first-construction view 7 targets (11,367 px); the two views differ only inside the B boxes. Recall of raw detections against the expert inventory: 52 % class-aware (RND-001).
- Reference runs: `outputs/s01_test_church/20261110_181914` (SAM-refined inventory, mask v3) and `outputs/s01_test_church/20261110_182638` (both views, latest). Develop from these, not from raw detections (oversized boxes).
- Re-run: `python run.py --structure s01_test_church --from damage --view both --no-review`.

## Where we stopped (2026-11-12)
- T19 done (see its status note in `docs/TASKS.md`): s01 synthetic test, 28 cases, preservation 0; partial 9/12 tier A, full 1/12 (B rules rarely fire), background abstention 4/4 in the restoration view, one B1 false positive in the first-construction view. Evaluation: `outputs/synthetic/s01_test_church/20261112_122615/eval_20261112_122636/completion.md`.
- Re-run: `python tools/synthetic_damage.py --run <run>` then `python evaluation/evaluate.py completion --cases <cases folder>` (about 3 minutes for 28 cases).
- T20 done (2026-11-12, `docs/rnd/RND-002-evidence-baseline.md`): `min_similarity` 0.80, `size_tolerance` 0.35, `row_tolerance` 0.15; s01 result unchanged; 4 of the 10 Stage 3C-C hypotheses found. Grid: `python evaluation/evaluate.py grid --cases <cases> --grid key=v1,v2 ...` (about 1 minute per point).
- T21 done (2026-11-12, `docs/rnd/RND-003-generative-fill.md`): one generation on s01, technically correct (3.3 GB, restoration byte-identical, only the open area changed) but not usable (invented frieze and portal, melted bases, seams); `first_construction.generative` stays false. Models come from `models.cache_dir` (`F:/huggingface/hub`), never downloaded during a run.
- T24 done (2026-11-12): README rewritten (English, Turkish summary, n = 1 results, limitations); no example images because the test photographs have no known licence.
- Next: T25 clean clone check (the photograph is not in git: copy `data/structures/s01_test_church/` into the clone by hand), then merge into `main` and tag `v0.1.0`, which the author decides. T22-T23 skipped (author decision 2026-11-12).
- Known weaknesses (after T20): alignment scale hits the search bounds; low NCC on pilasters; the size filter blocks `recovered_pilaster_046`, `recovered_pilaster_020` and, in the synthetic test, `recovered_arch_025` and `expert_cornice_021`; `min_similarity` cannot be tested without negatives (R-002a); the two B1 slots (x 343-366, 400-423) are doubtful and B1 made a slot on plain wall in `background__bg01`.
- Scale test (2026-11-12): `s01_test_church_upscaled` (Upscayl 3x, AI upscaler, not a new photo; see its `SOURCE.md`) showed that pixel settings did not follow image size; fixed the same day by `scale` in `config.yaml` (`runio.apply_pixel_scale`, reference 904 px; s01 outputs byte-identical). After the fix the copy gives the same targets, tiers and markings as s01; only 2 of 5 donors differ (upscaler texture). Not usable for calibration.

## Open author items
- Building identity/source/licence and larger copies of both photos (the author has found more photos of s01, not added yet: R5 in `docs/TASKS.md`); merge into `main`; deleting local branch `backup-before-rewrite`; redraw the s01 mask edge near `pilaster_010`/`pilaster_011` (mask v3 is provisional); confirm opening → window.
- Decision 2026-11-10: work on s01 only until the pipeline is complete; s02 gets its own later session.
  **Changed 2026-11-12** (after the roadmap check): T19 on s01, then a second structure under T22 rules
  (no code or config change, every correction logged) **before** T20, so thresholds are not tuned on one
  building. Upscaling a small photograph does not count: it adds no evidence.
  **Author decision 2026-11-12 (final): no other photograph.** s02 is dropped and no new or alternative
  photo is used; work continues on s01 only. T22-T23 are skipped for this stage. T20 calibrates on the
  original s01 image; `s01_test_church_upscaled` is a scale check only (its texture is invented). Every
  result stays n = 1 building, tuned and reported on the same building.
