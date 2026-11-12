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
`run.py` entry point; `config.yaml`; `i2r/` modules (`runio`, `detect`, `review`, `damage`, `evidence`, `complete`, `render`); `tools/` (`check_env`, `inventory_editor`, `editor_adapter`, `mask_editor`, `refine_masks`); `evaluation/evaluate.py`; `tests/`; `data/structures/<id>/`; `docs/`; `archive/`; `outputs/` and `assets/` (local, untracked).

## Current state
- Done: T1 to T18 plus T16b (detect OWL-ViT + SAM, review, editor adapter, mask editor, damage, evidence tiers A/B/C, donor transfer, restoration view, first-construction view, `tools/refine_masks.py`). All six `run.py` steps are real. 101 tests pass.
- s01 (n = 1): damage mask version 3 (provisional, see `data/structures/s01_test_church/SOURCE.md`); damage labels agree with the expert on 23 of 23 elements after two mask corrections made where they disagreed (19 of 23 with the author's first mask); 19 targets (5 A, 2 B, 12 C). Restoration view completes 5 targets (4,971 px); first-construction view 7 targets (11,367 px); the two views differ only inside the B boxes. Recall of raw detections against the expert inventory: 52 % class-aware (RND-001).
- Reference runs: `outputs/s01_test_church/20261110_181914` (SAM-refined inventory, mask v3) and `outputs/s01_test_church/20261110_182638` (both views, latest). Develop from these, not from raw detections (oversized boxes).
- Re-run: `python run.py --structure s01_test_church --from damage --view both --no-review`.

## Where we stopped (session ended 2026-11-10)
- Last commit `c669c5c feat(T18)` on `refactor/structure`, pushed; working tree clean.
- Next: **T19** synthetic damage and evaluation: `tools/synthetic_damage.py` (one case folder per intact element per variant partial/full/background, ARCHITECTURE §7.1; modified inventory, mask, `ground_truth.json`), `evaluation/evaluate.py completion` (runs the pipeline from `damage` with `--no-review`, metrics, CSV + Markdown table), `tests/test_synthetic_damage.py`. Accept: at least one case per intact element per variant; preservation 0 in every row. Keep the number of cases small (credit cost).
- Then: T20 calibrate thresholds (RND-002), T21 generative tier C (optional, cap the attempts), T22-T23 s02 (deferred, needs the author's s02 mask), T24 README/examples, T25 clean clone, merge, tag.
- Known weaknesses for T20: alignment scale hits the search bounds; low NCC on pilasters; the size filter blocks `recovered_pilaster_046` and `recovered_pilaster_020` (left as C); `min_similarity` 0.60 does not discriminate (scores 0.77-0.94); the two B1 slots (x 343-366, 400-423) are doubtful.

## Open author items
- Building identity/source/licence and larger copies of both photos; merge into `main`; deleting local branch `backup-before-rewrite`; s02 damage mask; redraw the s01 mask edge near `pilaster_010`/`pilaster_011` (mask v3 is provisional); confirm opening → window.
- Decision 2026-11-10: work on s01 only until the pipeline is complete; s02 gets its own later session.
