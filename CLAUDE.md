# ImageToRender — instructions for Claude

Research/portfolio repo. Input: a photograph of a damaged historical building. Output: a Venice-Charter
"restoration view" and a "first-construction view" with a provenance record. Use: early-stage decision
support, never a restoration decision. The owner is an architect and writes in Turkish.

## Read first
- `docs/TASKS.md`: ordered task list; continue at the first task without a "Status: done" note (next: T18).
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
- Work on branch `refactor/structure` (later `feat/pipeline`). Never commit to `main`. Commit and push only when asked or when the task list says so.
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
- Detect (OWL-ViT + SAM), review, editor adapter, mask editor, damage, evidence (tiers A, B, C) donor transfer and rendering of the restoration view are done (s01: 6 A, 2 B, 13 C; 6 of 6 A completed; T14 to T17 notes). The first-construction view is T18. s01 damage labels agree with the expert on 21 of 23 elements after the mask was extended (T13 note; mask version 2). s01 recall against the expert inventory: 52 % class-aware (RND-001, n = 1).
- Use the imported 23-element inventory of s01 with SAM masks (T16b note in `docs/TASKS.md`; run `outputs/s01_test_church/20261110_181914` or later, damage mask version 3), not raw detections, to develop T18 onwards.
- Open author items: building identity/source/licence and larger copies of both photos; merge into `main`; deleting local branch `backup-before-rewrite`; s02 damage mask. Decision 2026-11-10: work on s01 only until the pipeline is complete; s02 gets its own later session.
