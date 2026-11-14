# ImageToRender — instructions for Claude

Research/portfolio repo. Input: a photograph of a damaged historical building. Output: a Venice-Charter
"restoration view" and a "first-construction view" with a provenance record. Use: early-stage decision
support, never a restoration decision. The owner is an architect and writes in Turkish.

## Read first
- `docs/TASKS.md`: ordered task list; its "Before you start" section names the next task (none open in Phase 3; see `docs/STATUS.md`).
- `docs/STATUS.md`: state after Phase 3 (2026-11-14): problems with causes, and the author's manual steps in order.
- `docs/FAILURE_ANALYSIS.md`: the author's visual review of the s01 output and the causes found (2026-11-12); source of T26 to T30.
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
- `main` is tagged `v0.1.0` (2026-11-12). New work goes on a new branch from `main` (`refactor/structure` held everything up to v0.1.0). Never commit to `main`; it is updated by merging, which the author decides. Commit and push only when asked or when the task list says so.
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
`run.py` entry point; `config.yaml`; `i2r/` modules (`runio`, `detect`, `review`, `damage`, `evidence`, `complete`, `render`, `report`); `tools/` (`check_env`, `inventory_editor`, `editor_adapter`, `mask_editor`, `refine_masks`, `synthetic_damage`, `target_review`); `evaluation/evaluate.py`; `tests/`; `data/structures/<id>/`; `docs/`; `archive/`; `outputs/` and `assets/` (local, untracked).

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
- T25 done (2026-11-12): fresh clone reproduces the reference run byte for byte; `main` fast-forwarded and tagged `v0.1.0` (author decision). Local branch `backup-before-rewrite` deleted; its 3 commits are kept as the local tag `backup-before-rewrite` (not pushed). Further work goes on a new branch from `main`.
- Failure analysis (2026-11-12, `docs/FAILURE_ANALYSIS.md`): the author reviewed `outputs/s01_test_church/20261112_142811/first_construction.png`, which is the generative-fill trial and also the newest s01 run folder (use `20261112_141348` as the reference, fill off). Main findings: generative pixels are outlined and legended as donor pixels (bug from T21); the damage mask reaches the ground; `recovered_arch_001` does not describe the standing arch; 4 of 5 tier A alignments end on a search bound and nothing rejects a poor one; B1 put a slot beside a surviving pilaster fragment.
- T26 done (2026-11-14, branch `fix/failure-analysis`, see its status note in `docs/TASKS.md`): generated pixels are no longer outlined or legended as donor pixels; `first_construction.generative_display` `separate` (default) keeps them out of `first_construction.png` and writes `first_construction_ai_guess.png`; legends list only what occurs; `run_meta.json` records whether the pass ran. 136 tests. Run folders written before T26 keep the old images.
- T27 (2026-11-14, not committed unless the log shows it; status note in `docs/TASKS.md`): facade limit = convex hull of the element boxes, or the author's `data/structures/<id>/facade_mask.png` when present (`evidence.facade_limit`, run file `facade_mask.png`). Damage outside it: tier C kind `outside_facade`, never generated, hatched grey. s01: 4,811 px outside (4.8 %), 0 generated there; fill-off completed images byte-identical to `20261112_141348`. **Point 5 is not solved**: it lies inside the hull (`expert_base_022` box covers that ground). Needs the author: redraw the lower-left lobe or draw a facade mask; also judge the lower-right region the hull cuts. 143 tests.
- s01 runs of T27: `20261114_112130` (fill off) and `20261114_112408` (fill on), both before T28.
- Second photograph, same building (2026-11-14, author decision): `data/structures/s03_church_later` (900 x 1200, Wikiloc, licence unknown, later damage state than s01; see its `SOURCE.md`). Reference photos of the building are in `data/structures/s01_new_photos/` (not processed). Detection only so far: run `outputs/s03_church_later/20261114_122856`, 28 raw elements, same quality as on s01 (oversized boxes, scores below 0.03); the run stops at the damage step because no mask is drawn. Waiting for the author: damage mask (and facade mask), inventory correction in the editor. No code or config was changed for it.
- T28 done (2026-11-14, on s01; `docs/rnd/RND-004-alignment-acceptance.md`): every placed donor has `on_bound`, `limits`, `seam_agreement`; a tier A placement below `complete.min_alignment_score` (0.80) or `min_seam_agreement` (0.70) is `guess_needs_expert` (withheld in the restoration view, striped orange in the first-construction view). s01 now: restoration completes 1 target (`t_002`, 648 px), 4 tier A targets are guesses; first construction 3 completed + 4 guesses (11,367 px). The seam does not catch `t_003` (0.91); the score does. Thresholds from 9 synthetic cases, provisional. 151 tests. Run of T28: `outputs/s01_test_church/20261114_130553` (before T29, still with the two B slots).
- Author's working preference (2026-11-14): keep the structure buildable and not too complex; the author corrects records by hand where needed, so do not block on such items.
- T29 done (2026-11-14): a same-class element standing in a B1 gap off the implied positions stops that gap (`evidence.gap_intruders`, no new setting); every slot records `nearest_same_class`; tier B is painted as a guess in the first-construction view (`first_construction.tier_b_is_guess`). s01 now has **0 tier B targets** (21 targets: 5 A, 16 C); first construction changes 4,971 px (1 accepted + 4 tier A guesses) and differs from the restoration view only by the guesses. Synthetic: background 4/4 left C, full 1/12, partial 9/12, preservation 0. Left out on purpose: tier A second chance for half-height fragments (R-029a). 154 tests. Run of T29: `outputs/s01_test_church/20261114_144206`.
- T30 done (2026-11-14): `placement_box`, `donor_px` shares and `targets/<target_id>_<view>.png` panels (photograph, mask, donor as placed, result); outputs byte-identical.
- T31 done (2026-11-14, added task): expert decision per target. `python tools/target_review.py --run <run>` (window, or `--list` / `--approve` / `--reject` / `--clear`), stored in `data/structures/<id>/target_decisions.json` (tracked), applied by the next complete step; a decision is tied to one placement and goes stale when it changes. 163 tests.
- T32 done (2026-11-14, added task): the render step writes `report.html` + `input.png` per run (`i2r/report.py`): views, counts, every placed donor with its panel, open targets, commands for the expert. 164 tests.
- Expert review (2026-11-14): the author decided the five placements of run `20261114_150843` in `tools/target_review.py`: approve `t_003`, `t_007`; reject `t_001` (ends short), `t_002` (arch leg short of the capital), `t_006` (donor should be an intact pilaster). The tests agreed on 2 of 5 (RND-004). File: `data/structures/s01_test_church/target_decisions.json`.
- T33 done (2026-11-14, was R-029a, asked by the author): `evidence.extend_to_neighbour` gives a short fragment box the vertical extent of its nearest intact same-class neighbour when one end agrees, the other is short and the strip between is damaged (`evidence.extend_fragment_box`, `extend_min_damage_share` 0.5). Record keeps `element_box` + `box_extension`; `elements.json` unchanged. s01: 4 pilaster boxes extended, no arch; 20 targets (7 A, 13 C); `t_001` and `t_007` placements changed, so their decisions are stale; restoration completes only `t_003` (847 px), 4 pilaster guesses wait for the author (`t_005` scores -0.04). `t_002` not addressed. 169 tests.
- Current s01 reference: `outputs/s01_test_church/20261114_164027` (fill off, newest folder; open its `report.html`). After entering decisions: `python run.py --structure s01_test_church --from complete --no-review`.
- Status and open manual work for the author: `docs/STATUS.md`.
- Known weaknesses (after T20): alignment scale hits the search bounds; low NCC on pilasters; the size filter blocks `recovered_pilaster_046`, `recovered_pilaster_020` and, in the synthetic test, `recovered_arch_025` and `expert_cornice_021`; `min_similarity` cannot be tested without negatives (R-002a); the two B1 slots (x 343-366, 400-423) are doubtful and B1 made a slot on plain wall in `background__bg01`.
- Scale test (2026-11-12): `s01_test_church_upscaled` (Upscayl 3x, AI upscaler, not a new photo; see its `SOURCE.md`) showed that pixel settings did not follow image size; fixed the same day by `scale` in `config.yaml` (`runio.apply_pixel_scale`, reference 904 px; s01 outputs byte-identical). After the fix the copy gives the same targets, tiers and markings as s01; only 2 of 5 donors differ (upscaler texture). Not usable for calibration.

## Open author items
- Building identity/source/licence and larger copies of both photos (the author has found more photos of s01, not added yet: R5 in `docs/TASKS.md`); redraw the s01 mask edge near `pilaster_010`/`pilaster_011` (mask v3 is provisional); confirm opening → window.
- Decision 2026-11-10: work on s01 only until the pipeline is complete; s02 gets its own later session.
  **Changed 2026-11-12** (after the roadmap check): T19 on s01, then a second structure under T22 rules
  (no code or config change, every correction logged) **before** T20, so thresholds are not tuned on one
  building. Upscaling a small photograph does not count: it adds no evidence.
  **Author decision 2026-11-12 (final): no other photograph.** s02 is dropped and no new or alternative
  photo is used; work continues on s01 only. T22-T23 are skipped for this stage. T20 calibrates on the
  original s01 image; `s01_test_church_upscaled` is a scale check only (its texture is invented). Every
  result stays n = 1 building, tuned and reported on the same building.
  **Changed 2026-11-14 (author):** a second photograph of the same building is added as
  `s03_church_later` (found on the web; the largest version available is 900 x 1200). s01 stays as it
  is. Still n = 1 building; s03 is a second photograph and a later damage state, not a second building.
