# ImageToRender — Task List (this stage)

> **Türkçe özet.** Bu belge `docs/ARCHITECTURE.md` içindeki mimariyi Opus'un uygulayacağı küçük, sıralı
> görevlere böler. İlk görevler yeni klasör yapısını kurar ve eski denemeleri `archive/`'e taşır; sonra
> pipeline modül modül kurulur. Her görev tek oturumluktur ve "ne / hangi dosyalar / kabul ölçütü" içerir.
> "v1 bitti" noktası: tek komutla çalışan demo, iki yapıda iki çıktı ve README.

## Before you start

- Read `docs/EXPERIMENTS.md` (what exists) and `docs/ARCHITECTURE.md` (what to build). Section numbers
  such as "§5.2" refer to `ARCHITECTURE.md`.
- Do the tasks in order. Do not start a task until the previous one meets its acceptance criteria.
- Work on a branch: `refactor/structure` for Phase 0, then `feat/pipeline`. Do not commit to `main`.
- **The environment is fixed.** Use the conda env `ImageToRender_GPU`. Do not install, upgrade or remove
  any package. If a task seems to need a new package, stop and ask.
- **Archived files are never edited and never imported.** Copy the functions you need into the new
  module and adapt them there.
- No bare `except`. No hard-coded thresholds in modules; they go in `config.yaml`.
- Tasks marked **[needs author]** need a decision, a file or a manual step from the project owner. Stop
  and ask; do not guess.
- The test building's earlier runs live in `outputs/` (untracked). Useful ones:
  `restore_20260731_162315` (expert-locked inventory, 23 elements),
  `restore_20260804_141005` (10 approved hypotheses), `restore_20260804_143120` (donor selection).

---

## Phase 0 — New structure and archive

### T01. Prepare `.gitignore` and the skeleton
**What.** `.gitignore` currently ignores `archive/`, `data/` and `assets/`. Remove the `archive/` line.
Replace the `data/` line with rules that ignore `data/**` except `data/README.md` and
`data/structures/**/SOURCE.md` (images are decided per structure in T05). Create the empty layout from
§3: `i2r/__init__.py`, `tools/`, `evaluation/`, `tests/`, `data/structures/`, `docs/rnd/`.
**Files.** `.gitignore`, new empty package and folders (with `.gitkeep` where needed).
**Accept.** `git check-ignore archive/x.py` reports nothing. `git check-ignore data/structures/a/image.png`
reports ignored. `python -c "import i2r"` works from the repository root.

### T02. Archive the staged pipeline
**What.** Extract the 13 scripts from the local `ImageToRender_WORKFLOW_20260824.zip` and place them in
`archive/staged_pipeline_v8/` under their zip names. For the nine that are identical to a root file, use
`git mv` on the root file instead of adding a copy, so history is kept (mapping in §4). If the zip is not
present locally, take it from git history: `git show 84dd223:ImageToRender_WORKFLOW_20260824.zip`.
**Files.** `archive/staged_pipeline_v8/*.py` (13 files).
**Accept.** 13 files present. Each matches the SHA-256 in the zip's `MANIFEST.json` when line endings are
normalised to LF. `python -m py_compile` passes for all.

### T03. Archive everything else
**What.** `git mv`, no content change:

| From | To |
|---|---|
| `ImageToRender.py` | `archive/facadegen/facadegen_v1.py` |
| `ImageToRender1.py` | `archive/facadegen/facadegen_v2.py` |
| `GUIDE.md` | `archive/facadegen/GUIDE.md` |
| `ImageToRender2Restoration.py` | `archive/inpaint/inpaint_v3.py` |
| `ImageToRender2.py` | `archive/inpaint/inpaint_v6.py` |
| `ImageToRender3.py` | `archive/variants/stage1c_cleanup.py` |
| `ImageToRender4.py` | `archive/variants/stage3b_v2_root_variant.py` |
| `ImageToRender12_5.py` | `archive/stage5_v1/stage5_v1_ai_inpaint.py` |
| `ImageToRender_HybridFuzzyReconstruction_v1.py` | `archive/hybrid_v1/hybrid_fuzzy_v1.py` |

From `ImageToRender_WORKFLOW_EXACT_FILES_20260731.zip` extract only
`ImageToRender2Restoration_v5.py`, `..._stage3b_element_families.py` and
`..._stage3b_v5_expert_inventory_editor.py` into `archive/variants/`.
Leave `ImageToRenderEditorPage3B.py` in the root for T04.
**Files.** As listed.
**Accept.** The root contains exactly one `.py` file (`ImageToRenderEditorPage3B.py`). All archived files
compile.

### T04. Choose the inventory editor [needs author]
**What.** Two editor versions exist: root `ImageToRenderEditorPage3B.py` and
`archive/staged_pipeline_v8/09_stage3b_v5b_expert_inventory_editor.py`. Show the author a short summary of
how they differ (diff the two; list changed functions) and ask which to keep. Default if no answer: the
v5b version, because the zip README names it accepted. Put the chosen one at
`tools/inventory_editor.py` (copy if it is the archived one, `git mv` if it is the root one). Move the
root file to `archive/variants/` if it was not chosen. Do not edit the editor's code.
**Files.** `tools/inventory_editor.py`.
**Accept.** No `.py` files in the root. `tools/inventory_editor.py` compiles.

### T05. Data folder for the test building [needs author]
**What.** Create `data/structures/<id>/` (ask the author for a short ID). Move root `image.png` there and
root `completion_mask.png` there as `damage_mask.png`. Write `SOURCE.md` with: building name, location,
photographer, source URL, licence, date. Ask the author for these, and ask whether a larger copy of the
photograph exists (it is 628 x 904 px; §8 asks for 1500 px). Track the image in git only if the licence
allows redistribution, by adding a per-structure exception to `.gitignore`. Delete root
`auto_mask_preview.png` and `auto_mask_proposal.png`. Write `data/README.md` describing the folder
format and the photograph requirements from §8.
**Files.** `data/structures/<id>/*`, `data/README.md`, `.gitignore`.
**Accept.** `SOURCE.md` has no empty field. No image files remain in the root.

### T06. `archive/README.md` and clean-up
**What.** Write `archive/README.md`: one line per archived file with what it is, whether it worked and
the run that shows it. Take the text from `docs/EXPERIMENTS.md` §1 to §3, including the note on scripts
that are lost. Delete the two workflow zips from the working tree and the two
`setup/_old_requirements_facadegen_*.txt` files (all untracked; this is a local delete, so confirm with
the author first).
**Files.** `archive/README.md`.
**Accept.** Every file under `archive/` is mentioned in the README. `git status` is clean after commit.

### T07. Environment check
**What.** `tools/check_env.py` prints the interpreter path, CUDA availability and the versions of the
pinned packages in §1, and exits non-zero with a clear message if torch cannot be imported or a pinned
version differs.
**Files.** `tools/check_env.py`; add one line to `setup/SETUP.md`.
**Accept.** Exit code 0 in `ImageToRender_GPU`; non-zero under an interpreter without torch.

---

## Phase 1 — Pipeline, module by module

### T08. `config.yaml`, `i2r/runio.py`, `run.py` skeleton
**What.** Create `config.yaml` with every section named in §3 and the initial values given in §5 (mark
them `# initial guess, calibrate in T20`). `runio.py`: create `outputs/<structure>/<timestamp>/`, copy
the config into it, write `run_meta.json` (interpreter, package versions, git commit), read and write
JSON, and a `require_keys(obj, keys, where)` helper that raises with the file and key named.
`run.py`: parse `--structure`, `--view {restoration,first_construction,both}`, `--from <step>`,
`--no-review`; call the six steps in order; each step is a stub that logs its name.
**Files.** `config.yaml`, `i2r/runio.py`, `run.py`, `tests/test_runio.py`.
**Accept.** `python run.py --structure <id> --no-review` creates a run folder with `run_meta.json` and the
config copy and logs six step names. A test shows `require_keys` raising on a missing key.

### T09. `i2r/detect.py`
**What.** Port multi-scale OWL-ViT detection and SAM masking from
`archive/staged_pipeline_v8/01_stage1_multiscale_detection.py`. Read the class list and text prompts
from `config.yaml` (all classes in §1). Load one model at a time and free it before the next (6 GB).
Write `elements.json`: for each element `id`, `class`, `box` (xyxy), `score`, `mask_file`, `source:
"detector"`; and one mask PNG per element.
**Files.** `i2r/detect.py`, `config.yaml`, `tests/test_detect_schema.py` (checks the file format on a
small hand-made example; no GPU).
**Accept.** On the test building the step finishes within 6 GB and writes at least one element.
`elements.json` passes the schema test. Peak VRAM is logged.

### T10. Detection baseline (measure before changing anything)
**What.** `evaluation/evaluate.py detection`: compare an `elements.json` with a reference inventory and
print per-class recall, precision and mean box IoU (§7.3). Use the expert-locked inventory from
`outputs/restore_20260731_162315/` as the reference for the test building (read its format from the
file; elements are under `elements`, with `class`, `bbox_xyxy`, `status`). Write the result to
`docs/rnd/RND-001-detection-baseline.md`, including which of the new classes return nothing.
**Files.** `evaluation/evaluate.py`, `docs/rnd/RND-001-detection-baseline.md`.
**Accept.** The note has a per-class table and states n = 1 structure. No detector settings were tuned to
improve the numbers during this task.

### T11. `i2r/review.py`
*Note added after T10:* detector scores on s01 are all below 0.06, so with the initial threshold every
element is reviewed (48 on s01). The prompt therefore works on batches of numbers (reject several,
reclassify a whole class) rather than one question per element; the per-element outcome is the same.
**What.** Draw a numbered overlay of all elements. For each element with `score` below
`review.confidence_threshold`, ask in the terminal: confirm, change class (choose from the list), or
reject. Write decisions back to `elements.json` (`confirmed`, corrected `class`, `rejected`) and append
every decision to `review_log.json`. With `--no-review` nothing is asked and every element is kept with
`confirmed: false`.
**Files.** `i2r/review.py`, `tests/test_review.py` (feeds scripted answers).
**Accept.** The test passes with scripted input. `--no-review` never blocks. Rejected elements are kept
in the file with `rejected: true`, not deleted.

### T12. Adapter for the inventory editor
*Status: done. Commands: `edit` (export, open the editor, import), `export`, `import --run` (merge an
editor save into a run's `elements.json`) and `import --structure` (an old inventory becomes a new
run). Decisions: the editor is launched with the pipeline's class list (an editor `Config` argument,
so the editor file is unchanged); old editor classes go through `editor.class_map` (s01: `opening` to
`window`, reason in its `structure.yaml`); boxes the editor deleted or merged stay in `elements.json`
as rejected, because the editor's Delete removes a box outright; the editor's preset status is not
taken as an expert judgement, so `expert_condition` is written only for records the expert touched.
Result: the round trip on the 48-element s01 run returns an identical file; the locked inventory
imports as run `outputs/s01_test_church/20261110_171025` with 23 elements (12 intact, 11 fragmented,
8 expert-added), and the T10 command scores it 23/23 at IoU 1.00 against the original.
**For T13 to T15:** use that run (`python run.py --structure s01_test_church --from damage` while it is
the latest). Its masks are rectangles (`mask_kind: "box"`), so mask overlap in T13 overstates an
element's area; refining box masks with SAM is an open option, not done. The expert's own condition
label is in `expert_condition` and the donor flag in `editor_record.usable_as_prototype`; compare
both with what T13 and T15 compute.*
**What.** `tools/inventory_editor.py` reads and writes the old Stage 3B formats. Read its loading and
saving code, then write a small adapter with two commands: `export` turns a run's `elements.json` into
the files the editor expects; `import` turns the editor's locked inventory back into `elements.json`
(added elements get `source: "expert"`). Do not modify the editor.
**Files.** `tools/editor_adapter.py`, `tests/test_editor_adapter.py`.
**Accept.** A round trip (export then import, with no edits) returns the same elements. Importing
`outputs/restore_20260731_162315/expert_validated_inventory.json` (or the file name found there) yields
23 elements.

### T12b. Detector clean-up, measured against RND-001 (added after T10)
*Status: done before T12 (it does not depend on T12). Two deviations from the text below, both
deliberate: the class merge runs **before** NMS, otherwise two boxes stay on one element; and the
per-building choice lives in `data/structures/<id>/structure.yaml`, a new optional file merged over
`config.yaml`, because one settings file cannot hold a per-building decision. Result: class-aware
recall 30 % to 52 % on s01; see the re-measurement section of RND-001.*
**Why.** RND-001 found that five of eight pilasters are detected but labelled `column`, and that four
boxes each cover more than 15 % of the image and are always wrong.
**What.** (a) Add `detect.max_element_area_ratio` to `config.yaml` and drop elements whose box exceeds
it. (b) Add `detect.class_merge` (a mapping such as `column: pilaster`) applied after NMS, off by
default; decide per structure in `config.yaml` whether to enable it, and record the reason. Re-run the
T10 command. Change nothing else.
**Files.** `i2r/detect.py`, `config.yaml`, `tests/test_detect_schema.py`, append a "Re-measurement"
section to `docs/rnd/RND-001-detection-baseline.md`.
**Accept.** The re-measurement table sits next to the baseline in RND-001 with both commits named.
Neither change lowers class-agnostic recall.

### T13. `tools/mask_editor.py` and `i2r/damage.py`
*Status: done. `python tools/mask_editor.py --structure <id> [--propose] [--no-gui]` opens an existing
mask for editing, or starts from the proposal; it loads no torch (tested). Deviations, both deliberate:
OpenCV is an optional import in the proposal (as in the archived code; without it the Pillow fallback
runs), and the proposal and window settings live in new `mask_proposal` / `mask_editor` sections of
`config.yaml` (archived values, never calibrated). `damage` labels by element-mask overlap and also
records the box overlap for diagnosis; it copies the mask into the run folder, writes one PNG per
region, `damage_overlay.png`, and compares with `expert_condition` when present. `run.py` now updates a
step's `run_meta.json` entry instead of replacing it (the step's own figures were being overwritten).
Result on s01 (run `outputs/s01_test_church/20261110_172329`, from the imported 23-element inventory,
n = 1): mask 17.3 % of the image, 3 regions, 12 intact / 11 fragmented, **19 of 23 agree with the
expert label**. The four disagreements: `pilaster_010` and `pilaster_011` (expert intact) overlap the
mask by 0.07 and 0.11 because their masks are editor boxes and the drawn mask is loose at its edge;
`recovered_pilaster_020` and `recovered_pilaster_022` (expert fragmented) have 0 overlap, so the drawn
mask does not include whatever damage the expert saw on them. The threshold was not tuned to remove
these (n = 1); the second pair is for the author: either extend `damage_mask.png` over those two
pilasters or accept that they are not completed. The automatic proposal on s01 marks 2.2 % of the image
and has IoU 0.00 with the drawn mask (it finds dark openings, not losses): do not rely on it.
Follow-up (author's decision): the s01 mask was extended over the damaged lower ends of
`recovered_pilaster_020` and `recovered_pilaster_022` (mask version 2, details in the building's
`SOURCE.md`). Re-run `outputs/s01_test_church/20261110_173011`: 17.8 % of the image, 5 regions,
10 intact / 13 fragmented, **21 of 23 agree**; only the two box-mask pilasters remain. Use this run
for T14 onwards.*
**What.** (a) Copy `create_mask_with_editor()` and its helpers from `archive/inpaint/inpaint_v3.py` into a
standalone `tools/mask_editor.py` that imports only Pillow, numpy and Tkinter, with
`--image`, `--out` and an optional `--propose` flag that pre-loads the automatic dark-region proposal
(copy that function too). (b) `damage.py`: load `damage_mask.png`; fail if it is missing or nearly empty;
label each element `intact` or `fragmented` by its mask overlap (§5.1); split the damage mask into
connected regions; write `damage_regions.json` and the status into `elements.json`.
**Files.** `tools/mask_editor.py`, `i2r/damage.py`, `tests/test_damage.py`.
**Accept.** The mask editor runs without importing torch. On a synthetic test image with two boxes and a
mask covering half of one, exactly that one is labelled `fragmented`.

### T14. `i2r/evidence.py`, tier A
*Status: done. Filters and scorers are two lists (`DONOR_FILTERS`, `SCORERS`), as §9 asks. Decisions:
each element box is resized to 224 x 224 and the DINOv2 patch tokens are mean-pooled
(`evidence.pooling`, `crop_px`); the donor is resized to the target box and the target's hole is hidden
on it too (`mask_donor_like_target`), so both are compared on the part that survives; both orientations
are scored. Added after the first s01 run: `evidence.min_visible_fraction` (0.25, initial guess),
because `expert_opening_019`, 8 % visible, scored 0.95 against the drum slit: both crops were almost all
fill colour. Outputs `evidence.json` (every candidate with its score, every rejected same-class donor with
its reason), `evidence_overlay.png`, `evidence_sheet.png`. The model is downloaded on first use (88 MB).
Result on s01 (run `outputs/s01_test_church/20261110_173541`, n = 1): all 13 fragmented elements have a
record; 6 tier A, 7 tier C. Tier A: `recovered_arch_003` and `recovered_arch_001` from `arch_005` (the
second flipped; `arch_007` 0.92 is a near tie with 0.94), four pilasters from `recovered_pilaster_008`,
the only intact pilaster. Tier C: three pilasters (008 is 17 px wide, they are 25 to 27 px: size
filter), three bases (no intact base exists), `expert_opening_019` (visibility). What this does not
show: whether the similarity discriminates. Every scored candidate on s01 lies between 0.77 and 0.94,
so `min_similarity` 0.60 rejects nothing here; T20 must calibrate it. `pilaster_010` and `pilaster_011`
are targets, not donors, because T13 labels them fragmented (box masks, see T13); with the expert's
label they would be the donors the three tier C pilasters lack.*
**What.** Port DINOv2-small loading and embedding from
`archive/staged_pipeline_v8/02_stage2c_dinov2_repetition.py`. For each fragmented element: gather intact
elements of the same class, apply the size check, embed crops with damage pixels replaced by the crop
mean, rank by cosine similarity (also try the flipped donor), and write a tier A record with the best
donor and two alternatives when the best is at or above `evidence.min_similarity` (§5.2). Otherwise
record the target with `tier: "C"` and the reason. Structure the scoring as a list of functions (§9).
**Files.** `i2r/evidence.py`, `tests/test_evidence_a.py` (uses fixed fake embeddings; no GPU).
**Accept.** On the test building, with the 23-element inventory imported (T12), every fragmented element
gets a record. Use the imported inventory, not raw detector output, as the input for developing T14 to
T18: RND-001 shows the detector finds 1 of 11 fragmented elements, so detector output alone would leave
almost nothing to complete. The test confirms ranking order, the size filter and the threshold behaviour.

### T15. `i2r/evidence.py`, tier B and tier C
*Status: done. Rules are a table (`TIER_B_RULES`: B1 rows, B2 symmetry), each a function returning
targets and diagnostics. Decisions: rows are built from all observed same-class elements (intact and
fragmented, so a fragmented element blocks its slot and stays tier A's job); spacing is the median of
the centre distances close to the smallest one, so a row needs `min_row_size` (3) elements; B1 only
interpolates between neighbours; the B2 axis comes from `evidence.tier_b.symmetry_axis_x` (set it in
`structure.yaml`) or is estimated from midpoints of similar same-class pairs, scored by the share of
checkable elements whose mirror matches one (IoU `mirror_match_iou`); when B1 and B2 imply the same
slot one target is kept with `also_implied_by`. Tier C regions are the damage pixels outside every A and
B target box (masks in `evidence_c_regions/`); every C record has confidence 0, A uses the similarity,
B uses regularity or agreement times the donor's score. New settings in `evidence.tier_b`, all initial
guesses for T20. The acceptance layouts are in `tests/test_evidence_b.py`.
Result on s01 (run `outputs/s01_test_church/20261110_174241`, n = 1): 21 targets: 6 A, 2 B, 13 C (7
fragmented elements without a donor and 6 uncovered regions); tier C regions hold 81 % of the damage
pixels. B1 found one usable pilaster row (`recovered_pilaster_008`, `_022`, `pilaster_011`, `_010`,
regularity 0.73) and implies two slots in the central collapse, x 343-366 and 400-423. Doubtful: the
paired pilasters of this arcade are not evenly spaced, and the first slot lies 19 px right of the
fragmented `recovered_pilaster_046`, which joined no row because its box is half the height; it may be
the same pilaster. B2 estimated an axis at x = 355 (agreement 0.65 over 20 elements, just above 0.60)
and implied nothing. On a curved apse photographed obliquely, an image-space mirror axis is a weak
assumption (§10). The restoration view excludes tier B by default, so neither affects it.*
**What.** B1: group same-class intact elements into rows; find gaps between neighbours whose width is
near a whole multiple of the row spacing and whose implied slot lies in the damage mask with no element;
create a target with the nearest row member as donor. B2: read a symmetry axis from the structure folder
if present, else estimate it; mirror each intact element; create a target where the mirror falls in the
damage mask with no element. Switch B2 off and log it when no axis reaches the minimum agreement. Tier
C: all damage pixels not covered by an A or B target, written as regions. Never extrapolate past the end
of a row.
**Files.** `i2r/evidence.py`, `tests/test_evidence_b.py` (synthetic layouts).
**Accept.** A synthetic row of five boxes with the third removed yields one B1 target at the third
position. A row with the last removed yields none. A mirrored pair with one side removed yields one B2
target. `evidence.json` holds A, B and C records with `confidence`.

### T16. `i2r/complete.py`, donor transfer and the restoration rule
*Status: done. View rules are a table (`VIEW_RULES`; the first-construction rule is T18). Decisions:
donor pixels are its element mask minus its own damaged pixels; tier A alignment is an exhaustive search
over shifts up to `complete.refine_shift_frac` of the box and the scales in `refine_scales`, scored by
grey-level normalised correlation on the fragment's visible pixels (ties keep the unshifted placement);
the donor's mean colour is shifted to the fragment's visible part (`match_colour`); alpha ramps from 0
at the mask edge to 1 `feather_px` inside, and is 0 outside; overlapping targets are painted in
increasing confidence. Outputs `<view>_completed.png` (no marking yet, T17), a 16-bit
`<view>_source.png` label image and `completion.json`. `assert_preserved` raises `PreservationError`;
a test makes `paint` touch one outside pixel and expects the failure. No torch import (tested).
Result on s01 (run `outputs/s01_test_church/20261110_175223`, n = 1): 6 of 6 tier A targets completed,
9,793 pixels changed, none outside the mask. Visually, the two arch heads (`recovered_arch_003`,
`recovered_arch_001`) and the pilaster feet are restored in place. Weaknesses, stated plainly: the
element masks are boxes (T12), so each patch carries the wall around the element and ends in straight
edges, clearest at the right arch; four of six alignments chose a scale at the edge of the searched range
(1.06 or 0.94), so the range or the score is not settled; correlation after alignment is low for the
pilasters (0.16 to 0.62, against 0.79 and 0.86 for the arches), partly because their visible parts are
narrow strips. Up to 404 damage pixels per box stay unfilled where the scaled donor does not reach. All
for T20; SAM masks for the editor's boxes would address the first point.*
**What.** Implement donor transfer (§5.5): masked donor crop, affine map to the target box, optional
flip, alignment refinement for tier A, feathered blend inside the damage mask only, donor ID written to a
source layer. Implement the restoration rule (§5.3): tier A only, unless `restoration.include_tier_b`.
Assert at the end that no pixel outside the damage mask changed, and fail the run if one did.
**Files.** `i2r/complete.py`, `tests/test_complete.py`.
**Accept.** The preservation assertion is covered by a test that deliberately breaks it. On the test
building a restoration image is written and the count of completed targets equals the count of tier A
records. No torch import is needed for this path.

### T17. `i2r/render.py`
*Status: done. Deviations, both deliberate: the source map is written per view
(`restoration_source_map.png`, later `first_construction_source_map.png`), and the label and legend go
in a bar added under each image instead of on the photograph. Every marking (tone shift, inner outline,
blue hatch for open tier B, red hatch for every other open damage pixel, region labels) is drawn on a
layer copied in only inside the damage mask, so outside the mask each view image equals the photograph
(tested); a region label is drawn only where it fits wholly inside its region. Render also refuses a
changed pixel without a source label. `provenance.json` has the §5.6 fields plus `kind`, `rule`,
`element_id`, `evidence_reason`, per-view `views`, `view_reasons`, `pixels_changed` and `alignment`, and
the hashes of the photograph and the damage mask. Results on s01 (n = 1): from the imported inventory
(`python run.py --structure s01_test_church --from complete --view restoration --no-review`, run
`outputs/s01_test_church/20261110_175655`), 21 targets in `provenance.json`, 6 completed. The
acceptance command from raw detection (run `20261110_175724`) also produces every file: 48 detections,
43 targets (8 A, 6 B, 29 C), 7 completed; the eighth is `completed_but_empty` because a more confident
overlapping target painted over all its pixels. That run completes large areas from oversized detector
boxes (one target changes 18,570 pixels) and is not a result to show; it shows why T14 to T18 use the
inventory. The blue B hatch is hard to see over the dark interior of the central collapse, and the two
B slots (23 px wide) are too narrow for a label; the legend carries it.*
**What.** Produce, per run: `restoration.png` (additions with tone shift and thin outline; B and C
regions hatched and labelled as in §5.3), `provenance.json` (§5.6), `source_map.png`, an overlay of
elements and targets, and a legend. Stamp every image with "AI proposal, not a restoration decision".
**Files.** `i2r/render.py`, `tests/test_render.py` (checks `provenance.json` keys and that every changed
pixel has a source).
**Accept.** `python run.py --structure <id> --view restoration --no-review` produces all files. Every
target in `evidence.json` appears in `provenance.json` with its view status.

### T18. First-construction rule, tiers A and B
**What.** Add the first-construction rule (§5.4) for A and B by donor transfer. Render
`first_construction.png` with a separate `conjecture_overlay.png` (empty for now, tier C still unfilled
and shown as such). Update `provenance.json` with the second view's status.
**Files.** `i2r/complete.py`, `i2r/render.py`, tests.
**Accept.** `--view both` writes both images. The two images differ only where a tier B target exists.

### T19. Synthetic damage tool and evaluation
**What.** `tools/synthetic_damage.py`: from a structure with an inventory, create case folders for the
three variants in §7.1 (partial, full, background), each with a modified inventory, a modified mask and
`ground_truth.json`. `evaluation/evaluate.py completion`: run the pipeline from the damage step on every
case (`--no-review`) and compute the metrics in §7.1. Write a CSV and a Markdown table.
**Files.** `tools/synthetic_damage.py`, `evaluation/evaluate.py`, `tests/test_synthetic_damage.py`.
**Accept.** On the test building it generates at least one case per intact element per variant and writes
the table. Preservation is 0 in every row.

### T20. Calibrate thresholds and write the baseline
**What.** Using T19, vary `evidence.min_similarity`, `evidence.size_tolerance` and the B1 tolerance over a
small grid and record recovery rate, tier accuracy and abstention. Choose values, update `config.yaml`,
and remove the "initial guess" comments. Write `docs/rnd/RND-002-evidence-baseline.md` with the tables,
the chosen values, the failure cases with images, and a comparison with the 10 hypotheses approved in
`outputs/restore_20260804_141005/` (which of them the new pipeline also finds, and which it does not).
**Files.** `config.yaml`, `docs/rnd/RND-002-evidence-baseline.md`.
**Accept.** The note states n = 1 structure and says explicitly that the thresholds were tuned on the
same structure they are reported on.

### T21. Tier C generative fill
**What.** Copy the ControlNet inpaint set-up from `archive/inpaint/inpaint_v3.py` into `complete.py`
(pipeline loading, memory settings, lineart and depth conditioning). Run it **once** per image on a
context crop of at least 512 px around the tier C area, starting from the donor-completed
first-construction image, with conditioning computed from that image, at the strength in
`first_construction` config. Recomposite everything outside the tier C area from the donor-completed
image. Write the conjecture overlay and `method: "generative"` records. This step runs only for the
first-construction view.
**Files.** `i2r/complete.py`, `i2r/render.py`, `config.yaml`.
**Accept.** Runs within 6 GB. The restoration view is byte-identical with and without this step. Pixels
outside the tier C area are unchanged by it. Write `docs/rnd/RND-003-generative-fill.md` with the output
images and a plain verdict: usable or not. **If the result is not usable, leave the step switched off by
default (`first_construction.generative: false`) and continue; that outcome completes the task.**

---

## Phase 2 — Second structure and v1

### T22. Second structure, unchanged pipeline [needs author]
**What.** The author supplies a second photograph that meets §8, with `SOURCE.md`, and draws the damage
mask with `tools/mask_editor.py`. Run the full chain with review. **Do not change pipeline code or
config during this task.** Record every failure, every manual correction (review decisions, elements
added in the editor) and the time spent.
**Files.** `data/structures/<id2>/`, `docs/rnd/RND-004-second-structure.md`.
**Accept.** Either both views exist for structure 2, or the note states exactly which step failed and
why. Both outcomes complete the task.

### T23. Fix what the second structure broke
**What.** Address the failures from T22, one commit per fix, each with a test. Re-run T19's evaluation
on structure 1 and confirm the numbers did not get worse, or explain why they changed.
**Files.** As required in `i2r/`, `tests/`.
**Accept.** Both structures run end to end with one command each. The evaluation table covers both.

### T24. Examples and README
**What.** Put a small example set in `docs/examples/<id>/` for each structure whose licence allows it:
input, element overlay, restoration view, first-construction view, source map (long side 1000 px or
less). Add a `.gitignore` exception for `docs/examples/`. Rewrite `README.md`: what the project does and
does not do, the near-frontal repeating-facade assumption, the two views and their evidence rules in one
table, example images, install, the one-command demo, the results tables with n stated, limitations
(including the state of the generative fill), the "AI proposal" statement.
**Files.** `docs/examples/**`, `README.md`, `.gitignore`.
**Accept.** Every path and command in the README exists and works. Example images total under 4 MB.

### T25. Fresh-clone check and tag
**What.** Clone the repository into a new folder, activate the conda env, follow the README only, and
run the demo on both structures. Fix anything missing. Merge to `main` and tag `v0.1.0`.
**Files.** As needed.
**Accept.** The definition below is met.

---

## Definition of "v1 done"

All of the following hold on a fresh clone in the conda env:

1. `python tools/check_env.py` and `python -m unittest discover tests` pass.
2. `python run.py --structure <id>` produces, for **two structures**, `restoration.png`,
   `first_construction.png`, `provenance.json` and `source_map.png`.
3. Pixel difference outside the damage mask is 0 for both views on both structures.
4. Every completed region has a provenance entry naming its donor, tier and confidence, or is marked
   conjecture.
5. The restoration view contains no generative pixels.
6. The README shows the detection and synthetic-damage tables with the number of structures stated, the
   example images, and a limitations section.
7. The repository root holds `run.py` and `config.yaml` and no other scripts; everything older is in
   `archive/` and described in `archive/README.md`.

The generative fill (T21) is allowed to be switched off in v1 if it did not produce a usable image. In
that case the first-construction view shows tier C regions as marked and unfilled, and the README says so.

---

## Research notes (do not block v1)

| ID | Question | When | Output |
|---|---|---|---|
| R1 | Frozen ResNet-50, DINOv2-small or CLIP ViT-B/32 features for few-shot classification of heritage element crops (k-NN and a linear probe written in torch; 1, 5, 10 shots; Architectural Heritage Elements dataset). All three backbones are available in the pinned env. | Any time | `docs/rnd/RND-005-frozen-features.md` |
| R2 | Per-class recall of zero-shot detection on heritage facades, and how much expert correction it needs. | After T23 | Extends RND-001 |
| R3 | Does synthetic-damage performance predict acceptance on real damage? | After 5 structures | New note |
| R4 | Donor-only restoration view versus a harmonised image: which do experts prefer, and can they point to the additions? | After T21 | New note |
