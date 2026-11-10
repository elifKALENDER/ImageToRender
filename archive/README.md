# Archive

Earlier experiments, kept as a record. **Nothing here is edited or imported by the current code.** When
a function is reused, it is copied into `i2r/` or `tools/` and adapted there.

"Run" refers to a folder in the local, untracked `outputs/` directory. Full analysis:
`docs/EXPERIMENTS.md`.

**Hash-checked.** Most runs record the SHA-256 of the script that produced them. A run marked ✓ was
produced by exactly this file (allowing for line endings, a trailing newline and, for Stage 3B-v4, a
shebang line). A run marked ≈ carries the same stage label but was produced by a *different version*
of the script, so it shows the stage worked, not that this exact file did.

## `staged_pipeline_v8/` — the accepted 13-stage workflow (2026-07-25 to 2026-08-04)

Taken from `ImageToRender_WORKFLOW_20260824.zip`; every file matches that zip's `MANIFEST.json`.

| File | What it is | Worked? | Run |
|---|---|---|---|
| `01_stage1_multiscale_detection.py` | Multi-scale OWL-ViT boxes + SAM masks; classes arch, pilaster, window, door | An earlier version worked | `restore_20260725_212650` ≈ |
| `02_stage2c_dinov2_repetition.py` | DINOv2 embeddings to find repeating elements | Unknown (no run carries this label) | — |
| `03_stage2d_cleaned_axes.py` | Cleans rhythm axes from repetition groups | An earlier version worked | `restore_20260726_095000` ≈ |
| `04_stage2f_raw_candidate_fusion.py` | Fuses detector and repetition candidates | An earlier version worked | `restore_20260726_101903` ≈ |
| `05_stage3a_existing_evidence_skeleton.py` | Evidence skeleton from existing detections only | Yes | `restore_20260726_103302` ✓ |
| `06_stage3b_v2_perspective_families.py` | Perspective-normalised arch families | Unknown (run records no script hash) | `restore_20260726_133637` or `_133608` |
| `07_stage3b_v3_family_cleanup.py` | Family cleanup, reliable prototypes | Yes | `restore_20260726_135122` ✓ |
| `08_stage3b_v4_observed_instance_recovery.py` | Finds visible instances the detector missed | Yes | `restore_20260731_134812` ✓ |
| `09_stage3b_v5b_expert_inventory_editor.py` | Inventory editor, v5b (deletes boxes outright). The v5 editor in `tools/inventory_editor.py` was kept instead | Not used for the locked inventory | — |
| `10_stage3c_a_architectural_relations.py` | Relations between inventory elements | Yes | `restore_20260731_164727` ✓ |
| `11_stage3c_b_facade_sectors_and_bays.py` | Sectors, support and bay sequence, roles | Yes | `restore_20260731_172741` ✓ |
| `12_stage3c_c_reconstruction_hypotheses_expert_editor.py` | Rule-based hypotheses + approval editor | Yes: 10 hypotheses, all approved | `restore_20260804_141005` ✓ |
| `13_stage4_geometric_reconstruction.py` | Per-hypothesis masks, donor choice, source map | Yes, with warnings (2 of 10 hypotheses miss the mask) | `restore_20260804_143120` ✓ |

Retired because the chain is arcade-specific, tuned on one building and too long to maintain. Its ideas
live on in `i2r/` (see `docs/ARCHITECTURE.md` §4).

## `variants/` — superseded versions

| File | What it is | Worked? | Run |
|---|---|---|---|
| `stage1c_cleanup.py` | Stage 1C, detection cleanup and pattern prep (not the accepted Stage 1) | Yes | `restore_20260725_215919` ✓ |
| `stage3b_v2_root_variant.py` | Stage 3B-v2, a different version from the accepted one | Unknown (run records no script hash) | `restore_20260726_133608` or `_133637` |
| `ImageToRender2Restoration_v5.py` | Inpaint v5; recommended in the 2026-07-31 snapshot for drawing the damage mask | Unknown | — |
| `ImageToRender2Restoration_v8_stage3b_element_families.py` | Stage 3B families, first version | An earlier version worked | `restore_20260726_131547` ≈ |
| `ImageToRender2Restoration_v8_stage3b_v5_expert_inventory_editor.py` | Earlier copy of the v5 editor (layout differs from `tools/inventory_editor.py`) | Unknown | — |

## `inpaint/` — diffusion-first restoration attempts

| File | What it is | Worked? | Run |
|---|---|---|---|
| `inpaint_v3.py` | ControlNet inpaint producing both views by prompt and strength; mask editor; automatic dark-region mask | Runs; output invents architecture (columns, red door, horseshoe arch) | `run_20260731_222817` ✓ |
| `inpaint_v6.py` | First-construction view using pixels mirrored across a symmetry axis as a guide | Runs; not kept as a success | `restore_20260724_223345` ✓ |

## `stage5_v1/`

| File | What it is | Worked? | Run |
|---|---|---|---|
| `stage5_v1_ai_inpaint.py` | Per-element crop, donor guide composite, ControlNet inpaint | **Rejected**: noise and unrelated fragments in every region. Kept as a negative result | `restore_20260804_150935` (no hash recorded) |

## `hybrid_v1/`

| File | What it is | Worked? | Run |
|---|---|---|---|
| `hybrid_fuzzy_v1.py` | DINOv2 + small CNN + structural features + fuzzy fusion to propose missing arches | **No**: 0 hypotheses. Primary cause is a schema mismatch (`left_x`/`right_x` vs `x_range`) hidden by a bare `except`; see `docs/EXPERIMENTS.md` §5 | `restore_20260824_004739` |

## `facadegen/` — a different project

| File | What it is | Worked? | Run |
|---|---|---|---|
| `facadegen_v1.py` | Photo to design-proposal render, depth ControlNet | Yes | `run_202605*` (no hash recorded) |
| `facadegen_v2.py` | Renovation proposals, depth + lineart img2img, programme presets | Yes | `i2r_bakim_evi_*` (no hash recorded) |
| `GUIDE.md` | Client handbook for FacadeGen | — | — |

## Lost scripts

These ran, but their source is in neither the repository nor any snapshot:

| Run folders | Stage label |
|---|---|
| `restore_20260725_190749` to `_191351` | `evidence_composite_only_no_diffusion` (the only evidence composite that ever ran) |
| `restore_20260725_194716` to `_210145` | Early Stage 1 versions |
| `restore_20260726_100846`, `_100854` | Stage 2E evidence fusion |
| `restore_20260726_102557`, `_102604` | Early Stage 3A completion skeleton |
| `restore_20260804_145413` | Non-AI Stage 5 material integration |
