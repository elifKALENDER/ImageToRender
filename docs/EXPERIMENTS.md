# ImageToRender — Experiment Inventory

> **Türkçe özet.** Bu belge repodaki her denemeyi tek tek listeler: ne yapmaya çalışıyor, çalışıyor mu, bu
> aşamanın amacıyla ilgisi ve önerilen akıbeti (yeni yapıya taşı / arşivle / sil). "Çalışıyor mu" sütunu
> tahmine değil, `outputs/` klasöründeki gerçek çalıştırma kayıtlarına dayanır. Repoda notebook yoktur.
> Kısa sonuç: analiz tarafı tek yapıda çalışıyor, üretim tarafı henüz kabul edilebilir bir görsel vermedi.

Basis: repository at commit `85a33f0` (2026-11-10), the local untracked `outputs/` directory (62 run
folders), and the two local workflow zips. Nothing was executed for this inventory; "works" means a run
folder exists whose metadata or log reports completion.

**Stage goal used for the "relevance" column.** From one photograph of a damaged historical building:
recognise architectural elements, complete missing and damaged regions from the building's own intact
parts, and produce two views (restoration view, first-construction view) with a provenance record.

**Script-version check (added in T06).** Run metadata records the SHA-256 of the producing script.
Checked against the archived files: Stage 3A, 3B-v3, 3B-v4, 3C-A, 3C-B, 3C-C, 4, Stage 1C, inpaint v3,
inpaint v6 and the inventory editor are confirmed. For Stage 1, 2D, 2F and the first 3B families
version, the cited run was produced by a *different* version of the script. Details:
`archive/README.md`.

**Fate values.** *Port* = its logic moves into the new structure (`docs/ARCHITECTURE.md` §3).
*Archive* = moved to `archive/` unchanged, kept as a record. *Delete* = removed from the working tree
(git history keeps it).

---

## 1. Scripts in the repository root

| File | What it tries to do | Does it work? | Relevance | Fate |
|---|---|---|---|---|
| `ImageToRender.py` | FacadeGen v1. Photo to design-proposal render: depth ControlNet text-to-image, Realistic Vision V6, optional LoRA. | Yes. 9 runs, `outputs/run_202605*`. | None. Different purpose (client design proposals). | Archive (`archive/facadegen/`). |
| `ImageToRender1.py` | FacadeGen v2. Renovation proposals: depth + lineart ControlNet img2img, programme presets, three strength levels. | Yes. 7 runs, `outputs/i2r_bakim_evi_*`. | None. | Archive (`archive/facadegen/`). |
| `ImageToRender2Restoration.py` | Inpaint v3. Both views by prompt and strength only; manual mask editor; automatic dark-region mask proposal; ControlNet inpaint. | Runs (`run_20260731_222817`, completed). Output is not usable: it invents columns, a red door and a horseshoe arch, and seeds disagree. | High for three parts: the mask editor, the automatic mask proposal, the 6 GB inpaint set-up. The "two views differ only by prompt" idea is weak. | Port the three parts. Archive the file (`archive/inpaint/`). |
| `ImageToRender2.py` | Inpaint v6. First-construction view only; mirrors intact pixels across a symmetry axis into the hole as an "evidence guide" for the inpaint. | Runs (`restore_20260724_223345`). Result was not kept as a success. | Medium. The mirror-across-axis idea is reused as one evidence rule. | Archive (`archive/inpaint/`). |
| `ImageToRender3.py` | Stage 1C. OWL-ViT + SAM detection with class-specific cleanup and pattern preparation. | Yes (`restore_20260725_215919`). | Medium. It is a variant of Stage 1, and not the one the accepted workflow uses. | Archive (`archive/variants/`). |
| `ImageToRenderSkeleton3A.py` | Stage 3A. Draws an "evidence skeleton" from existing detections only. | Yes (`restore_20260726_103302`). | Low. Visualisation step of the arcade-specific chain. | Archive (`archive/staged_pipeline_v8/`). |
| `ImageToRender4.py` | Stage 3B-v2. Groups arches into perspective-normalised families. | Yes (`restore_20260726_1336*`). Differs from the zip's accepted version by about 2100 lines. | Low. | Archive (`archive/variants/`). |
| `ImageToRender5.py` | Stage 3B-v3. Cleans families, picks reliable prototypes. | Yes (`restore_20260726_135122`). | Low. | Archive (`archive/staged_pipeline_v8/`). |
| `ImageToRender6.py` | Stage 3B-v4. Uses prototypes to recover visible instances the detector missed; optional DINOv2 check. | Yes (`restore_20260731_134812`). | Medium. "Search for more instances of a known intact element" is useful for recall. | Archive now (`archive/staged_pipeline_v8/`); candidate for a later port. |
| `ImageToRenderEditorPage3B.py` | Stage 3B-v5. Tkinter editor where the expert accepts, rejects, reclassifies, moves and adds element boxes, then locks the inventory. | Yes (`restore_20260731_162315`). Differs from the zip's v5b by about 550 lines. | High. It is the only way to add elements the detector missed. | Keep as a tool, unchanged (`tools/inventory_editor.py`), after choosing between this and v5b. |
| `ImageToRender8_3C_A.py` | Stage 3C-A. Builds relations between inventory elements: families, courses, alignments, arch and pilaster pairs. | Yes (`restore_20260731_164727`). | Low for this stage. Arcade-specific rules. | Archive (`archive/staged_pipeline_v8/`). |
| `ImageToRender9_3C_B.py` | Stage 3C-B. Facade sectors, ordered supports and bays, element roles. | Yes (`restore_20260731_172741`). | Low for this stage. | Archive (`archive/staged_pipeline_v8/`). |
| `ImageToRender10_3C_C.py` | Stage 3C-C. Generates reconstruction hypotheses by rule and opens a Tkinter editor for expert approval. | Yes (`restore_20260804_141005`): 10 hypotheses, all approved. | Medium. The best-working inference step so far, but written around arches, pilasters and courses on one building. Its three hypothesis types inform the new evidence tiers. | Archive (`archive/staged_pipeline_v8/`). |
| `ImageToRender11_4.py` | Stage 4. Turns approved hypotheses into per-element masks, selects a donor for each, writes a source map. No image synthesis. | Yes (`restore_20260804_143120`, completed with warnings: 2 of 10 hypotheses do not touch the damage mask). | High as an idea: donor per element plus source map. The code depends on the Stage 3C file formats. | Port the idea. Archive the file (`archive/staged_pipeline_v8/`). |
| `ImageToRender12_5.py` | Stage 5 V1. Per-element crop, donor "guide composite", ControlNet inpaint for material and light. | Runs (`restore_20260804_150935`). Output rejected: multicoloured noise in the cornice, unrelated fragments in pilasters. | Negative result worth keeping. Shows that per-element inpaint on 20 to 130 px elements fails. | Archive (`archive/stage5_v1/`). |
| `ImageToRender_HybridFuzzyReconstruction_v1.py` | Hybrid prototype: DINOv2 similarity, small CNN, structural features, fuzzy fusion, expert feedback, to propose missing arches. | Runs, but produced 0 hypotheses (`restore_20260824_004739`). See §5. | Low for this stage (learning models are out of scope). Its "several evidence channels, record feedback" idea is a door left open. | Archive (`archive/hybrid_v1/`). |

## 2. Scripts that exist only inside `ImageToRender_WORKFLOW_20260824.zip`

The zip is local and gitignored. Its README calls these 13 files the accepted workflow. Nine of them are
identical to root files (see the mapping in `docs/ARCHITECTURE.md` §4). Four exist nowhere else:

| File in zip | What it tries to do | Does it work? | Relevance | Fate |
|---|---|---|---|---|
| `01_STAGE1/01_stage1_multiscale_detection.py` | Stage 1. Multi-scale OWL-ViT boxes, SAM masks, classes arch / pilaster / window / door. | Yes (`restore_20260725_212650`, stage label 1B). | High. This is the detector the new pipeline starts from. | Port into `i2r/detect.py`. Archive the file. |
| `02_STAGE2/02_stage2c_dinov2_repetition.py` | Stage 2C. DINOv2 embeddings to find repeating elements. | Unknown. No run folder carries a 2C stage label. | High. DINOv2 similarity between elements is the core of donor finding. | Port into `i2r/evidence.py`. Archive the file. |
| `02_STAGE2/03_stage2d_cleaned_axes.py` | Stage 2D. Cleans rhythm axes from repetition groups. | Yes (`restore_20260726_095000`). | Medium. Row and spacing logic informs the tier B rule. | Archive. |
| `02_STAGE2/04_stage2f_raw_candidate_fusion.py` | Stage 2F. Fuses raw candidates from detection and repetition. | Yes (`restore_20260726_1019*`, `_1020*`). | Low. | Archive. |

## 3. Runs whose script is not in the repository

| Run folders | Stage label in metadata | Note | Fate |
|---|---|---|---|
| `restore_20260725_190749` to `_191351` (4 runs) | `evidence_composite_only_no_diffusion` | An early "evidence composite before diffusion" step. Three cancelled, one approved. The script is in neither the repository nor the zips. It is the closest thing to the planned donor system that ever ran. | Nothing to move. Record in `archive/README.md` that the script is lost. |
| `restore_20260725_194716` to `_210145` (4 runs) | `v8_stage1_detect_elements_only`, `v8_stage1_architectural_element_detection` | Earlier Stage 1 versions; three failed, one completed. Scripts not in the repository. | Nothing to move. |
| `restore_20260726_1008*` (2 runs) | `v8_stage2e_architectural_evidence_fusion` | Stage 2E, replaced by 2F. Script not in the repository. | Nothing to move. |
| `restore_20260726_1025*`, `_102604` | `v8_stage3a_completion_skeleton` | Earlier Stage 3A that drew completions; replaced by the "existing evidence only" version. | Nothing to move. |
| `restore_20260804_145413` | `v8_stage5_material_texture_light_integration` | A non-AI Stage 5 variant run 15 minutes before Stage 5 V1. Script not in the repository. | Nothing to move. |

## 4. Other files

| Path | What it is | Fate |
|---|---|---|
| `ImageToRender_WORKFLOW_20260824.zip` | Snapshot of the accepted workflow (13 scripts + manifest). Contains stray `.pyc` files. | Extract the 13 scripts into `archive/staged_pipeline_v8/`, then delete the zip. |
| `ImageToRender_WORKFLOW_EXACT_FILES_20260731.zip` | Earlier snapshot. Every script in it is either identical to one in the newer zip or an older version of one. It also holds `ImageToRender2Restoration_v5.py`, an inpaint version not otherwise present. | Extract only `ImageToRender2Restoration_v5.py` and the two older variants into `archive/variants/`, then delete the zip. |
| `GUIDE.md` | Client handbook for FacadeGen. | Archive with FacadeGen. |
| `setup/` (current files) | Environment definition and install notes. | Keep in place. |
| `setup/_old_requirements_facadegen_*.txt` | Old FacadeGen requirement lists. Untracked, UTF-16. | Delete. |
| `image.png`, `completion_mask.png` (root) | The test photograph and its damage mask. | Move to `data/structures/<id>/`. |
| `auto_mask_preview.png`, `auto_mask_proposal.png` (root) | Leftover outputs of the automatic mask proposal. | Delete. |
| `assets/` (8 images) | Reference images of unknown provenance. | Ask the author. Candidates for new structures only if source and licence are known. |
| `archive/` (existing, empty) | Empty folder, currently gitignored. | Becomes the archive; remove it from `.gitignore`. |
| `loras/` (3 files, 14 MB) | LoRA weights used only by FacadeGen. | Leave on disk (untracked). Not used by this stage. |
| `models/ip-adapter_sd15.bin`, `models/models--h94--IP-Adapter` | IP-Adapter weights. Cannot be loaded in diffusers 0.20.2 (`load_ip_adapter` does not exist there). | Leave on disk. Not used by this stage. |
| `models/mobile_sam.pt` | MobileSAM weights. A pattern search found no script that loads it. | Leave on disk. Verify before deleting. |
| `outputs/` (62 run folders, 200 MB) | All run results. Untracked. Metadata holds absolute `E:\` paths that no longer exist. | Keep on disk as the record. Do not move. |
| `desktop.ini` | Windows folder file. | Already ignored. |

## 5. Claims checked against the code

| Remembered as | What the code and runs show |
|---|---|
| Element recognition and classification is complete | Partly. Detection is zero-shot OWL-ViT + SAM with four classes (arch, pilaster, window, door); cornice, base and opening exist only as expert-drawn boxes. There is no trained classifier and no accuracy figure. On the test building the expert added 8 of 23 final elements by hand and corrected the box of 9 more. |
| `StableDiffusionInpaintPipeline` + dark-threshold mask | Present, as `StableDiffusionControlNetInpaintPipeline`. The automatic mask uses a dark percentile, local contrast and connected components. It runs; its output quality is poor. |
| Donor system (Grounded SAM, DINOv2 donor search, homography, evidence composite approval) | Mostly not implemented. No Grounded SAM, no homography or warp call, no LightGlue trace. DINOv2 is used for repetition, not donor matching. Stage 4 picks donors by class and geometry. The one evidence-composite script that ran is lost (§3). |
| Hybrid prototype failed because of one structure, missing torch, 0 training rows | Those are true but secondary. The primary cause is a schema mismatch: the script reads `left_x` / `right_x` from bay intervals, Stage 3C-B writes `x_range`; a bare `except` drops all 7 intervals silently. Even when fixed, its only rule ("bay with no arch at all") matches nothing on this building, and its row thresholds (8 and 16) exceed what one building provides (7 and 10). |
| TRELLIS / Blender multi-view line | No code in the repository. |
| FacadeGen | Present and working. |

## 6. Summary

- **Works on one building:** detection with heavy expert correction, the inventory editor, rule-based
  hypotheses, donor selection and source maps.
- **Runs but output rejected:** both generative attempts (inpaint v3, Stage 5 V1).
- **Does not work:** hybrid prototype.
- **Missing from git:** four scripts of the accepted workflow; five earlier scripts are lost entirely.
- **No tests, no evaluation, no dataset, no entry point.**
- **One test photograph, 628 x 904 px.** Every threshold in every script was tuned on it.
