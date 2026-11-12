# ImageToRender

> **Türkçe özet.** Hasarlı bir tarihi yapının tek fotoğrafından ve hasar maskesinden iki görünüm üretir:
> yalnızca yapının kendi sağlam elemanlarından (donör) kopyalanan kanıta dayalı bir "restorasyon
> görünümü" (Venedik Tüzüğü) ve konumu ima edilen elemanları da ekleyen bir "ilk inşa görünümü". Her
> eklenen pikselin kaynağı kayıtlıdır. Tek bir test yapısında (n = 1) geliştirildi; erken aşama karar
> desteğidir, hiçbir çıktı restorasyon kararı değildir.

**Every image this pipeline produces is an AI proposal, not a restoration decision.**

Research prototype by an architect. Input: one photograph of a damaged historical facade and a mask of
the damaged area. Output: two views of the facade and a provenance record that names, for every changed
pixel, the target it belongs to, the evidence tier and the intact element it was copied from.

Status: all six pipeline steps work, the evaluation exists, and the thresholds were chosen on **one
building** (`s01_test_church`, n = 1). Nothing here has been validated on a second building.

## What it does and does not do

It does:

- find architectural elements (arches, pilasters, windows, cornices, ...) with zero-shot detection, let an
  expert correct the inventory, and label each element intact or fragmented against the damage mask;
- give every damaged place one evidence tier (below) and complete it, where the rules allow, by copying an
  intact element of the same building into the damage mask (no generative model);
- leave every pixel outside the damage mask unchanged (checked on every run; the run fails otherwise);
- record where every added pixel came from (`provenance.json`, `*_source_map.png`).

It does not:

- decide anything about a real building, date it, or know its history;
- invent elements that the photograph gives no evidence for (the generative fill exists but is off; see
  Limitations);
- handle oblique views, several photographs, or 3D: it assumes **one near-frontal photograph of a facade
  with repeating elements** (several arches, several pilasters), because donors come from the same image.

## The two views

Evidence tiers (`docs/ARCHITECTURE.md` §5.2):

| Tier | Meaning | Condition |
|---|---|---|
| **A** direct | Part of the element survives and the building shows what the whole looks like | A fragmented element and an intact donor of the same class, similar size, DINOv2 similarity >= 0.80 |
| **B** positional | The element left no trace; only the pattern says it was there | A gap in a row of at least three same-class elements (B1), or the mirror of an intact element across the facade's axis (B2) |
| **C** none | No evidence in this photograph | Every other damaged pixel |

| View | Tier A | Tier B | Tier C |
|---|---|---|---|
| Restoration (Venice Charter) | completed by donor transfer, marked by a tone shift and outline | **not** completed, hatched "position implied" | not completed, hatched "no evidence" |
| First construction | completed by donor transfer | completed by donor transfer | not completed (the generative fill is off) |

The restoration view stops where conjecture begins, as the Venice Charter asks; tier B is an inference
about the building, not a surviving piece of it.

## Example images

None are included. The only test photographs (`s01_test_church`, `s02_test_arch`) have an unknown source
and licence, so neither they nor images derived from them are in the repository (`data/README.md`). Run
the pipeline on your own photograph to see the outputs listed under "Outputs".

## Results (n = 1 building)

All figures are for `s01_test_church` (628 x 904 px) and come from the notes in `docs/rnd/`. The
thresholds were tuned on the same building they are reported on.

Detection (`docs/rnd/RND-001-detection-baseline.md`), raw detector output against the expert inventory of
23 elements, IoU >= 0.5:

| | Value |
|---|---|
| Recall, class-aware (after the building's pilaster/column merge) | 52 % (12 of 23) |
| Recall, class ignored | 57 % |
| Precision on the classes the inventory covers | 26 % |

Raw detection is not good enough on its own; the inventory used below was corrected by the expert in the
inventory editor.

Synthetic damage (`docs/rnd/RND-002-evidence-baseline.md`): intact elements are hidden (mask plus black
paint) and the pipeline is asked to bring them back; 28 cases; recovered = a tier A or B target of the
same class at box IoU >= 0.5.

| Case | Cases | Expected | Result |
|---|---|---|---|
| Partial (top half hidden) | 12 | A | 9 recovered |
| Full (whole element hidden) | 12 | B | 1 recovered |
| Plain wall | 4 | C | 3 left C; the restoration view left all 4 open |
| Pixels changed outside the damage mask | 28 | 0 | 0 |

Real damage of s01: 19 targets (5 A, 2 B, 12 C). The restoration view completes 5 targets (4,971 px);
the first-construction view 7 (11,367 px); 79.5 % of the damaged pixels have no evidence and stay open.
Of the 10 reconstruction hypotheses an expert approved in an earlier rule-based experiment on the same
photograph, the pipeline finds 4.

## Install

Windows or Linux, an NVIDIA GPU with 6 GB (developed on an RTX 2060), conda. The package versions are
pinned and must not be changed (torch 2.4.0+cu121, xformers 0.0.27.post2, diffusers 0.20.2, transformers
4.33.2, numpy 1.26.4, opencv-python 4.8.1.78). Details: [`setup/SETUP.md`](setup/SETUP.md).

```bash
cd setup
conda env create -f environment.yml
conda activate ImageToRender_GPU
pip install -r torch_install.txt      # torch + xformers from the cu121 index
pip install -r requirements.txt
cd ..
python tools/check_env.py             # checks versions and CUDA
python -m unittest discover tests     # 130 tests, no GPU needed
```

Model weights are not in the repository. The detection and evidence models (OWL-ViT, SAM, DINOv2-small;
IDs in `config.yaml`) are downloaded to the Hugging Face cache on first use. The generative models
(`config.yaml` › `models`) are read only from `models.cache_dir` and are never downloaded during a run;
set that path for your machine, or leave the generative fill off.

## Use

1. Put the building in `data/structures/<id>/`: `image.png` (long side at least 1500 px recommended),
   `SOURCE.md` (source and licence), optionally `structure.yaml` for settings that differ for this
   building (`data/README.md`).
2. Draw the damage mask (white = damaged or missing):

   ```bash
   python tools/mask_editor.py --structure <id>
   ```

3. Run the whole chain, both views, with the interactive review:

   ```bash
   python run.py --structure <id>
   ```

   Every run writes one folder `outputs/<id>/<timestamp>/` with all files, `run.log` and `run_meta.json`
   (settings, package versions, git commit, step times).

4. Because the detector misses about half of the elements, correct the inventory in the expert editor
   and continue from the damage step:

   ```bash
   python tools/editor_adapter.py edit --run outputs/<id>/<timestamp>
   python tools/refine_masks.py --run outputs/<id>/<timestamp>     # SAM masks for boxes drawn in the editor
   python run.py --structure <id> --from damage --no-review
   ```

   `--from <step>` copies the files of the latest run of that building and starts at that step;
   `--view restoration` or `--view first_construction` renders one view only.

Pixel settings (outline width, hatch spacing, minimum region size, ...) were set on a 904 px image and are
scaled automatically to the photograph's size (`config.yaml` › `scale`).

### Outputs

| File | Content |
|---|---|
| `restoration.png`, `first_construction.png` | the two views, with markings and legend |
| `restoration_source_map.png`, `first_construction_source_map.png` | every changed pixel coloured by tier, with its target ID |
| `first_construction_conjecture_overlay.png` | generative pixels (none while the fill is off); always delivered with the first-construction view |
| `provenance.json` | one entry per target: tier, donor, similarity, confidence, alternatives, method, pixels per view |
| `evidence_overlay.png`, `evidence_sheet.png` | donor -> target arrows; per target the fragment beside the chosen donor and alternatives |
| `damage_overlay.png` | intact (green) and fragmented (red) elements |

### Evaluation

```bash
python tools/synthetic_damage.py --run outputs/<id>/<timestamp>          # cases from a finished run
python evaluation/evaluate.py completion --cases outputs/synthetic/<id>/<timestamp>
python evaluation/evaluate.py grid --cases outputs/synthetic/<id>/<timestamp> \
    --grid evidence.min_similarity=0.60,0.80 --grid evidence.size_tolerance=0.35,0.50
python evaluation/evaluate.py detection --pred <run>/elements.json --ref <inventory.json>
```

## Limitations

- **One building.** Everything was developed, tuned and reported on one photograph of one building
  (n = 1). A second building was planned and dropped for this stage.
- **Small photograph.** The test image is 628 x 904 px, below the 1500 px the method asks for; narrow
  elements are only 20 px wide.
- **Raw detection is weak** (52 % recall); a usable result needs the expert's corrections.
- **Tier B rarely fires.** It needs regular rows or a clear symmetry axis; on the test building it
  recovers 1 of 12 fully hidden elements, and it once implied a pilaster on plain wall.
- **Donor transfer is a 2D copy.** No perspective correction; alignment is a small shift and scale
  search, which reaches its bound on several pilasters, and arch heads come back as thin rings.
- **The generative fill is off.** One ControlNet inpaint pass for tier C was implemented and tried once
  (`docs/rnd/RND-003-generative-fill.md`): it stays within 6 GB and changes only the open area, but it
  invents architecture (a frieze, an entrance portal) and was judged not usable. The first-construction
  view therefore shows tier C hatched and unfilled.
- **Thresholds.** Only `evidence.min_similarity`, `evidence.size_tolerance` and
  `evidence.tier_b.row_tolerance` were chosen with the synthetic test; every other threshold in
  `config.yaml` is an uncalibrated initial guess and is marked so.

## Repository

| Path | Content |
|---|---|
| `run.py`, `config.yaml` | entry point; every setting and threshold |
| `i2r/` | the six steps: `detect`, `review`, `damage`, `evidence`, `complete`, `render`; `runio` for run folders and settings |
| `tools/` | `mask_editor`, `editor_adapter` (+ the unchanged `inventory_editor`), `refine_masks`, `synthetic_damage`, `check_env` |
| `evaluation/evaluate.py` | detection, completion and grid evaluation |
| `tests/` | unit and step tests (no GPU) |
| `docs/` | `ARCHITECTURE.md` (design), `TASKS.md` (task list and status), `EXPERIMENTS.md` (earlier experiments), `rnd/` (research notes) |
| `archive/` | every earlier script, kept unchanged and described in `archive/README.md` |
| `data/structures/<id>/` | one folder per building; only `SOURCE.md` and `structure.yaml` are tracked |

## Licence

Not specified yet.
