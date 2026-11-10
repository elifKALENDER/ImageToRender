# RND-001 — Detection baseline

> **Türkçe özet.** Yeni tespit adımı (OWL-ViT + SAM, hiçbir ayar değiştirilmeden) uzmanın onayladığı
> 23 elemanlı listeyle karşılaştırıldı. Sınıfı doğru ve kutusu yeterince örtüşen eleman oranı %30.
> Sınıf yok sayılınca %57'ye çıkıyor, çünkü pilasterlerin çoğu bulunuyor ama "sütun" diye etiketleniyor.
> Tek yapı, tek fotoğraf; sonuçlar genellenemez.

| | |
|---|---|
| Date | 2026-11-10 |
| Task | T10 (`docs/TASKS.md`) |
| Code | commit `57c3330`, branch `refactor/structure` |
| Run | `outputs/s01_test_church/20261110_140325` (local) |
| Reference | `outputs/restore_20260731_162315/expert_validated_inventory.json` (locked expert inventory, 23 elements) |
| Command | `python evaluation/evaluate.py detection --pred <run>/elements.json --ref <reference>` |
| Structures | **n = 1** (s01_test_church); s02_test_arch has no reference, only a visual check |
| Tuning | **None.** `config.yaml` detector values are the archived Stage 1 values; nothing was changed to improve these numbers |

## Question

How much of the expert's final inventory does zero-shot detection find on its own, before any review or
manual addition?

## Method

- Detector: multi-scale OWL-ViT base (7 views) with 29 text prompts in 12 classes, class-aware NMS,
  SAM ViT-B masks. Peak VRAM 1.73 GB.
- A prediction matches a reference element when their classes are compatible and box IoU is at least
  0.5. Matching is one-to-one, greedy by IoU.
- Class alias: the reference class `opening` matches a predicted `door` or `window`.
- Precision is computed only for predicted classes the reference covers (arch, pilaster, cornice, base,
  door, window). Predictions of column and roof cannot be judged against this reference.

## Results

| Overall | Value |
|---|---|
| Reference elements | 23 |
| Predicted elements | 48 |
| Matched (class-aware) | 7 |
| **Recall (class-aware)** | **30 %** |
| Recall, class ignored | 57 % |
| Recall at IoU 0.3 (class-aware) | 35 % |
| Precision on classes the reference covers | 18 % (38 predictions judged) |
| Mean IoU of matched pairs | 0.77 |

| Reference class | Reference | Found | Recall | Mean IoU |
|---|---|---|---|---|
| arch | 8 | 3 | 38 % | 1.00 |
| base | 3 | 0 | 0 % | — |
| cornice | 2 | 2 | 100 % | 0.65 |
| opening | 2 | 1 | 50 % | 0.51 |
| pilaster | 8 | 1 | 12 % | 0.54 |

| Predicted class | Predicted | Correct | Precision |
|---|---|---|---|
| arch | 10 | 3 | 30 % |
| base | 10 | 0 | 0 % |
| column (not in reference) | 9 | — | — |
| cornice | 8 | 2 | 25 % |
| door | 4 | 0 | 0 % |
| pilaster | 4 | 1 | 25 % |
| roof (not in reference) | 1 | — | — |
| window | 2 | 1 | 50 % |

**Classes with no detection at all:** eave, jamb, console, parapet (s01). On s02: jamb, console.

Other observations on s01:

- Scores are very low: median 0.015, maximum 0.055. A score threshold cannot separate right from wrong
  boxes at this level.
- 4 boxes cover more than 15 % of the image each (labelled arch, base, cornice). They span the whole
  building and are wrong.
- 7 of 48 elements are "fallback" boxes (a prompt's single best box kept although below the floor).

## Findings

1. **Pilasters are found but called columns.** Five of the eight reference pilasters have a predicted
   box with IoU 0.60 to 1.00, all labelled `column`. This single confusion explains most of the gap
   between 30 % and 57 %. Architecturally the distinction (engaged versus free-standing) is subtle at
   this resolution, and OWL-ViT does not make it.
2. **Fragmented elements are mostly missed.** Of the 11 fragmented reference elements, 1 is matched. Of
   the 12 intact ones, 6 are matched. This matters because fragments are exactly the tier A targets.
3. **Base courses are not found** (0 of 3). Predicted `base` boxes land elsewhere.
4. **Localisation is good when the class is right** (mean IoU 0.77).
5. **The new fine-grained classes return nothing on s01** (eave, jamb, console, parapet). On s02 a few
   eave and parapet boxes appear; by visual inspection they are wrong (the parapet boxes sit on the roof
   apex and on vegetation).
6. **s02 visual check (no reference):** the conical roof, the lower arched doorway and two lower arches
   are plausibly right; several boxes again span the whole building.

## Limits of this measurement

- **One building, one photograph, 628 x 904 px.**
- **The reference is not independent of the detector.** Five reference boxes are unmodified detector
  boxes from the July runs of a closely related model and settings (`source` = `ai_known_prototype` or
  `ai_recovered_observed`, `expert_action` = `unmodified`). The three arches matched with IoU 1.00 are
  such boxes. This inflates recall for those elements.
- **The reference is not exhaustive.** The expert annotated the lower arcade; for example the drum
  pilasters and the upper drum window are not in the inventory. Correct detections there count as
  false positives, so precision is underestimated.
- **Five reference classes only.** Seven of the twelve configured classes cannot be evaluated.

## Implications for later tasks

- Review (T11) and the inventory editor (T12) are not optional on this evidence; completion quality
  would otherwise be capped near 30 %.
- A class-merging rule for pilaster and column, or prompts that separate them, is the cheapest
  improvement. Any such change must be re-measured with this command, against this baseline.
- Removing boxes larger than a set share of the image would remove four clear errors.
- Neither change was made in this task, by design.

## RND questions raised

- Per-class zero-shot recall on heritage facades, over more buildings (R2 in `docs/TASKS.md`).
- Do frozen DINOv2 or CLIP features separate pilaster from column on crops better than OWL-ViT's text
  prompts? This connects to the existing frozen-features question (R1).

---

## Re-measurement after detector clean-up (T12b)

| | |
|---|---|
| Date | 2026-11-10 |
| Baseline code | commit `57c3330` |
| Clean-up code | the commit that adds this section (`fix(T12b)`) |
| Run | `outputs/s01_test_church/20261110_165746` (local) |
| Structures | **n = 1** (s01_test_church) |

Two changes, nothing else:

1. **Oversized boxes dropped.** `detect.max_element_area_ratio: 0.15`. On the baseline no box above
   6.6 % of the image matched any reference element, and the largest reference element covers 2.7 %.
   0.15 is a cautious cut that removes only the boxes spanning the whole building. A class can override
   it (`classes.<name>.max_area_ratio`), for example for roofs.
2. **`column` merged into `pilaster` for this building only**, in
   `data/structures/s01_test_church/structure.yaml`. The main `config.yaml` keeps the two classes
   apart. The merge runs before NMS, so two boxes on the same support become one.

| Metric | Baseline | After clean-up |
|---|---|---|
| Recall (class-aware) | 30 % (7 of 23) | **52 % (12 of 23)** |
| Recall, class ignored | 57 % | 57 % |
| Recall at IoU 0.3 (class-aware) | 35 % | 57 % |
| Precision on classes the reference covers | 18 % (38 judged) | 26 % (46 judged) |
| Mean IoU of matched pairs | 0.77 | 0.76 |
| Oversized candidates dropped | — | 9 |

| Reference class | Baseline recall | After clean-up |
|---|---|---|
| arch | 38 % (3 of 8) | 38 % (3 of 8) |
| base | 0 % (0 of 3) | 0 % (0 of 3) |
| cornice | 100 % (2 of 2) | 100 % (2 of 2) |
| opening | 50 % (1 of 2) | 50 % (1 of 2) |
| pilaster | 12 % (1 of 8) | **75 % (6 of 8)** |

### Reading the result

- The whole gain comes from the class merge: pilaster recall rises from 1 to 6 of 8. Class-aware
  recall now nearly equals class-agnostic recall (52 % against 57 %), so class confusion is no longer
  the main loss on this building.
- Dropping oversized boxes did not raise recall. It removed wrong boxes; the freed places under the
  48-element limit were taken by other low-score candidates, so the element count stayed at 48 and
  precision improved only from 18 % to 26 %.
- **Unchanged weaknesses:** arches 3 of 8, bases 0 of 3, and fragmented elements. These are missed
  boxes, not wrong labels, and no setting changed here addresses them. They remain work for the expert
  editor (T12) or for a later instance-recovery step.

### Caution

Both changes were chosen by looking at this one building and are reported on the same building. The
pilaster merge is an architectural judgement about s01 (engaged supports), not a general rule; it is
deliberately not in the main configuration. On s02 (no reference) the size limit dropped the boxes
that spanned the whole building; whether it also dropped a correct large element there was checked by
eye only.
