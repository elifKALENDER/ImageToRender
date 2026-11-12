# Failure analysis — s01 first-construction output with the generative fill on

> **Türkçe özet.** Yazarın numaraladığı 8 nokta kod ve koşu kayıtlarıyla incelendi; kod değiştirilmedi.
> İncelenen çıktı, yapay zekâ dolgusu **açık** koşunun ilk inşa görünümü (`20261112_142811`): 3, 4, 5, 6,
> 8 ve soldaki 7 tamamen üretken dolgu. En önemli bulgu bir etiket hatası: üretken pikseller donör gibi
> çerçeveleniyor ve göstergede "kanıt yok" satırı taranmış piksel olmadan yazılıyor. 5 numara hasar
> maskesinin zemine taşmasından, 1 numara envanterdeki kemer kaydının ayakta duran kemeri değil yıkıntıyı
> kapsamasından ve hizalamada ret eşiği olmamasından, 2 numara B1 yuva kuralından geliyor. 3 ve 8
> belirsiz bırakıldı. Sonunda beş düzeltme görevi önerildi (T26-T30); hiçbiri uygulanmadı.

| | |
|---|---|
| Date | 2026-11-12 |
| Requested by | the author (visual review of one output, 8 numbered points) |
| Assessed image | `outputs/s01_test_church/20261112_142811/first_construction.png` (generative fill **on**, the T21 trial judged not usable in RND-003) |
| Reference | `outputs/s01_test_church/20261112_141348` (same settings, generative fill off) |
| Code | `main` = `refactor/structure` at `32bb654` (v0.1.0 + release note) |
| Read | `docs/ARCHITECTURE.md`, `docs/TASKS.md`, `i2r/*.py`, `config.yaml`, the two run folders, the locked expert inventory `outputs/restore_20260731_162315/expert_validated_inventory.json`. `docs/BRIEF.md` does not exist. |
| Structures | n = 1 (s01_test_church, 628 x 904 px) |
| Changes made | none (analysis only) |

Statements marked **(guess)** are my reading, not something a file or the code states.

## 1. What the author marked

| No. | Author's note | Author's confidence |
|---|---|---|
| 5 | Grass and rubble in the input; the output has a flat beige patch. Ground treated as architecture. | clear error |
| 1 | The arch of the right bay stands in the input; a donor arch lies over it, not aligned to it. | probably an error |
| 2 | One column in the input; the system put a second one beside it. | doubtful |
| 3, 8 | Unclear. 1, 2, 3 and 8 together might be one window composition; to be checked against the 1920 Kalantar photographs. Not to be counted as errors. | undecided |
| 4, 6 | Place may be right; tone and texture are wrong. | error in appearance |
| 7 | Patch edges are visible. | error in appearance |

## 2. Which pixels each point is made of

Read from `first_construction_source.png` (label image) and `completion.json` of both runs, in a
13 x 13 px window at each marker. Coordinates are image pixels.

| No. | Pixel (x, y) | Generative on (142811) | Generative off (141348) | Evidence record there |
|---|---|---|---|---|
| 1 | 394, 371 | generative 139 px, donor `t_003` 30 px | photograph 139 px (hatched), donor `t_003` 30 px | `t_003` tier A, element `recovered_arch_001`, donor `arch_005` flipped |
| 2 | 343, 499 | generative 117 px, donor `t_012` 52 px | photograph 117 px, donor `t_012` 52 px | `t_012` tier B (B1 slot); beside `recovered_pilaster_046` (tier C) |
| 3 | 299, 297 | generative 169 px | photograph (hatched) | only the tier C region `t_014` |
| 4 | 324, 629 | generative 169 px | photograph (hatched) | `expert_base_023`, tier C (no intact base exists) |
| 5 | 145, 694 | generative 169 px | photograph (hatched) | `expert_base_022`, tier C; tier C region `t_015` |
| 6 | 557, 667 | generative 169 px | photograph (hatched) | only the tier C region `t_016` |
| 7 left | 82, 589 | generative 169 px | photograph (hatched) | tier C region `t_017` = box [70, 562, 97, 616] |
| 7 right | 240, 587 | donor `t_006` 169 px | donor `t_006` 169 px | `t_006` tier A, `recovered_pilaster_022`, donor `pilaster_011` |
| 8 | 430, 518 | generative 169 px | photograph (hatched) | only `t_014`; the B1 slot `t_013` [400, 410, 423, 618] is next to it |

So 3, 4, 5, 6, 8 and the left 7 are entirely the generative pass. With the fill off, those places are
hatched "no evidence" and unchanged. 1 and 2 are donor transfer surrounded by generated pixels. The right
7 is donor transfer.

## 3. Findings

### F0. The legend and the outline misreport the generative pixels (highest priority)

*Certain; introduced in T21.*

- `i2r/render.py` `mark_view`: `changed = labels > 0` includes the generative label, so generated pixels
  get the same tone shift and white inner outline as donor-transferred ones.
- `view_legend` has no item for generative pixels and appends "hatched: no evidence" unconditionally.
- Effect in the assessed image: 88,991 conjectural pixels sit inside an outline whose legend says
  "completed from a donor (outlined)", and the legend shows a hatch that occurs nowhere
  (`open_no_evidence_px` = 0 in `provenance.json`).
- The separate `first_construction_conjecture_overlay.png` and `provenance.json` › `generative_fill` are
  correct, but the main image, which is what a reader looks at, is not.
- Related trap: the generative trial is the newest run folder of s01, so "the latest output" is the one
  with the fill on, although `config.yaml` has it off.

### No. 5 — ground filled as architecture

*Certain.*

- The damage mask is the only statement of where the building is missing. Its lower-left lobe lies on
  grass and rubble (damage rows reach y = 723; the lowest element box ends at y = 714). The mask was drawn
  by hand (version 1 by the author; versions 2 and 3 edited by the assistant, see
  `data/structures/s01_test_church/SOURCE.md`).
- `i2r/evidence.py` `tier_c_regions`: every damage pixel outside A/B boxes becomes a tier C region; region
  `t_015` = [89, 493, 214, 724] covers the lobe. Nothing distinguishes facade from ground.
- `i2r/complete.py` `open_area` returns `damage & (labels == 0)`; `generative_fill` fills all of it and
  `paste_area` pastes it. No ground, sky or "outside the facade" concept exists anywhere in the code; I
  searched `i2r/` and `config.yaml` and found none.
- The prompt asks for "the masked missing architectural portions", so the model paints masonry or a floor
  wherever the mask is.
- With the fill off the lobe is hatched and unchanged: the donor path never touches it.

### No. 1 — donor arch over the standing arch

*Probably an error, as the author says. Three causes stack; the first is upstream of the transfer.*

1. **The inventory record is not the standing arch (see §9).** `recovered_arch_001` has the box
   [391, 341, 468, 408], widened by the expert from the detector's [413, 358, 468, 408] to the left and
   up, into the collapse. Its SAM mask (`element_masks/recovered_arch_001_sam.png`, 2,018 px, made by
   `tools/refine_masks.py`) covers the rubble core inside the collapse and part of the tympanum; 80 % of
   it lies in the damage mask. The standing arch ring is outside the damage mask and outside this mask.
   The record is labelled fragmented, so the pipeline tries to complete it.
2. **The alignment ends on its search bounds.** `complete.align` chose scale 1.06 (the largest of
   `refine_scales` [0.94, 1.0, 1.06]) and dy = 7 (the limit, `refine_shift_frac` 0.10 x box height 67).
   NCC rose from 0.33 to 0.71. An optimum on the bound means the true optimum is probably outside the
   search.
3. **No acceptance test.** `complete.transfer` uses whatever `align` returns; there is no minimum score
   and no check that the donor continues the surviving piece. The donor (`arch_005`, flipped, 84 x 65 px,
   mapped box to box) is then blended only inside the damage mask (`fill = mask & damage_w`), so only the
   strip that falls in the mask appears: a thin ring segment beside the standing arch that does not meet
   it.

The generative pass then draws its own arch forms around that segment.

Of the 5 tier A targets on the real damage, 4 end on a bound:

| Target | Element | Box w x h | Shift limit | dx, dy, scale | NCC before -> after | On bound |
|---|---|---|---|---|---|---|
| t_001 | pilaster_013 | 27 x 161 | 3, 16 | -2, 13, 1.06 | 0.26 -> 0.73 | scale |
| t_002 | recovered_arch_003 | 99 x 59 | 10, 6 | -3, 4, 1.00 | 0.54 -> 0.85 | none |
| t_003 | recovered_arch_001 | 77 x 67 | 8, 7 | 4, 7, 1.06 | 0.33 -> 0.71 | scale, dy |
| t_006 | recovered_pilaster_022 | 24 x 203 | 2, 20 | -2, -4, 1.06 | 0.34 -> 0.64 | scale, dx |
| t_007 | expert_pilaster_017 | 18 x 158 | 2, 16 | 2, 8, 1.00 | 0.29 -> 0.69 | dx |

### No. 2 — second column beside the surviving one

*Doubtful, as the author says; the mechanism is certain, the architectural truth is not.*

- The surviving fragment `recovered_pilaster_046` (box [324, 406, 349, 505], 99 px high) is tier C:
  `evidence.filter_size` rejects every pilaster donor (about 200 px high) because the box covers only the
  standing part ("no usable donor of class 'pilaster': 4 not intact, 3 size").
- `evidence.tier_b1_rows` then implies a slot `t_012` at [343, 410, 366, 619], 6 px into the fragment's
  box. `slot_is_free` accepts it: the overlap is 12 % of the slot and `slot_free_max_overlap` is 0.30.
- `rule_first_construction` completes tier B, so a whole pilaster from `pilaster_011` is placed next to
  the fragment. The restoration view does not complete it (hatched "position implied").
- The same rule made the second slot `t_013` at point 8's left. Both slots were already recorded as
  doubtful in T15/T18. In the synthetic test B1 also put a slot on plain wall (`background__bg01`).
- Whether a pilaster pair stood here is not something the code can know **(guess: the pilasters of this
  arcade are paired, so a second shaft is plausible, but the slot position comes from row spacing, not
  from the pair rhythm)**.

### No. 3 and No. 8 — undecided, not counted as errors

- Both are 100 % generative pixels. The system's own record says there is no evidence for them in this
  photograph (tier C). That is a statement about provenance, not about correctness.
- If 1, 2, 3 and 8 form a window composition, the pipeline did not derive it: the frieze at 3 and the
  opening at 8 come from the diffusion model, the two shafts from B1.
- Checking against the 1920 photographs is outside the pipeline (§11).

### No. 4 and No. 6 — tone and texture

*Certain as a mechanism.*

- `complete.generative_fill` generates at 512 x 504 px (SD 1.5, strength 0.96) and resizes back;
  `paste_area` copies the result into the open area with no colour matching and no feathering.
  `complete.match_colour` and `feather_alpha` exist only on the donor path (`transfer`).
- The tone shift of `render.mark_view` (`restoration.tone_shift` 0.08) is then added on top, because of F0.
- **(guess)** Most of the texture difference is the model itself (smooth, lighter stone), not the paste;
  the cheapest check is in §5.

### No. 7 — visible patch edges

*Certain; three sources.*

1. Two rectangles were added to the damage mask in version 2 (x 70-96, y 562-615 and x 227-254,
   y 555-617; `SOURCE.md`). A rectangular mask gives a rectangular patch. The left 7 is the first one,
   filled by the generative pass; the right 7 is the second, filled by donor `t_006`.
2. `render.mark_view` draws a 1 px outline and a tone shift around every label on purpose (T18 decision:
   additions are marked in both views).
3. The generative paste has a hard edge (see No. 4/6).

## 4. Intermediate results: what is saved and what is missing

Saved in every run folder today:

| File | Content |
|---|---|
| `evidence.json` | per target: tier, chosen donor, donor and target boxes, flip, similarity, confidence, every candidate with its score, every rejected donor with the reason, `tier_b_diagnostics` (rows, axis) |
| `completion.json` | per target and view: alignment (dx, dy, scale, NCC before and after, visible pixels), colour shift, pixels painted and changed, damage pixels left unfilled in the box; generative record (crop box, size, seed, settings, models, VRAM) |
| `provenance.json` | the two merged per target |
| `evidence_overlay.png`, `evidence_sheet.png` | donor -> target arrows; fragment beside donor |
| `<view>_source.png`, `*_source_map.png` | label of every changed pixel |
| `generative_raw.png`, `generative_mask.png`, `generative_depth.png`, `generative_lineart.png` | the generative pass |
| `run.log`, `run_meta.json`, `config.yaml` | log, timings, effective settings |

Missing (to be written in `complete.transfer` into `completion.json` unless stated otherwise):

1. **Final placement rectangle** of the donor in image pixels (today derivable from box, scale, dx, dy).
2. **On-bound flags**: which of scale, dx, dy ended on the limit of the search, and the limits used.
3. **Seam agreement**: how well donor and photograph agree in a band just outside the damage mask around
   the filled pixels (the donor is known there but not painted). This is the number that would have
   caught No. 1.
4. **Share of the donor used**: usable donor pixels, pixels that fell inside the mask, pixels discarded
   because they fell on surviving stone.
5. **Per-target crops**: `targets/<target_id>_<view>.png` with photograph, damage mask, donor as placed,
   result (the comparison made by hand for this analysis).
6. In `evidence.json`: for each B1 slot the nearest same-class element and the distance to it (would have
   shown the slot 6 px into `recovered_pilaster_046`).
7. In `run_meta.json`: whether the generative fill ran, so a resumed or "latest" run can be told apart.

## 5. Ranking and the cheapest check for each

| Rank | Finding | Likelihood that the stated cause is right | Cheapest check (one image, one setting) |
|---|---|---|---|
| 1 | F0 legend | certain (read in code and in `provenance.json`) | none needed |
| 2 | 5 ground | certain | open `20261112_141348/first_construction.png`: the lobe is hatched and unchanged |
| 3 | 7 edges | certain | re-render only with `restoration.outline_px: 0`, `tone_shift: 0`: outline gone, rectangles of the mask remain |
| 4 | 4, 6 tone | certain as mechanism; share of model versus paste is a guess | compare `generative_raw.png` with the pasted result in the same crop |
| 5 | 1 arch | probable: record and mask verified; the weight of each of the three causes is a guess | overlay `recovered_arch_001_sam.png` and its box on the photograph (done for this note: the mask is on the rubble, the ring is outside it) |
| 6 | 2 column | mechanism certain, correctness unknown | `evidence.tier_b.slot_free_max_overlap: 0.10`, evidence step only: does `t_012` disappear? |
| 7 | 3, 8 | undecided | not checkable in code; needs the archive photographs |

## 6. Does the system ever say "no evidence"? Can it defer to the expert?

- It does, when the generative fill is off: 88,991 px (79.5 % of the damage) are hatched "no evidence" in
  both views and 12 of 19 targets are tier C with a written reason.
- It does not in the assessed image, because the generative pass fills every open pixel and F0 then hides
  the difference.
- Deferring is possible with what exists: every target has a tier, a confidence, a reason and a
  provenance entry, and `i2r/review.py` already asks the expert about low-score elements. What is missing
  is (a) a per-target status "guess, needs expert approval" between "completed" and "not completed",
  (b) a rule that sets it (tier B always; tier A when alignment is on a bound or below the threshold;
  generative always), (c) a marking for it in the views, and (d) a place to record the expert's decision
  per target (the review step works on elements, not on targets). This is the content of T28 and T26.

## 7. Coverage against accuracy

Measured on s01, share of the 100,358 damage pixels changed in the first-construction view:

| Setting | Filled | Of which with a named source in this photograph |
|---|---|---|
| Restoration view | 5 % (4,971 px) | all (tier A donors) |
| First construction, fill off | 11 % (11,367 px) | all (tier A 4,971 px; tier B 6,396 px from a pattern) |
| First construction, fill on | 100 % | 11 %; 89 % generated |

What accuracy costs:

- Removing the generative pass: back to 11 %.
- Rejecting poorly aligned tier A targets: between 1 and 3 of the 5 would be withheld **(guess; depends
  on the threshold, §8.3)**, that is up to about 3,500 of the 4,971 px.
- Tightening the slot rule: the two B1 pilasters (6,396 px) would be withheld or flagged.
- A strict setting could therefore leave about 1,400 px (1.4 %) completed on this photograph **(guess,
  same assumption)**.

The two goals, "fill the gaps" and "what is filled is right", cannot both be met with this photograph:
79.5 % of the damage has no counterpart in the image. The workable middle is three visible classes instead
of two: completed with evidence; proposed as a guess for the expert (tier B, weak alignment, generative,
each marked differently); open. Coverage then stays visible without being presented as fact.

## 8. Alignment: threshold, bound flag, calibration, resolution

### 8.1 What happens today

`complete.align` searches 3 scales and shifts up to 10 % of the box, keeps the best NCC over the visible
fragment pixels, and returns it. Nothing rejects a result. On the real damage 4 of 5 targets end on a
bound (§3, No. 1). On the 9 recovered synthetic partial cases **all 9** end on a bound:

| Case | NCC before -> after | dx, dy, scale | On bound | SSIM over hidden px |
|---|---|---|---|---|
| partial__arch_004 | 0.85 -> 0.92 | -2, 2, 0.94 | scale | 0.13 |
| partial__arch_005 | 0.75 -> 0.85 | 0, 1, 0.94 | scale | 0.21 |
| partial__arch_007 | 0.47 -> 0.88 | 2, -6, 0.94 | scale, dy | 0.29 |
| partial__recovered_arch_016 | 0.48 -> 0.86 | 3, -3, 0.94 | scale | 0.20 |
| partial__recovered_arch_065 | 0.75 -> 0.89 | -2, -6, 1.00 | dy | 0.05 |
| partial__pilaster_011 | 0.08 -> 0.22 | 1, -13, 0.94 | scale | 0.05 |
| partial__pilaster_010 | -0.02 -> 0.17 | -2, -6, 1.06 | scale, dx | 0.00 |
| partial__recovered_pilaster_008 | 0.07 -> 0.58 | 2, 4, 1.06 | scale, dx | 0.22 |
| partial__expert_cornice_020 | 0.14 -> 0.40 | -16, -3, 0.94 | scale, dx | 0.07 |

Two things follow. A wider search alone is not the answer: when every case is on a bound, the box-to-box
mapping itself is off (one isotropic scale for donors whose boxes differ in proportion) **(guess for the
cause; the fact is the 9 of 9)**. And NCC alone does not predict fidelity: `recovered_arch_065` has NCC
0.89 and SSIM 0.05.

### 8.2 Proposed checks (no code written)

1. **On-bound flag.** Record per target which parameters ended on the limit. A flagged target is not
   failed by that alone; it is reported and counts toward "guess".
2. **Acceptance threshold on the alignment score**, `complete.min_alignment_score`. Below it the target is
   not completed in the restoration view and is marked "guess, needs expert" in the first-construction
   view, with the reason in `completion.json`.
3. **Seam test**, `complete.min_seam_agreement`: agreement between donor and photograph in a band of
   surviving pixels around the fill. This targets No. 1 directly: a donor that does not continue the
   standing piece fails it even when the NCC over the box is fair.
4. Only then widen the search, and judge by whether optima move off the bounds, not by the score.

### 8.3 How to calibrate the threshold

The synthetic partial cases are the only data with ground truth (the hidden half is known).

1. The present fidelity number is unusable for this: SSIM is averaged over all hidden pixels, and pixels
   the donor did not fill are black, so it mostly measures coverage. First add a fidelity measured **only
   on pixels that were both hidden and filled** (NCC or SSIM against the original).
2. For every partial case record alignment score, seam agreement, on-bound flags and that fidelity.
3. Plot fidelity against each candidate measure. Take as threshold the value below which fidelity is no
   better than an unaligned placement (scale 1, no shift) of the same donor. That baseline is already
   computed (`score_before`).
4. Report it with the count: 9 cases, 3 classes, one building. With n = 9 the threshold is a provisional
   value, to be written in `config.yaml` as "chosen on s01, 9 cases", like the T20 values.
5. Check it against the real damage: it must flag `t_003` (No. 1). If it cannot separate `t_003` from
   `t_002` (the one target not on a bound, NCC 0.85), say so and keep the on-bound flag as the only test.

### 8.4 At what resolution does the pipeline work, and would a larger input help?

- Everything works at the native size of the input, here 628 x 904 px. `detect` passes views to OWL-ViT
  and SAM, whose processors resize internally; `evidence` resizes each element crop to 224 x 224 for
  DINOv2; `complete` transfers, aligns and blends in native pixels with 1 px shift steps; the generative
  pass works on a 521 x 512 crop resized to 512 x 504. Pixel settings are scaled to the long side
  (`config.yaml` › `scale`, reference 904).
- A **real** larger photograph would help the alignment in two ways: the shift step becomes finer (today
  1 px is 4 to 6 % of the width of an 18 to 27 px pilaster, and the three pilasters have a horizontal
  shift range of only +/- 2 or 3 px), and NCC is computed over more pixels (today 570 to 2,309).
- It would not fix the bound problem, the box-to-box scale, a wrong inventory record (No. 1) or the
  curvature of the apse.
- **Upscaling does not help.** The 3x Upscayl copy (`s01_test_church_upscaled`) still ended on scale 1.06
  for the same two pilasters and its scores were not higher (0.66 and 0.66 against 0.73 and 0.64).
- Cost: the search grows with the fourth power of the scale factor (range and pixel count both grow); the
  complete step took about 1 s per view on s01 and about 30 s on the 3x copy.

## 9. Who labelled the right arch "fragmented", and is it right?

What the files say:

- Locked expert inventory (`outputs/restore_20260731_162315/expert_validated_inventory.json`, archived
  Stage 3B-v5 editor), record `recovered_arch_001`: `source: ai_recovered_observed` (found by the old
  recovery stage, not drawn by the expert), `ai_status: observed_fragmented` (the status was proposed by
  that stage), `expert_action: bbox_corrected`, `expert_reviewed: true`, `original_bbox_xyxy`
  [413, 358, 468, 408], `bbox_xyxy` [391, 341, 468, 408], `usable_as_prototype: false`, no note.
  So the status came from the old AI stage; the expert reviewed the record, widened the box 22 px to the
  left and 17 px up, and left the status as it was.
- `tools/editor_adapter.py` imported it as `expert_condition: fragmented` (it records a condition only
  for records the expert touched; a corrected box counts).
- `tools/refine_masks.py` replaced the box mask by a SAM mask prompted with the widened box.
- `i2r/damage.py` `assess_elements` computes its own label from mask and damage mask: overlap 0.80,
  status fragmented, reason "overlap". Expert and computed labels agree (this is one of the "23 of 23").

What the photograph shows **(my reading, guess)**:

- The arch ring of the right bay stands complete and lies outside the damage mask.
- The SAM mask of the record is not on that ring. It covers the rubble core inside the collapse (all of
  its 614 px left of x = 413 are in the damage mask) and part of the tympanum under the arch.
- The widened box spans the collapse and the left half of the standing arch.

Conclusion: the record does not describe the standing arch. Either it was meant as the remnant of the
neighbouring collapsed arch, or the box was widened too far. The "23 of 23 agreement" does not test this:
it compares two labels that both follow from the same box and mask. Only the author can say what the
record was meant to be; it is listed as an author item in T28.

## 10. Showing the generative output as an AI guess

Options, none implemented:

1. **Separate deliverable (recommended default).** `first_construction.png` never contains generated
   pixels; open areas stay hatched. The generative result is written as a second file,
   `first_construction_ai_guess.png`, with a banner "AI guess: no evidence in the photograph" across the
   image, not only in the bar.
2. **Inline, marked differently.** Generated pixels get no donor outline and no donor tone shift. They
   get their own marking: a dashed outline in the conjecture colour already used in the overlay
   (magenta), a light diagonal stripe so the area reads as provisional at a glance, and a legend item
   "AI guess (generative), no evidence". The donor legend item stays for donor pixels only.
3. **Three panels.** Photograph | evidence only | evidence + AI guess, in one image, so the guess is never
   seen without the evidence-only state beside it.

A setting `first_construction.generative_display: separate | inline | panels` would choose; `separate`
keeps the main image honest by construction. In every option the legend lists only what occurs in the
image.

## 11. What this data and method cannot solve

- What stood in the collapsed central bay (window, door or wall), the frieze at 3 and the opening at 8:
  the photograph holds no trace. It needs another source (the 1920 photographs, a survey); the pipeline
  can at most show a guess marked as such.
- Continuation of the base and cornice courses: no intact base exists to copy and no course rule exists
  (the three archived hypotheses the new pipeline does not find, RND-002).
- Correct transfer on the curved apse: a box-to-box 2D copy cannot follow the foreshortening.
- Fine alignment of elements 18 to 27 px wide at 628 x 904 px.
- Stone texture and style faithful to this building from SD 1.5 at 512 px (RND-003).
- Any claim beyond this building: n = 1.

## 12. Fix order and proposed tasks

Order given by the author: (a) legend and labels, (b) mask reaching the ground, (c) alignment threshold
and bound check, (d) slot rule, (e) logging. One dependency: (c) needs the on-bound flag and the seam
figure from (e); those two items are included in T28 so the order can stand.

The tasks below were added to `docs/TASKS.md` (Phase 3) on 2026-11-12, as written here; **none is
implemented**. Size: S = under one session, M = one session, L = more than one.

### T26. Honest marking of generative pixels and legend (a)
**What.** In `render.mark_view` treat the generative label separately from donor labels: no donor
outline or tone shift; marking per `first_construction.generative_display` (§10; implement `separate`
and `inline`). `view_legend` lists an item only when its pixel count is above 0, including "hatched: no
evidence". Record in `run_meta.json` whether the generative pass ran.
**Files.** `i2r/render.py`, `i2r/complete.py` (meta), `config.yaml`, `tests/test_render.py`.
**Accept.** (1) A view with a generative label: no pixel of that label is inside a donor outline, and the
legend has an "AI guess" item. (2) A view with 0 hatched pixels has no "hatched" legend line. (3) With
`separate`, `first_construction.png` of a generative run is byte-identical to that of the same run with
the fill off. (4) `restoration.png` of the reference run `20261112_141348` is byte-identical before and
after. Tests for each.
**Size.** S.

### T27. Keep the fill off the ground (b)
**What.** Give the pipeline a notion of "outside the facade". Options to decide when starting: a
per-structure `facade_mask.png` or ground line in `structure.yaml` drawn by the author, or the convex hull
of the element boxes (already used in `tools/synthetic_damage.py`). Damage pixels outside it become a
separate tier C kind `outside_facade`, are never passed to `generative_fill`, and are hatched with their
own legend text. **[needs author]** Redraw or confirm the lower-left lobe of the s01 mask.
**Files.** `i2r/evidence.py`, `i2r/complete.py`, `i2r/render.py`, `config.yaml`, `data/README.md`, tests.
**Accept.** On s01 with the fill on: 0 generated pixels below the facade limit; the lobe at point 5 is
unchanged from the photograph; `provenance.json` lists the excluded area and its pixel count. With the
fill off, outputs change only in the legend text for that area. A test with a synthetic mask that
reaches below the hull.
**Size.** S to M (M if a new mask file and its editor support are chosen).

### T28. Alignment acceptance: bound flag, seam test, threshold, "guess" status (c)
**What.** In `complete.align` and `transfer`: record on-bound flags and the search limits; compute seam
agreement; add `complete.min_alignment_score` and `complete.min_seam_agreement`; a target failing
them gets status `guess_needs_expert`, is not completed in the restoration view, and is completed but
marked as a guess in the first-construction view (same marking family as T26). The seam agreement is item
3 of §4. Add the fidelity over
filled pixels to `evaluation/evaluate.py completion` and calibrate as in §8.3; write the result as
`docs/rnd/RND-004-alignment-acceptance.md`. Only after that, try a wider search and anisotropic scale and
report whether optima leave the bounds. **[needs author]** Decide what `recovered_arch_001` is (§9):
correct its box in the editor, or reject the record.
**Files.** `i2r/complete.py`, `i2r/render.py`, `evaluation/evaluate.py`, `config.yaml`, tests, RND-004.
**Accept.** (1) `completion.json` has `on_bound`, `limits`, `seam_agreement` for every completed target.
(2) The note states the threshold, the 9 cases it came from, and whether it flags `t_003`; if it cannot
separate good from bad, it says so. (3) No tier A target that ends on a bound is presented as plain
"completed" without the flag. (4) Preservation stays 0; tests cover a donor deliberately misplaced by 10
px (must be flagged) and a correct one (must pass).
**Size.** M to L.

### T29. Slot rule beside a surviving fragment (d)
**What.** `evidence.slot_is_free` should not call a slot free when a same-class fragment stands beside
it. Proposal: test against the fragment's box extended along the element's long axis to the row's height
(the fragment box covers only the standing part), or require a minimum centre distance as a share of the
row spacing. Give `recovered_pilaster_046`-like fragments a second chance as tier A with the size filter
applied to width only when the element is cut by the damage mask. All tier B targets carry
`guess_needs_expert` in the first-construction view (see T28).
**Files.** `i2r/evidence.py`, `config.yaml`, `tests/test_evidence_b.py`.
**Accept.** On s01: `t_012` is either not produced or flagged as a guess, with the reason logged; the
nearest same-class element and its distance are in `evidence.json` for every slot. Synthetic test: full
recovery not below 1 of 12, `background__bg01` no longer tier B, or the note says why it still is.
**Size.** S to M.

### T30. Remaining logging (e)
**What.** The items of §4 not covered by T26 to T29: final placement rectangle, share of the donor used
and discarded, per-target crops `targets/<target_id>_<view>.png` (photograph, mask, donor as placed,
result), nearest-element distance for slots if not done in T29.
**Files.** `i2r/complete.py`, `i2r/evidence.py`, tests.
**Accept.** For every completed target of the reference run the crop exists and `completion.json` holds
the rectangle and the two shares; outputs of both views are byte-identical to before (logging only).
**Size.** S.
