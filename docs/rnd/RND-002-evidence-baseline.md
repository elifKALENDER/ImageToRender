# RND-002 — Evidence thresholds and synthetic-damage baseline

> **Türkçe özet.** Üç eşik (DINOv2 benzerliği, boyut toleransı, B1 sıra toleransı) 27 kombinasyonla,
> s01'in 28 sentetik vakası ve gerçek hasarı üzerinde denendi. Benzerlik eşiği 0,60'tan 0,80'e çıktı;
> diğer ikisi aynı kaldı, çünkü gevşetmek yalnızca şüpheli donörler ve gerçek yıkıntıda üst üste binen
> hayali plasterler ekledi. Kısmi gizlemede 12 elemandan 9'u A ile geri geliyor, tam gizlemede yalnızca
> 1'i. Eşikler aynı binada seçildi ve aynı binada raporlanıyor (n = 1); başka binaya genellenemez.

| | |
|---|---|
| Date | 2026-11-12 |
| Task | T20 (`docs/TASKS.md`) |
| Code | branch `refactor/structure`, commit after `b191018` (grid command in `evaluation/evaluate.py`) |
| Cases | `outputs/synthetic/s01_test_church/20261112_122615` (local): 12 partial, 12 full, 4 background |
| Grid | `.../grid_20261112_135246/grid.md`, `grid.json` (local) |
| Real damage | inventory of `outputs/s01_test_church/20261112_121244` (SAM-refined expert inventory, mask v3) |
| Command | `python evaluation/evaluate.py grid --cases <cases> --grid evidence.min_similarity=0.60,0.80,0.88 --grid evidence.size_tolerance=0.25,0.35,0.50 --grid evidence.tier_b.row_tolerance=0.10,0.15,0.25` (27 points, about 30 minutes on the RTX 2060) |
| Structures | **n = 1** (s01_test_church, 628 x 904). By author decision (2026-11-12) there is no second building or photograph in this stage. |
| Tuning | **The thresholds were tuned on the same building they are reported on.** Nothing here shows they hold for another building. |

## Question

Which values of the three evidence thresholds the task names (`evidence.min_similarity`,
`evidence.size_tolerance`, the B1 tolerance `evidence.tier_b.row_tolerance`) recover the most hidden
elements on s01 without placing evidence where there is none?

## Method

- Synthetic damage (ARCHITECTURE §7.1, T19): every intact element hidden in two ways. **Partial**: the
  top half of its box is added to the mask and painted black; expected tier A. **Full**: the box plus
  4 px is hidden and the element is marked rejected; expected tier B. **Background**: 4 rectangles of
  28 x 48 px on plain wall; expected tier C.
- "Recovered" = a tier A or B target of the same class with box IoU >= 0.5 to the hidden element.
  "Background left C" = no A/B target covers half of the rectangle.
- Each grid point runs the damage and evidence steps on all 28 cases and on the real damage. The DINOv2
  model is loaded once and shared; settings it depends on cannot be varied by the grid.
- **What the test cannot measure.** The cases contain no wrong-class or wrong-element candidates for tier
  A: the class filter removes them before similarity is computed. A high `min_similarity` can therefore
  only lose recoveries here, never prevent a false one. Whether a donor is a true counterpart (donor
  validity, §7.1) needs the author; the donor of each case is listed in `grid.json`.

## Results

Partial = share of the 12 partial cases recovered (tier A); Full = share of the 12 full cases recovered
(tier B); Background = share of the 4 rectangles left tier C; Accuracy = cases given their expected tier
(of 28); Real = tier A/B/C targets on the real damage of s01.

| Point | min_similarity | size_tolerance | row_tolerance | Partial | Full | Background | Accuracy | Real A/B/C |
|---|---|---|---|---|---|---|---|---|
| p01 | 0.60 | 0.25 | 0.10 | 50 % | 8 % | 75 % | 36 % | 5/2/12 |
| p02 | 0.60 | 0.25 | 0.15 | 50 % | 8 % | 75 % | 36 % | 5/2/12 |
| p03 | 0.60 | 0.25 | 0.25 | 50 % | 58 % | 75 % | 57 % | 5/7/14 |
| p04 | 0.60 | 0.35 | 0.10 | 75 % | 8 % | 75 % | 46 % | 5/2/12 |
| p05 | 0.60 | 0.35 | 0.15 | 75 % | 8 % | 75 % | 46 % | 5/2/12 |
| p06 | 0.60 | 0.35 | 0.25 | 75 % | 58 % | 75 % | 68 % | 5/7/14 |
| p07 | 0.60 | 0.50 | 0.10 | 92 % | 8 % | 75 % | 54 % | 6/2/11 |
| p08 | 0.60 | 0.50 | 0.15 | 92 % | 8 % | 75 % | 54 % | 6/2/11 |
| p09 | 0.60 | 0.50 | 0.25 | 92 % | 58 % | 75 % | 75 % | 6/7/13 |
| p10 | 0.80 | 0.25 | 0.10 | 42 % | 8 % | 75 % | 32 % | 5/2/12 |
| p11 | 0.80 | 0.25 | 0.15 | 42 % | 8 % | 75 % | 32 % | 5/2/12 |
| p12 | 0.80 | 0.25 | 0.25 | 42 % | 58 % | 75 % | 54 % | 5/7/14 |
| p13 | 0.80 | 0.35 | 0.10 | 75 % | 8 % | 75 % | 46 % | 5/2/12 |
| **p14** | **0.80** | **0.35** | **0.15** | **75 %** | **8 %** | **75 %** | **46 %** | **5/2/12** |
| p15 | 0.80 | 0.35 | 0.25 | 75 % | 58 % | 75 % | 68 % | 5/7/14 |
| p16 | 0.80 | 0.50 | 0.10 | 83 % | 8 % | 75 % | 50 % | 5/2/12 |
| p17 | 0.80 | 0.50 | 0.15 | 83 % | 8 % | 75 % | 50 % | 5/2/12 |
| p18 | 0.80 | 0.50 | 0.25 | 83 % | 58 % | 75 % | 71 % | 5/7/14 |
| p19 | 0.88 | 0.25 | 0.10 | 17 % | 8 % | 75 % | 21 % | 3/2/14 |
| p20 | 0.88 | 0.25 | 0.15 | 17 % | 8 % | 75 % | 21 % | 3/2/14 |
| p21 | 0.88 | 0.25 | 0.25 | 17 % | 58 % | 75 % | 43 % | 3/7/16 |
| p22 | 0.88 | 0.35 | 0.10 | 42 % | 8 % | 75 % | 32 % | 3/2/14 |
| p23 | 0.88 | 0.35 | 0.15 | 42 % | 8 % | 75 % | 32 % | 3/2/14 |
| p24 | 0.88 | 0.35 | 0.25 | 42 % | 58 % | 75 % | 54 % | 3/7/16 |
| p25 | 0.88 | 0.50 | 0.10 | 58 % | 8 % | 75 % | 39 % | 4/2/13 |
| p26 | 0.88 | 0.50 | 0.15 | 58 % | 8 % | 75 % | 39 % | 4/2/13 |
| p27 | 0.88 | 0.50 | 0.25 | 58 % | 58 % | 75 % | 61 % | 4/7/15 |

The highest accuracy (p09, 75 %) is **not** chosen; see below.

## Chosen values (p14)

| Setting | Before | Chosen | Reason |
|---|---|---|---|
| `evidence.min_similarity` | 0.60 | **0.80** | At size 0.35, 0.60 and 0.80 give identical results: every donor chosen scored 0.83-0.93 (partial cases) and 0.85-0.94 (real damage). At size 0.50, 0.80 removes the two doubtful donors the looser size filter lets in (`recovered_arch_016` for `recovered_arch_025`, 0.78; `recovered_pilaster_008` for the real `recovered_pilaster_020`, 0.73). 0.88 loses true donors (partial 75 % -> 42 %; real A 5 -> 3). The margin to the lowest true donor (0.83) is small, and there are no negatives to test against. |
| `evidence.size_tolerance` | 0.35 | 0.35 | 0.25 loses 3 of 9 partial recoveries. 0.50 adds 2 (`recovered_arch_025`, `expert_cornice_021`) only through donors of clearly different proportions (an arch of 36 x 48 px for one of 70 x 34 px), and changes 2 of 5 donors on the real damage. |
| `evidence.tier_b.row_tolerance` | 0.15 | 0.15 | 0.10 and 0.15 are identical. 0.25 raises full recovery from 1 to 7 of 12, but on the real damage it adds 6 pilaster slots about 20 px apart that overlap each other (x 310-457) and an arch slot in the collapse: B1 then reads the pilaster pairs as one dense row. These slots are not evidence. |

With the chosen values the s01 run is unchanged (`outputs/s01_test_church/20261112_141348`: 5 A, 2 B,
12 C; restoration 4,971 px, first construction 11,367 px). The other thresholds in `config.yaml` were not
varied and stay marked "initial guess, not calibrated".

## Failure cases (chosen values)

Images: each case's `evidence_overlay.png` and `evidence_sheet.png` in
`outputs/synthetic/s01_test_church/20261112_122615/eval_20261112_122636/<case>/` (local; the s01
photograph has no known licence, so no image is copied into the repository).

| Case | Result | Cause |
|---|---|---|
| `partial__expert_opening_018` | C | The only window on the facade: no donor exists. |
| `partial__expert_cornice_021` | C | The other cornice fails the size filter (box widths 161 and 108 px). |
| `partial__recovered_arch_025` | C | No arch of similar proportion (70 x 34 px is the flattest arch). |
| `full__*` (11 of 12) | C | B1 needs a row of at least 3 same-class elements and only interpolates; the arches and pilasters of s01 form short rows broken by the collapse. B2 mirrors at the estimated axis x = 355 do not land on the hidden element (the view is not frontal, the apse is curved). Only `full__expert_cornice_020` comes back, by B2. |
| `background__bg01` | B (B1) | A pilaster slot is implied on plain wall between two pilasters: a B1 false positive. The restoration view still leaves it open (B is not restored); the first-construction view fills it. |

## Comparison with the 10 hypotheses approved in Stage 3C-C

`outputs/restore_20260804_141005/expert_approved_reconstruction_hypotheses.json` (10 approved, 0
rejected). A hypothesis counts as found when a tier A or B target of the same class has box IoU >= 0.5
with it.

| Hypothesis | Class | Found by the new pipeline | IoU | Note |
|---|---|---|---|---|
| hyp_arch_001 | arch | yes, `recovered_arch_003` A | 0.72 | |
| hyp_arch_002 | arch | no | 0.44 | Same arch (`recovered_arch_001`, A), but the hypothesis box starts at x 336, the inventory box at x 391. |
| hyp_pilaster_002 | pilaster | yes, `pilaster_013` A | 0.79 | |
| hyp_pilaster_003 | pilaster | yes, `recovered_pilaster_022` A | 0.99 | |
| hyp_pilaster_005 | pilaster | yes, `expert_pilaster_017` A | 0.67 | |
| hyp_pilaster_001 | pilaster | no | — | `recovered_pilaster_020` stays C: its only near-size donor fails the size filter. |
| hyp_pilaster_004 | pilaster | no | 0.14 | `recovered_pilaster_046` stays C (size filter); a B1 slot lies beside it, not on it. |
| hyp_base_001, hyp_base_002 | base | no | — | No intact base exists; course continuation is not a rule of the new pipeline (D1). |
| hyp_cornice_001 | cornice | no | — | Cornice course continuation is not a rule of the new pipeline (D1). |

The new pipeline finds **4 of 10** (5 of 10 if the arch with IoU 0.44 is counted), and proposes 2 B1
pilaster slots the old run did not (x 343-366 and 400-423), which the expert has not judged. As D1
predicted, the simpler evidence step proposes less than Stage 3C on this building; the missing part is
course continuation (bases, cornice) and two pilasters blocked by the size filter.

## Open questions

- R-002a: `min_similarity` cannot be tested without negatives. A test with wrong-but-same-class donors
  (e.g. a pilaster donor for a hidden arch relabelled as pilaster) would give it a lower bound.
- R-002b: B1 on short, interrupted rows: should the spacing come from the paired-pilaster rhythm (two
  spacings) rather than one? The 0.25 result shows the rule is close to failing.
- R-002c: a course rule (B3) for bases and cornices, as the Stage 3C hypotheses had.
- R3 (TASKS): whether these synthetic numbers predict acceptance on real damage cannot be answered on one
  building.
