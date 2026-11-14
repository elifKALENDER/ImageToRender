# RND-004 — Acceptance of a tier A placement

> **Türkçe özet.** Donör yerleştirmesinin kabulü için üç ölçü eklendi: aramanın sınıra dayanması, hizalama
> puanı ve "dikiş uyumu" (dolgunun çevresindeki sağlam taşta donör ile fotoğrafın benzerliği). 9 sentetik
> durumda dikiş uyumu iyi ve kötü yerleştirmeleri ayırıyor; hizalama puanı tek başına ayırmıyor; 9 durumun
> 9'u sınıra dayandığı için sınır tek başına ret nedeni olamıyor. Seçilen eşikler: puan 0,80, dikiş 0,70.
> Gerçek s01 hasarında 5 A hedefinden 1'i kabul ediliyor, 4'ü "tahmin, uzman onayı gerekir" oluyor:
> restorasyon görünümü 4.971 pikselden 648 piksele düşüyor. Dikiş uyumu 1 numaralı hatayı (`t_003`)
> yakalamıyor; onu hizalama puanı işaretliyor. Aramayı genişletmek puanı yükseltip sadakati düşürdü,
> aralık değiştirilmedi. Tek yapı, 9 durum: eşikler geçici.

| | |
|---|---|
| Date | 2026-11-14 |
| Task | T28 (`docs/TASKS.md`), from `docs/FAILURE_ANALYSIS.md` §8 |
| Code | `i2r/complete.py` (`align`, `seam_agreement`, `judge`, `transfer`), `i2r/render.py`, `evaluation/evaluate.py`; branch `fix/failure-analysis` |
| Data | 12 synthetic partial cases of s01 (`outputs/synthetic/s01_test_church/20261112_122615`), 9 of them with a tier A target; the 5 tier A targets of the real damage |
| Structures | **n = 1** (s01_test_church, 628 x 904). 9 cases, 3 classes (5 arches, 3 pilasters, 1 cornice). Tuned and reported on the same building. |

## What is measured

For every placed donor, written to `completion.json` and `provenance.json`:

| Figure | Meaning |
|---|---|
| `alignment.on_bound`, `alignment.limits` | which of scale, dx, dy ended on the edge of the search, and the edges |
| `alignment.score` | grey-level NCC between donor and photograph over the visible fragment pixels in the target box (what the search maximises) |
| `seam_agreement` | the same NCC over the surviving pixels within `complete.seam_band_px` (4) of the filled ones: does the donor continue the stone next to the fill? Empty when that band has fewer than `seam_min_px` (30) pixels |

For the synthetic cases, `evaluation/evaluate.py completion` now also gives **fidelity**: NCC with the
original photograph over the pixels that were hidden **and** filled. The earlier SSIM was averaged over all
hidden pixels and mostly measured coverage.

## The 9 cases

First-construction view; "unaligned" is the same donor placed box to box with the search off
(`refine_shift_frac` 0, `refine_scales` [1.0]). Evaluations `eval_20261114_125042` (search on) and
`eval_20261114_125241` (off).

| Case | Fidelity | Fidelity unaligned | Score before -> after | On bound | Seam |
|---|---|---|---|---|---|
| arch_007 | 0.78 | 0.18 | 0.47 -> 0.88 | scale, dy | 0.77 |
| arch_005 | 0.77 | 0.41 | 0.75 -> 0.85 | scale | 0.91 |
| recovered_arch_016 | 0.71 | 0.50 | 0.48 -> 0.86 | scale | 0.90 |
| arch_004 | 0.65 | 0.78 | 0.85 -> 0.92 | scale | 0.93 |
| recovered_pilaster_008 | 0.47 | 0.35 | 0.07 -> 0.58 | scale, dx | 0.07 |
| pilaster_010 | 0.46 | 0.20 | -0.02 -> 0.17 | scale, dx | 0.63 |
| recovered_arch_065 | 0.29 | 0.67 | 0.75 -> 0.89 | dy | 0.17 |
| pilaster_011 | 0.10 | 0.19 | 0.08 -> 0.22 | scale | 0.42 |
| expert_cornice_020 | -0.06 | 0.07 | 0.14 -> 0.40 | scale, dx | -0.40 |

What the table shows:

- **Fidelity falls into two groups** with a gap: four arches at 0.65 to 0.78, the other five at 0.47 or
  below.
- **The seam separates the same two groups**: 0.77 to 0.93 against 0.63 or below.
- **The score does not**: `recovered_arch_065` has 0.89, the second highest, and fidelity 0.29. Without
  that case the score has a wide empty range, 0.58 to 0.85.
- **The bound flag separates nothing**: 9 of 9 end on a bound, the four good ones included.
- **The search helps on average, not always**: fidelity is higher with the search in 5 of 9 cases and lower
  in 4 (`arch_004`, `recovered_arch_065`, `pilaster_011`, `expert_cornice_020`). The rule proposed in the
  failure analysis ("threshold = where fidelity is no better than unaligned") therefore gives no clean
  value; the gap between the two groups was used instead.
- All three pilasters and the only cornice are in the low group. Whether that is the class or the size
  (18 to 27 px wide) cannot be told from 4 cases.

## Chosen values

| Setting | Value | Basis |
|---|---|---|
| `complete.min_seam_agreement` | 0.70 | middle of the gap 0.63 to 0.77 |
| `complete.min_alignment_score` | 0.80 | the cautious end of the empty range 0.58 to 0.85; **not a measured boundary** |
| `complete.guess_when_on_bound` | false | a bound alone cannot fail a placement here; it is recorded for every target |

A placement that fails either test gets the status `guess_needs_expert` with the reasons. The restoration
view does not paint it (its box is hatched orange, "tier A guess withheld"). The first-construction view
paints it with orange stripes and a dashed outline instead of the donor outline ("donor placed as a
guess"). A missing seam value (too few pixels) does not fail a placement.

With these values the four high-fidelity cases are accepted and the other five are guesses: the rule
reproduces the split it was chosen from, which is not a test of it.

## Real damage of s01

Run `outputs/s01_test_church/20261114_130553` (generative fill off).

| Target | Element | Score | Seam | On bound | Result |
|---|---|---|---|---|---|
| t_001 | pilaster_013 | 0.73 | 0.69 | scale | guess (score, seam) |
| t_002 | recovered_arch_003 | 0.85 | 0.89 | none | **accepted** |
| t_003 | recovered_arch_001 | 0.71 | 0.91 | scale, dy | guess (score) |
| t_006 | recovered_pilaster_022 | 0.64 | -0.33 | scale, dx | guess (score, seam) |
| t_007 | expert_pilaster_017 | 0.69 | none (band too small) | dx | guess (score) |

- Restoration view: 1 target completed, 648 px (before: 5 targets, 4,971 px). First-construction view:
  3 completed (t_002 and the two tier B slots), 4 placed as guesses (4,323 px); 11,367 px changed, as
  before.
- **`t_003` is flagged, but only by the score.** Its seam is 0.91, the highest of the five: the seam test
  does not catch point 1 of the failure analysis, contrary to what that analysis expected. The cause of
  point 1 is upstream (the inventory record does not describe the standing arch, §9 there), and a
  placement can agree well with the stone beside it and still be the wrong element.
- **All four real scores that fail (0.64 to 0.73) lie in the range where there is no synthetic case.**
  With `min_alignment_score` at 0.60 instead of 0.80, t_003 and t_007 would be accepted and only t_001
  and t_006 would fail (on the seam). The choice of 0.80 decides them, and it is a cautious choice, not a
  finding. `t_001` misses the seam threshold by 0.01.
- `t_002` is the only target that passes both tests and the only one that is not on a bound.

## Wider search

`refine_shift_frac` 0.20 and `refine_scales` 0.82 to 1.18 in steps of 0.06 (`eval_20261114_125645`):

| | Search as configured | Wider search |
|---|---|---|
| Cases on a bound | 9 of 9 | 6 of 9 |
| Mean score | 0.64 | 0.72 |
| Mean fidelity | 0.46 | 0.39 |

The score rises or stays in every case and the fidelity falls in five (most in `arch_007`, 0.78 to 0.38,
and `pilaster_010`, 0.46 to 0.13) and rises in three. A larger search fits the visible fragment better and the hidden part worse, so the range in
`config.yaml` is unchanged. An anisotropic scale (separate x and y factors) was **not** tried: it adds a
search dimension, and this result gives no reason to expect a gain from more freedom.

## Synthetic test with the thresholds on

28 cases, `eval_20261114_130637`: preservation 0 in every row; tiers and recovery unchanged (partial 9 of
12 tier A, full 1 of 12, background 3 of 4 left C). The filled share of hidden pixels in the partial cases
falls from 48 % to 21 % in the restoration view (five guesses withheld) and stays 48 % in the
first-construction view.

## Limits

- One building, 9 cases, 3 classes; thresholds chosen and reported on the same cases.
- The synthetic damage is a straight cut through an intact element; real fragments are irregular and the
  real scores fall where the synthetic set is empty.
- The tests judge the placement, not the record: a wrong inventory entry (`recovered_arch_001`) can pass
  the seam test.
- Tier B targets are not judged (no fragment); marking them as guesses is part of T29.

## Open questions

- **R-004a.** Does the seam threshold hold on another photograph or building? Needs cases from one.
- **R-004b.** Is the low group a matter of class or of size? Needs larger pilasters (a larger photograph).
- **R-004c.** Synthetic cases whose score falls between 0.58 and 0.85 (irregular cuts, smaller visible
  parts) to place `min_alignment_score` on data.
