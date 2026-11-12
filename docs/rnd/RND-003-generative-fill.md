# RND-003 — Tier C generative fill

> **Türkçe özet.** Kanıtsız (C) alanlar ilk inşa görünümünde tek bir ControlNet inpaint geçişiyle
> dolduruldu (arşivdeki inpaint_v3 kurulumu, 6 GB'ta 3,3 GB, 125 sn). Teknik ölçütler sağlandı:
> restorasyon görünümü bayt bayt aynı, açık alanın dışında hiçbir piksel değişmedi. Ama görüntü
> kullanılamaz: model binada olmayan bir friz, merdivenli bir kapı ve iç mekân uyduruyor, kaideler
> erimiş gibi, sınırlar görünür. Karar: **kullanılamaz**; adım kapalı kalıyor (`generative: false`).

| | |
|---|---|
| Date | 2026-11-12 |
| Task | T21 (`docs/TASKS.md`) |
| Code | `i2r/complete.py` `generative_fill`, branch `refactor/structure` |
| Run | `outputs/s01_test_church/20261112_142811` (local), with a config copy in which only `first_construction.generative` is `true` |
| Reference | `outputs/s01_test_church/20261112_141348` (same settings, generative off) |
| Attempts | **one** generation. An earlier start stopped while loading the models (the configured cache lacked the `Intel/dpt-hybrid-midas` weights) before anything was generated; the cache was corrected and the run repeated. |
| Structures | **n = 1** (s01_test_church, 628 x 904) |

## Set-up

Copied from `archive/inpaint/inpaint_v3.py`, its first-construction mode ("ozgun_hal"); nothing tuned:
Realistic Vision 6.0 B1 (SD 1.5) inpaint with the `sd-vae-ft-mse` VAE and two ControlNets (depth from
DPT-hybrid-MiDaS at 0.24, guidance end 0.42; lineart at 0.34, guidance end 0.56), control images
neutralised inside the mask, strength 0.96, 40 DPM++ Karras steps, guidance 6.5, seed 42, xformers,
model CPU offload. One pass over the context crop [70, 230, 591, 742] (521 x 512 px, resized to
512 x 504) of the **donor-completed** first-construction image; conditioning computed from that crop.
The generation mask is the open area grown 5 px and blurred 3 px; the result is pasted back **only** into
the open area (every damage pixel the donors left unfilled: 88,991 px, 15.7 % of the image).

## Technical criteria (T21)

| Criterion | Result |
|---|---|
| Runs within 6 GB | Yes: peak 3.32 GB allocated (RTX 2060), 124.8 s |
| Restoration view byte-identical with and without the step | Yes: `restoration.png`, `restoration_completed.png`, `restoration_source.png` have the same SHA-256 in both runs |
| Pixels outside the tier C area unchanged | Yes: 88,990 pixels differ from the donor-only image, all inside the open area; none in donor-filled or intact pixels (also checked in code: `PreservationError`) |
| Conjecture overlay and records | `first_construction_conjecture_overlay.png` shows the generated area in magenta; `completion.json` and `provenance.json` carry a `generative_fill` record (method `generative`, no donor, evidence `conjecture`, all settings and the seed); its pixels have their own label in the source map |

## Visual verdict: not usable

Images (local; the photograph has no known licence, so none is copied into the repository):
`first_construction_completed.png`, `first_construction.png`, `first_construction_conjecture_overlay.png`
and the debug files `generative_raw.png`, `generative_mask.png`, `generative_depth.png`,
`generative_lineart.png` in the run folder.

At a glance the building looks closed and whole. On a closer look:

1. **Invented architecture.** A bead or egg-and-dart frieze appears in the cornice zone that exists
   nowhere on the building; the central collapsed bay becomes an entrance portal with steps and a lit
   interior room. These are exactly the conjectures the
   project forbids (D2, D8): they look like evidence and are not.
2. **Melted forms.** Pilaster bases and plinths in the lower collapse are rounded, smooth blobs; the
   negative prompt ("melted masonry, warped columns") did not prevent them.
3. **Visible seams and tone.** The pasted area is lighter, smoother and pinker than the weathered stone;
   its boundary reads as a torn-paper edge along several pilasters.
4. **The mask reaches the ground.** Part of the open area at the lower left is rubble and soil; it is
   filled with a smooth floor.

The donor-filled pilasters and arches next to it are untouched, so the failure is confined to the
generated pixels, which the overlay marks. That does not make it acceptable for an image meant to read as
a first-construction proposal.

## Decision

`first_construction.generative` stays **false** (T21: "If the result is not usable, leave the step
switched off by default and continue; that outcome completes the task"). The code stays, tested with a
stand-in generator, so the step can be tried again with another model or on a larger photograph without
touching the restoration view. The first-construction view keeps tier C hatched and unfilled.

## Open questions

- R-003a: does a smaller generation area (one tier C region at a time, still one pass per region) avoid
  the invented portal? D8 rejected per-element crops on this image size; a per-region crop of at least
  512 px is a different case.
- R-003b: would conditioning on the donor-completed image *with* tier B filled and the restoration
  markings removed help, or is SD 1.5 at 512 px simply too weak here (D8 risk)?
- R4 (TASKS): experts' preference between the donor-only view and a harmonised one is still untested.
