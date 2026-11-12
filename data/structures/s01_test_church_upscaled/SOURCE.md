# s01_test_church_upscaled

> **Türkçe özet.** s01 fotoğrafının Upscayl (Real-ESRGAN ailesi, "high-fidelity-4x" modeli, 3 kat)
> ile büyütülmüş hali. Yeni bir bina veya yeni bir fotoğraf değildir; yalnızca pipeline'ın görüntü
> boyutuna duyarlılığını sınamak için vardır. Büyütme ayrıntı üretir (uydurur), bu yüzden buradaki
> benzerlik puanları kamera fotoğrafını temsil etmez ve T20 eşik ayarında ikinci yapı yerine sayılmaz.

| Field | Value |
|---|---|
| Building | Same as `s01_test_church` (unknown). |
| Source | `data/structures/s01_test_church/image.png` (628 x 904) upscaled by the author with Upscayl, model "high-fidelity-4x" (Real-ESRGAN family; 4x then resized to 3x), saved as `assets/chrome_LGhPjvSfrm_upscayl_3x_high-fidelity-4x.png`. |
| Licence | Unknown, as s01. Not redistributable: `image.png` stays out of git. |
| Date | 2026-11-12 |
| Image size | 1884 x 2712 px (exactly 3 x s01). Downscaled back to 628 x 904 it differs from s01 by 3.9 grey levels on average (99th percentile 17): same view, pixel-aligned. |

## How the folder was made

- `image.png`: the upscaled file, unchanged.
- `damage_mask.png`: s01 mask version 3 resized x3, nearest neighbour.
- `structure.yaml`: copied from s01.
- Seed run `outputs/s01_test_church_upscaled/20261112_132908`: `elements.json` and element masks of
  `outputs/s01_test_church/20261112_121244`, boxes x3, masks resized x3 (nearest), damage fields removed.
- Run: `python run.py --structure s01_test_church_upscaled --from damage --view both --no-review`
  -> `outputs/s01_test_church_upscaled/20261112_132918`. config.yaml unchanged.

## Result against s01 (n = 1 building; one upscaled copy)

- Unchanged: damage labels 23/23 with the expert, 12 intact / 11 fragmented; tiers 5 A, 2 B (the same two
  B1 pilaster slots), the same fragments stay tier C for the same reasons; restoration 5 targets, first
  construction 7.
- Changed: 2 of 5 tier A targets took another donor (`pilaster_013`: 010 -> 011; `recovered_arch_001`:
  arch_005 -> arch_007), and similarities moved by up to 0.10 (`expert_pilaster_017` 0.87 -> 0.77). The
  donor choice is sensitive to the upscaler's invented texture; this is the reason the copy cannot be used
  to calibrate `min_similarity`.
- Pixel-based settings did not scale with the image: 8 tier C regions instead of 6 (`damage.min_region_px`
  64 is 9 times smaller relative to the image), and the hatch (8 px), the 1 px outline and the labels
  became too fine to read at 1884 x 2712 (run `outputs/s01_test_church_upscaled/20261112_132918`).
  **Fixed the same day:** `scale` in `config.yaml` now scales every pixel setting by long side / 904
  (areas by the square). s01 outputs are byte-identical to before; the copy (run
  `outputs/s01_test_church_upscaled/20261112_134410`, factor 3) now gives 19 targets (5 A, 2 B, 12 C),
  6 tier C regions and the same markings as s01. The only remaining difference is the 2 donors above.
- Alignment still hits the scale bound 1.06 on two pilasters, as on s01; the low resolution was not the
  cause of that.
