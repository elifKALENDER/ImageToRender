# s01_test_church

| Field | Value |
|---|---|
| Building | Unknown. Centrally planned medieval church with a blind arcade (arches on paired pilasters) around a polygonal drum; partially collapsed on the photographed side. |
| Location | Unknown |
| Photographer | Unknown |
| Source URL | Unknown. The image appears to be taken from a web page: `assets/chrome_LGhPjvSfrm.png` is a browser screenshot of the same picture at the same size. |
| Licence | Unknown. Treat as **not redistributable**: `image.png` is kept out of git. |
| Date obtained | Before 2026-07-24 (first run using it: `outputs/restore_20260724_204426`) |
| Image size | 628 x 904 px (below the 1500 px long-side requirement in `data/README.md`) |

## Files

| File | Tracked in git | Notes |
|---|---|---|
| `image.png` | No | RGB. Pixel-identical to `input.png` of every earlier run. |
| `damage_mask.png` | No | Version 3 (2026-11-10, SHA-256 starts `fa5a28af6afcdf1c`), 17.7 % of pixels marked: version 2 minus the SAM masks of `pilaster_010` and `pilaster_011` grown by 3 px (841 px), so the mask no longer runs over two pilasters the expert labelled intact. Done by the assistant at the author's request as a provisional fix; the author will redraw the edge later. |
| `damage_mask_v2_before_T16b.png` | No | Version 2 (SHA-256 starts `0fcd1467c78183ec`), 17.8 % of pixels marked: version 1 plus two rectangles added at the author's request after T13, over the damaged lower ends of the two pilasters the expert labelled fragmented but the mask missed: x 70-96, y 562-615 (`recovered_pilaster_020`, shaft foot and missing base) and x 227-254, y 555-617 (`recovered_pilaster_022`, broken shaft foot and eroded base block). Extents are the assistant's reading of the 628 x 904 photograph; the author may refine them with `tools/mask_editor.py`. |
| `damage_mask_v1_before_T13.png` | No | Version 1 (SHA-256 starts `bf0c797e88b9593c`): drawn by the author; 17.3 % of pixels marked. Byte-identical to the `completion_mask.png` used by every earlier run. |

## Open actions

- Identify the building and find the original photograph, its author and licence.
- Look for a larger copy (long side at least 1500 px). If one is found under a licence that allows
  redistribution, replace `image.png`, redraw `damage_mask.png` at the new size and add a `.gitignore`
  exception for this folder.
