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
| `damage_mask.png` | No | Drawn by the author; 17.3 % of pixels marked. Byte-identical to the `completion_mask.png` used by every earlier run. |

## Open actions

- Identify the building and find the original photograph, its author and licence.
- Look for a larger copy (long side at least 1500 px). If one is found under a licence that allows
  redistribution, replace `image.png`, redraw `damage_mask.png` at the new size and add a `.gitignore`
  exception for this folder.
