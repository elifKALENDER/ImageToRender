# s02_test_arch

| Field | Value |
|---|---|
| Building | Probably the Church of St Gregory of Abughamrents, Ani (identification by appearance, not confirmed): circular plan with radiating niches, drum with an arcade of windows, conical roof. Lower walls and arches partly collapsed and patched. |
| Location | Probably Ani archaeological site, Kars, Türkiye (unconfirmed) |
| Photographer | Unknown |
| Source URL | Unknown. Original file name was a hash (`dea39debd066abf9e44d3767e1728477.jpg`), typical of an image saved from a website. Kept as `assets/test_arch.jpg`. |
| Licence | Unknown. Treat as **not redistributable**: `image.png` is kept out of git. |
| Date obtained | 2026-07-15 (file date in `assets/`) |
| Image size | 1045 x 1080 px (below the 1500 px long-side requirement in `data/README.md`) |

## Files

| File | Tracked in git | Notes |
|---|---|---|
| `image.png` | No | RGB conversion of `assets/test_arch.jpg`. |
| `damage_mask.png` | — | **Not drawn yet.** Draw with `tools/mask_editor.py` (task T13). |

## Notes for the pipeline

- The building is round, not a flat facade. Donor transfer assumes a near-frontal plane, so elements
  near the left and right edges are foreshortened. Expect weaker results there; this is a useful test of
  the limits stated in `docs/ARCHITECTURE.md` §10.

## Open actions

- Confirm the building and find the original photograph, its author and licence.
