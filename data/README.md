# Data

One folder per building: `data/structures/<id>/`.

| File | Required | Content |
|---|---|---|
| `image.png` | Yes | The photograph, RGB. |
| `damage_mask.png` | Yes | Single channel, same size as the image. White (255) = damaged or missing, black (0) = intact. Draw it with `tools/mask_editor.py`. |
| `SOURCE.md` | Yes | Building, location, photographer, source URL, licence, date obtained. Write "Unknown" rather than leaving a field empty. |
| `facade_mask.png` | No | Single channel, same size as the image. White (255) = the building's facade, black (0) = ground, sky, rock. Damage pixels outside it are never filled and are hatched as "outside the facade limit". Without the file the limit is the convex hull of the element boxes (`evidence.facade_limit`), which follows the inventory and keeps ground covered by an element box. Can be drawn with `tools/mask_editor.py --image ... --out ...`. |
| `target_decisions.json` | No | The expert's approve / reject decisions per target, written by `tools/target_review.py` and applied by the complete step. Tracked in git (it holds no image). |
| `structure.yaml` | No | Settings that differ from `config.yaml` for this building only (same layout, only the keys that differ). State the reason in a comment. |

A known vertical symmetry axis is given in `structure.yaml` as `evidence.tier_b.symmetry_axis_x`
(pixel column); without it the axis is estimated from the elements.

## What is tracked in git

Only `README.md`, `SOURCE.md`, `structure.yaml` and `target_decisions.json` are tracked by default. Images are tracked only
when `SOURCE.md` states a licence that allows redistribution; add a per-structure exception to
`.gitignore` in that case.

## Photograph requirements

- Near-frontal view of one facade.
- Long side at least 1500 px.
- At least three intact instances of each element class that is damaged, so that donors exist.
- Known source and licence.

## Where to find buildings

- The author's own photographs (preferred: full rights and known conditions).
- Wikimedia Commons, with the licence recorded.
- National and museum heritage inventories that publish under open licences.
- Public facade datasets (CMP Facade, eTRIMS) show intact urban buildings. They are useful for the
  synthetic damage test on windows and doors, but not as examples of ruins.
