# Data

One folder per building: `data/structures/<id>/`.

| File | Required | Content |
|---|---|---|
| `image.png` | Yes | The photograph, RGB. |
| `damage_mask.png` | Yes | Single channel, same size as the image. White (255) = damaged or missing, black (0) = intact. Draw it with `tools/mask_editor.py`. |
| `SOURCE.md` | Yes | Building, location, photographer, source URL, licence, date obtained. Write "Unknown" rather than leaving a field empty. |
| `symmetry_axis.json` | No | `{"x": <pixel column>}` if the facade has a known vertical symmetry axis. |

## What is tracked in git

Only `README.md`, `SOURCE.md` and `symmetry_axis.json` are tracked by default. Images are tracked only
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
