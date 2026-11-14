# s03_church_later

> **Türkçe özet.** s01 ile aynı yapının, aynı hasarlı cephesinin ikinci bir fotoğrafı (900 x 1200).
> Hasar s01'dekinden ileri görünüyor (tahmin: daha geç tarihli). İnternetten (Wikiloc) alındı; fotoğrafçı,
> tarih ve lisans henüz bilinmiyor, bu yüzden görüntü git dışında. Hasar maskesi ve envanter henüz yok.

Second photograph of the building of `s01_test_church`, same damaged side, almost the same viewpoint.
Added on 2026-11-14 by the author's decision, which changes the decision of 2026-11-12 ("no other
photograph"). Same building: results on s01 and s03 together are still n = 1 building.

| Field | Value |
|---|---|
| Building | Unknown by name. Same building as `s01_test_church` (same graffiti on the drum, same collapse). |
| Location | Digor district, Kars, Turkey (from the Wikiloc listing the author gave). Exact site not recorded. |
| Photographer | Unknown (a Wikiloc user; not recorded yet). |
| Source URL | `https://s1.wklcdn.com/image_245/7377499/109191099/70481209Master.jpg` (Wikiloc image server). The author found it through `https://tr.wikiloc.com/rotalari/gezi-yuruyus/turkiye/kars/digor`. The route page itself is not recorded yet; Wikiloc refuses automated access (HTTP 403). |
| Licence | Unknown. Wikiloc user photograph, no licence statement seen. Treat as **not redistributable**: images stay out of git. |
| Date taken | Unknown. No EXIF. Guess: later than the s01 photograph, because the central collapse is larger and a graffito ("HAMZA") is present that s01 does not show. |
| Date obtained | 2026-11-14 |
| Image size | 900 x 1200 px (below the 1500 px long-side requirement in `data/README.md`). The largest version the server offers: the name without "Master" gives 700 x 933. The building is about 590 px wide here, about 500 px in s01. |

## Files

| File | Tracked in git | Notes |
|---|---|---|
| `source_70481209Master.jpg` | No | The file as downloaded (SHA-256 starts `4346940e1738dfcd`). |
| `image.png` | No | The same pixels saved as PNG, not resized. |
| `structure.yaml` | Yes | The `column` to `pilaster` merge of s01, same building, same reason. |
| `damage_mask.png` | n/a | **Not drawn yet** (author). Without it only the detect and review steps run. |

## Other photographs of the building (reference only, not processed)

In `data/structures/s01_new_photos/` (not in git): `88168418Master.jpg` (same side, oblique, wide-angle
distortion), `470553649_533637929672305_3536149461209620897_n.jpg` (the other side, arcade largely intact,
pilasters with their bases; 567 x 800) and `70690849Master.jpg` (the other side from far away).

## Open actions (author)

- Route page address, photographer, date and, if stated, the building's name.
- Draw `damage_mask.png` (and, if wanted, `facade_mask.png`) with `tools/mask_editor.py`.
- Correct the detected inventory in the editor (`tools/editor_adapter.py edit --run <run>`).
