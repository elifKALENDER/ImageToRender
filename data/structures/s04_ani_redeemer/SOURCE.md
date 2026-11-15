# s04_ani_redeemer

> **Türkçe özet.** İkinci yapı: Ani'deki Kurtarıcı Kilisesi (Church of the Holy Redeemer), ayakta kalan
> yarısının dıştan görünüşü. Wikimedia Commons'tan, lisansı belli (CC BY-SA 2.0). Alt kemer dizisinde bir
> bölüm yıkık, iki yanında sağlam kemer ve sütunçeler var. Görüntü, 4094 x 2729 fotoğraftan yapıyı
> içeren 1501 x 1956 kesit. Hasar maskesi ve envanter henüz yok (yazarın işi). s01 için ayarlanan hiçbir
> eşik bu yapı için değiştirilmedi.

Second **building** of the project (s01 and s03 are two photographs of one building). Chosen on 2026-11-15
from six Wikimedia photographs the author collected in `data/incoming/`; the choice of this one is the
assistant's (reasons below), the author left it open.

| Field | Value |
|---|---|
| Building | Church of the Holy Redeemer (Surb Prkich), Ani ruins, Kars Province, Turkey. Name as given on the file page; not checked against another source. |
| Source URL | `https://commons.wikimedia.org/wiki/File:Church_of_the_Holy_Redeemer_-_Ani_Ruins.jpg` |
| Photographer | Flickr user "Panegyrics of Granovetter" as named on the Commons page (the page's Flickr link shows another user name, "sarah_c_murray"; not resolved). |
| Licence | CC BY-SA 2.0 (Creative Commons Attribution-Share Alike 2.0 Generic), as stated on the Commons page on 2026-11-15. Redistribution with attribution and the same licence is allowed. |
| Date taken | 2012-06-16 (EXIF, per the Commons page) |
| Date obtained | 2026-11-15 (downloaded by the author) |
| Image size | Source 4094 x 2729 px. `image.png` is the crop x 500 to 2001, y 409 to 2365 of it: 1501 x 1956 px, not resized. |

## Why this photograph

- One bay of the lower blind arcade is broken out; arches, colonnettes and capitals of the same arcade
  stand intact on both sides of it in the same photograph. That is the case the method is built for.
- The licence is known, so results can be shown.
- The building is about 1,500 px wide here, three times s01.
- Limits seen before any processing: the arcade runs round a polygonal or round body, so the bays are seen
  at different angles (the cause of the `t_002` problem on s01); the bases are hidden by rubble.

## Files

| File | Tracked in git | Notes |
|---|---|---|
| `source_Church_of_the_Holy_Redeemer_-_Ani_Ruins.jpg` | No | The file as downloaded (SHA-256 starts `af0e5192789e01bc`). |
| `image.png` | No | The crop named above. Not tracked by default; the licence would allow it with attribution, which is the author's decision. |
| `structure.yaml` | Yes | The `column` to `pilaster` merge of s01, same element type. |
| `damage_mask.png` | n/a | **Not drawn yet** (author). Without it only the detect and review steps run. |

## The other photographs in `data/incoming/` (not processed)

| File | What it shows | Use |
|---|---|---|
| `Church_of_the_Redeemer,_Ani_-_panoramio_(1).jpg` | the same church from the collapsed side, interior exposed (1536 x 2048; CC BY 3.0, Panoramio user "Gonzo Gooner", 2008-05-12, per its Commons page) | same building, but shows no intact exterior arcade: not usable as reference donor |
| `20110419_Georgian_Church_Ani_Turkey.jpg` | a wall with three blind arches, frontal, behind red steel shoring (4256 x 2832; CC BY-SA 3.0, Commons user Ggia, 2011-04-19, per its Commons page) | candidate for a later structure; the shoring hides parts of the wall |
| `AniCathedral.jpg` | the cathedral from a distance, largely intact | little damage to complete; licence not looked up |
| `Ani_Harabeleri.jpg` | a domed church seen from above, largely intact | little damage to complete; licence not looked up |
| `20110419_dome_inside_Saint_Gregory_Aburaments_Ani_Turkey.jpg` | a dome from inside | not a facade; licence not looked up |

## Open actions (author)

- Draw `damage_mask.png` with `tools/mask_editor.py`.
- Correct the element inventory in the editor.
- Decide whether `image.png` is committed (the licence allows it with attribution).
