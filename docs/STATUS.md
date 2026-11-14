# Status after Phase 3 — what works, what does not, what needs the author

> **Türkçe özet.** Hata analizinden çıkan beş görev (T26-T30) ve iki ek (uzman kararı aracı, rapor sayfası)
> tamamlandı. Sistem artık az ama savunulabilir dolduruyor: s01'de restorasyon görünümünde tek bir kemer
> parçası (648 piksel) kabul ediliyor, dört yerleştirme "tahmin" olarak sizin onayınızı bekliyor, hasarın
> % 86'sı için fotoğrafta kanıt yok. Sizden beklenen elle işler sırasıyla: (1) beş yerleştirmeyi
> `tools/target_review.py` ile onaylamak ya da reddetmek, (2) cephe maskesi çizmek, (3) `recovered_arch_001`
> kaydını düzeltmek, (4) yayınlanabilir bir örnek için lisansı belli bir fotoğraf. Bölüm 4'te komutlarıyla.

| | |
|---|---|
| Date | 2026-11-14 |
| Branch | `fix/failure-analysis`, ahead of `main` (`v0.1.0`); not merged, which is the author's decision |
| Structures | **n = 1 building** (`s01_test_church`, 628 x 904). `s03_church_later` is a second photograph of the same building: detection only so far |
| Reference run | `outputs/s01_test_church/20261114_150757` (generative fill off). Open `report.html` in it |
| Tests | 164 pass (`python -m unittest discover tests`, no GPU) |

Statements marked **(guess)** are the assistant's reading, not a measurement.

## 1. What exists now

One command produces, for one photograph with a damage mask and an element inventory:

```
python run.py --structure s01_test_church --from damage --view both --no-review
```

| Output in the run folder | Content |
|---|---|
| `report.html` | one page: photograph, both views, what was filled and on what basis, every placed donor with its comparison image, what stayed open and why, the commands for the expert |
| `restoration.png` | only donor placements that passed the acceptance tests or that the expert approved |
| `first_construction.png` | the same, plus guesses, marked orange |
| `targets/<id>_<view>.png` | per placed donor: photograph, damage mask, donor as placed, result |
| `provenance.json` | every target with donor, tier, figures, verdict and the expert's decision |

Three things can be entered by hand, each with a tool:

| What | Tool | Stored in |
|---|---|---|
| damage mask, facade mask | `tools/mask_editor.py` | `data/structures/<id>/damage_mask.png`, `facade_mask.png` |
| element inventory | `tools/editor_adapter.py edit` | the run's `elements.json` |
| approve / reject a placed donor | `tools/target_review.py` | `data/structures/<id>/target_decisions.json` |

## 2. Result on s01 today

| | Restoration view | First-construction view |
|---|---|---|
| Accepted donor pixels | 648 (0.6 % of the damage) | 648 (0.6 %) |
| Guesses | 4 targets withheld (boxes hatched orange) | 4 targets painted and marked, 4,323 px (4.3 %) |
| No evidence, open | 86.2 % | 90.3 % |
| Outside the facade limit | 4.8 % | 4.8 % |

21 targets: 5 tier A, 0 tier B, 16 tier C. In `v0.1.0` the same photograph gave 4,971 px in the
restoration view and 11,367 px in the first-construction view; the difference is what the acceptance tests
(T28) and the slot rule (T29) now withhold.

## 3. Problems, and what causes each

Ordered by how much they limit a presentable result.

### P1. Most of the damage cannot be filled from this photograph
- **What.** 86 % of the damage has no evidence. The collapsed centre bay, the frieze above it and the
  bases are open.
- **Cause.** Not a defect of the code: the method copies what the photograph shows elsewhere, and the
  photograph shows no intact base and nothing of the centre bay. The three bases are tier C because no
  intact base exists on this side.
- **What would change it.** Another source: an archive photograph, a survey, or the other side of the
  building (`470553649_...n.jpg` in `data/structures/s01_new_photos/` shows intact pilasters with bases).
  Using a donor from another photograph is not implemented (R5c in `docs/TASKS.md`).
- **Needs the author.** A decision on whether such a cross-photograph donor is worth building.

### P2. Four of five placements are "guess", on thresholds that are not settled
- **What.** Only `t_002` passes. `t_001`, `t_003`, `t_006`, `t_007` are guesses.
- **Cause.** `min_alignment_score` 0.80 is the cautious end of a range (0.58 to 0.85) in which the 9
  synthetic cases have no example; the four real scores are 0.64 to 0.73, inside that range
  (`docs/rnd/RND-004-alignment-acceptance.md`). The tests cannot be made more exact with this data.
- **Reading of the panels (guess).** `t_003` (right arch) and `t_001` (pilaster) look plausible in their
  comparison images. The three pilaster placements are 18 to 27 px wide and hard to judge at this size,
  also for a person; `t_006` has the lowest seam agreement (-0.33). This is for the expert, not for a
  threshold.
- **Needs the author.** Review the five placements (section 4, step 1). This is the step that turns the
  output from "what the tests allow" into "what the expert accepts".

### P3. The inventory has to be made by hand, and one record is wrong
- **What.** Raw detection is not usable: on s01 52 % class-aware recall, every score below 0.06; on
  `s03_church_later` 28 boxes, most of them covering a whole bay. The pipeline runs on the expert's
  inventory (23 elements). In it, `recovered_arch_001` covers rubble and tympanum, not the standing arch.
- **Cause.** OWL-ViT is a general zero-shot detector, not trained on heritage elements (RND-001). The
  wrong record came from the old recovery stage; the expert widened its box and kept its status
  (`docs/FAILURE_ANALYSIS.md` §9).
- **Effect now.** `t_003` is withheld from the restoration view by the score test, so the wrong record no
  longer reaches that view unapproved.
- **Needs the author.** Correct or reject `recovered_arch_001` in the editor (step 3). For any new
  photograph: the whole inventory.

### P4. The damage mask reaches the ground (point 5 of the failure analysis)
- **What.** The lower-left lobe of the mask lies on grass and rubble. With the generative fill on it is
  still painted as masonry.
- **Cause.** The mask is the only statement of where the building is missing, and it is drawn loosely
  there. The automatic facade limit (convex hull of the element boxes) does not help: the box of
  `expert_base_022` covers that ground, so point 5 is inside the hull. The hull also cuts a lower-right
  region that may be the foot of the apse **(guess)**.
- **Effect now.** With the fill off (the default) the lobe is only hatched "no evidence": nothing is
  painted there. It matters when the fill is on.
- **Needs the author.** A `facade_mask.png` (step 2), or a redrawn mask. Mask version 3 is provisional in
  two more places (near `pilaster_010` and `pilaster_011`).

### P5. Alignment ends on its search bounds
- **What.** 9 of 9 synthetic placements and 4 of 5 real ones end on a bound of the search.
- **Cause.** Probably the box-to-box mapping with one scale for both directions on a curved apse seen at
  an angle **(guess)**; a wider search raised the score and lowered the fidelity (RND-004), so it is not a
  matter of range. Elements are small: pilasters are 18 to 27 px wide, so 1 px is 4 to 6 % of the width.
- **What would change it.** A larger real photograph (finer steps), or a placement that follows the
  element's own axis instead of the box. Neither is done.
- **Needs the author.** Nothing now; it is why the expert's review (P2) matters.

### P6. A surviving fragment cannot be completed when its box covers only the standing part
- **What.** `recovered_pilaster_046` and `recovered_pilaster_020` stay tier C ("no usable donor: size").
- **Cause.** The size filter compares the fragment's box (99 px high) with whole pilasters (about 200 px).
  Completing such a fragment needs its box extended to the row's height first. Left out of T29 to keep the
  structure simple (R-029a).
- **Needs the author.** Either redraw those two boxes at full height in the editor (then the existing
  path applies), or ask for R-029a.

### P7. Position rules (tier B) almost never fire
- **What.** s01 has no tier B target now. In the synthetic test 1 of 12 fully hidden elements is found.
- **Cause.** B1 needs a row of at least three evenly spaced elements; the pilasters of this arcade are
  paired, not evenly spaced. B2 mirrors across an image axis, which a curved, obliquely seen apse does not
  have. T29 removed the two slots that existed because a fragment contradicted their spacing.
- **What would change it.** A rule for paired rhythm (bays), listed as B3 in `docs/ARCHITECTURE.md` §9.

### P8. The generative fill is not usable
- **What.** Off by default. When on it invents a frieze and a portal (RND-003).
- **Cause.** SD 1.5 at 512 px with no knowledge of this building. It is now at least shown honestly:
  apart, striped, under an "AI guess" banner, and never outside the facade limit.

### P9. Nothing can be published as an example
- **What.** The README has no image; `report.html` holds the photograph and must stay local.
- **Cause.** Every photograph in use (s01, s03, the reference photos) comes from the web with no known
  licence.
- **Needs the author.** One photograph with a known licence, ideally the author's own (step 4).

### P10. Everything is measured on one building
- **What.** Thresholds were chosen and are reported on s01. `s03_church_later` is the same building.
- **Effect.** No number here says how the method does elsewhere.

### Smaller points
- A placement rejected by the expert is hatched like "no evidence"; it has no marking of its own.
- The label "outside facade" is drawn only where it fits in the region; on s01 it does not, the legend
  carries it.
- `restoration.include_tier_b` paints tier B without the guess marking (unchanged behaviour, off by
  default).
- Run folders written before 2026-11-14 keep their old images and legends.
- The review window was opened and stepped through by a script; it has not been used by a person yet.

## 4. Manual work for the author, in order

**Step 1 — decide on the five placed donors.** Changes the result at once; no drawing needed.

```
python tools/target_review.py --run outputs/s01_test_church/20261114_150757
python run.py --structure s01_test_church --from complete --no-review
```

Keys: A approve, R reject, C clear, arrows to move; the note is saved with the decision. Then open
`report.html` of the new run folder (the newest under `outputs/s01_test_church/`).

**Step 2 — draw the facade mask** (white = the building's facade, black = ground, sky, rock).

```
python tools/mask_editor.py --image data/structures/s01_test_church/image.png --out data/structures/s01_test_church/facade_mask.png
python run.py --structure s01_test_church --from damage --no-review
```

**Step 3 — correct `recovered_arch_001`** (and, if wanted, draw `recovered_pilaster_046` and
`recovered_pilaster_020` at full height, P6).

```
python tools/editor_adapter.py edit --run outputs/s01_test_church/<newest run>
python tools/refine_masks.py --run outputs/s01_test_church/<newest run>
python run.py --structure s01_test_church --from damage --no-review
```

`refine_masks.py` prints the run folder it wrote; `--from damage` then continues from the newest folder.
After a change of the inventory or the mask, placements may change and earlier decisions of step 1 are
then reported as stale: review those targets again.

**Step 4 — source and licence.** For s01 and s03: page address, photographer, date, the building's name
(`SOURCE.md` of each). For a publishable example: one photograph the author may publish.

**Step 5 — decisions only the author can take.** Merge `fix/failure-analysis` into `main` and tag it;
whether to continue with `s03_church_later` (it needs a damage mask and an inventory); which of the open
items below to build next.

## 5. Open coding items, not started

| ID | What | Why it is open |
|---|---|---|
| R-029a | extend a fragment's box to the row height so it can take a donor | P6 |
| R5c | donor from another photograph of the building | P1; needs registration between photographs |
| B3 | paired-rhythm (bay) rule | P7 |
| R-004a to c | seam and score thresholds on more data | P2 |
| T27 rest | point 5 with the author's facade mask | P4; waits for step 2 |
| — | marking for expert-rejected placements | smaller points |
