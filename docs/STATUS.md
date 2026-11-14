# Status after Phase 3 — what works, what does not, what needs the author

> **Türkçe özet.** Hata analizinden çıkan beş görev (T26-T30) ve iki ek (uzman kararı aracı, rapor sayfası)
> tamamlandı; yazar beş yerleştirmeyi inceledi (2 onay, 3 ret) ve ret notları üzerine kutu uzatma kuralı
> (T33) eklendi. Şu an s01'de 7 A hedefi var: restorasyon görünümünde yalnızca onaylı sağ kemer (847
> piksel) boyanıyor, dört pilaster "tahmin" olarak kararınızı bekliyor (ikisi yeni, ikisinin yerleşimi
> değiştiği için eski kararı geçersiz), iki yerleştirme reddedildi; hasarın % 85'i için kanıt yok. Sizden
> beklenen: (1) dört pilasteri `tools/target_review.py` ile yeniden değerlendirmek, (2) cephe maskesi,
> (3) `recovered_arch_001` kaydı, (4) lisansı belli bir fotoğraf. Bölüm 4'te komutlarıyla.

| | |
|---|---|
| Date | 2026-11-14 |
| Branch | `fix/failure-analysis`, ahead of `main` (`v0.1.0`); not merged, which is the author's decision |
| Structures | **n = 1 building** (`s01_test_church`, 628 x 904). `s03_church_later` is a second photograph of the same building: detection only so far |
| Reference run | `outputs/s01_test_church/20261114_164027` (generative fill off, the author's decisions applied). Open `report.html` in it |
| Tests | 173 pass (`python -m unittest discover tests`, no GPU) |

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
| approve / reject a placed donor, name another donor | `tools/target_review.py` | `data/structures/<id>/target_decisions.json` |

## 2. Result on s01 today

| | Restoration view | First-construction view |
|---|---|---|
| Completed (approved by the expert) | 847 px (0.8 % of the damage): `t_003` | the same |
| Guesses | 4 targets withheld (boxes hatched orange) | 4 targets painted and marked, 6,394 px (6.4 %) |
| Rejected by the expert | `t_002`, `t_006`: not painted | not painted |
| No evidence, open | 84.6 % | 88.1 % |
| Outside the facade limit | 4.5 % | 4.7 % |

20 targets: 7 tier A, 0 tier B, 13 tier C. In `v0.1.0` the same photograph gave 4,971 px in the
restoration view and 11,367 px in the first-construction view; the difference is what the acceptance tests
(T28), the slot rule (T29) and the expert's rejections now withhold.

**The expert's review (2026-11-14).** The author decided on the five placements of run `20261114_150843`:
approve `t_003`, `t_007`; reject `t_001` (ends short), `t_002` (arch leg does not reach the capital),
`t_006` (donor not an intact pilaster). The automatic tests agreed on 2 of 5
(`docs/rnd/RND-004-alignment-acceptance.md`). The box extension (T33) followed from the note on `t_001`;
it changed the placements of `t_001` and `t_007`, so those two decisions are stale and both wait again.

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

### P2. The automatic acceptance tests do not agree with the expert
- **What.** Of the first five placements the tests passed only `t_002`, which the expert rejected; the
  expert approved two the tests held back. None of the three rejections was about alignment as measured:
  two about extent, one about the choice of donor. After T33 four pilasters (`t_001`, `t_004`, `t_005`,
  `t_007`) are guesses and wait for the expert.
- **Cause.** `min_alignment_score` 0.80 is the cautious end of a range (0.58 to 0.85) in which the 9
  synthetic cases have no example; the four real scores are 0.64 to 0.73, inside that range
  (`docs/rnd/RND-004-alignment-acceptance.md`). The tests cannot be made more exact with this data.
- **Reading of the panels (guess).** `t_003` (right arch) and `t_001` (pilaster) look plausible in their
  comparison images. The three pilaster placements are 18 to 27 px wide and hard to judge at this size,
  also for a person; `t_006` has the lowest seam agreement (-0.33). This is for the expert, not for a
  threshold.
- **Needs the author.** Review the four pilasters (section 4, step 1). For `t_006` another donor can
  now be named in the review tool (T34); in a scripted trial `pilaster_010` scored 0.77 / seam 0.72 there
  against 0.64 / -0.33 for the automatic `pilaster_011`. Which element is the right donor is the expert's
  call.

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

### P6. A fragment whose box covers only the standing part (addressed by T33, results unjudged)
- **What.** Four pilaster boxes were shorter than their row; the donor was squeezed into the short box
  (`t_001`, rejected for it) or no donor passed the size filter (`recovered_pilaster_046`, `_020`).
- **Now.** T33 extends such a box to the vertical extent of the nearest intact same-class neighbour when
  the strip in between is damaged. All four are extended and placed; all four are guesses.
  `t_005` (`recovered_pilaster_020`) has alignment score -0.04 and is probably not usable **(guess)**.
- **Not covered.** `t_002`: the arch's box already reaches the capital, the placed arch leg does not.
  That is placement (one scale for both directions), see P5.
- **Needs the author.** The review of step 1. The extension can be overruled by drawing the box in the
  editor: a box that already has the row's extent is not changed.

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
- The review window was used by the author on 2026-11-14 (five decisions, with notes).
- The donor row of the review window (T34) was stepped through by a script; it has not been used by a
  person yet. The run folder `20261114_165810` holds that trial's donor choice; it is not a result.
- The decision flow was run once end to end on s01 by a script (approve `t_003`, reject `t_002`, run,
  clear, run): the approved guess was completed in the restoration view, the rejected one withheld in
  both, and after clearing the images were byte-identical to before. The run folder `20261114_150837`
  holds that test's decisions (noted "round-trip test, not an expert decision"); it is not a result.

## 4. Manual work for the author, in order

**Step 1 — decide on the four pilasters that wait (`t_001`, `t_004`, `t_005`, `t_007`).** Changes the
result at once; no drawing needed. The earlier decisions on `t_002`, `t_003`, `t_006` still apply.

For `t_006` (rejected because of its donor): pick another element in the donor row of the window and
press "Use this donor", run the second command, then open the window again on the new run folder and
approve or reject the new placement.

```
python tools/target_review.py --run outputs/s01_test_church/20261114_164027
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
| — | arch leg that stops short of the capital (`t_002`) | P5, P6 |
| R5c | donor from another photograph of the building | P1; needs registration between photographs |
| B3 | paired-rhythm (bay) rule | P7 |
| R-004a to c | seam and score thresholds on more data | P2 |
| T27 rest | point 5 with the author's facade mask | P4; waits for step 2 |
| — | marking for expert-rejected placements | smaller points |
