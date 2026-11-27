# ALPR Colombia

Automatic license plate recognition for Colombian *particular* plates, for vehicle
access control at a residential building.

The project is planned in 8 phases and 20 sprints. This README covers only how to
run what currently exists.

**Status: Phases 1–3 complete; Phase 4 in progress (Sprint 4.1).** `data_sources`
has all eleven sources registered, licence-audited and reachable through one
command. `plates` has the MT-001 geometry, rectification, the `annotate`
labelling tool, a frozen provisional test set, the 36 extracted glyphs, and the
plate renderer. `synth` has the degradation pipeline, the background corpus and the
hash-identified corpus builder. `recognizer` has the model architecture.
Nothing else exists yet, deliberately — one phase creates one module, and nothing
is scaffolded ahead.

Own photographs are deferred until just before Phase 5. Until then, the glyph
source and a provisional test set come from the Roboflow images, labelled with
`annotate`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Phase 1 needs `requests`, `tqdm` and `gdown`; Phase 2 adds `numpy` and
`opencv-python-headless`; Phase 4 adds `torch` (used only by `alpr/recognizer`).

## Credentials

Copy the template and fill in what you need:

```bash
cp .env.example .env      # .env is gitignored; .env.example holds no values
acquire keys              # which credentials resolve, and from where
```

`acquire keys` never prints a value — only whether one is set and which file or
variable it came from. A real environment variable always beats the file, so a
one-off override works without editing anything:

```bash
ROBOFLOW_API_KEY=other_key acquire fetch roboflow_placas_colombianas
```

Only `ROBOFLOW_API_KEY` is needed in Phase 1, and only for the three Roboflow
sources. Everything else works without any credentials.

## Test

```bash
.venv/bin/pytest          # ~10 s; only tests/recognizer needs PyTorch
.venv/bin/ruff check .
.venv/bin/ruff format .
```

The acquisition layer imports nothing heavier than the standard library at load
time — `requests` and friends are imported inside the functions that download. That
property is enforced by a test (`test_phase_contract.py`), not just intended.
`alpr.plates` needs numpy and OpenCV but no ML framework, also enforced by a test.

## The `acquire` command

```bash
acquire list                       # every source with licence, access mode, purpose
acquire list --commercial-only     # the commercially deployable subset
acquire list --purpose real_test   # sources serving one purpose
acquire paths --ensure             # show / create the data layout

acquire fetch divipola mt001       # download named sources
acquire fetch --all --skip-manual  # everything obtainable unattended
acquire index --all                # normalize what's on disk into JSON Lines

acquire manifest --audit           # rebuild data/manifest.json, print the audit
acquire verify                     # does the manifest still match the disk?
acquire audit                      # can what we have be commercially deployed?
acquire keys                       # which credentials resolve, and from where
acquire email ufpr_alpr --write    # draft the licence-request email

acquire import-captures --dry-run  # validate the capture app's output
acquire import-captures            # ...then write labels.csv
```

`acquire audit` exits 0 when clean and 2 when blocked, so it can gate a script.

`data/` is never in version control. Point it elsewhere with `ALPR_DATA_ROOT`:

```bash
ALPR_DATA_ROOT=/Volumes/external/alpr acquire fetch ccpd
```

### Sources

| Source | Licence | Commercial | Access | State |
|---|---|---|---|---|
| `ccpd` | MIT | yes | automatic (13.2 GB) | 70,516 indexed (20% sample) |
| `divipola` · `mt001` | Colombian public government data | yes | automatic | 1,021 municipalities |
| `roboflow_placas_colombia_itm` | MIT | yes | API key | 100 images |
| `roboflow_proyecto_placas` | CC BY 4.0 | yes | API key | 274 images |
| `roboflow_placas_colombianas` | CC BY 4.0 | yes | API key | **blocked upstream** |
| `own_photos` | own work | yes | **you, with a phone** | |
| `platesmania` | site terms, per-image backlink | no | manual | |
| `dashcam` | YouTube ToS, per video | no | manual | |
| `ufpr_alpr` · `rodosol_alpr` | academic non-commercial | no | signed agreement by email | |

Licences for the two projects the plan listed as "unverified" were read from the
Roboflow API on 2026-08-09 — MIT and CC BY 4.0 — so both are on the clean path.

`roboflow_placas_colombianas` has 1,770 images but **zero generated versions**
upstream, and only a project's owner can generate one. Nothing to export until
someone forks it into their own workspace and generates a version there (with no
augmentation).

Mapillary is deliberately absent: plates are auto-blurred at upload, which makes it
structurally useless here.

### Never horizontal-flip

Roboflow bakes augmentation into images at *export* time, so a mirrored version
cannot be un-mirrored downstream. `fetch` inspects each version's augmentation
settings and **refuses** any that enables horizontal flip (plan §2 — a front plate
is never seen mirrored, and flipping destroys the identity of `2`/`5`, `E`/`3` and
the asymmetric MinTransporte logo).

### CCPD on a small disk

The archive is 13.2 GB and extracts to roughly the same again. On a laptop that
does not have 27 GB spare:

```bash
acquire fetch ccpd --no-extract                              # 13.2 GB archive only
acquire fetch ccpd --keep-fraction 0.2 --delete-archive      # ~60k images, ~2.6 GB
acquire fetch ccpd --subsets ccpd_base ccpd_np               # only what a stage needs
```

Sampling is a stable hash of the filename, not a random draw, so re-running after an
interruption converges on the same images rather than an overlapping set. The
archive is deleted only after the completion marker is written.

The full tree is **~24 GB**, not ~13 GB — xz compresses tar's per-member padding
across ~350k small files, so the archive is much smaller than what it unpacks to.
The preflight check uses the measured figure.

**Box and corners disagree in `ccpd_base`.** Across the 69,895 indexed records with
corners, every subset except `base` publishes a box that is exactly the hull of the
four vertices. All 39,847 `base` records disagree — median IoU 0.91, min 0.54, with
deltas of inconsistent sign — so the two were annotated independently. Both are kept
as published. Phase 6 should derive the box from the corners rather than mixing the
two conventions.

## Labelling plates — the `annotate` command

```bash
annotate serve roboflow_proyecto_placas   # opens the labelling page (127.0.0.1 only)
annotate status roboflow_proyecto_placas  # progress, plate types, character coverage
annotate sheet roboflow_proyecto_placas   # contact sheet of rectified crops
annotate measure roboflow_proyecto_placas # measure plate layout from labelled plates
annotate freeze roboflow_proyecto_placas roboflow_placas_colombia_itm   # freeze the test set
annotate verify                           # recheck the frozen set's checksums
annotate glyphs roboflow_proyecto_placas roboflow_placas_colombia_itm   # extract the 36 glyphs
annotate render ABC123 --municipality MEDELLIN   # draw one plate
annotate overlay roboflow_proyecto_placas roboflow_placas_colombia_itm  # renderer vs photos
```

The page shows each plate zoomed to its source box. Click the plate's four
**outer** corners in any order (the outer edge of the metal, black border included)
— they are ordered TL/TR/BR/BL by the same helper the sources and the rectifier
use. A magnifier follows the cursor; drag a corner to
adjust it. As soon as four corners are down, the rectified 192 × 96 crop appears
with the MT-001 character cells drawn over it: if it is rotated, mirrored, or the
characters sit outside their cells, the click is wrong — fix it before saving.

Type the plate, pick the type (`1` particular, `2` público, `3` moto, `4` no plate,
`5` unknown), tag it, and press Enter to save and move to the next unlabelled plate.
Every save is written immediately; stop and resume at any time.

Tag the sharpest, most frontal plates `glyph_source`: Sprint 2.2 cuts the 36
characters from them, and they must stay out of the provisional test set. The
coverage panel counts characters over **distinct** plates — the Roboflow sets
photograph the same car many times — against the gate of five per character.

Some published photos arrive sideways or upside down (their orientation was lost
before upload). Corner order is decided by image position, so it only works on an
upright plate: turn the photo first with **⟳ / ⟲** (`R` / `Shift+R`), then click.
The turn is saved as a `rot90`, `rot180` or `rot270` tag and the corners are stored
in the upright frame. Anything that uses a label's corners must therefore read the
image with `load_labelled_image(path, label)` from `alpr.plates.labels`, which applies the
turn. A plate that is merely tilted (under ~45°) needs no rotation.

Labels go to `data/labels/<source>.csv` (or `data/raw/own_photos/labels.csv`), in
full-image pixel coordinates, in the column layout `own_photos` already reads.

### The provisional test set — `annotate freeze`

Until the own photos exist, the V0 recognizer is scored on labelled Roboflow
plates, frozen into `data/provisional_test/provisional/`:

- **Every photo of a `glyph_source` car is left out.** Those cars supply the
  renderer's characters; testing on them would measure memory, not reading.
- **Results are per car.** Each entry's `group` is its plate text, and near-identical
  photos of the same car (perceptual hash within 4 bits) are dropped.
- **Unreadable plates are left out**; they carry no ground truth.
- Images are copied byte for byte with source, licence and attribution per entry,
  made read-only and listed in `SHA256SUMS`. `annotate verify` rechecks them, and
  `freeze` refuses to overwrite an existing set.

Read entries with `load_entries` and images with `entry_image` from
`alpr.plates.testset` (it applies any rotation tag). These images must never be
trained on, including in a later detector fine-tuning split.

### The glyphs — `annotate glyphs`

The plate typeface is not published, so the renderer's characters are cut from
plates tagged `glyph_source`. Each character is rebuilt from **all** its
instances, one photo per car (the sharpest): the crops are rectified at 4 px/mm,
masked to the character's own ink, aligned to each other and median-combined, so
blur, glare and dirt that differ between photos cancel out.

Output goes to `data/assets/glyphs/`: one PNG per character plus `logo.png`
(255 = ink), `glyphs.json` with the source cars, confidence and measured ink size
of each, and `qa_sheet.png` with each glyph beside its source crops. Glyphs from
fewer than three cars are marked `low`; replace them from the own close-ups. Read
glyphs with `load_glyphs` from `alpr.plates.glyphs`.

Glyphs are stored as photographed: their ink is ~6% taller than the 80 mm cell
(the embossing seen from above). Scaling them into the cell is the renderer's job.

### The renderer — `render(PlateSpec)`

`alpr.plates.render.render(PlateSpec("ABC123", "MEDELLIN"))` returns a clean
1320 x 640 RGB plate (4 px/mm) drawn from the extracted glyphs at their true size.
Characters, municipality letters and the border rim are embossed; the plate is lit
with diffuse and specular shading from `PlateSpec.lighting`, raised parts cast a
shadow, and `PlateSpec.view` adds the parallax of seeing the relief at an angle.

That parallax explains a measurement from earlier: photographed from above, the
characters' top edge stays put while the raised face slides down, so their ink
measures ~85 mm instead of 80. The municipality letters barely move, so they are
modelled lower (`MUNI_RELIEF`). Phase 3 samples lighting and view per image.

`annotate overlay` is the Sprint 2.2 gate. It measures the labelled photos and
renders of the same plate strings the same way, and compares ten landmarks
(character top and bottom, six slot centres, municipality top and bottom):

- at the viewing tilt that matches the Roboflow photos (`--view-y 3`): worst
  landmark 1.25 mm, slot centres within 0.2 mm (gate 2 mm);
- frontal: the character bottom differs by 4.6 mm, as expected from the above;
- render time ~11-13 ms on an M-series Mac (gate 15 ms).

It also writes `overlay.png`, each photographed plate beside its render.

### Plate geometry — where each number comes from

`alpr/plates/geometry.py` is the single source of truth for plate layout, in
millimetres, and records the source of every field:

| Source | Fields |
|---|---|
| MT-001 as published (confirmed with a tape measure) | plate 330 x 160, character height 80, stroke, character gap, municipality letter width |
| Tape measure on a real plate (the photos agree) | space above the characters (30), municipality letter height (23), character width (41) |
| `annotate measure` on 147 rectified photos | logo gutter (27), gap above the municipality (15) |

In photos the characters' bottom edge reads about 5 mm low, because the embossed
characters cast a shadow below them. Measure vertical layout from top edges, and
expect the renderer to need that shadow to look real.

## Synthetic plates — the `generate` command

```bash
generate backgrounds        # ~19k vehicle-front crops (CCPD + Roboflow), ~20 s on 8 cores
generate demo               # data/synth/demo_sheet.png: rows from severity 0 to 1
generate compare            # data/synth/compare.png: 20 real + 20 synthetic, shuffled
generate corpus             # 20k training crops → data/corpus/<hash>/ (~75 s on 8 cores)
generate ablation           # four 20k corpora that differ only in stage order
```

A sample is a rendered plate taken through seven stages, **in this order**, each
where a real camera does it (one module each under `alpr/synth/stages/`):

1. **warp** — perspective at full render resolution (resampling only once, later)
2. **composite** onto a vehicle-front background, still at render resolution, so
   the plate's edge is band-limited like everything else
3. **downsample** to the sampled plate width with an area filter — the one band-limit
4. **photometric** — exposure, white balance, shadow, specular band, headlight
   bloom, infrared mode, then the tone curve
5. **blur** — motion and defocus
6. **noise** — Poisson shot + Gaussian read, per pixel, after blur
7. **JPEG** — last

then it is rectified with slightly jittered corners into the 192 × 96 crop the
recognizer reads. Before the warp, worn paint, grime, decorative frames and bolt
heads go on the flat plate; grille bars go on the scene before the downsample.

Every random choice is drawn first into a `SampleParams` (`alpr/synth/params.py`),
so a sample is reproducible from its parameters. One **severity** scalar in [0, 1]
scales every magnitude: 0 is a clean, frontal, large plate; 1 lets every effect
reach the harsh end of its range. Plate and ink colours are sampled from the range
measured on real labelled plates (amber sheet, warm dark ink).

Backgrounds are crops around real plates with the plate painted out, placed so the
synthetic plate lands where the real one was. **Photos in the frozen test set are
never used.** `generate compare` writes a numbered sheet and a separate answer key:
if you can sort the crops at a glance, the difference is visible to the model too.

### The corpus — `generate corpus`

A `CorpusSpec` (size, seed, class mix, severity range, stage order, sampling
ranges) plus the digests of every input (glyphs, background index, municipality
list, CCPD index, generator version) hash to a 16-character id; the corpus lives
in `data/corpus/<id>/`. Same spec and inputs, same id, same pixels: sample `i`
draws from its own seeded stream, so worker count does not matter, and a finished
corpus is reused instead of rebuilt. **Every trained model must record the id of
the corpus it saw.**

Each sample is written already rectified (corner jitter baked in) as
`<index>_<class>_<text|none>.png`, with `labels.jsonl` alongside. The mix:

| Class | Share | What it is |
|---|---|---|
| particular | 88% | the plates to read — the only `valid` class |
| público | 5% | white sheet, same layout |
| moto | 2% | `LLLDDL` with COLOMBIA in the strip, as in the labelled photos |
| ccpd | 3% | real Chinese plates, rectified from their corners |
| non-plate | 2% | textured, plate-shaped patches of vehicle fronts |

Letters and digits are uniform and balanced per position (spread 3.5% / 1.7%);
plate width is log-uniform 70–260 px. Municipalities are uniform: the DIVIPOLA list
has no population column, so the plan's population weighting waits for one.

To keep 250+ samples/s, each plate is rendered at the coarsest of 1, 2 or 4 px/mm
that keeps the warp at least 4x its final width — the warp still happens well
above the final resolution, before the single band-limiting downsample.

`generate ablation` builds the correct order plus three wrong ones
(composite-after-downsample, noise-before-blur, JPEG-before-noise). Training them
is Phase 4's first job: the correct order must win by at least 3 points.

## The recognizer — `alpr/recognizer`

`Recognizer.read(crop)` takes a rectified 96 × 192 RGB crop and returns a
`Reading(plate, per_position_confidence, validity)`. The model (plan §9):

| Part | Shape | Parameters |
|---|---|---|
| Trunk: ten 3×3 conv+BN+ReLU, 32→64→128→256→256; pooling 2,2,2 then (2,1),(2,1),(3,1) | 3×96×192 → 24×256 | 2,353,888 |
| Positional attention head: 6 learned queries over 24 columns; letter classifiers (26) at positions 1–3, digit classifiers (10) at 4–6; validity head | → 6 logit sets + validity | 183,533 |
| CTC head (the Sprint 5.3 ablation): BiGRU 256→2×128, 37-way | → 24×37 | 305,957 |

The parameter counts are tested exactly: if one is off, a layer is wrong. The
letter/digit split makes the classic plate confusions (0/O, 1/I, 8/B, 5/S, 2/Z,
6/G) impossible rather than unlikely — position 2 has no `0` to output.

Attention starts on the expected columns (2, 6, 10, 14, 18, 22): the queries are
initialised to those columns' positional embeddings and the key projection to the
identity. `alpr.recognizer.diagnostics.AttentionLog` records each epoch's peaks,
flags any that are not left-to-right ordered, and draws a per-epoch figure.

**Apple-GPU (MPS) runs are for shape and loss-direction sanity only.** MPS and CUDA
differ numerically; every reported number comes from a CUDA run.

## Two contracts that cause silent failures if ignored

**Corners are always ordered TL, TR, BR, BL.** CCPD publishes its four vertices
starting at the *right-bottom*. Consumed in file order they have the same winding
as the canonical order — it is a rotation of the same cycle, not a reversal — so a
winding check alone will not catch the mistake. Nothing errors: the detector trains
happily, the rectifier emits upside-down plates, and the recognizer plateaus at
chance. `order_corners()` converts geometrically and `Record` normalizes on
construction, so a Record that exists carries canonical corners.

**Box-only sources set `corners=None`.** There is deliberately no
`box_to_corners()` helper. Fabricating corners from a bounding box teaches the
detector that every plate is axis-aligned; mask the keypoint loss on those records
instead.

## Licence discipline

Provenance is attached at acquisition time, never reconstructed later. Records
carry a licence *key*; the licence table is the single place that decides
`commercial_ok`. Anything unverified defaults to the `unknown` licence, which is
**not** commercially usable — so forgetting to check a licence shrinks the dataset
rather than silently poisoning a checkpoint.

Licences that require per-image attribution (Platesmania) are enforced in
`Record.__post_init__`: a record without its backlink cannot be constructed.

The commercially clean path is: own synthetic + own captures + CCPD (MIT) +
Roboflow CC-BY.

## Layout

```
alpr/data_sources/   [phase 1]  acquisition — one module per source
  base                          Record, License, Status, corner ordering
  storage                       paths, atomic download, extraction, checksums
  registry                      registration, filtering
  manifest                      what's on disk, when, under what terms + the audit
  imagesize                     header-only image dimensions, no image library
  assets · ccpd · roboflow · gated · platesmania · dashcam · own_photos
  acquire            (CLI)
alpr/plates/         [phase 2]  the plate domain model
  geometry                      MT-001 in millimetres — single source of truth
  rectify                       4 corners → canonical 192 × 96 crop
  labels                        label storage, items to label, coverage
  testset                       freeze and verify the provisional test set
  glyphs                        extract the 36 characters + logo from labelled plates
  render                        PlateSpec → embossed, lit 1320 × 640 plate
  annotate           (CLI)      click-4-corners labelling tool
alpr/synth/          [phase 3]  the synthetic corpus
  params                        every random choice for a sample; severity
  stages/                       warp · composite · downsample · photometric · blur · noise · jpeg
  augment                       bolts, frames, dirt, wear, grille, bloom, infrared
  backgrounds                   vehicle-front crops, plate painted out, test set excluded
  pipeline                      render → occluders → the seven stages → 192 × 96 crop
  negatives                     CCPD and non-plate crops (público and moto are rendered)
  corpus                        spec → hash → directory; parallel, deterministic builder
  generate           (CLI)      backgrounds, demo, compare, corpus, ablation
alpr/recognizer/     [phase 4]  the from-scratch recognizer (the only PyTorch user)
  model                         trunk, attention head, CTC head, Reading, read()
  diagnostics                   attention drift: per-epoch peaks and figure
tests/                          mirrors the module layout
data/                           never in version control
docs/
  shooting_guide.md             field manual for the four photo sessions
  capture_app_spec.md           build spec for the phone capture app
  legal_privacy_co.md           started here, finished in Phase 8
```

## Shooting the real test set

The test photographs and glyph close-ups are the only irreplaceable data in the
project. Two documents:

- **[docs/shooting_guide.md](docs/shooting_guide.md)** — how to plan and run the
  four sessions. Read this before shooting.
- **[docs/capture_app_spec.md](docs/capture_app_spec.md)** — build spec for the
  phone capture app. `alpr/data_sources/captures.py` enforces its output contract,
  so build against `acquire import-captures --dry-run`, not against the prose.

The app records the plate string and field-only conditions. **Corners are labelled
later, on a desktop**, by the Sprint 2.1 annotation tool: corner precision bounds
the recognizer's accuracy, and a phone-tapped corner is not good enough.
