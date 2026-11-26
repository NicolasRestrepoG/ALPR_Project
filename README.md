# ALPR Colombia

Automatic license plate recognition for Colombian *particular* plates, for vehicle
access control at a residential building.

The project is planned in 8 phases and 20 sprints. This README covers only how to
run what currently exists.

**Status: Phase 1 complete; Phase 2 in progress (Sprint 2.1).** `data_sources`
has all eleven sources registered, licence-audited and reachable through one
command. `plates` has the MT-001 geometry, rectification, and the `annotate`
labelling tool. Nothing else exists yet, deliberately — one phase creates one
module, and nothing is scaffolded ahead.

Own photographs are deferred until just before Phase 5. Until then, the glyph
source and a provisional test set come from the Roboflow images, labelled with
`annotate`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Phase 1 needs `requests`, `tqdm` and `gdown`; Phase 2 adds `numpy` and
`opencv-python-headless`. PyTorch arrives in Phase 4, not on day one.

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
.venv/bin/pytest          # a few seconds; no ML framework required
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
```

The page shows each plate zoomed to its source box. Click the plate's four
**outer** corners in any order — they are ordered TL/TR/BR/BL by the same helper
the sources and the rectifier use. A magnifier follows the cursor; drag a corner to
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

Labels go to `data/labels/<source>.csv` (or `data/raw/own_photos/labels.csv`), in
full-image pixel coordinates, in the column layout `own_photos` already reads.

### Plate geometry — where each number comes from

`alpr/plates/geometry.py` is the single source of truth for plate layout, in
millimetres, and records the source of every field:

| Source | Fields |
|---|---|
| MT-001 as published (confirmed with a tape measure) | plate 330 x 160, characters 40 x 80, stroke, character gap, municipality letter width |
| Tape measure on a real plate | space above the characters (30), municipality letter height (23) |
| `annotate measure` on 147 rectified photos | logo gutter (28), gap above the municipality (15) |

In photos the characters' bottom edge reads about 5 mm low, because the embossed
characters cast a shadow below them. Measure vertical layout from top edges, and
expect the renderer to need that shadow to look real.

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
  annotate           (CLI)      click-4-corners labelling tool
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
