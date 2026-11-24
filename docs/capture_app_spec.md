# Capture app — build specification

An Expo (React Native) app for photographing and labelling Colombian plates in the
field. Built once, used for three shooting sessions, then retired.

**Status:** specification. The output contract is enforced by
`alpr/data_sources/captures.py` and `acquire import-captures` — build against those,
not against your memory of this document.

---

## 1. What it is for

The ~240 test-set photographs and ~20 glyph close-ups are the only irreplaceable
data in the project. Everything else can be re-downloaded. Two things make an app
worth building rather than shooting on the stock camera:

**Hard labels are unrecoverable later.** A dirty plate at dusk, 25° off-axis, 90 px
wide is among the most valuable images in the test set — but only if you know what
it says. Standing in front of the car you can read it. Three weeks later at a laptop
you cannot, so you would silently drop exactly those images and your test set drifts
easy. An easy test set inflates every number you ever report.

**Character coverage has to be steered while you can still act on it.** The Sprint
2.1 exit gate needs every character A–Z and 0–9 at least five times. Finding out at
home that you have two `Q`s means another trip. A live counter turns that into a
five-minute detour.

### Explicitly out of scope

| Not in the app | Why |
|---|---|
| **Corner labelling** | Precision is load-bearing — Sprint 6.2 targets mean corner error ≤1.5% of plate width. Tapping four corners outdoors on a phone is worse than clicking with a mouse, zoom and undo. A second corner implementation also doubles the chance of getting TL/TR/BR/BL order wrong, and that failure is completely silent. Desktop tool does corners. |
| Accounts, sync, backend | It is a tool, not a product. |
| Cloud upload | These are photographs of residents' vehicles — personal data under Ley 1581. Keeping them off third-party storage removes the international-transfer question entirely (plan §22 item 9). Device → cable/AirDrop → Mac. |
| GPS coordinates | Do not collect location data on neighbours' cars while building a privacy-sensitive system. A session label is enough provenance. |
| Any image editing | No crop, no filter, no HDR stacking, no "beautify". |

---

## 2. Output contract

**This section is the actual specification.** Everything else is a suggestion about
how to make it pleasant to use.

### 2.1 Directory layout

The app produces one folder, transferred wholesale into the project:

```
alpr_capture/
├── testset/
│   ├── s1_20260809_193042_0001.jpg
│   └── s1_20260809_193118_0002.jpg
├── close_ups/
│   └── s1_20260809_194511_0003.jpg
└── captures.jsonl
```

It maps directly onto `data/raw/own_photos/` in the project.

### 2.2 Filenames

```
{session}_{YYYYMMDD}_{HHMMSS}_{seq:04d}.jpg
```

`seq` is a monotonic counter across the whole app install, never reset per session.
Filenames are assigned at capture and **never changed afterwards** — the JSONL
references them, and a rename orphans the label.

Lowercase `.jpg`. No spaces. No plate string in the filename (it changes when you
correct a typo, and the file must not).

### 2.3 captures.jsonl

**One JSON object per line, UTF-8, append-only.** Never rewrite or reorder the file.

Append-only is what makes it crash-safe: a killed app loses at most the line it was
writing, and every earlier label survives. It also gives edits for free.

```json
{"schema":1,"filename":"s1_20260809_193042_0001.jpg","kind":"testset","plate":"ABC123","plate_type":"particular","tags":["night","yaw_high"],"consent":"visitor-signage","session":"s1","captured_at":"2026-08-09T19:30:42-05:00","image_width":4032,"image_height":3024,"device":"iPhone 15","app_version":"1.0.0"}
```

**Editing a label**: append a new line with the same `filename`. **Last line wins.**
Do not modify the earlier line — the history is a useful audit trail and rewriting
risks corrupting the file.

**Deleting a capture**: append `{"schema":1,"filename":"…","deleted":true}`. Leave
the image file in place; the importer ignores both.

### 2.4 Field reference

| Field | Type | Required | Rules |
|---|---|---|---|
| `schema` | int | yes | Always `1`. |
| `filename` | string | yes | As §2.2. Must exist in `kind`'s directory. |
| `kind` | string | yes | `testset` or `close_ups`. |
| `plate` | string | conditional | **Required and must match `^[A-Z]{3}[0-9]{3}$` when `plate_type` is `particular`.** Optional otherwise; no format check. Omit or `null` when unreadable. |
| `plate_type` | string | yes | Exactly one of: `particular`, `publico`, `moto`, `diplomatic`, `official`, `trailer`, `temporary`, `non_plate`, `unknown`. |
| `tags` | string[] | yes (may be `[]`) | From §2.5 only. Unknown tags are rejected. |
| `consent` | string | yes | From §2.6. |
| `session` | string | yes | `s1`, `s2`, `s3`, … |
| `captured_at` | string | yes | ISO 8601 **with offset**, e.g. `2026-08-09T19:30:42-05:00`. Not UTC-only — local time is what tells you it was dusk. |
| `image_width` | int | no | Pixels as stored. Cross-checked against the file on import. |
| `image_height` | int | no | Same. |
| `device` | string | no | e.g. `iPhone 15`. |
| `app_version` | string | no | |
| `deleted` | bool | no | `true` marks a tombstone; all other fields optional. |
| `note` | string | no | Free text, for anything odd. |

**Corners are absent by design.** The desktop tool adds them later. Do not emit
corner fields, even empty ones.

### 2.5 Tag vocabulary

Only these strings. The importer rejects anything else, so a typo cannot quietly
create a tag that no slice ever matches.

| Tag | Meaning | Sticky? |
|---|---|---|
| `night` | Shot after dark / under artificial light | **sticky** |
| `flash` | Camera flash fired | **sticky** |
| `yaw_high` | Standing more than ~10° off the plate's axis | per-photo |
| `frame` | Decorative or dealer frame around the plate | per-photo |
| `bolt` | Bolt heads intruding on the characters | per-photo |
| `dirt` | Visibly dirty, faded or damaged | per-photo |
| `wet` | Rain on the plate or lens | per-photo |

**Sticky** tags persist to the next photo until you turn them off; they describe the
session, not the plate. **Per-photo** tags reset after every save — they describe
this plate, and carrying them forward would mislabel the next one.

Do **not** tag blur or plate size. Blur is judged from the image; plate width in
pixels is computed from the corners in Phase 2. Tagging them by eye adds noise to
the very slices they define.

### 2.6 Consent values

Plan §22. Pick the one that applies:

| Value | When |
|---|---|
| `visitor-signage` | Public/semi-public parking, no identifiable owner association |
| `resident-signed` | Owner is a resident and has given **written** permission |
| `owner-verbal` | Verbal permission only — usable for development, flag before publishing anything |
| `own-vehicle` | Your own car |

If none apply, do not take the photo.

---

## 3. Screens

### 3.1 Session (launch screen)

```
┌────────────────────────────────┐
│  ALPR Capture                  │
│                                │
│  Session   [ s1 ] [ s2 ] [+]   │
│                                │
│  ── this session ────────────  │
│  testset      47               │
│  close-ups     3               │
│  started   19:12               │
│                                │
│  ── all sessions ────────────  │
│  testset     112 / 240         │
│  positives    98 / 200         │
│  negatives    14 / 40          │
│                                │
│  [    START SHOOTING     ]     │
│                                │
│  [ Coverage ]  [ Review ]      │
│  [ Export ]                    │
└────────────────────────────────┘
```

Sessions must be distinct occasions — different hours, ideally different days. The
plan's named risk is "all photos from one spot at one time of day": the model looks
excellent and fails at night. Three sessions minimum.

### 3.2 Capture

Full-bleed camera preview. Everything else is a thin overlay so the plate stays
visible.

```
┌────────────────────────────────┐
│ s1 · 47   [TESTSET|close-up]   │  ← mode toggle, sticky
│ ⚡flash  🌙night                │  ← sticky tags, tap to toggle
│                                │
│                                │
│         (live preview)         │
│                                │
│      ┌──────────────┐          │  ← guide box, 2.06:1
│      │              │          │     advisory only, not enforced
│      └──────────────┘          │
│                                │
│  missing: Q×3 W×5 X×2 7×1      │  ← live, from §5
│                                │
│           (  ●  )              │  ← shutter
└────────────────────────────────┘
```

- Rear camera only. **Never the front camera** (see §6).
- Mode toggle is sticky between shots.
- The 2.06:1 guide box helps you frame consistently; it is guidance, not a crop.
- The missing-characters line is the reason the app exists. Keep it visible.

### 3.3 Label — appears immediately after the shutter

```
┌────────────────────────────────┐
│  ┌──────────────────────────┐  │
│  │      (photo thumbnail)   │  │  ← tap to zoom; check focus
│  └──────────────────────────┘  │
│                                │
│  PLATE                         │
│  ┌──────────────────────────┐  │
│  │  A B C 1 2 3          ✓  │  │  ← autofocused, keyboard up
│  └──────────────────────────┘  │
│                                │
│  TYPE                          │
│  [particular] publico  moto    │
│  no-plate  other…              │
│                                │
│  THIS PLATE                    │
│  ( yaw>10° )( frame )( bolt )  │
│  ( dirt )( wet )               │
│                                │
│  CONSENT  [ visitor-signage ▾] │  ← sticky
│                                │
│  [ Retake ] [ Discard ]        │
│  [      SAVE & NEXT       ]    │
└────────────────────────────────┘
```

- The plate field is **autofocused with the keyboard already up**. This is the single
  biggest speed win across 240 photos.
- Auto-uppercase. Autocorrect, autocomplete and spellcheck **off** — iOS will happily
  "fix" `ABC123`.
- Live validation: green check at exactly `LLLDDD`, amber while incomplete, red on a
  clear violation (a digit in positions 1–3). Do not block saving on red for
  non-particular types.
- Selecting a non-`particular` type makes the plate field optional and drops the
  format check. Selecting `non_plate` disables it entirely.
- `SAVE & NEXT` writes the JSONL line, then returns straight to the camera. Per-photo
  tags clear; sticky ones persist.

### 3.4 Coverage

Two separate targets, because they serve different phases.

```
┌────────────────────────────────┐
│  TEST SET — every char ≥ 5     │
│  A6 B7 C9 D5 E8 F5 G6 H7 I5    │
│  J5 K5 L8 M9 N6 O5 P7 Q2 R8    │  ← Q amber (2 of 5)
│  S9 T6 U5 V5 W0 X1 Y5 Z5       │  ← W red, X red
│  0:9 1:12 2:8 3:7 4:6 5:9      │
│  6:5 7:1 8:6 9:7               │  ← 7 red
│                                │
│  CLOSE-UPS — every char ≥ 2    │
│  …same grid, target 2…         │
│                                │
│  ── slices ──────────────────  │
│  night        18%  ✓ (≥15%)    │
│  yaw > 10°    21%  ✗ (≥25%)    │
│  negatives    14   ✗ (≥40)     │
└────────────────────────────────┘
```

Colour: red `0`, amber below target, green at or above.

`Q`, `W` and `X` are rare in Colombian series and will be the last to fill. Expect to
hunt for them specifically, and to top up from Platesmania close-ups.

### 3.5 Review

Scrollable list — thumbnail, filename, plate, type, tags. Tap to reopen the label
form; saving appends a corrected line (§2.3). Swipe to delete appends a tombstone.

Filter to "unreadable / needs attention" so you can find anything you skipped.

### 3.6 Export

Zip or share the `alpr_capture/` folder, or expose it via the Files app. Show a
final integrity summary before export:

- images on disk vs. lines in the JSONL
- captures with no label / labels with no image
- particular plates failing the format check
- duplicate filenames

Refuse nothing — just report it. The importer checks the same things and is the
authority.

---

## 4. Capture flow

```
camera → shutter → write JPEG → label form (plate autofocused)
   ↑                                        │
   └──────────── SAVE & NEXT ───────────────┘
                 (appends one JSONL line)
```

Target: **under 12 seconds per photo**, so 240 photos is about an hour of actual
work spread across three sessions.

Rules:

1. Write the image file **before** showing the form. If the app dies during
   labelling, an unlabelled image is recoverable; a label for a nonexistent image is
   not.
2. Append the JSONL line on save, flushed immediately. One line, one `fsync`.
3. Never buffer labels in memory to write "later".
4. `Retake` discards the image and reopens the camera with the same sticky state.

---

## 5. Coverage logic

Read your own `captures.jsonl` (last-wins, tombstones excluded) and count:

```
for each record where plate_type == "particular" and plate is set:
    for each character in plate:
        counts[kind][character] += 1
```

Two independent tallies, `testset` and `close_ups`.

- **testset target: 5** per character — Sprint 2.1 exit gate. Below this, Phase 5's
  per-glyph error attribution is statistical noise rather than signal.
- **close_ups target: 2** per character — Sprint 2.2. These become the 36 extracted
  glyphs, and the glyphs are the single highest-risk item in the project.

Display the shortfall, not the count: `Q×3` means *three more needed*.

---

## 6. Camera settings — the three that would ruin the shoot

### Never mirror

Expo's camera mirrors the front camera by default, and some setups mirror the
preview *and* the saved file.

A mirrored plate is catastrophic. Plan §2 forbids horizontal flips outright: they
destroy the identity of `2`/`5` and `E`/`3`, and reverse the asymmetric
MinTransporte logo. A mirrored image in the test set is a permanently wrong label.

- Rear camera only. Do not offer a camera-switch control at all.
- `mirror={false}` explicitly.
- **Before shooting 240 photos**, take one picture of a plate and compare it letter
  by letter with the real thing.

### Bake rotation into pixels

If images are saved landscape with an EXIF orientation flag, the project's header
reader returns the *stored* dimensions while a desktop viewer shows the *rotated*
ones — so corner coordinates will not match image dimensions, and rectification
silently produces garbage.

Write pixels in their true orientation and strip the orientation tag. `image_width`
and `image_height` in the JSONL must equal what a raw header read returns.

### Full quality

- Maximum JPEG quality (`quality: 1.0`). The degradation pipeline models compression
  itself; starting from an already-compressed image stacks two different JPEG
  histories and produces artefacts no real camera makes.
- Full sensor resolution. Close-ups need roughly 400×800 px **per character**.
- No HDR stacking, no night mode, no portrait mode, no filters. Any smoothing
  destroys the stroke edges the glyph extraction depends on.

### Flash — shoot some deliberately

Colombian plates are **retroreflective**, and under flash they look nothing like a
diffuse render. Phase 5 holds "add a retroreflective term to the renderer" in reserve
as the highest-value unbuilt feature for exactly the moment the sim2real gap
plateaus.

Shoot roughly **15% of the night photos with flash on**, tagged `flash`. They cost
nothing now and are the reference images that work becomes possible from. The
eventual deployment uses co-axial IR illumination, which is the same physics.

---

## 7. Transfer into the project

```bash
# 1. Copy alpr_capture/ off the phone (cable or AirDrop) into:
#      data/raw/own_photos/
#    so you end up with:
#      data/raw/own_photos/testset/…
#      data/raw/own_photos/close_ups/…
#      data/raw/own_photos/captures.jsonl

# 2. Validate without writing anything
acquire import-captures --dry-run

# 3. Write labels.csv
acquire import-captures

# 4. Progress against the exit gate
acquire fetch own_photos
```

`import-captures` validates every rule in §2, resolves last-wins and tombstones,
cross-checks images against labels, and writes `labels.csv` with the eight corner
columns **empty**. The Phase 2 annotation tool fills those in.

Run `--dry-run` against a handful of test captures **before** the real shoot. Finding
out the schema is wrong after 240 photos is avoidable.

---

## 8. Acceptance checklist

Before the first real session:

- [ ] One photo of a known plate, compared character by character — **not mirrored**
- [ ] `image_width`/`image_height` in the JSONL match a raw header read of the file
- [ ] A capture, an edit, and a delete all round-trip through `--dry-run` cleanly
- [ ] `ABC123` survives the plate field with autocorrect off
- [ ] A `publico` capture saves with no plate string
- [ ] Killing the app mid-form leaves a valid JSONL and an orphan image, not corruption
- [ ] Coverage screen counts a newly added plate immediately
- [ ] Full-resolution output, no filter applied

During the shoot:

- [ ] Three sessions, different hours, at least one after dark
- [ ] ≥200 `particular`, ≥40 negatives — **any** non-`particular` class counts
      (`publico`, `moto`, `non_plate`, `diplomatic`, `official`, `trailer`,
      `temporary`). `unknown` counts as neither.
- [ ] ≥15% night, ≥25% `yaw_high`
- [ ] Some deliberately ugly plates — dirty, framed, bolted, faded
- [ ] ~15% of night shots with `flash`
- [ ] ~20 close-ups covering every character at least twice
