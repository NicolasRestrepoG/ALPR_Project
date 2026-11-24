# Shooting guide — how to get excellent data, fast

Field manual for the four photo sessions. Read once before you start; keep §2 and
§9 open on your phone while shooting.

**Total: about 290 photos, roughly 2½ hours spread over 3–4 occasions.**

---

## 1. The 60-second version

You are collecting two completely different things.

**CLOSE-UPS (~20–30 photos)** — perfect, boring photos of plates.
Sharp, square-on, filling the frame, clean plates, good light.
These become the *font* the system draws with.

**TEST-SET PHOTOS (~270 photos)** — deliberately varied photos of cars.
Different distances, angles, times of day. Dirty plates. Ugly plates.
These become the *exam* the system is graded on.

> **The trap:** your instinct will be to take nice photos. For the test set that is
> exactly wrong. A test set full of clean, close, straight-on plates makes the model
> look brilliant and tells you nothing about whether it works at the gate.

Then there is one rule that can silently ruin everything:

> ### ⚠️ Check for mirroring before photo #1
> Take one photo of a plate you can read. Compare it letter by letter with the real
> plate. If it comes out backwards, **stop** and fix the app. A mirrored photo is a
> permanently wrong label and there is no way to detect it later.

---

## 2. The walking rhythm (this is the whole trick)

Don't think about hitting quotas. Just walk a row of parked cars and follow this
rhythm. The variety happens automatically.

```
   Walk down a row. Photograph every car you pass.

   DEFAULT          stand ~4–5 m back, straight at the plate
   every 3rd car    first take 2–3 paces sideways     → the "angle" shots
   every 5th car    first back up to ~10 m            → the "small plate" shots
```

That's it. One photo per car, keep moving. Around **12 seconds per car**, so 120
photos is about 25 minutes of shooting plus walking time.

**Do not** stop to find a better angle, clean the plate, or wait for a car to move.
Anything awkward is exactly what you want.

---

## 3. What the two photo types are for

### Close-ups → become the 36 letters and digits

The typeface used on Colombian plates **is not published anywhere**. You cannot
download it. Every lookalike font produces a system that hits an invisible accuracy
ceiling, and it looks like a software bug for weeks before anyone suspects the font.

So the letters have to be rebuilt from your photographs. That is what these ~20–30
photos are.

**What good looks like:** the plate fills most of the frame, taken from directly in
front, sharp enough to see the edge of every stroke, no glare across the characters,
plate in decent condition.

### Test-set photos → become the measuring instrument

These are used to answer one question, over and over for a year: *how well does it
actually work?* They are **never** used to train the model — only to grade it.

That's why variety matters more than quality. Every condition missing from this set
is a condition you are blind to.

**What good looks like:** roughly what a camera at a gate would see. Some easy, some
hard, some barely readable.

---

## 4. Four rules

**1. Never guess a plate.** If you can't read it, mark it `unknown` in the app. A
guessed label silently corrupts the only thing you measure against. Marking
`unknown` is free and honest.

**2. Don't clean the plate.** Dirt, fading and damage are the point.

**3. Vary the distance more than feels natural.** Standing 10 m back feels too far.
It isn't — that's where the system gets hard, and it's what needs measuring.

**4. Don't zoom.** Use the normal 1× lens. Phone "2×" is usually a digital crop —
an upscaled blur, not a real distant plate.

---

## 5. The four sessions

Three *different* occasions for the test set — different hours, ideally different
days and different car parks. One session's photos all share the same light and
weather, and a model tested only on those looks great right up until the first
cloudy evening.

| | When | Photos | Time | Focus |
|---|---|---|---|---|
| **s1** | Daytime, overcast or mid-morning | **120** | ~50 min | Bulk of the set. Most of your negatives. |
| **s2** | Golden hour / dusk | **80** | ~35 min | Long shadows, half-lit plates, lots of angles |
| **s3** | After dark | **70** | ~35 min | Turn flash on for ~30 of them |
| **s4** | Any good even light | **20–30 close-ups** | ~25 min | Separate trip. Take your time. |

That totals about **270 test-set + 25 close-ups**, which leaves comfortable margin
above the minimums after you throw away the blurry ones.

### Per session, roughly

Of every 10 test-set photos: **8 normal cars, 2 "negatives"** (see §6).

Don't count precisely. The app shows you where you are.

### Session 3 (night) — the flash half

Halfway through the night session, **turn the flash on and leave it on** for the
rest. Aim for ~30 flash photos.

Why this matters more than it sounds: Colombian plates are *retroreflective* — under
a flash they light up like a road sign, completely unlike a normal photo. The real
system will use an infrared light that does exactly the same thing. If the model
later struggles and nobody has photos of what a plate looks like under flash, there
is no way to fix it. Thirty photos now is cheap insurance.

### Session 4 (close-ups) — do it on a different day

Shoot the close-ups on a different day from when you check them. Fresh eyes catch
crooked and blurry crops that you'll wave through on the day.

Take about 8–10 easy ones first, then **open the coverage screen** and shoot
specifically for whatever letters are still missing. Expect the last few letters to
take longer than the first twenty.

---

## 6. "Negatives" — the odd ones out

About **1 in 5** of your test-set photos should be something the system must
*refuse* to read. You need at least 40 of these.

| Type | What to look for | How common |
|---|---|---|
| **`non_plate`** | Any car with **no front plate** at all | Easy — very common |
| **`publico`** | **White** plate instead of yellow (taxis, buses, delivery) | Worth hunting |
| **`moto`** | Motorcycle plate — two lines instead of one | Easy |
| others | Diplomatic, official, trailer, temporary | Rare — grab if seen |

**White `publico` plates are the important ones.** They have the same three letters,
same three numbers, same size and same font as a resident's plate — only the colour
and the small print differ. If the system can't tell them apart, it will read one
confidently and open the gate for it. That is the exact failure this whole project
exists to prevent.

For `non_plate` and `moto` you don't type a plate number at all. Just set the type
and save.

---

## 7. Where to stand

At about 5 metres from the car:

```
        CAR
         │
    2 m  │  2 m
   ┌─────┼─────┐         straight on   → normal shot
   │     │     │         2 paces aside → about 10°  ✓ counts as an "angle" shot
   └─────┼─────┘         5 paces aside → about 25°  ✓ the limit — don't go past
         │
      5 m│
         │
        YOU
```

- **Sideways** is the one that's tracked. Two to five paces to the left or right.
- **Height** varies for free — SUVs and low sports cars give it to you automatically.
- **Keep the phone level.** Don't tilt it sideways.
- Don't go beyond ~5 paces aside — past that is outside what the system will ever see.

---

## 8. Deliberately collect these

They're easy to miss and disproportionately useful:

- [ ] Plates **half in sun, half in shade** — very common at a gate, and brutal
- [ ] Plates with a **decorative frame** around them (frames often cut off the first
      and last character — a specific failure the system needs to be tested on)
- [ ] Plates with **bolt heads** over the numbers
- [ ] **Dirty, faded, sun-bleached** plates
- [ ] **Backlit** — sun behind the car
- [ ] A **mix of vehicle heights** — sedans and SUVs
- [ ] **Old and new** plates — faded vs bright reflective

---

## 9. Field card

Screenshot this.

```
┌──────────────────────────────────────────────┐
│  BEFORE PHOTO #1                             │
│  □ one test shot — NOT mirrored?             │
│  □ flash off (except late in session 3)      │
│  □ 1× lens, no zoom                          │
│                                              │
│  RHYTHM                                      │
│  walk a row · one photo per car · keep going │
│    default        4–5 m, straight on         │
│    every 3rd      2–3 paces to the side      │
│    every 5th      back up to ~10 m           │
│                                              │
│  EVERY 10 PHOTOS                             │
│    8 normal cars                             │
│    2 odd ones (no plate / white / moto)      │
│                                              │
│  CAN'T READ IT?  →  mark "unknown"           │
│                     never guess              │
│                                              │
│  DON'T                                       │
│    clean the plate · skip the ugly ones      │
│    zoom · shoot everything from up close     │
└──────────────────────────────────────────────┘
```

---

## 10. After every session

Copy the folder off the phone into `data/raw/own_photos/`, then:

```bash
acquire import-captures --dry-run
```

It prints exactly what's still missing:

```
Sprint 2.1 exit gate (test set):
  OK positives       218 / 200
  -- negatives        31 / 40        ← still need 9 odd ones
  OK night            26% / 15%
  OK |yaw| > 10 deg   33% / 25%
  -- test set chars >= 5
       still needed: Qx3 Wx5 Xx2     ← hunt for these next time
```

Anything with `--` is not done yet. Plan the next session around it.

**Do this after session 1, not at the end.** It's how you find out you're short on
white plates while there are still two sessions left to fix it.

---

## 11. Before the real thing: a five-photo rehearsal

Worth twenty minutes.

1. Take 5 photos: 3 normal, 1 with no plate, 1 you deliberately mark `unknown`.
2. Edit one of them in the app; delete another.
3. Copy to the computer and run `acquire import-captures --dry-run`.
4. It should say **"dry run: valid"**.

If it complains, fix the app now. Discovering a format problem after 290 photos is
the one avoidable disaster here.

---

## 12. If something goes wrong

| Problem | What to do |
|---|---|
| Photos come out mirrored | **Stop.** Fix the app, discard everything shot so far. |
| Can't find white `publico` plates | Taxi ranks, delivery bays, bus stops. Weekday mornings. |
| Can't find `Q`, `W`, `X` | Genuinely rare here. Scan plates *without* photographing until you spot one — much faster. If you truly can't, we top up from an online plate archive. |
| Ran out of light mid-session | Stop and start a new session. Don't mix a session's conditions. |
| Photographed the same car twice | Harmless — duplicates get removed automatically. |
| Not sure if a plate counts as an "angle" shot | If you took paces sideways, tag it. Roughly right is fine. |

---

## 13. Privacy, briefly

- Frame the **plate plus about one car width**. Not people, not car interiors.
- Public or shared parking → `visitor-signage`.
- A specific neighbour's car → ask, get it **in writing**, mark `resident-signed`.
- Don't shoot inside private garages without asking.
- Don't share the raw photos. Only results and anonymised crops leave your machine.

Full detail in [legal_privacy_co.md](legal_privacy_co.md).

---

## 14. What "done" looks like

- [ ] ~270 test-set photos across **three** genuinely different sessions
- [ ] ≥200 normal yellow plates, ≥40 odd ones
- [ ] ≥40 taken after dark, ~30 of those with flash
- [ ] ~1 in 3 taken from an angle
- [ ] ~1 in 5 taken from far back
- [ ] ~20–30 close-ups covering **every letter and digit at least twice**
- [ ] `acquire import-captures --dry-run` shows **all OK**, no `--`

Then Phase 2 opens: the letters get extracted from your close-ups, and the plate
generator gets built.
