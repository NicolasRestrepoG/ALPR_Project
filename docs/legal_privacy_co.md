# Legal and privacy — Colombia

**Not legal advice.** Have a Colombian data-protection lawyer review before any
commercial deployment.

Started in Phase 1, finished in Phase 8. This document tracks obligations as they
become actionable; the full checklist is in the project plan, §22, and every
item on it must be checked **before the relay is connected**.

---

## Why this starts now and not in month ten

Two things decay if left: **licence provenance** and **consent records**. Six
months from now nobody will remember whether a checkpoint touched UFPR, or whether
the owner of that white pickup agreed to be photographed. Both are trivial to
record at the moment of acquisition and impossible to reconstruct afterwards.

## Legal basis

Plate images captured by a camera and linked to a determinable person are
**personal data** under **Ley Estatutaria 1581 de 2012**. The Superintendencia de
Industria y Comercio has repeatedly held that capture, recording, transmission,
storage, conservation and reproduction of video-surveillance imagery is
*tratamiento de datos personales* and fully subject to the regime.

For a *propiedad horizontal*, the responsible party is the community of owners,
represented by the administrator — not the person who built the system.

## Phase 1 obligations (live now)

### Licence lineage — §22 item 10

Tracked in code, not on paper:

- every `Record` carries a licence key; `alpr/data_sources/base.py` holds the one
  table that decides `commercial_ok`;
- unverified licences default to `unknown`, which is not commercially usable, so
  forgetting to check a source **excludes** it;
- `acquire audit` answers "can this be commercially deployed" and names the
  blocker;
- per-checkpoint lineage (config + corpus hash + every dataset touched) lands in
  Phase 4, when there is a checkpoint to attach it to.

**UFPR-ALPR and RodoSol-ALPR are academic non-commercial.** Any model whose weights
were influenced by them cannot be commercially deployed. They are marked
"evaluation only, if at all" and the plan does not depend on them.

### Attribution — §22 item 12

- **Platesmania** requires an active backlink *per image*. Enforced at the type
  level: `Record.__post_init__` refuses to construct a Platesmania record without
  an `attribution_url`. An image that cannot be attributed does not enter the
  corpus.
- **Roboflow CC-BY-4.0** requires dataset-level attribution. Belongs in the final
  writeup. Currently owed for `roboflow_proyecto_placas`
  (`placas-colombia/proyecto-placas-8arfj`, CC BY 4.0), and for
  `roboflow_placas_colombianas` if it is ever obtained.
  `roboflow_placas_colombia_itm` is MIT and needs no attribution, though crediting
  it costs nothing.
- **Dashcam frames** carry their video URL for the same reason, enforced the same
  way. Creative-Commons videos are preferred (`--cc-only` is the default), but "CC
  on YouTube" is a claim by an uploader who may not hold the rights, so the source
  stays marked non-commercial regardless.

### Gated datasets — requests to send

`acquire email ufpr_alpr --write` and `acquire email rodosol_alpr --write` generate
ready-to-send drafts in `data/licence_requests/` (gitignored: they carry the
sender's name and email from `.env`). Both go to the same contact and
have a 1–5 day turnaround, so they should go out early even though the plan expects
to make little or no use of them (risk R4: a request from a non-institutional
address is likely to be declined, and nothing depends on it).

**Not yet sent.** This is the one Sprint 1.2 exit-gate item that needs a human.

### Own photographs

The ~240 test-set images and ~20 glyph close-ups are the only irreplaceable data in
the project. When shooting (Sprint 1.2 / 2.1):

- [ ] photograph plates, not people — frame to exclude faces and building interiors;
- [ ] where a vehicle is identifiable to a specific resident, obtain **written**
      permission and record it alongside the image provenance;
- [ ] record per-image provenance at capture time: date, location, session,
      consent status;
- [ ] do not publish the raw test set. Metrics and anonymised crops only.

### Bystanders — §22 item 13

Frame to exclude the sidewalk where possible. Event crops store the plate plus
about one vehicle width; everything else is discarded.

## Deferred to later phases

| Obligation | Phase | Note |
|---|---|---|
| Aviso de privacidad signage (Spanish, legible from a vehicle) | 8 | Before any capture at the site |
| Written policy adopted by the assembly | 8 | |
| Retention: 30 days crops/video, 12 months audit log | 8 | Automated lifecycle rule + partition drop, not a good intention |
| Data-subject rights channel (*consulta* ≤ 10 días hábiles, *reclamo* ≤ 15) | 8 | Query endpoint must actually be answerable |
| Security: encryption, per-user roles, view auditing, camera VLAN, no default creds | 8 | |
| RNBD registration determination | 8 | Decreto 090 likely exempts residential *propiedad horizontal*; verify and **document the determination** |
| Processor agreements | 8 | Avoided entirely by staying on-premise |
| Breach notification procedure + named responsible person | 8 | |
| AGPL block (Ultralytics YOLOv8) | 6 | Removed in Sprint 6.3; CI-enforced |

## Purpose limitation

Access control only. **No** third-party sharing, marketing, mobility analytics,
speed estimation, pico y placa enforcement, or parking billing. Adding any of them
changes the declared purpose and voids the privacy notice. Enforced in the API's
authorization model, not just on paper.

Face recognition and occupant identification are deliberately out of scope —
biometric data falls under a much stricter consent regime, and the exclusion is a
selling point in the privacy notice.

---

## Change log

| Date | Change |
|---|---|
| 2026-08-08 | Started. Licence-lineage and attribution controls implemented in `data_sources` (Sprint 1.1). |
| 2026-08-08 | All 11 sources registered with licences attached. `data/manifest.json` + `acquire audit` answer the commercial-deployment question and name the blocker (Sprint 1.2). Licence-request drafts generated; **not yet sent**. |
| 2026-08-09 | Licences of the two "unverified" Roboflow projects read from the Roboflow API: `itm-mprof/placas-colombia` = **MIT**, `placas-colombia/proyecto-placas-8arfj` = **CC BY 4.0**. Both moved off the `unknown` key onto the commercially clean path; CC-BY attribution now owed for the latter. 376 Colombian images on disk, audit reports YES. |
