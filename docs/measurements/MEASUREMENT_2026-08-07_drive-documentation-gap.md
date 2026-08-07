# MEASUREMENT 2026-08-07 — the documentation tree was not on Google Drive

**Measured, remediated and re-measured on 2026-08-07, between 02:42 and 04:18.
Repository at `8ff555f`, then `db61455`.**

## What was measured

The standing durability requirement puts the roadmap and the documentation on
Google Drive via the rclone remote `genvarcla:`. A dry run of

    rclone copy docs genvarcla:genomic-variant-classifier/docs --include "*.md"

reported `Transferred: 272 / 272` beside `Checks: 1 / 1`. That counter is
ambiguous about which side was listed, so it was not treated as an answer.

`rclone check --one-way --combined` itemises every file as identical, missing or
differing, and returned:

    +  272   present locally, MISSING on Drive
    =    1   identical

Corroborated twice over. `rclone lsjson` on the destination returned a single
entry — the `sessions` directory — and `rclone size` reported **1 object,
12,421 bytes** for the whole tree. The Drive folder's modification time was
**2026-07-06**, five weeks stale.

## What that meant

`ROADMAP.md`, `CHANGELOG.md`, all 50 incident records, all 55 measurements, both
errata and every session document from 2026-08-01 onward — including the OP-1
step 4 record written that morning — existed in exactly two places: one laptop's
disk and GitHub. Neither is the durability path the project specifies.

## Remediation, and its verification

`rclone copy` at 02:47, 3.648 MiB in 2m33.7s, 272 transferred. Then the same
`--combined` check:

    =  273   identical
    +    0

with rclone's own summary reading `0 differences found` and `273 matching
files`. `rclone size` returned **273 objects, 3.386 MiB**. Three files were
listed by name to confirm the session's own work had landed: `ROADMAP.md` at
411,040 bytes, `CHANGELOG.md` at 439,625, and
`SESSION_2026-08-06_op1-step4-selectors.md` at 12,757, all timestamped
2026-08-07.

## Why DRIVE-1 stays OPEN

At **04:18**, ninety minutes after the verified-clean sync, the same check
returned `* 1` and `= 272`: `CHANGELOG.md: sizes differ`, because commit
`db61455` had appended to it. `ROADMAP.md` still matched, having not moved.

The 272-document gap is closed. The condition that produced it is not: **nothing
enforces the sync**, and it has already recurred once within the same session.
Discharging the item on the strength of the remediation would close the symptom
and lose the condition — which is how the carried-item register records item
CI-l reading OPEN for eleven commits after it had been discharged, in the other
direction.

## RCLONE-1, raised alongside it

Every `rclone` invocation against `genvarcla:` prints:

> This remote uses rclone's shared Google Drive client_id, which is being
> retired and will stop working during 2026.

**rclone itself is not being retired and needs no replacement.** What is being
retired is one credential: the OAuth client identifier that ships inside rclone
and is shared by every rclone user. Google notified the project that it will
begin charging for application-programming-interface requests made against that
identifier, on a timescale of "later in 2026, following 90 days of notice", and
the documentation now states that creating your own is required rather than
recommended.

Two side benefits follow. Each identifier carries its own Google-imposed rate
limit — the default quota is ten transactions per second — and `genvarcla:`
currently shares that ceiling with every other rclone user in the world.

**The dependency:** creating a client identifier requires a Google Cloud
project, and `INCIDENT_2026-04-29` records that the project was deleted after
the billing incident. A new one is needed. It does not require billing to be
enabled; the charge Google announced applies to rclone's shared identifier
specifically, not to normal Drive application-programming-interface use.

The work is an afternoon. The exposure is every durability path in the project —
the roadmap, the documentation, the gnomAD exomes, the SpliceAI parquet. It
should be done well before the 90-day notice lands, not after.

Author: Monzia Moodie
