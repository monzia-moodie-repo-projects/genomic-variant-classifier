# MEASUREMENT 2026-08-30 -- gate timing has no established cause

**Author: Monzia Moodie**
**Measured at:** `c7e26eb`
**Status:** an OBSERVATION. Two plausible causes are refuted; none is established.

---

## 0. Why this exists

`GATE-TIMING-NOISE-EXCEEDS-TREND-1` has been carried as a finding since
2026-08-28. Across 2026-08-29 and 2026-08-30 I attributed it twice, in
conversation, to causes that the data refutes. Neither attribution was ever
committed, so no record is wrong -- but the refutation is worth keeping, so a
later session cannot borrow the same explanations.

---

## 1. The method: group by SUITE IDENTITY, not by date

An acceptance-gate run is only comparable to another run over the SAME
collected identity set. Two runs on different sets differ for a reason; two
runs on one set differ for none that the work explains.

Thirteen of the fourteen install attestations from 2026-08-29 and 2026-08-30
are grouped below by `suite_transition.after_digest`. `95f6c44`'s attestation
was not available when this was written, and its absence is stated rather than
smoothed over.

```
identity 17c32d1da8f78ecd   collected 5,682
    b67e30f  D-SESSION-17                  1103.8 s
    482c0c9  INCIDENT-ARTIFACT-IDENTITY    1126.9 s
    02c13b4  CORRECTION-REGISTRY-MISSED    1131.4 s
    54989dc  CORRECTION-PART-2             1205.6 s
    4 runs | spread 1.09x

identity 14339e6e37abcb84   collected 5,690
    81f6c4f  D-SESSION-18                  1111.6 s
    62d0a33  ALIAS-MERGE-DIGEST            1537.9 s
    2 runs | spread 1.38x

identity 60c7535c9a4ffeea   collected 5,705
    c8c3240  INCIDENT-GENCODE              1084.4 s
    6f9ae43  CORRECTION-PART-2             1106.2 s
    b3619f2  D-SESSION-19                  1108.8 s
    6545b64  CORRECTION-TOOL-ALREADY-KNEW  1114.5 s
    ac14ab5  RETIRE-SOURCENAME             1312.2 s
    5 runs | spread 1.21x

identity 02535ddfc579feab   collected 5,719
    fd6cd4e  DATA-TREE-GATE                1087.1 s
    c7e26eb  D-SESSION-20                  1577.6 s
    2 runs | spread 1.45x

identity 6937cf8536417101   collected 5,709
    69e8524  SOURCE-REGISTRY               1042.3 s
    1 run
```

---

## 2. REFUTED: "the longer run added tests that spawn subprocesses"

Said at `62d0a33`, whose unit added eight tests that each start a subprocess
and write files.

`62d0a33` and `81f6c4f` share identity `14339e6e37abcb84`. **The same 5,690
tests**, 1537.9 seconds against 1111.6 -- 1.38x apart. The subprocess tests were
present in both.

---

## 3. REFUTED: "consistent with the manifest's 1.87x at 8.9 per cent free disk"

`configs/data_manifest.yaml` records, dated 2026-07-21, that the suite showed a
1.87x wall-clock range over nine runs while the volume was 8.925 per cent free.
That observation is sound for its own conditions.

MEASURED 2026-08-30 by `shutil.disk_usage`: the volume is **25.67 per cent
free** -- 240.15 gibibytes of 935.59 -- and the largest within-identity spread
is 1.45x. `storage_gate` reports the same figure as an `OK` row.

Disk pressure at 8.9 per cent does not explain variance at 25.67 per cent.

---

## 4. What IS established

Within-identity spread is real, and ranges 1.09x to 1.45x across four groups
whose collected identity sets did not change.

No cause is established. Candidates NOT tested here, and named so they are not
mistaken for eliminated:

- background load on a shared workstation
- filesystem cache warmth between consecutive runs
- Windows Defender or another scanner touching `data/`
- thermal or power-profile behaviour over a long session
- pytest collection order effects, which the identity digest does NOT pin

---

## 5. Why the earlier attributions failed

Both compared runs by DATE and by narrative, not by what they collected.
Grouping is what refuted them, and grouping took one pass over data already in
hand.

The second is the same shape as `QUOTED-A-FINDING-PAST-ITS-OWN-REPAIR-1`,
recorded four hours earlier the same day: **a finding that is true elsewhere,
asserted here.** The manifest's 2026-07-21 diagnosis was correct at 8.9 per
cent free; borrowing it at 25.67 per cent was not a measurement.

---

## 6. What this does NOT claim

That the variance matters. Every run in the table passed, and the gate's
purpose is the verdict rather than the duration.

That 1.45x is the ceiling. Two runs bound it, and two runs bound very little.

That the manifest's note should change. It records a different volume state and
remains accurate for it.

That `95f6c44` would not shift the picture. Its attestation was unavailable;
thirteen of fourteen runs are here.
