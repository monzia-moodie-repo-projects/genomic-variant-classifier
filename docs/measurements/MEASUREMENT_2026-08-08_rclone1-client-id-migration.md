# MEASUREMENT 2026-08-08 — RCLONE-1: the Google Drive client identifier migration, and five wrong explanations

**Author: Monzia Moodie**
**rclone v1.75.0. Google Cloud project `genvarcla-rclone`, project number
1052485832379. No repository file was modified; this records an operational
change and its verification.**

RCLONE-1 was open since `d208240` (2026-08-07). The `genvarcla:` remote used
rclone's shared Google Drive client identifier, which **is being retired and
will stop working during 2026**. It was the only item on the register with an
external deadline nobody here controls: when it expired, the Drive copy of the
documentation — the only off-machine copy of 282 files — would have stopped
updating, and the failure would have been silent.

**Result: migrated and verified. `rclone check` reports 282 matching files, 0
differences, with MD5 verification. The retirement notice no longer appears.**

**Cost: four configuration defects, five wrong explanations from this author,
and five defects in this author's own verification scripts.** The migration
itself took minutes; establishing that it had worked took two hours, almost
entirely because of the instruments rather than the thing being measured.

---

## 1. What was done

Following rclone's own documentation (`https://rclone.org/drive/`, last updated
2026-07-31) rather than recollection, because the Google Cloud console
interface changes and this author's training data does not.

1. A Google Cloud project created. Billing not required.
2. The Google Drive application programming interface enabled.
3. An OAuth consent screen configured, audience **External**.
4. Three scopes added — `.../auth/docs`, `.../auth/drive` and
   `.../auth/drive.metadata.readonly`. The second is what permits editing,
   creating and deleting.
5. Self added as a test user.
6. An OAuth client of type **Desktop app** created.
7. The application **published**, not left in Testing.
8. `rclone config` edited to carry the new client identifier and secret, then
   `rclone config reconnect genvarcla:` run to mint a token against them.

## 2. Four configuration defects, in the order they were hit

### 2.1 The application was nearly left in Testing

This author's first instructions said publishing was unnecessary. rclone's
documentation says otherwise: **grants expire after one week** in Testing mode.
That would have produced a weekly re-authorisation, indefinitely, for no
reason. Verification is not required — Google exempts personal-use apps under
100 users, which need only click through the "unverified app" warning.

### 2.2 The scopes were omitted from the instructions entirely

Without `.../auth/drive` the client would have been created successfully and
the authorisation would have granted no write access. The failure would have
appeared later, as a permissions error partway through a sync, when the cause
was no longer in front of anyone.

### 2.3 `service_account_file = n`

`rclone config` prompts for a service account credentials file with a stated
default of `n`. Pressing Enter stores the **literal string** `"n"` as a file
path. rclone then tries to open a file called `n`, fails, and cannot construct
the file system at all:

```
CRITICAL: failed when making oauth client: error opening service account
credentials file: open n: The system cannot find the file specified.
```

This also explains why `rclone config reconnect` appeared to do nothing: it
never reached the browser flow. Cleared with
`rclone config update genvarcla service_account_file ""`.

### 2.4 The Google Drive application programming interface was not enabled

```
Error 403: Google Drive API has not been used in project 1052485832379
before or it is disabled.   reason: SERVICE_DISABLED
```

Step 2 had not taken effect. Notably, **the scopes were added successfully
regardless** — so the consent screen looked complete while the interface itself
was off. This author had earlier offered "scopes will not add if the interface
is disabled" as a diagnostic; that is false, and the console gives no signal.

## 3. A secret was exposed, and rotated

The `rclone config` prompt echoes the existing client secret as its default
value, so the terminal transcript contained it, and the transcript was shared.
The secret was rotated in the console immediately afterwards.

The instruction "do not send me the secret" was given and then undermined by
this author presenting a table of prompts and answers that made pasting the
whole transcript the natural response. **An instruction that competes with the
format it is written in is not an instruction.**

## 4. Five wrong explanations, none measured before it was offered

| # | claim | why it was wrong |
|---|---|---|
| 1 | the mistyped secret `G0CSPX-` broke authentication | the stored value was correct; the terminal rendering misled |
| 2 | the delete failure was Drive listing propagation lag | delete worked; the comparison in the script never held |
| 3 | `--include "*.md"` misses root-level files under `-R` | all three filter variants returned identical sets |
| 4 | the `PLAYBOOK` grep result was a misread | it was correct; a bad list was used to overrule it |
| 5 | root identifier `0A…` indicates a Shared Drive | `backend drives` shows a different Shared Drive; `team_drive` empty |

Each was plausible. Each cost a round trip. **The correct instrument —
`rclone check --include "**.md" -vv`, reading its own summary — was available
from the first message and is what the project's sync procedure already uses.**

## 5. Five defects in this author's verification scripts

1. **Absence read as success.** A probe reported
   `NOTICE present: False` after the command had failed with a CRITICAL. A
   check that passes because the operation never ran is the defect this project
   exists to remove.
2. **A conditional printed as prose, not enforced as a gate.** The script
   printed *"THE COMMAND FAILED. Everything below is meaningless"* and then ran
   the remaining sections anyway, grouping a 403 error's JSON by first character
   and presenting it as a table of markers.
3. **A case-insensitive error match that fired on filenames.** The detector
   matched `CRITICAL|ERROR|Failed`, and this repository contains
   `INCIDENT_2026-04-30_gnn-gene-symbol-keyerror.md` and
   `RUN_11_FINDINGS_AND_CRITICAL_FIXES.md`. A successful check was reported as
   a failure. Repaired by anchoring on rclone's log format —
   `^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} (CRITICAL|ERROR)` — and case-sensitive
   matching, then **proved against those two exact filenames** before use.
4. **`lsf --include` without `--files-only`.** Directory names are listed
   regardless of the include filter, so a probe for leftover test files
   returned 22 directory names and answered nothing.
5. **A count comparison that could never hold**, because PowerShell's empty
   command output is not an empty array in the way the script assumed.

## 6. DRIVELIST-1 — the finding worth keeping

`rclone lsf` returned **inconsistent results for the same file, minutes
apart**:

```
lsf --files-only        -> PLAYBOOK_STALE_NOTICE.md present
lsf --files-only        -> absent (56 files listed)
lsf -R --files-only     -> absent (281)
lsf -R --fast-list      -> absent (281)
ls  (recursive)         -> absent (281)
rclone check            -> 282 matching, 0 differences, MD5 verified
```

`rclone copy` on that single file reported `size = 4137 OK`, *"Size and
modification time the same"*, *"Unchanged skipping"* — it could only compare
against a remote object that exists.

rclone documents a Google Drive listing defect in this family and ships
`--drive-fast-list-bug-fix` (default on) because Drive's parent-based search
*"returns nothing sometimes"*. Whether this is the same defect is not
established here.

**The operational consequence is what matters: sync verification must use
`rclone check`, which compares by content. `lsf` counts must not be treated as
evidence.** Every count in this session's exchange, including this author's,
came from the one command that cannot be trusted for it.

`SYNCFILTER-1` was proposed during this work and is **withdrawn** — it was a
hypothesis about `--include` anchoring, refuted one round later when all three
filter variants returned identical sets. It is recorded here so that a future
reader finding it in the transcript knows it was tested and rejected.

## 7. Verified final state

| check | result |
|---|---|
| retirement notice on `lsd` | **absent** |
| `client_id` in config | set, project 1052485832379 |
| `client_secret` | rotated after exposure, starts `GOCSPX-` |
| `service_account_file` | empty |
| token | minted 2026-08-08 against the project's own client |
| directory listing | 9 directories under `genomic-variant-classifier/` |
| upload | succeeded, file visible |
| delete | succeeded — `INFO : Deleted`, then absent from listing |
| probe files left behind | none in `docs/`; three in Drive trash, as `use_trash` intends |
| **`rclone check --include "**.md" --one-way`** | **282 matching files, 0 differences, MD5 verified** |

## 8. Register

**RCLONE-1 closes.** The remote runs on the project's own client identifier;
the retirement no longer threatens the documentation backup.

**DRIVELIST-1 is filed.** `rclone lsf` is not a reliable enumeration on this
remote; `rclone check` is the authority.

**SYNCFILTER-1 is withdrawn** — proposed and refuted within this measurement.

52 carried in, one closed, one filed: **52 open.**

## 9. Method

Every statement above comes from a command executed on 2026-08-08 against the
live remote. No claim here rests on a hypothesis that was not subsequently
measured — which is precisely the standard §4 records this author failing five
times before meeting.
