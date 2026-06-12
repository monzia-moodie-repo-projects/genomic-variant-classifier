# Launch Failure Registry (living -- append every run)

Purpose: record, for every launch, exactly what went wrong, WHY, and the fix -- so a footgun
is designed out once instead of rediscovered. Append a row the moment something fails or
surprises during a launch; also mirror it into docs/CHANGELOG.md. Author: Monzia Moodie.

Format: ID | first seen | symptom | root cause | fix | status (FIXED / MITIGATED / STANDING / OPEN).

## Shell-context footguns (the recurring time sink)

- L01 | runs 9-16 | `vastai create instance <OFFER_ID> ...` errors `The '<' operator is reserved`
  | PowerShell treats `<` as a redirection operator, so angle-bracket placeholders never paste
  | orchestrator builds every argv in Python -- no placeholders ever reach the shell | FIXED.
- L02 | run 16 | pasting the KAN block (`IMODELSX_KAN=...`, `sed`, `fi`, `&&`) into PowerShell
  errors `not recognized` / `&& not a valid separator` | it is bash meant for the box, not the
  laptop shell | orchestrator sends all box bash via `ssh ... "<block>"` from Python; the user
  never pastes bash | FIXED.
- L03 | run 16 | preflight RED `Connection to UNKNOWN port -1` against `ssh5.vast.ai:23456`
  | that endpoint was a placeholder; no instance existed | orchestrator reads the real
  ssh_host/ssh_port from `vastai show instance --raw` after the box is running | FIXED.

## Instance selection

- L04 | run 16 | a "fresh" offer search returned 2x/4x/8x machines | the search query omitted
  `num_gpus = 1` | SEARCH_QUERY now pins `num_gpus = 1`; pick_offer also excludes multi-GPU | FIXED.
- L05 | runs 10-16 | data-prep OOM; leftover `*.OOMbak` files (AlphaMissense/dbNSFP full index
  OOM'd ~16 GiB) | full-scoring indexes exceed small-RAM boxes | ship cohort-filtered indexes
  only; do-not-ship the `.OOMbak`; select boxes with >= 64 GB RAM (prefer ~128) | MITIGATED.

## Staging / transfer

- L06 | run 16 | `staging failed (rc=124)` during the ESM-2 cache upload | the orchestrator ran
  stage as a subprocess with the default 600 s timeout while ~4.3 GB uploaded; the manifest
  finished but the 2.5 GB ESM cache crossed 600 s and the parent killed the child | (a) box
  downloads ESM-2 from HuggingFace instead of a laptop upload; (b) staging/fetch use
  TRANSFER_TIMEOUT=14400 s; (c) stage skips byte-present files for a cheap resume | FIXED (d7797f4).
- L07 | run 16 | uploading a 2.5 GB HF cache from a residential uplink is slow and fragile
  | wrong direction of transfer -- the box has datacenter bandwidth | default to box-side HF
  download (validated: ESM-2 loads via `from_pretrained("facebook/esm2_t33_650M_UR50D")`);
  `--scp-hf-cache` retained only for an offline box | FIXED.

## Teardown safety

- L08 | run 9-era | risk of destroying an instance before results were scp'd back (lost results)
  | destroy and fetch were manual, unguarded | `down` fetches + verifies before any destroy and
  refuses to destroy unless gates AND fetch succeed | FIXED.
- L09 | run 16 | `up` printed `status` and `down --destroy` on adjacent lines -- an accidental
  paste could destroy a live run | monitoring and teardown were co-located | `up` prints only
  `status` + a non-destructive `down`; the destroy command is surfaced only after a clean fetch;
  `down --destroy` refuses while training is alive or with no completion marker, and requires a
  typed instance-id confirmation (unless --yes) | FIXED (v3).

## Model / env

- L10 | run 11-14 | KAN errors at fit(): imodelsx v1.0.13 references bare `test_size` etc.
  | upstream bug | kan.py sets the attrs before fit (in repo) + a self-guarding, idempotent sed
  patch on the installed package runs on the box during setup | FIXED (bf2f665).
- L11 | <= run 9 | LightGBM silently absent from the ensemble | CPU/build issue | LightGBM CPU
  fix; trained from Run 13 onward | FIXED.

## Operational hygiene

- L12 | run 16 | `vastai create` prints `instance_api_key` to the console | it is a live secret
  | keep it private; never commit/paste it; the orchestrator parses only `new_contract` | STANDING.
- L13 | all runs | PowerShell 5.1 encoding traps (Set-Content writes BOM; `[System.IO.File]`
  uses .NET CWD; em-dash -> mojibake) | platform behavior | ASCII-only, no-BOM, newline-
  preserving, count-guarded idempotent patchers | STANDING.

## Diagnosis / state integrity

- L14 | run 16 | `status` reported `cannot reach instance ... it may be down` when the box was
  actually reachable | it treated any non-zero ssh exit as unreachable; a missing log made
  `tail` exit 1 (the `Welcome to vast.ai` banner in the output was a SUCCESSFUL login) | `status`
  is now a marker-based diagnostic (`PROBE_OK`): it separates SSH-unreachable from a reachable
  box, and reports train-process state, data/splits presence, and BOTH logs (the nohup redirect
  /workspace/run16_full.log and train.py's own /workspace/.../logs/train.log) | FIXED (v4).
- L15 | run 16 | step-tracked state (`steps.bootstrap/staged/launched`) would let a re-`up` SKIP
  staging on a box whose /workspace had been wiped, trusting stale "done" flags | premature
  optimization; the flags did not reflect actual box state | removed step-skipping; `up` is now
  idempotent + self-healing -- it always runs bootstrap (clone-if-absent + pull), stage (skips
  byte-present files, re-uploads if wiped), and a pgrep-guarded launch, so it converges any box
  state and is a safe no-op on a healthy run | FIXED (v4).

- L16 | run 16 | `status` showed almost nothing ("no log file found") even though training was
  running | it tailed one fixed log PATH, but that path had been unlinked (the box restarted
  once, deleting /workspace/run16_full.log while the train process kept the fd open), and it did
  not inspect the process, GPU, or output files | `status` is now a rich monitor matching the
  Run 10/11/14 SSH probe: it recovers the LIVE log from /proc/PID/fd/1 (works after unlink),
  shows GPU util/mem, greps progress markers, counts models that reported AUROC, reads the
  outputs/splits file state, infers the PHASE, and computes cost from process elapsed time | FIXED (v5).

## Known OPEN / watch (carry forward)

- W03 | run 16 | instance 40728494 was reachable after launch but /workspace content (log, and
  likely data + repo) was GONE -- most consistent with a container/host restart on that host,
  not a training crash (a crash leaves the redirect log behind). Resolution: run the v4 `status`
  diagnostic to confirm (RESET vs running vs crashed); `up` self-heals a reset box; if the host
  resets repeatedly, destroy and `up` fresh onto a new offer. Watching whether this recurs.

- W01 | gnomAD `constraint_index.parquet` absent -> box regenerates from the TSV (slower).
  Pre-building and shipping it would shave data-prep time. Non-blocking.
- W02 | Sec.4 gates run POST-train (train.py has no `--prep-only`), so a broken prep is caught
  only after paying for training. Pre-train gate needs `train.py --prep-only` + `--splits-dir`.

## What WORKS (proven launch template)

Image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`; public repo cloned to
/workspace/genomic-variant-classifier and checked out at HEAD; ship-manifest scp'd in (cohort
indexes only); ESM-2 pulled on the box from HF; symlink bridge /workspace/{data,outputs} ->
repo; the three-command protocol (up / status / down) from PowerShell via launch_runNN.py.
Run 14 reference: AUROC 0.9975 at $2.17 (no full ESM-2). Run 16 adds full ESM-2 650M.
