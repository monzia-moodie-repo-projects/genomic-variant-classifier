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

- L17 | run 16 | a "rich status" commit changed status output not at all -- the deployed
  launcher stayed on the prior version | `Copy-Item "$HOME\Downloads\launch_run16.py"` copied a
  STALE same-named file from Downloads (many prior downloads of the same name); the commit then
  reported "1 file changed, 8 insertions" (the docs entry only) and nobody checked that
  scripts/launch_run16.py was in the diff -- a silent no-op deploy | launcher now carries
  `__version__` + a `version` command + a banner on up/status; deploy uses a VERSIONED download
  filename (launch_run16_v5.py) to avoid the Downloads name-collision; deploy procedure verifies
  the version AND that git shows the file modified BEFORE committing | FIXED (v5 + procedure).

- L18 | run 16 | the v5 rich `status` HUNG for the full 180 s ssh timeout | it set
  LOGSRC=/proc/PID/fd/1 unconditionally and ran grep/tail on it, but on this box that fd is a
  PIPE (readlink shows `pipe:[...]`), and grep/tail on a pipe block waiting for EOF the live
  process never sends. The sandbox proof had only covered the regular-file (deleted-but-open)
  case, which is seekable -- the blocking path was untested | v6: choose LOGSRC only from sources
  that pass `[ -f ]` (regular file, incl. deleted-but-open -- a pipe fails `[ -f ]`), prefer the
  FileHandler log at the process's REAL cwd (`$(readlink /proc/PID/cwd)/logs/train.log`), wrap
  every read in `timeout`, dump fd0/1/2 + any *.log fd for diagnosis, and drop the ssh timeout
  to 90 s. Validated: `[ -f ]` distinguishes deleted-regular (true) from pipe (false); bash -n;
  all reads timeout-wrapped; parse/verdict for running/pipe/no-log without hang | FIXED (v6).

- L19 | run 16 | v6 `status` returned fast but reported a confident "PHASE 1 ... GPU activity =
  ESM-2 work" verdict for a process that was almost certainly doing nothing | the box showed
  GPU 0%/1MiB/23C (idle), CWD=/root (our launch runs from the repo), FD1/FD2=pipe, no log file
  anywhere, no outputs/splits at 25 min -- yet the verdict ASSERTED GPU activity it never read
  (same class of bug as asserting state without verifying). status also could not say WHAT the
  process was doing because it only tried to tail a log | v7: `status` is now a forensic probe --
  it reads the process CMDLINE, STATE, sampled CPU%, RSS, threads, wchan, open data-file fds, and
  socket count, parses the GPU utilization number, and discovers any *.log via find. The verdict
  synthesizes GPU+CPU: idle GPU AND low CPU AND no splits/outputs/log => *** SUSPECT *** (stalled
  or started outside the orchestrator) with a pkill + `up` relaunch remedy; genuine CPU- or
  GPU-active prep => PHASE 1 ACTIVE; STATE=Z => zombie. Validated: bash -n, 6 verdict scenarios,
  live-process forensic block. Also: cost line now labeled proc-time with a billed-time caveat;
  empty-dir `|| echo` dead-code replaced with explicit `[ -d ]` tests | FIXED (v7).

- L20 | run 16 | ROOT CAUSE of the whole session: `status` reported TRAIN=RUNNING and `up`
  refused to (re)launch for ~hours, but Run 16 was NEVER training | `pgrep -f "scripts/train.py"`
  matches ANY process whose cmdline contains that string -- including the status probe itself
  (it runs `pgrep -f scripts/train.py`) and ORPHANED probes left stuck on `pipe_read` by the v5
  hang (reparented to init, PPID=1). So: (a) status matched a dead probe and called it training;
  (b) up\'s pgrep guard matched the same phantom and printed TRAIN_ALREADY_RUNNING, skipping every
  relaunch; (c) v7\'s broad `find ... *.log` then surfaced outputs/run14/run14_master.log (May 26)
  as if it were Run 16; (d) the `crashed` check regex-scanned the whole probe output, matching the
  probe\'s OWN echoed grep pattern (Traceback|...|ABORT) in CMDLINE | v8: match ONLY a process whose
  /proc/PID/comm is python* (the probes are bash) in up/status/down; `up` sweeps orphaned probes
  (`pkill -f \'echo PROBE_OK\'`) before the guard; log discovery is Run-16-scoped (no cross-run find);
  `crashed` comes from a CRASH_HITS count grepped from the log only; status reports ORPHANS.
  Validated: live comm-filter picks python and rejects a bash proc whose cmdline contains
  scripts/train.py; bash -n; 5 verdict scenarios incl. no-real-train and the crash false positive | FIXED (v8).

- L21 | run 16 | the run16 vm.sh launched scripts/train.py straight into
  ModuleNotFoundError (catboost, then pandas) | TWO causes: (a) project deps were NEVER installed
  on this box -- up's phantom-pgrep guard (L20) exited at TRAIN_ALREADY_RUNNING *before* its pip
  step every time, and a restart had wiped the env; the pytorch image python is /opt/conda
  (3.11.10) with only torch present. (b) REGRESSION: my slim port of launch_run11_vm.sh DROPPED
  its [4/7] import-smoke gate that ABORTS on failure, so the launcher ran train.py despite the
  env check printing ModuleNotFoundError | vm.sh now has an ENV GATE that (i) checks the EXACT
  deps train.py imports -- pandas,numpy,sklearn,catboost,lightgbm,xgboost,imodelsx,transformers,
  torch + torch.cuda (NOT torch_geometric: Run_Preflight_VM.sh gates GNN deps the run16 train.py
  path does not use, so it must not be the run16 gate), (ii) self-heals via pip install -r
  requirements.txt if deps are missing, (iii) ABORTS (exit 4) before launch if still not green,
  (iv) applies the KAN patch only after install, (v) verifies train.py is alive 3s post-launch.
  requirements.txt does NOT pin torch, so the install does not clobber the image CUDA build.
  LESSON: a port must be >= the original -- never silently drop a safeguard when reusing a proven
  script; carry forward the gates, add to them | FIXED (vm.sh env gate).

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
