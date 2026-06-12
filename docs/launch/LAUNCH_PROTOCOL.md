# Launch Protocol (formalized -- all future runs)

Goal: every run launches, executes, and tears down with a fixed, automated, three-command
protocol -- no pasted bash, no angle-bracket placeholders, no manual host/port, no reading
the offer table. One orchestrator (`scripts/launch_runNN.py`) drives vastai / ssh / scp /
remote-bash through Python subprocess, run entirely from PowerShell. Author: Monzia Moodie.

## The three commands (per run)

```powershell
# 1. UP -- auto-select cheapest single-GPU 4090 (R>=99, RAM>=64, dlperf>=95), create,
#    bootstrap (clone @ HEAD), preflight-gate, stage data, env+KAN patch, launch training.
#    Preview first with --dry-run; resumable; adopt an existing box with --instance-id.
python scripts\launch_run16.py up --dry-run
python scripts\launch_run16.py up

# 2. STATUS -- tail the remote training log; reports running / DONE / FAILED.
python scripts\launch_run16.py status

# 3. DOWN -- run the Sec.4 gates on the box, fetch outputs, verify, then (only if gates
#    AND fetch succeed) destroy. Never destroys on failure.
python scripts\launch_run16.py down --destroy
```

Hands-off variant: `python scripts\launch_run16.py watch --destroy` does up -> poll -> down.

## What the orchestrator guarantees (the friction it removes)

- Single shell context: you only ever run `python ...` in PowerShell. vastai/ssh/scp/bash
  are subprocess arg-lists, so PowerShell never parses `<`, `&&`, `sed`, or CRLF heredocs.
- Instance selection is automatic and correct: the search query pins `num_gpus = 1` (the
  omission that returned 2x/4x/8x offers) and ranks by price among reliability/RAM/dlperf
  gates. Override with `--offer-id` or adopt a live box with `--instance-id`.
- The imodelsx KAN package patch and `pip install` run on the box automatically.
- Teardown is gated: outputs are fetched and the Sec.4 gates must pass before destroy; a
  failure leaves the instance up for debugging (no lost results, the Run-9 failure mode).

## Standardizing a NEW run (e.g. Run 17)

Copy `scripts/launch_run16.py` to `scripts/launch_run17.py` and edit only the constants
block at the top: `LABEL`, `OUT_REL`, `TRAIN_FLAGS`, `TRAIN_LOG`, and (if the manifest
changed) point preflight/stage at the new run. Everything else is unchanged.

## Known follow-ups (tracked, not silently dropped)

- PRE-train gate: train.py has no `--prep-only`, so the schema-drift + feature-population
  gates currently run POST-run (before fetch/destroy). Adding `train.py --prep-only`
  (exit after PHASE 1 split-write) + `--splits-dir` reuse would let the orchestrator gate
  BEFORE the expensive training. Worth doing for Run 17.
- A fully generic `launch_run.py` reading a per-run YAML (instead of copy+edit constants)
  is the natural next step once a second run uses this protocol.
