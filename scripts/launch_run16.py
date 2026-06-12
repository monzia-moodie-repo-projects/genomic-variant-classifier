#!/usr/bin/env python3
"""launch_run16.py (v2) -- robust one-command Run 16 launch orchestrator (run from PowerShell).

Drives vastai / ssh / scp / remote-bash entirely through Python subprocess arg-lists, so there
is NO pasted bash, NO angle brackets, NO manual host/port, NO offer-table reading.

v2 changes (robustness / responsiveness to staging failures + abrupt changes):
  * ESM-2 is downloaded ON THE BOX from HuggingFace (datacenter speed, fail-fast) instead of
    uploading a ~2.5 GB cache from the laptop. (--scp-hf-cache still available for offline boxes.)
  * staging + output-fetch get a 4 h transfer timeout (the old 600 s default killed multi-GB
    uploads -> the rc=124 we hit). stage itself now skips byte-present files (cheap resume).
  * SETUP guards against a double train launch (pgrep) and prefetches+validates ESM-2 weights.
  * incremental, step-tracked state: a failed 'up' leaves the box UP and re-running 'up' resumes
    (skips bootstrap/stage/launch already done). wait-loop detects a destroyed/preempted box.

Phases: up | status | down | watch.  Always preview with --dry-run first.  Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/monzia-moodie-repo-projects/genomic-variant-classifier.git"
REMOTE_REPO = "/workspace/genomic-variant-classifier"
IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
LABEL = "run16"
TRAIN_LOG = "/workspace/run16_full.log"
OUT_REL = "outputs/run16"
DONE_MARK = "Training complete"
HF_MODEL = "facebook/esm2_t33_650M_UR50D"
TRANSFER_TIMEOUT = 14400   # 4 h -- staging + output fetch (multi-GB over a home uplink)
SETUP_TIMEOUT = 3600       # 1 h -- pip install + ESM prefetch; nohup train returns immediately
SSH_TIMEOUT = 1800         # 30 min -- bootstrap clone
# single-GPU gate set (num_gpus=1 is REQUIRED -- omitting it returns 2x/4x/8x offers)
SEARCH_QUERY = ("reliability > 0.99 dlperf >= 95 pcie_bw >= 12 gpu_name = RTX_4090 "
                "num_gpus = 1 cuda_max_good >= 12.0 disk_space >= 200 cpu_ram >= 64 "
                "rentable = true")

TRAIN_FLAGS = (
    "--clinvar data/processed/clinvar_grch38_clean_seq.parquet "
    "--alphamissense data/external/alphamissense/AlphaMissense_hg38.tsv.gz "
    "--gnomad data/processed/gnomad_v4_exomes.parquet "
    "--gnomad-constraint data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv "
    "--dbnsfp-path data/external/dbnsfp/dbnsfp_clinvar_index.parquet "
    "--lovd-path data/external/lovd/lovd_all_variants.parquet "
    "--esm2-model esm2_t33_650M_UR50D "
    "--esm2-uniprot-index data/external/uniprot/uniprot_human_reviewed.parquet "
    "--esm2-device cuda --out-dir " + OUT_REL
)

BOOTSTRAP = f"""set -e
cd /workspace
if [ ! -d genomic-variant-classifier ]; then git clone {REPO_URL}; fi
cd genomic-variant-classifier
git fetch origin && git checkout main && git pull --ff-only
echo "REMOTE_HEAD=$(git rev-parse --short HEAD)"
echo BOOTSTRAP_DONE"""

SETUP_TRAIN = f"""set -e
cd {REMOTE_REPO}
if pgrep -f "scripts/train.py" >/dev/null 2>&1; then echo TRAIN_ALREADY_RUNNING; exit 0; fi
pip install -r requirements.txt --break-system-packages
IMODELSX_KAN=$(python -c "import imodelsx.kan.kan_sklearn as m; print(m.__file__)" 2>/dev/null || true)
if [ -n "$IMODELSX_KAN" ] && grep -q "test_size=test_size" "$IMODELSX_KAN"; then
  sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"
  sed -i 's/random_state=random_state/random_state=self.random_state/g' "$IMODELSX_KAN"
  sed -i 's/shuffle=shuffle/shuffle=self.shuffle/g' "$IMODELSX_KAN"
  echo "imodelsx_patch applied"
fi
python -c "import catboost,lightgbm,xgboost,torch;print('ENV_OK',torch.cuda.is_available())"
python -c "from huggingface_hub import snapshot_download; print('ESM2_CACHED', snapshot_download('{HF_MODEL}'))"
nohup python scripts/train.py {TRAIN_FLAGS} > {TRAIN_LOG} 2>&1 &
echo "TRAIN_PID=$!"
echo TRAIN_LAUNCHED"""

GATES = f"""cd {REMOTE_REPO}
python scripts/run_schema_drift_check.py --matrix {OUT_REL}/splits/X_train.parquet; echo "DRIFT_RC=$?"
python scripts/audit_smoke_feature_population.py {OUT_REL}/splits; echo "POP_RC=$?\""""

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new"]


# ---------- subprocess plumbing ----------
def sh(cmd: list[str], dry: bool, capture: bool = True, stdin: str | None = None,
       timeout: int = 600) -> tuple[int, str, str]:
    if dry:
        print("  DRY> " + " ".join(cmd))
        return 0, "", ""
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True, input=stdin, timeout=timeout)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout {timeout}s"


def vastai(args: list[str], dry: bool, stdin: str | None = None) -> tuple[int, str, str]:
    return sh(["vastai", *args], dry, stdin=stdin)


def ssh_bash(st: dict, key: str, block: str, dry: bool, timeout: int = SSH_TIMEOUT) -> tuple[int, str, str]:
    base = ["ssh", "-i", key, "-p", str(st["ssh_port"]), *SSH_OPTS,
            f"{st['ssh_user']}@{st['ssh_host']}"]
    if dry:
        print(f"  DRY ssh> [{block.splitlines()[0][:50]} ...] ({len(block.splitlines())} lines)")
        return 0, "", ""
    return sh(base + [block], dry=False, timeout=timeout)


# ---------- pure helpers (unit-tested) ----------
def pick_offer(offers: list[dict]) -> dict | None:
    def price(o): return float(o.get("dph_total", o.get("dph", 1e9)))
    def rel(o): return float(o.get("reliability2", o.get("reliability", 0)))
    def ram(o): return float(o.get("cpu_ram", 0))
    usable = [o for o in offers if int(o.get("num_gpus", 1)) == 1]
    if not usable:
        return None
    return sorted(usable, key=lambda o: (price(o), -rel(o), -ram(o)))[0]


def parse_contract(text: str) -> int | None:
    m = re.search(r"new_contract'?\s*:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def parse_show(text: str) -> dict:
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            obj = obj[0] if obj else {}
    except Exception:  # noqa: BLE001
        return {}
    return {"actual_status": obj.get("actual_status"),
            "ssh_host": obj.get("ssh_host"), "ssh_port": obj.get("ssh_port")}


def parse_rate(text: str) -> float | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            obj = obj[0] if obj else {}
        v = obj.get("dph_total")
        return float(v) if v is not None else None
    except Exception:  # noqa: BLE001
        return None


# ---------- state ----------
def state_path(root: Path) -> Path:
    return root / OUT_REL / ".launch_state.json"


def save_state(root: Path, st: dict) -> None:
    p = state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=2), encoding="utf-8")


def read_state(root: Path) -> dict | None:
    p = state_path(root)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_state(root: Path) -> dict:
    st = read_state(root)
    if st is None:
        raise SystemExit(f"no state at {state_path(root)}; run 'up' first.")
    return st


# ---------- phases ----------
def resolve_instance(args, dry: bool) -> int:
    if args.instance_id:
        print(f"[up] adopting instance {args.instance_id}")
        return int(args.instance_id)
    offer = args.offer_id
    if not offer:
        print("[up] auto-selecting cheapest single-GPU 4090 (R>=99, RAM>=64, dlperf>=95)")
        rc, out, err = vastai(["search", "offers", SEARCH_QUERY, "--order", "dph_total", "--raw"], dry)
        if dry:
            offer = "AUTO"
        else:
            try:
                chosen = pick_offer(json.loads(out))
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"could not parse offers ({e}); pass --offer-id.")
            if not chosen:
                raise SystemExit("no offer matched gates; relax filters or pass --offer-id.")
            offer = str(chosen.get("id"))
            print(f"[up] selected offer {offer}  ${chosen.get('dph_total')}/hr  "
                  f"RAM {chosen.get('cpu_ram')}  R {chosen.get('reliability2', chosen.get('reliability'))}")
    rc, out, err = vastai(["create", "instance", str(offer), "--image", IMAGE,
                           "--disk", str(args.disk), "--ssh", "--direct", "--label", LABEL], dry)
    if dry:
        return 0
    cid = parse_contract(out)
    if not cid:
        raise SystemExit(f"create did not return a contract id: {out or err}")
    print(f"[up] created instance {cid}")
    return cid


def wait_running(cid: int, dry: bool, tries: int = 40, every: int = 15) -> dict:
    if dry:
        print(f"  DRY> poll 'vastai show instance {cid} --raw' until actual_status=running")
        return {"ssh_host": "DRY_HOST", "ssh_port": "0"}
    misses = 0
    for i in range(tries):
        rc, out, _ = vastai(["show", "instance", str(cid), "--raw"], dry=False)
        if rc != 0 or not out:
            misses += 1
            if misses >= 3:
                raise SystemExit(f"instance {cid} not queryable (destroyed/preempted?).")
            time.sleep(every)
            continue
        info = parse_show(out)
        status = info.get("actual_status")
        if status in ("exited", "offline", "error"):
            raise SystemExit(f"instance {cid} status={status} -- not usable; create a fresh one.")
        if status == "running" and info.get("ssh_host") and info.get("ssh_port"):
            print(f"[up] instance running: {info['ssh_host']}:{info['ssh_port']}")
            return info
        print(f"[up] waiting for running ({i+1}/{tries}) status={status}")
        time.sleep(every)
    raise SystemExit("instance did not reach 'running' in time.")


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def phase_up(args) -> int:
    root = Path(args.repo_root).resolve()
    dry = args.dry_run
    prev = read_state(root)
    if args.instance_id:
        cid = int(args.instance_id)
    elif prev and not args.fresh and prev.get("instance_id"):
        cid = int(prev["instance_id"])
        print(f"[up] resuming from state: instance {cid}")
    else:
        cid = resolve_instance(args, dry)
    same = bool(prev) and not args.fresh and int(prev.get("instance_id", -1)) == int(cid or -1)
    steps = dict(prev.get("steps", {})) if same else {}

    info = wait_running(cid if not dry else 0, dry)
    st = {"instance_id": cid, "ssh_user": "root", "ssh_host": info["ssh_host"],
          "ssh_port": str(info["ssh_port"]), "image": IMAGE, "train_log": TRAIN_LOG,
          "steps": steps, "launched_at": (prev or {}).get("launched_at") or time.time()}
    url = f"ssh://{st['ssh_user']}@{st['ssh_host']}:{st['ssh_port']}"
    save_state(root, st)

    def fail(msg):
        raise SystemExit(f"{msg}\n  instance {cid} LEFT UP; fix, then re-run 'up' to resume.")

    if steps.get("bootstrap"):
        print("[up] bootstrap: already done (state); skipping")
    else:
        print("[up] bootstrap (clone @ HEAD)")
        rc, out, err = ssh_bash(st, args.ssh_key, BOOTSTRAP, dry)
        if not dry and "BOOTSTRAP_DONE" not in out:
            fail(f"bootstrap failed: {err or out}")
        steps["bootstrap"] = True
        save_state(root, st)

    print("[up] preflight gate")
    rc, *_ = sh([sys.executable, str(_scripts_dir() / "preflight_run16.py"), "--repo-root", str(root),
                 "--ssh-url", url, "--ssh-key", args.ssh_key], dry, capture=False)
    if rc != 0:
        fail(f"preflight RED (rc={rc}); not staging")

    if steps.get("staged"):
        print("[up] stage: already done (state); skipping")
    else:
        print("[up] stage data (box pulls ESM-2 from HF; no 2.5 GB cache upload)")
        cmd = [sys.executable, str(_scripts_dir() / "stage_run16.py"), "--repo-root", str(root),
               "--ssh-url", url, "--ssh-key", args.ssh_key, "--yes"]
        if args.scp_hf_cache:
            cmd.append("--scp-hf-cache")
        rc, *_ = sh(cmd, dry, capture=False, timeout=TRANSFER_TIMEOUT)
        if rc != 0:
            fail(f"staging failed (rc={rc}); re-run 'up' to resume (present files skip)")
        steps["staged"] = True
        save_state(root, st)

    if steps.get("launched"):
        print("[up] train: already launched (state); skipping. Use 'status'.")
    else:
        print("[up] env setup + ESM-2 prefetch + launch training")
        rc, out, err = ssh_bash(st, args.ssh_key, SETUP_TRAIN, dry, timeout=SETUP_TIMEOUT)
        if not dry and "TRAIN_LAUNCHED" not in out and "TRAIN_ALREADY_RUNNING" not in out:
            fail(f"train launch failed: {err or out}")
        if not dry:
            m = re.search(r"TRAIN_PID=(\d+)", out)
            st["train_pid"] = m.group(1) if m else None
        steps["launched"] = True
        save_state(root, st)

    name = Path(__file__).name
    print(f"[up] DONE. training on instance {cid}.")
    print(f"     monitor:       python {name} status")
    print(f"     fetch+verify:  python {name} down      (non-destructive; prints the destroy "
          "command only after a clean fetch)")
    return 0


def phase_status(args) -> int:
    root = Path(args.repo_root).resolve()
    st = load_state(root)
    if not args.dry_run:
        rc0, out0, _ = vastai(["show", "instance", str(st["instance_id"]), "--raw"], dry=False)
        rate = parse_rate(out0)
        if rate and st.get("launched_at"):
            hrs = (time.time() - st["launched_at"]) / 3600.0
            print(f"[status] uptime ~{hrs:.1f}h  rate ${rate:.4f}/hr  "
                  f"cost so far ~${hrs * rate:.2f}  (approx; billed from instance create)")
    rc, out, err = ssh_bash(st, args.ssh_key, f"tail -n 25 {st['train_log']}", args.dry_run, timeout=120)
    if args.dry_run:
        return 0
    if rc != 0:
        print(f"[status] cannot reach instance {st['instance_id']} ({err or rc}); it may be down.")
        return 2
    print(out or err)
    if DONE_MARK in out:
        print("[status] DONE -- run 'down --destroy' to fetch + teardown.")
    elif re.search(r"Traceback|Error|FAILED", out):
        print("[status] possible FAILURE -- inspect the log before teardown.")
    else:
        print("[status] still running.")
    return 0


def _train_state(st: dict, key: str, dry: bool) -> tuple[bool, bool]:
    """Return (running, completed) by inspecting the box: pgrep for the train process and a
    completion-marker count in the log."""
    if dry:
        return False, True
    probe = (f"if pgrep -f 'scripts/train.py' >/dev/null 2>&1; then echo TRAIN_RUNNING; "
             f"else echo TRAIN_IDLE; fi; "
             f"echo DONE_COUNT=$(grep -c '{DONE_MARK}' {st['train_log']} 2>/dev/null || echo 0)")
    rc, out, _ = ssh_bash(st, key, probe, dry=False, timeout=120)
    running = "TRAIN_RUNNING" in out
    m = re.search(r"DONE_COUNT=(\d+)", out)
    completed = bool(m and int(m.group(1)) > 0)
    return running, completed


def phase_down(args) -> int:
    root = Path(args.repo_root).resolve()
    dry = args.dry_run
    st = load_state(root)
    name = Path(__file__).name

    running, completed = _train_state(st, args.ssh_key, dry)
    if not dry:
        print(f"[down] training: {'RUNNING' if running else 'not running'}; "
              f"completion marker: {'present' if completed else 'ABSENT'}")
    # Never touch a live run when a destroy was requested.
    if args.destroy and running:
        print("[down] training is STILL RUNNING -- refusing to fetch/destroy. "
              "Wait until 'status' shows DONE, then re-run.")
        return 3

    print("[down] Sec.4 gates on the box")
    rc, out, err = ssh_bash(st, args.ssh_key, GATES, dry, timeout=1800)
    gates_ok = dry or ("DRIFT_RC=0" in out and "POP_RC=0" in out)
    if not dry:
        print(out)
    if not gates_ok:
        print("[down] GATES NOT GREEN.")

    print("[down] fetch outputs")
    local_out = str(root / "outputs")
    remote_out = f"{st['ssh_user']}@{st['ssh_host']}:{REMOTE_REPO}/{OUT_REL}"
    scp = ["scp", "-i", args.ssh_key, "-P", str(st["ssh_port"]), *SSH_OPTS, "-r",
           remote_out, local_out]
    rc, out, err = sh(scp, dry, timeout=TRANSFER_TIMEOUT)
    fetch_ok = dry or (rc == 0 and (Path(local_out) / "run16").exists())
    if not fetch_ok:
        print(f"[down] FETCH FAILED ({err or rc}) -- NOT destroying.")
        return 2

    # Non-destructive default: fetch + verify, leave the box up, surface the destroy command.
    if not args.destroy:
        print(f"[down] outputs fetched + verified. Instance {st['instance_id']} LEFT UP.")
        print("[down] when results look right, destroy as a SEPARATE deliberate step:")
        print(f"         python {name} down --destroy")
        return 0

    # Destroy path: every guard must pass.
    if not (gates_ok and fetch_ok):
        print(f"[down] gates/fetch not clean -- NOT destroying. Instance {st['instance_id']} left up.")
        return 2
    if not completed and not args.force:
        print("[down] no completion marker and --force not set -- NOT destroying "
              "(run may have crashed; inspect). Instance left up.")
        return 2
    if not args.yes and not dry:
        ans = input(f"[down] type the instance id ({st['instance_id']}) to DESTROY, "
                    "or press Enter to keep it up: ").strip()
        if ans != str(st["instance_id"]):
            print("[down] destroy NOT confirmed; instance left up.")
            return 0
    print(f"[down] destroying instance {st['instance_id']}")
    vastai(["destroy", "instance", str(st["instance_id"])], dry, stdin="y\n")
    print("[down] destroyed.")
    return 0


def phase_watch(args) -> int:
    phase_up(args)
    if args.dry_run:
        print("[watch] (dry) would poll status to completion, then down --destroy")
        return 0
    root = Path(args.repo_root).resolve()
    st = load_state(root)
    for _ in range(args.max_polls):
        time.sleep(args.poll_every)
        rc, out, _ = ssh_bash(st, args.ssh_key, f"tail -n 5 {st['train_log']}", False, timeout=120)
        if rc == 0 and DONE_MARK in out:
            break
        if rc == 0 and re.search(r"Traceback|FAILED", out):
            print("[watch] failure detected; stopping (instance left up).")
            return 2
    args.destroy = True
    args.yes = True
    return phase_down(args)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 16 one-command launch orchestrator (v2).")
    ap.add_argument("phase", choices=["up", "status", "down", "watch"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--ssh-key", default=os.path.expanduser(r"~/.ssh/id_lambda_run8"))
    ap.add_argument("--offer-id", help="explicit offer (skip auto-select)")
    ap.add_argument("--instance-id", help="adopt an already-created instance")
    ap.add_argument("--fresh", action="store_true", help="ignore prior state; create a new instance")
    ap.add_argument("--scp-hf-cache", action="store_true",
                    help="upload the local ESM-2 cache instead of box-side HF download (offline boxes)")
    ap.add_argument("--disk", type=int, default=200)
    ap.add_argument("--destroy", action="store_true", help="(down/watch) destroy after verified fetch")
    ap.add_argument("--yes", action="store_true", help="(down) skip the typed destroy confirmation")
    ap.add_argument("--force", action="store_true", help="(down) destroy even without a completion marker")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-every", type=int, default=120)
    ap.add_argument("--max-polls", type=int, default=600)
    args = ap.parse_args()
    return {"up": phase_up, "status": phase_status, "down": phase_down, "watch": phase_watch}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
