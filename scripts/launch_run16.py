#!/usr/bin/env python3
"""launch_run16.py -- one-command Run 16 launch orchestrator (run from PowerShell).

Drives vastai / ssh / scp / remote-bash entirely through Python subprocess arg-lists, so
there is NO pasted bash, NO angle-bracket placeholders, NO manual host/port splitting, and
NO offer-table reading. Phased + resumable via a state file.

Phases:
  up      auto-select+create (or --offer-id / adopt --instance-id) -> wait running ->
          read ssh_host/ssh_port -> bootstrap clone@HEAD -> preflight gate -> stage data ->
          env setup (pip + imodelsx KAN patch) -> launch train (nohup) -> save state.
  status  tail the remote log; report running / DONE / FAILED.
  down    run Sec.4 gates on the box -> fetch outputs -> verify -> (with --destroy and only
          if gates+fetch OK) destroy the instance. Never destroys on failure.
  watch   up, then poll status to completion, then down --destroy. Holds the terminal.

Always preview with --dry-run first. Author: Monzia Moodie.
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
pip install -r requirements.txt --break-system-packages
IMODELSX_KAN=$(python -c "import imodelsx.kan.kan_sklearn as m; print(m.__file__)" 2>/dev/null || true)
if [ -n "$IMODELSX_KAN" ] && grep -q "test_size=test_size" "$IMODELSX_KAN"; then
  sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"
  sed -i 's/random_state=random_state/random_state=self.random_state/g' "$IMODELSX_KAN"
  sed -i 's/shuffle=shuffle/shuffle=self.shuffle/g' "$IMODELSX_KAN"
  echo "imodelsx_patch applied"
fi
python -c "import catboost,lightgbm,xgboost,torch;print('ENV_OK',torch.cuda.is_available())"
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


def ssh_bash(st: dict, key: str, block: str, dry: bool, timeout: int = 1800) -> tuple[int, str, str]:
    base = ["ssh", "-i", key, "-p", str(st["ssh_port"]), *SSH_OPTS,
            f"{st['ssh_user']}@{st['ssh_host']}"]
    if dry:
        print(f"  DRY ssh> [{block.splitlines()[0][:50]} ...] ({len(block.splitlines())} lines)")
        return 0, "", ""
    return sh(base + [block], dry=False, timeout=timeout)


# ---------- pure helpers (unit-tested) ----------
def pick_offer(offers: list[dict]) -> dict | None:
    """Offers are pre-filtered by SEARCH_QUERY gates; choose cheapest, tiebreak by
    reliability then RAM (both descending)."""
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


# ---------- state ----------
def state_path(root: Path) -> Path:
    return root / OUT_REL / ".launch_state.json"


def save_state(root: Path, st: dict) -> None:
    p = state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=2), encoding="utf-8")


def load_state(root: Path) -> dict:
    p = state_path(root)
    if not p.exists():
        raise SystemExit(f"no state at {p}; run 'up' first.")
    return json.loads(p.read_text(encoding="utf-8"))


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
    for i in range(tries):
        rc, out, _ = vastai(["show", "instance", str(cid), "--raw"], dry=False)
        info = parse_show(out)
        if info.get("actual_status") == "running" and info.get("ssh_host") and info.get("ssh_port"):
            print(f"[up] instance running: {info['ssh_host']}:{info['ssh_port']}")
            return info
        print(f"[up] waiting for running ({i+1}/{tries}) status={info.get('actual_status')}")
        time.sleep(every)
    raise SystemExit("instance did not reach 'running' in time.")


def phase_up(args) -> int:
    root = Path(args.repo_root).resolve()
    dry = args.dry_run
    cid = resolve_instance(args, dry)
    info = wait_running(cid, dry)
    st = {"instance_id": cid, "ssh_user": "root", "ssh_host": info["ssh_host"],
          "ssh_port": str(info["ssh_port"]), "image": IMAGE, "train_log": TRAIN_LOG}
    url = f"ssh://{st['ssh_user']}@{st['ssh_host']}:{st['ssh_port']}"

    print("[up] bootstrap (clone @ HEAD)")
    rc, out, err = ssh_bash(st, args.ssh_key, BOOTSTRAP, dry)
    if not dry and ("BOOTSTRAP_DONE" not in out):
        raise SystemExit(f"bootstrap failed: {err or out}")

    print("[up] preflight gate")
    pf = Path(__file__).resolve().parent / "preflight_run16.py"
    rc, *_ = sh([sys.executable, str(pf), "--repo-root", str(root),
                 "--ssh-url", url, "--ssh-key", args.ssh_key], dry, capture=False)
    if rc != 0:
        raise SystemExit(f"preflight RED (rc={rc}); not staging.")

    print("[up] stage data")
    stg = Path(__file__).resolve().parent / "stage_run16.py"
    rc, *_ = sh([sys.executable, str(stg), "--repo-root", str(root), "--ssh-url", url,
                 "--ssh-key", args.ssh_key, "--scp-hf-cache", "--yes"], dry, capture=False)
    if rc != 0:
        raise SystemExit(f"staging failed (rc={rc}).")

    print("[up] env setup + launch training")
    rc, out, err = ssh_bash(st, args.ssh_key, SETUP_TRAIN, dry, timeout=3600)
    if not dry and "TRAIN_LAUNCHED" not in out:
        raise SystemExit(f"train launch failed: {err or out}")
    if not dry:
        m = re.search(r"TRAIN_PID=(\d+)", out)
        st["train_pid"] = m.group(1) if m else None
    save_state(root, st)
    print(f"[up] DONE. training running on instance {cid}.")
    print(f"     monitor:  python {Path(__file__).name} status --ssh-key {args.ssh_key}")
    print(f"     finish:   python {Path(__file__).name} down --destroy --ssh-key {args.ssh_key}")
    return 0


def phase_status(args) -> int:
    root = Path(args.repo_root).resolve()
    st = load_state(root)
    rc, out, err = ssh_bash(st, args.ssh_key, f"tail -n 25 {st['train_log']}", args.dry_run)
    if args.dry_run:
        return 0
    print(out or err)
    if DONE_MARK in out:
        print("[status] DONE -- run 'down --destroy' to fetch + teardown.")
    elif re.search(r"Traceback|Error|FAILED", out):
        print("[status] possible FAILURE -- inspect the log before teardown.")
    else:
        print("[status] still running.")
    return 0


def phase_down(args) -> int:
    root = Path(args.repo_root).resolve()
    dry = args.dry_run
    st = load_state(root)

    print("[down] Sec.4 gates on the box")
    rc, out, err = ssh_bash(st, args.ssh_key, GATES, dry)
    gates_ok = dry or ("DRIFT_RC=0" in out and "POP_RC=0" in out)
    if not dry:
        print(out)
    if not gates_ok:
        print("[down] GATES NOT GREEN -- NOT destroying. Inspect on the box.")

    print("[down] fetch outputs")
    local_out = str(root / "outputs")
    remote_out = f"{st['ssh_user']}@{st['ssh_host']}:{REMOTE_REPO}/{OUT_REL}"
    scp = ["scp", "-i", args.ssh_key, "-P", str(st["ssh_port"]), *SSH_OPTS, "-r",
           remote_out, local_out]
    rc, out, err = sh(scp, dry, timeout=3600)
    fetch_ok = dry or (rc == 0 and (Path(local_out) / "run16").exists())
    if not fetch_ok:
        print(f"[down] FETCH FAILED ({err or rc}) -- NOT destroying.")
        return 2

    if args.destroy and gates_ok and fetch_ok:
        print(f"[down] destroying instance {st['instance_id']}")
        vastai(["destroy", "instance", str(st["instance_id"])], dry, stdin="y\n")
        print("[down] destroyed.")
    else:
        print(f"[down] outputs fetched. Instance {st['instance_id']} LEFT UP "
              f"(destroy manually: vastai destroy instance {st['instance_id']}).")
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
        rc, out, _ = ssh_bash(st, args.ssh_key, f"tail -n 5 {st['train_log']}", False)
        if DONE_MARK in out:
            break
        if re.search(r"Traceback|FAILED", out):
            print("[watch] failure detected; stopping (instance left up).")
            return 2
    args.destroy = True
    return phase_down(args)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 16 one-command launch orchestrator.")
    ap.add_argument("phase", choices=["up", "status", "down", "watch"])
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--ssh-key", default=os.path.expanduser(r"~/.ssh/id_lambda_run8"))
    ap.add_argument("--offer-id", help="explicit offer (skip auto-select)")
    ap.add_argument("--instance-id", help="adopt an already-created instance")
    ap.add_argument("--disk", type=int, default=200)
    ap.add_argument("--destroy", action="store_true", help="(down/watch) destroy after verified fetch")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--poll-every", type=int, default=120)
    ap.add_argument("--max-polls", type=int, default=600)
    args = ap.parse_args()
    return {"up": phase_up, "status": phase_status, "down": phase_down, "watch": phase_watch}[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
