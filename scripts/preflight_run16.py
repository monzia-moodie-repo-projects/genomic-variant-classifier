#!/usr/bin/env python3
"""preflight_run16.py -- single launch-readiness gate for Run 16 (Vast.ai).

Read-only. Validates LOCAL staging readiness always, and REMOTE box readiness when
SSH vars are supplied. GREEN (exit 0) only if no hard FAILs; WARNs are listed but do
not block. Pure stdlib (no pandas/torch import) so it runs in a bare venv.

Two-phase usage:
  1) before creating the instance -- local readiness only:
       python scripts/preflight_run16.py --skip-remote
  2) after `vastai create` + ssh-url -- full gate:
       python scripts/preflight_run16.py ^
         --ssh-host sshN.vast.ai --ssh-port 12345 ^
         --ssh-key C:\\Users\\monzi\\.ssh\\id_lambda_run8

Requires the OpenSSH client (Win10+ ships it) for the remote phase.
Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

Result = namedtuple("Result", "name status detail")  # status: PASS WARN FAIL INFO

# ---- manifests (from docs/launch/LAUNCH_CONTRACT_run16.md Sec.3) -----------------
# (relpath, min_MB)  min_MB=0 => existence-only (catches missing, not truncation)
SHIP = [
    ("data/processed/clinvar_grch38_clean_seq.parquet", 400),
    ("data/external/alphamissense/AlphaMissense_hg38.tsv.gz", 500),
    ("data/external/alphamissense/alphamissense_protein_index.parquet", 10),
    ("data/processed/gnomad_v4_exomes.parquet", 20),
    ("data/external/gnomad/gnomad.v4.1.constraint_metrics.tsv", 70),
    ("data/external/dbnsfp/dbnsfp_clinvar_index.parquet", 25),
    ("data/external/lovd/lovd_all_variants.parquet", 0),
    ("data/external/uniprot/uniprot_human_reviewed.parquet", 8),
    ("data/raw/cache/spliceai_scores_snv.parquet", 350),
    ("data/reference/schema/schema_baseline.json", 0),
]
# ship to save box regen time; WARN (not FAIL) if absent
SHIP_OPTIONAL = [
    ("data/external/gnomad/gnomad.v4.1.constraint_index.parquet", 0),
]
# allowed to exist locally but MUST be excluded from staging (WARN reminder)
DO_NOT_SHIP = [
    "data/external/dbnsfp/dbnsfp_full_index.parquet",
    "data/external/dbnsfp/dbnsfp_full_index.parquet.OOMbak",
    "data/raw/cache/alphamissense_scores_hg38.parquet.OOMbak",
]
EXPECTED_SCHEMA_COLS = 81
EXPECTED_RUN_TAG = "run16b"


def _mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


# ---- LOCAL checks ----------------------------------------------------------------
def check_ship_files(root: Path) -> list[Result]:
    out = []
    for rel, min_mb in SHIP:
        p = root / rel
        if not p.exists():
            out.append(Result(f"ship: {rel}", "FAIL", "MISSING -- train.py/gate needs it"))
        elif min_mb and _mb(p) < min_mb:
            out.append(Result(f"ship: {rel}", "FAIL",
                              f"TRUNCATED? {_mb(p):.1f} MB < {min_mb} MB expected"))
        else:
            out.append(Result(f"ship: {rel}", "PASS", f"{_mb(p):.1f} MB"))
    for rel, _ in SHIP_OPTIONAL:
        p = root / rel
        if p.exists():
            out.append(Result(f"ship?: {rel}", "PASS", f"{_mb(p):.1f} MB (cache present)"))
        else:
            out.append(Result(f"ship?: {rel}", "WARN",
                              "absent -- box will regen from the constraint TSV (slower)"))
    return out


def check_do_not_ship(root: Path) -> list[Result]:
    out = []
    for rel in DO_NOT_SHIP:
        p = root / rel
        if p.exists():
            out.append(Result(f"exclude: {rel}", "WARN",
                              f"present ({_mb(p):.0f} MB) -- DO NOT scp this"))
    if not out:
        out.append(Result("exclude: do-not-ship set", "PASS", "none present locally"))
    return out


def check_schema_baseline(root: Path) -> Result:
    p = root / "data/reference/schema/schema_baseline.json"
    if not p.exists():
        return Result("schema baseline", "FAIL", "schema_baseline.json missing")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return Result("schema baseline", "FAIL", f"unparseable JSON: {e}")
    ncols = obj.get("n_columns")
    if ncols is None:
        for k in ("columns", "schema", "dtypes"):
            v = obj.get(k)
            if isinstance(v, (list, dict)):
                ncols = len(v)
                break
    run = str(obj.get("run_label", obj.get("run", "")))
    if ncols != EXPECTED_SCHEMA_COLS:
        return Result("schema baseline", "FAIL",
                      f"n_columns={ncols}, expected {EXPECTED_SCHEMA_COLS}")
    if EXPECTED_RUN_TAG not in run:
        return Result("schema baseline", "WARN",
                      f"{ncols} cols OK but run_label='{run}' (expected ~{EXPECTED_RUN_TAG})")
    return Result("schema baseline", "PASS", f"{ncols} cols, run_label='{run}'")


def _git(root: Path, *args: str) -> tuple[int, str]:
    try:
        r = subprocess.run(["git", "-C", str(root), *args],
                           capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return 127, str(e)


def check_git(root: Path) -> list[Result]:
    rc, _ = _git(root, "rev-parse", "--is-inside-work-tree")
    if rc != 0:
        return [Result("git", "WARN", "not a git work tree (skipping clean/push checks)")]
    out = []
    rc, st = _git(root, "status", "--porcelain")
    if st:
        n = len(st.splitlines())
        sample = ", ".join(line[2:].strip() for line in st.splitlines()[:6])
        out.append(Result("git clean", "WARN",
                          f"{n} uncommitted/untracked path(s): {sample}"
                          + (" ..." if n > 6 else "")))
    else:
        out.append(Result("git clean", "PASS", "working tree clean"))
    rc_h, head = _git(root, "rev-parse", "HEAD")
    rc_u, up = _git(root, "rev-parse", "@{u}")
    if rc_u != 0:
        out.append(Result("git pushed", "WARN", "no upstream tracking branch set"))
    elif head != up:
        out.append(Result("git pushed", "WARN", "local HEAD differs from upstream (unpushed?)"))
    else:
        out.append(Result("git pushed", "PASS", f"HEAD==upstream ({head[:9]})"))
    return out


def check_hf_cache(hf_cache: Path) -> Result:
    # ESM-2 650M: SCP the HF cache OR download on box. Informational either way.
    snap = hf_cache / "snapshots"
    if hf_cache.exists() and snap.exists() and any(snap.iterdir()):
        return Result("esm2 hf-cache", "INFO",
                      f"present ({hf_cache}); you may scp it to skip the box download")
    return Result("esm2 hf-cache", "INFO",
                  "absent -- box will download facebook/esm2_t33_650M_UR50D (~2.5 GB)")


def check_ssh_key(key: Path) -> Result:
    if key.exists():
        return Result("ssh key", "PASS", str(key))
    return Result("ssh key", "FAIL", f"{key} not found (remote phase cannot run)")


# ---- REMOTE checks ---------------------------------------------------------------
def ssh_base(key: Path, port: str, user: str, host: str) -> list[str]:
    return ["ssh", "-i", str(key), "-p", str(port),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{user}@{host}"]


def _run_remote(base: list[str], remote_cmd: str, timeout: int = 25) -> tuple[int, str, str]:
    try:
        r = subprocess.run(base + [remote_cmd], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except FileNotFoundError:
        return 127, "", "ssh client not found on PATH (install OpenSSH)"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return 1, "", str(e)


def check_connectivity(base: list[str]) -> Result:
    rc, out, err = _run_remote(base, "echo PREFLIGHT_OK_7731")
    if rc == 0 and "PREFLIGHT_OK_7731" in out:
        return Result("ssh connect", "PASS", "handshake + echo OK")
    return Result("ssh connect", "FAIL", err or out or f"rc={rc}")


def check_gpu(base: list[str]) -> Result:
    rc, out, err = _run_remote(
        base, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    if rc != 0 or not out:
        return Result("gpu", "FAIL", err or "nvidia-smi returned nothing")
    first = out.splitlines()[0]
    if "4090" not in first:
        return Result("gpu", "WARN", f"not a 4090-class card: {first}")
    return Result("gpu", "PASS", first)


def check_disk(base: list[str], workdir: str, min_gb: int = 25) -> Result:
    rc, out, err = _run_remote(base, f"df -BG {workdir} 2>/dev/null || df -BG /")
    if rc != 0 or not out:
        return Result("disk", "WARN", err or "df returned nothing")
    try:
        avail = int(out.splitlines()[1].split()[3].rstrip("G"))
    except Exception:  # noqa: BLE001
        return Result("disk", "WARN", f"could not parse df: {out.splitlines()[-1]}")
    if avail < min_gb:
        return Result("disk", "FAIL", f"{avail} GB free < {min_gb} GB needed")
    return Result("disk", "PASS", f"{avail} GB free")


def check_env(base: list[str]) -> Result:
    rc, out, err = _run_remote(
        base, "python -c \"import torch;print(torch.__version__, torch.cuda.is_available())\"")
    if rc != 0:
        return Result("remote torch", "INFO", "torch not importable yet (env not built -- expected pre-setup)")
    return Result("remote torch", "INFO", out)


# ---- driver ----------------------------------------------------------------------
def render(results: list[Result]) -> int:
    sym = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "INFO": "[info]"}
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  {sym[r.status]}  {r.name.ljust(width)}  {r.detail}")
    n_fail = sum(r.status == "FAIL" for r in results)
    n_warn = sum(r.status == "WARN" for r in results)
    print("-" * 78)
    if n_fail:
        print(f" VERDICT: RED -- {n_fail} FAIL, {n_warn} WARN. Resolve all FAILs before launch.")
        return 2
    print(f" VERDICT: GREEN -- 0 FAIL, {n_warn} WARN. Cleared for staging/launch.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 16 launch preflight (local + remote).")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--skip-remote", action="store_true", help="local readiness only")
    ap.add_argument("--ssh-host")
    ap.add_argument("--ssh-url", help="ssh://user@host:port from 'vastai ssh-url'; overrides --ssh-host/--ssh-port/--remote-user")
    ap.add_argument("--ssh-port", default="22")
    ap.add_argument("--ssh-key",
                    default=os.path.expanduser(r"~/.ssh/id_lambda_run8"))
    ap.add_argument("--remote-user", default="root")
    ap.add_argument("--remote-workdir", default="/workspace")
    ap.add_argument("--hf-cache",
                    default=os.path.expanduser(
                        "~/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D"))
    ap.add_argument("--check-env", action="store_true",
                    help="also probe remote torch/cuda (only meaningful post-setup)")
    ap.add_argument("--min-disk-gb", type=int, default=25)
    args = ap.parse_args()
    if getattr(args, "ssh_url", None):
        _u = args.ssh_url.strip()
        if _u.startswith("ssh://"):
            _u = _u[6:]
        if "@" in _u:
            args.remote_user, _u = _u.split("@", 1)
        _host, _sep, _port = _u.partition(":")
        if _host:
            args.ssh_host = _host
        if _port:
            args.ssh_port = _port

    root = Path(args.repo_root).resolve()
    print("=" * 78)
    print(f" Run 16 preflight  (repo={root})")
    print("=" * 78)
    print("[LOCAL staging readiness]")
    results: list[Result] = []
    results += check_ship_files(root)
    results += check_do_not_ship(root)
    results.append(check_schema_baseline(root))
    results += check_git(root)
    results.append(check_hf_cache(Path(args.hf_cache)))

    do_remote = not args.skip_remote and args.ssh_host
    if do_remote:
        key = Path(args.ssh_key)
        results.append(check_ssh_key(key))
        if key.exists():
            base = ssh_base(key, args.ssh_port, args.remote_user, args.ssh_host)
            conn = check_connectivity(base)
            results.append(conn)
            if conn.status == "PASS":
                results.append(check_gpu(base))
                results.append(check_disk(base, args.remote_workdir, args.min_disk_gb))
                if args.check_env:
                    results.append(check_env(base))
            else:
                results.append(Result("gpu/disk", "FAIL", "skipped -- no ssh connectivity"))
    elif not args.skip_remote:
        results.append(Result("remote phase", "WARN",
                              "no --ssh-host given; ran local-only (use --skip-remote to silence)"))

    print()
    rc = render(results)
    if rc == 0 and (args.skip_remote or not args.ssh_host):
        print(" NEXT: stage the ship-manifest, then re-run WITH --ssh-host/--ssh-port to clear")
        print("       the remote phase; then run the on-box Sec.4 gates after data-prep:")
        print("       run_schema_drift_check.py (green) + audit_smoke_feature_population.py")
        print("       (LOVD must be > 0 at full scale).")
    return rc


if __name__ == "__main__":
    sys.exit(main())
