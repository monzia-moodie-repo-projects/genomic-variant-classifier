#!/usr/bin/env python3
"""stage_run16.py -- preflight-gated staging for Run 16 on Vast.ai.

Pipeline: (0) run preflight_run16.py as a hard gate -> (1) verify the box has the
repo at the expected commit -> (2) scp the ship-manifest INTO the cloned repo (the
do-not-ship set is excluded by construction) with per-file remote size verification
-> (3) optionally scp the ESM-2 HF cache -> (4) safely (re)create the /workspace
symlink bridge, escalating any destructive rm -rf to a verified manual block -> STOP
and print the on-box Sec.4 gates + launch commands. It never starts training and
never runs an unguarded rm -rf.

All ssh/scp go through subprocess arg-lists (no shell string), which avoids the
PowerShell quote-stripping / CRLF-heredoc failures seen in earlier launches.

  Dry-run (prints the exact plan, runs nothing):
    python scripts/stage_run16.py --dry-run --ssh-host ssh5.vast.ai --ssh-port 12345

  Real:
    python scripts/stage_run16.py --ssh-host ssh5.vast.ai --ssh-port 12345 ^
      --ssh-key C:\\Users\\monzi\\.ssh\\id_lambda_run8 [--scp-hf-cache] [--yes]

Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---- single source of truth for the manifest: import from preflight ---------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import preflight_run16 as pf  # noqa: E402
    SHIP = list(pf.SHIP)
    DO_NOT_SHIP = list(pf.DO_NOT_SHIP)
except Exception as e:  # noqa: BLE001
    print(f"ABORT: cannot import preflight_run16 for the manifest ({e}). "
          "Keep stage_run16.py and preflight_run16.py together in scripts/.")
    sys.exit(1)

SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
            "-o", "StrictHostKeyChecking=accept-new"]


class Transport:
    """Thin ssh/scp wrapper. In dry_run, prints the command and returns success."""

    def __init__(self, key: str, port: str, user: str, host: str, dry_run: bool):
        self.key, self.port, self.user, self.host, self.dry = key, port, user, host, dry_run

    def _ssh_cmd(self, remote: str) -> list[str]:
        return ["ssh", "-i", self.key, "-p", self.port, *SSH_OPTS,
                f"{self.user}@{self.host}", remote]

    def _scp_cmd(self, local: str, remote: str, recursive: bool) -> list[str]:
        c = ["scp", "-i", self.key, "-P", self.port, *SSH_OPTS]
        if recursive:
            c.append("-r")
        c += [local, f"{self.user}@{self.host}:{remote}"]
        return c

    def ssh(self, remote: str, timeout: int = 60) -> tuple[int, str, str]:
        cmd = self._ssh_cmd(remote)
        if self.dry:
            print("  DRY ssh> " + remote)
            return 0, "", ""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        except FileNotFoundError:
            return 127, "", "ssh client not found on PATH"
        except subprocess.TimeoutExpired:
            return 124, "", f"timeout after {timeout}s"

    def scp(self, local: str, remote: str, recursive: bool = False) -> tuple[int, str, str]:
        cmd = self._scp_cmd(local, remote, recursive)
        if self.dry:
            print(f"  DRY scp> {local}  ->  {remote}")
            return 0, "", ""
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            return r.returncode, r.stdout.strip(), r.stderr.strip()
        except FileNotFoundError:
            return 127, "", "scp client not found on PATH"
        except subprocess.TimeoutExpired:
            return 124, "", "scp timeout (2h)"


def remote_path(remote_repo: str, rel: str) -> str:
    return f"{remote_repo.rstrip('/')}/{rel}"


def gate_preflight(args) -> int:
    """Run preflight_run16.py as a hard gate. Returns its exit code."""
    pf_path = Path(__file__).resolve().parent / "preflight_run16.py"
    cmd = [sys.executable, str(pf_path), "--repo-root", args.repo_root,
           "--ssh-host", args.ssh_host, "--ssh-port", args.ssh_port,
           "--ssh-key", args.ssh_key, "--remote-user", args.remote_user,
           "--remote-workdir", "/workspace"]
    print(">> preflight gate: " + " ".join(cmd[1:]))
    r = subprocess.run(cmd)
    return r.returncode


def check_remote_repo(t: Transport, remote_repo: str, local_head: str) -> bool:
    rc, out, err = t.ssh(f"test -d {remote_repo}/.git && git -C {remote_repo} rev-parse HEAD || echo NOREPO")
    if t.dry:
        return True
    if out == "NOREPO" or rc != 0:
        print(f"  [FAIL] no git repo at {remote_repo} -- clone it first "
              f"(box image normally auto-clones to /workspace/genomic-variant-classifier).")
        return False
    if local_head and out != local_head:
        print(f"  [WARN] remote HEAD {out[:9]} != local {local_head[:9]} -- "
              f"run `git -C {remote_repo} pull` on the box before training.")
    else:
        print(f"  [ OK ] remote repo present at HEAD {out[:9]}")
    return True


def upload(t: Transport, root: Path, remote_repo: str) -> bool:
    # 1) make all remote parent dirs in one shot
    dirs = sorted({os.path.dirname(remote_path(remote_repo, rel)) for rel, _ in SHIP})
    t.ssh("mkdir -p " + " ".join(dirs))
    ok = True
    for rel, _ in SHIP:
        local = root / rel
        rp = remote_path(remote_repo, rel)
        if not local.exists():
            print(f"  [FAIL] local missing: {rel}")
            ok = False
            continue
        rc, out, err = t.scp(str(local), rp)
        if not t.dry and rc != 0:
            print(f"  [FAIL] scp {rel}: {err or out}")
            ok = False
            continue
        # verify remote size matches local
        if not t.dry:
            rc2, rout, _ = t.ssh(f"stat -c %s {rp} 2>/dev/null || echo MISSING")
            lsize = local.stat().st_size
            if rout == "MISSING" or not rout.isdigit() or int(rout) != lsize:
                print(f"  [FAIL] size mismatch {rel}: local {lsize} vs remote {rout}")
                ok = False
            else:
                print(f"  [ OK ] {rel}  ({lsize/1048576:.1f} MB verified)")
    return ok


def upload_hf_cache(t: Transport, hf_cache: Path, remote_hf_dir: str) -> bool:
    if not t.dry and not hf_cache.exists():
        print(f"  [WARN] HF cache {hf_cache} absent -- box will download ESM-2 instead.")
        return True
    t.ssh(f"mkdir -p {remote_hf_dir}")
    rc, out, err = t.scp(str(hf_cache), remote_hf_dir.rstrip("/") + "/", recursive=True)
    if not t.dry and rc != 0:
        print(f"  [FAIL] HF cache scp: {err or out}")
        return False
    print(f"  [ OK ] HF cache -> {remote_hf_dir}")
    return True


def setup_symlinks(t: Transport, remote_repo: str) -> bool:
    """Safe /workspace/{data,outputs} bridge. Auto-creates only non-destructively;
    escalates any required rm -rf to a printed verified block."""
    all_ok = True
    for d in ("data", "outputs"):
        link = f"/workspace/{d}"
        want = f"{remote_repo}/{d}"
        insp = (f'if [ -L {link} ]; then echo "LINK:$(readlink {link})"; '
                f'elif [ -e {link} ]; then echo REALDIR; else echo ABSENT; fi')
        rc, out, err = t.ssh(insp)
        if t.dry:
            print(f"  DRY symlink> ensure {link} -> {want} (inspect, then ln -s / ln -sfn / escalate)")
            continue
        if out.startswith("LINK:"):
            target = out[5:]
            if target == want:
                print(f"  [ OK ] {link} already -> {want}")
                continue
            t.ssh(f"ln -sfn {want} {link}")
            print(f"  [ OK ] {link} repointed {target} -> {want}")
        elif out == "ABSENT":
            t.ssh(f"ln -s {want} {link}")
            print(f"  [ OK ] {link} -> {want} (created)")
        elif out == "REALDIR":
            print(f"  [STOP] {link} is a real directory. Destructive removal required -- "
                  f"NOT auto-running. Review and run on the box:")
            print(f"         rm -rf {link} && ln -s {want} {link} && ls -la {link}")
            all_ok = False
        else:
            print(f"  [WARN] {link}: unexpected inspect result '{out}' ({err}); skipping")
            all_ok = False
    return all_ok


def print_next_steps(remote_repo: str) -> None:
    print("\n" + "=" * 78)
    print(" STAGING COMPLETE -- stopping at the on-box gates (no training started).")
    print("=" * 78)
    print(f" On the box (cd {remote_repo}), in order:")
    print(" 1) data-prep + train (Sec.1 flag set):  the LAUNCH_CONTRACT_run16.md command")
    print(" 2) BEFORE trusting the run, the Sec.4 gates on the produced splits:")
    print("      python scripts/run_schema_drift_check.py --matrix outputs/run16/splits/X_train.parquet  # green")
    print("      python scripts/audit_smoke_feature_population.py outputs/run16/splits                    # LOVD must be > 0")
    print(" 3) checkpoint each base estimator + OOF; verify < 30 min/model else ABORT.")
    print(" Teardown (SEPARATE, after scp-back + manual verify):  echo y | vastai destroy <id>")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run 16 preflight-gated staging (scp + symlinks).")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--ssh-host")
    ap.add_argument("--ssh-url", help="ssh://user@host:port from 'vastai ssh-url'; overrides --ssh-host/--ssh-port/--remote-user")
    ap.add_argument("--ssh-port", default="22")
    ap.add_argument("--ssh-key", default=os.path.expanduser(r"~/.ssh/id_lambda_run8"))
    ap.add_argument("--remote-user", default="root")
    ap.add_argument("--remote-repo", default="/workspace/genomic-variant-classifier")
    ap.add_argument("--remote-hf-dir", default="/root/.cache/huggingface/hub")
    ap.add_argument("--hf-cache", default=os.path.expanduser(
        "~/.cache/huggingface/hub/models--facebook--esm2_t33_650M_UR50D"))
    ap.add_argument("--scp-hf-cache", action="store_true")
    ap.add_argument("--no-symlinks", action="store_true",
                    help="skip the /workspace bridge (Run-16 train.py is repo-relative)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="NOT recommended; bypasses the readiness gate")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
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
    if not args.ssh_host:
        ap.error("provide --ssh-host (with --ssh-port) or --ssh-url from 'vastai ssh-url'")

    root = Path(args.repo_root).resolve()
    t = Transport(args.ssh_key, args.ssh_port, args.remote_user, args.ssh_host, args.dry_run)

    print("=" * 78)
    print(f" Run 16 staging  (repo={root} -> {args.remote_user}@{args.ssh_host}:{args.remote_repo})")
    print("=" * 78)

    # 0) preflight gate
    if args.dry_run:
        print(">> [dry-run] would gate on preflight_run16.py (local+remote) and abort if RED")
    elif args.skip_preflight:
        print(">> WARNING: --skip-preflight set; readiness NOT verified.")
    else:
        rc = gate_preflight(args)
        if rc != 0:
            print(f"\nABORT: preflight returned {rc} (not GREEN). Fix FAILs, then re-stage.")
            return rc

    # local HEAD for the remote-clone freshness check
    local_head = ""
    try:
        local_head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    # plan summary
    total_mb = sum((root / rel).stat().st_size for rel, _ in SHIP if (root / rel).exists()) / 1048576
    print(f"\n[plan] {len(SHIP)} files (~{total_mb:.0f} MB) -> {args.remote_repo}")
    print(f"[plan] excluded do-not-ship: {', '.join(os.path.basename(x) for x in DO_NOT_SHIP)}")
    if args.scp_hf_cache:
        print(f"[plan] + ESM-2 HF cache -> {args.remote_hf_dir}")
    print(f"[plan] symlink bridge: {'SKIPPED' if args.no_symlinks else 'safe (escalates rm -rf)'}")

    if not args.dry_run and not args.yes:
        try:
            if input("\nProceed with upload? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Aborted by user.")
                return 1
        except EOFError:
            print("No TTY for confirmation; re-run with --yes.")
            return 1

    print("\n[1] remote repo check")
    if not check_remote_repo(t, args.remote_repo, local_head):
        return 2
    print("\n[2] upload ship-manifest")
    if not upload(t, root, args.remote_repo):
        print("\nABORT: upload verification failed.")
        return 2
    if args.scp_hf_cache:
        print("\n[3] upload ESM-2 HF cache")
        if not upload_hf_cache(t, Path(args.hf_cache), args.remote_hf_dir):
            return 2
    if not args.no_symlinks:
        print("\n[4] symlink bridge")
        setup_symlinks(t, args.remote_repo)

    print_next_steps(args.remote_repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
