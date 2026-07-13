#!/usr/bin/env python
"""audit_postcommit.py (2026-07-11) -- READ-ONLY post-commit gap audit. After the 4 housekeeping
commits, resolve: (1) tests/conformal/ contents + tracked state (conformal package shipped without
tests?); (2) is the cohort-build/ingest pipeline (build_cohort_from_source, ingest_clinvar_snapshot,
rebuild_cohort_v3*, rederive_pathogenicity) genuinely UNCOMMITTED product, or is a tracked version
already in git history?; (3) do the untracked status docs exist + their intent; (4) run the committed
machinery's test suite to confirm GREEN post-commit. Commits nothing. ASCII-safe.
"""
from __future__ import annotations
import io, subprocess, sys
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-", n=78): print(c*n)
def sh(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or ""), (r.stderr or "")
    except Exception as e:
        return -1, "", f"(failed: {e})"

def tracked(path):
    rc,_,_ = sh(["git","ls-files","--error-unmatch",path], timeout=30)
    return rc == 0

def in_history(path):
    """Has this path EVER been committed (appears in any commit)?"""
    rc,out,_ = sh(["git","log","--all","--oneline","--",path], timeout=30)
    return bool(out.strip())

def main() -> int:
    print("="*78); print("POST-COMMIT GAP AUDIT (READ-ONLY) -- 2026-07-11"); print("="*78)

    print("\n### 1. tests/conformal/ -- contents + tracked state ###")
    tc = Path("tests/conformal")
    if tc.exists():
        for p in sorted(tc.rglob("*.py")):
            print(a(f"  {'[tracked]' if tracked(str(p)) else '[UNTRACKED]'}  {p}"))
    else:
        print("  (tests/conformal/ does not exist)")

    print("\n### 2. cohort-build / ingest pipeline -- product or scratch? tracked/in-history? ###")
    pipeline = [
        "scripts/build_cohort_from_source.py",
        "scripts/ingest_clinvar_snapshot.py",
        "scripts/inventory_clinvar_snapshots.py",
        "scripts/rebuild_cohort_v3.py",
        "scripts/rebuild_cohort_v3_by_sid.py",
        "scripts/rebuild_cohort_v3_final.py",
        "scripts/rederive_pathogenicity.py",
        "scripts/finalize_fresh_parquet.py",
    ]
    for f in pipeline:
        if Path(f).exists():
            t = "tracked" if tracked(f) else "UNTRACKED"
            h = "in-history" if in_history(f) else "NEVER-committed"
            print(a(f"  {t:9} {h:16} {f}"))
        else:
            print(a(f"  MISSING              {f}"))

    print("\n### 2b. is there a TRACKED cohort builder already (committed pipeline)? ###")
    rc,out,_ = sh(["git","ls-files","scripts/"], timeout=30)
    cohort_tracked = [l for l in out.splitlines() if any(k in l.lower() for k in
                      ["cohort","clinvar","ingest","pathogenic","rederive"])]
    print(a("  tracked scripts matching cohort/clinvar/ingest/pathogenic:"))
    for l in (cohort_tracked or ["  (none)"]):
        print(a(f"    {l}"))

    print("\n### 3. status docs ###")
    for d in ["docs/status/ALLELELESS_PROVENANCE_2026-07-09.md",
              "docs/status/ALLELELESS_PROVENANCE_2026-07-09_FINAL.md"]:
        p = Path(d)
        if p.exists():
            print(a(f"  {'[tracked]' if tracked(d) else '[UNTRACKED]'}  {d}  ({p.stat().st_size} B)"))

    print("\n### 4. POST-COMMIT test suite -- the committed machinery's tests (confirm GREEN) ###")
    # Run ONLY the tests that were committed for the new machinery + the split module's existing suite.
    targets = [
        "tests/test_split_protocol_v2.py",
        "tests/test_delta_window_builder.py",
        "tests/test_seq_window_manifest.py",
    ]
    existing = [t for t in targets if Path(t).exists()]
    if existing:
        print(a(f"  running: pytest {' '.join(existing)} -q"))
        rc,out,err = sh([sys.executable,"-m","pytest",*existing,"-q","--no-header"], timeout=600)
        tail = (out or "").splitlines()[-15:]
        for l in tail:
            print(a(f"    {l}"))
        if err.strip():
            for l in err.splitlines()[-8:]:
                print(a(f"    [stderr] {l}"))
        print(a(f"  pytest exit = {rc}"))
    else:
        print("  (no committed machinery tests found to run)")

    line("=")
    print("POST-COMMIT GAP AUDIT COMPLETE. Read-only.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
