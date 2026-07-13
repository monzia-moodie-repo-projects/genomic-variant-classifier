#!/usr/bin/env python
"""verify_conformal_and_fullsuite.py (2026-07-11) -- READ-ONLY verify gate before committing
tests/conformal/. (1) run the untracked conformal test suite -> prove GREEN before we commit it
(never commit a failing/unverified test suite). (2) run the committed re-baseline machinery tests +
the existing split/seq-window/ensemble suites as a broader post-commit regression gate. Commits
nothing. Reports pass/fail counts + exit codes. ASCII-safe.
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

def run_pytest(targets, label, timeout=1200):
    existing = [t for t in targets if Path(t).exists()]
    missing = [t for t in targets if not Path(t).exists()]
    print(a(f"\n### {label} ###"))
    if missing:
        print(a(f"  (not present, skipped: {', '.join(missing)})"))
    if not existing:
        print("  (no targets present to run)")
        return None
    print(a(f"  pytest {' '.join(existing)} -q"))
    try:
        r = subprocess.run([sys.executable,"-m","pytest",*existing,"-q","--no-header","-p","no:cacheprovider"],
                           capture_output=True, text=True, timeout=timeout)
        out = r.stdout or ""
        for l in out.splitlines()[-18:]:
            print(a(f"    {l}"))
        if r.returncode != 0 and r.stderr.strip():
            for l in r.stderr.splitlines()[-10:]:
                print(a(f"    [stderr] {l}"))
        print(a(f"  exit = {r.returncode}  ({'GREEN' if r.returncode==0 else 'NOT GREEN'})"))
        return r.returncode
    except Exception as e:
        print(a(f"  (pytest failed to run: {e})"))
        return -1

def main() -> int:
    print("="*78); print("CONFORMAL + FULL-SUITE VERIFY (READ-ONLY) -- 2026-07-11"); print("="*78)

    # 1. the untracked conformal test suite -- MUST be green before we commit it
    rc_conf = run_pytest(["tests/conformal"], "1. CONFORMAL TEST SUITE (gate before committing tests/conformal/)")

    # 2. the committed machinery tests (regression -- should stay green)
    rc_mach = run_pytest([
        "tests/test_split_protocol_v2.py",
        "tests/test_delta_window_builder.py",
        "tests/test_seq_window_manifest.py",
    ], "2. COMMITTED RE-BASELINE MACHINERY TESTS (regression)")

    # 3. existing seq-window / split / ensemble suites (broader regression, if present)
    rc_exist = run_pytest([
        "tests/unit/test_seq_window_join.py",
        "tests/unit/test_seq_windows.py",
        "tests/test_rekey_seq_windows_v2.py",
    ], "3. EXISTING SEQ-WINDOW SUITES (broader regression)")

    line("=")
    print("VERDICT:")
    print(a(f"  conformal suite: {'GREEN -> safe to commit tests/conformal/' if rc_conf==0 else 'NOT GREEN -> DO NOT commit; report failures' if rc_conf is not None else 'absent'}"))
    print(a(f"  machinery tests: {'GREEN' if rc_mach==0 else 'NOT GREEN' if rc_mach is not None else 'absent'}"))
    print(a(f"  existing suites: {'GREEN' if rc_exist==0 else 'NOT GREEN' if rc_exist is not None else 'absent'}"))
    print("READ-ONLY. Nothing committed. Commit tests/conformal/ ONLY if its suite is GREEN above.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
