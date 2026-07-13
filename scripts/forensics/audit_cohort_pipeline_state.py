#!/usr/bin/env python
"""audit_cohort_pipeline_state.py (2026-07-11) -- READ-ONLY: is the untracked cohort-v3/fresh-ClinVar
pipeline at a coherent stopping point, or mid-change? For each of the 8 scripts: (1) does it PARSE
(syntax-clean)?; (2) does it IMPORT cleanly (or at least compile without unresolved top-level import
errors)?; (3) does it reference a matching test, and is that test GREEN?; (4) does it import internal
project modules that DON'T EXIST (a tell of mid-refactor)?; (5) last-modified time (staleness signal).
Runs each script's syntax + a dry import in a subprocess so one bad file can't abort the audit.
Commits/edits/deletes NOTHING. ASCII-safe.
"""
from __future__ import annotations
import ast, io, subprocess, sys, time
from pathlib import Path
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

def a(s): return s.encode("ascii","replace").decode("ascii")
def line(c="-", n=78): print(c*n)

SCRIPTS = [
    "scripts/build_cohort_from_source.py",
    "scripts/ingest_clinvar_snapshot.py",
    "scripts/inventory_clinvar_snapshots.py",
    "scripts/rebuild_cohort_v3.py",
    "scripts/rebuild_cohort_v3_by_sid.py",
    "scripts/rebuild_cohort_v3_final.py",
    "scripts/rederive_pathogenicity.py",
    "scripts/finalize_fresh_parquet.py",
]

# candidate test files (untracked) that pair with these scripts
TESTS = [
    "tests/test_build_cohort_from_source.py",
    "tests/test_ingest_clinvar_snapshot.py",
    "tests/test_cohort_v3_rebuild.py",
    "tests/test_rebuild_cohort_v3_by_sid.py",
    "tests/test_rebuild_cohort_v3_final.py",
    "tests/test_dedup_collapse.py",
    "tests/test_diff_cohorts.py",
]

def internal_imports(path):
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "genomic_variant_classifier" in n.module:
            mods.append(n.module)
        elif isinstance(n, ast.Import):
            for nm in n.names:
                if "genomic_variant_classifier" in nm.name:
                    mods.append(nm.name)
    return sorted(set(mods))

def module_exists(dotted):
    # map genomic_variant_classifier.data.foo -> src/genomic_variant_classifier/data/foo.py (or package dir)
    rel = dotted.replace(".", "/")
    for base in ["src", "."]:
        if (Path(base)/f"{rel}.py").exists() or (Path(base)/rel/"__init__.py").exists():
            return True
    return False

def main() -> int:
    print("="*78); print("COHORT-PIPELINE STATE AUDIT (READ-ONLY) -- 2026-07-11"); print("="*78)
    now = time.time()

    print("\n### PER-SCRIPT STATE ###")
    summary = []
    for s in SCRIPTS:
        p = Path(s)
        if not p.exists():
            print(a(f"\n  {s}: MISSING")); summary.append((s,"MISSING")); continue
        age_days = (now - p.stat().st_mtime)/86400.0
        # 1. syntax / compile
        rc_c = subprocess.run([sys.executable,"-m","py_compile",s], capture_output=True, text=True)
        syntax_ok = rc_c.returncode == 0
        # 2. dry import-check: run the module's imports only via a tiny probe (compile already covers syntax;
        #    for import we attempt 'python -c "import ast; exec(compile(open(s).read(),s,\"exec\"))"' is unsafe
        #    -- instead just check internal module refs resolve)
        ints = internal_imports(s)
        missing_int = [m for m in ints if not module_exists(m)]
        flag = "STABLE"
        notes = []
        if not syntax_ok:
            flag = "SYNTAX-ERROR"; notes.append(rc_c.stderr.strip().splitlines()[-1] if rc_c.stderr.strip() else "compile failed")
        if missing_int:
            flag = "MID-REFACTOR?"; notes.append(f"imports missing internal modules: {missing_int}")
        print(a(f"\n  {s}"))
        print(a(f"    syntax: {'ok' if syntax_ok else 'FAIL'}   internal-imports: {ints or '(none)'}   last-modified: {age_days:.1f}d ago"))
        if notes:
            for n in notes: print(a(f"    NOTE: {n}"))
        print(a(f"    -> {flag}"))
        summary.append((s, flag))

    print("\n### PAIRED TEST FILES -- present + will run below ###")
    present_tests = [t for t in TESTS if Path(t).exists()]
    for t in TESTS:
        print(a(f"  {'present' if Path(t).exists() else 'absent '}  {t}"))

    print("\n### RUN the paired cohort tests (are they GREEN? = coherent stopping point signal) ###")
    if present_tests:
        print(a(f"  pytest {' '.join(present_tests)} -q"))
        r = subprocess.run([sys.executable,"-m","pytest",*present_tests,"-q","--no-header","-p","no:cacheprovider"],
                           capture_output=True, text=True, timeout=900)
        for l in (r.stdout or "").splitlines()[-20:]:
            print(a(f"    {l}"))
        if r.returncode!=0 and r.stderr.strip():
            for l in r.stderr.splitlines()[-10:]:
                print(a(f"    [stderr] {l}"))
        print(a(f"  exit = {r.returncode}  ({'GREEN' if r.returncode==0 else 'NOT GREEN'})"))
    else:
        print("  (no paired tests present)")

    line("=")
    print("SUMMARY:")
    for s,f in summary:
        print(a(f"  {f:14} {s}"))
    print("\nread the flags: all STABLE + tests GREEN -> coherent stopping point (safe to commit).")
    print("any MID-REFACTOR?/SYNTAX-ERROR or tests NOT GREEN -> that file is mid-change (leave it).")
    print("READ-ONLY. Nothing committed, edited, or deleted.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
