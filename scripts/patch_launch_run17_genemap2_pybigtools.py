#!/usr/bin/env python3
r"""patch_launch_run17_genemap2_pybigtools.py

Close the last two silent-zeros in scripts/launch_run17_baseline.sh:

EDIT 1 (OMIM genemap2 wiring + hard-fail guard):
  After the connector rewrite, genemap2.txt is the SOLE source for all three OMIM
  columns; mim2gene is inert. The launch script wires only --omim-path (mim2gene)
  and has NO guard on genemap2.txt -> OMIM columns silent-zero AND the script's own
  preflight guards the wrong file. This adds a genemap2 file-pick, a [ -f ] hard-fail
  (exit 8, loud echo, mirroring PhyloP/dbSNP/ClinGen), and --omim-genemap2-path to $ARGS.
  --omim-path (mim2gene) is kept for backward-compat (connector ignores it; harmless).

EDIT 2 (pybigtools install + verify):
  launch_run17 activates a prebuilt /venv/main and never pip-installs. If that venv
  lacks pybigtools, PhyloPConnector hits ImportError -> silent phylop_score=0.0.
  This adds an idempotent `pip install pybigtools>=0.3.0` + a hard `import pybigtools`
  verify (exit 4) to step 4, covering the prebuilt-venv scenario regardless of the
  requirements files. (PyPI-confirmed cp39-cp313 manylinux wheels => binary install.)

Anchors verified against reads 24a/25a/25b. ANCHOR-BASED, IDEMPOTENT, LF-only (bash).
"""
from __future__ import annotations
import argparse
from pathlib import Path

TARGET = Path("scripts/launch_run17_baseline.sh")
MARKER_GENEMAP2 = "--omim-genemap2-path"
MARKER_PYBIG = "import pybigtools"

# ---- EDIT 1: insert genemap2 block right after the OMIM block's closing fi ----
# Anchor: the OMIM block end + the PhyloP comment that follows (read 25a, exact).
OMIM_OLD = '''    ARGS="$ARGS --omim-path $OMIM_FILE"; echo "==> OMIM wired: $OMIM_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: OMIM file missing under $DATA/external/omim/" | tee -a "$LOG"; exit 8
fi
# PhyloP: single source file.'''

OMIM_NEW = '''    ARGS="$ARGS --omim-path $OMIM_FILE"; echo "==> OMIM wired: $OMIM_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: OMIM file missing under $DATA/external/omim/" | tee -a "$LOG"; exit 8
fi
# OMIM genemap2: the SOLE source for omim_n_diseases / omim_n_diseases_molecular /
# omim_is_autosomal_dominant after the connector rewrite (mim2gene is inert now).
# Without --omim-genemap2-path, all three OMIM columns silent-zero across the cohort.
OMIM_GENEMAP2_FILE="$(ls "$DATA"/external/omim/*genemap2* 2>/dev/null | head -n1 || true)"
if [ -n "$OMIM_GENEMAP2_FILE" ] && [ -f "$OMIM_GENEMAP2_FILE" ]; then
    ARGS="$ARGS --omim-genemap2-path $OMIM_GENEMAP2_FILE"; echo "==> OMIM genemap2 wired: $OMIM_GENEMAP2_FILE" | tee -a "$LOG"
else
    echo "==> ABORT: OMIM genemap2.txt missing under $DATA/external/omim/ (omim_n_diseases/omim_n_diseases_molecular/omim_is_autosomal_dominant would silent-zero)" | tee -a "$LOG"; exit 8
fi
# PhyloP: single source file.'''

# ---- EDIT 2: insert pybigtools install+verify after the step-4 dep-check heredoc ----
# Anchor: the heredoc close (lightgbm smoke fit) + the step-5 comment (read 25b, exact).
PYBIG_OLD = '''print('lightgbm smoke fit OK')
" 2>&1 | tee -a "$LOG"

# -- 5. KG + rnaseq + STRING wiring sanity (read-only column probes) -----------'''

PYBIG_NEW = '''print('lightgbm smoke fit OK')
" 2>&1 | tee -a "$LOG"

# -- 4b. pybigtools (PhyloP BigWig reader): install-if-missing + HARD verify.
#     launch activates a prebuilt /venv/main that may predate this dep; without it
#     PhyloPConnector ImportErrors -> silent phylop_score=0.0. Idempotent install,
#     then a hard import gate (exit 4) so a failed install is LOUD, never silent.
echo "==> [4b/6] pybigtools (PhyloP) install + verify" | tee -a "$LOG"
$PY -m pip install 'pybigtools>=0.3.0' --quiet 2>&1 | tail -3 | tee -a "$LOG" || true
if ! python -c "import pybigtools; print('pybigtools', pybigtools.__version__ if hasattr(pybigtools,'__version__') else 'OK')" 2>&1 | tee -a "$LOG"; then
    echo "==> ABORT (exit 4): pybigtools import failed -- PhyloP would silent-zero. Install pybigtools>=0.3.0 on the VM." | tee -a "$LOG"; exit 4
fi

# -- 5. KG + rnaseq + STRING wiring sanity (read-only column probes) -----------'''


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")

    done_gm = MARKER_GENEMAP2 in src
    done_pb = MARKER_PYBIG in src
    if done_gm and done_pb:
        print("OK (idempotent): both genemap2 wiring and pybigtools verify already present."); return 0

    anchors = []
    if not done_gm:
        anchors.append(("EDIT1 OMIM block", OMIM_OLD))
    if not done_pb:
        anchors.append(("EDIT2 dep-check", PYBIG_OLD))
    ok = True
    for label, anc in anchors:
        c = src.count(anc)
        if c != 1:
            print(f"FAIL: anchor '{label}' occurs {c}x (need 1)."); ok = False
    if not ok:
        return 3
    if ns.check:
        print(f"CHECK: {len(anchors)} anchor(s) found exactly once."); print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".sh.pre_genemap2_pybigtools.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")

    patched = src
    if not done_gm:
        patched = patched.replace(OMIM_OLD, OMIM_NEW, 1)
    if not done_pb:
        patched = patched.replace(PYBIG_OLD, PYBIG_NEW, 1)
    # bash => LF only
    TARGET.write_text(patched, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = [
        ("--omim-genemap2-path in ARGS", "--omim-genemap2-path $OMIM_GENEMAP2_FILE" in after),
        ("genemap2 hard-fail guard", "OMIM genemap2.txt missing" in after),
        ("genemap2 file-pick", "*genemap2*" in after),
        ("pybigtools install", "pip install 'pybigtools>=0.3.0'" in after),
        ("pybigtools verify+exit4", "ABORT (exit 4): pybigtools import failed" in after),
        ("step 4b label", "[4b/6] pybigtools" in after),
    ]
    allok = True
    for label, present in checks:
        print(f"  {'OK' if present else 'MISSING'}  {label}"); allok &= present
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 5


if __name__ == "__main__":
    raise SystemExit(main())
