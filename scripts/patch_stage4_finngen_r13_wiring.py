#!/usr/bin/env python3
r"""patch_stage4_finngen_r13_wiring.py

Stage 4 of the FinnGen R12+R13 dual-release experiment: wire R13's REAL data into the
pipeline as a second, independent FinnGenConnector pass (column_prefix="r13_") reading
a separate ~30GB R13 file. This is where R13 live data actually enters the pipeline.

Mirrors the existing R12 wiring (verified verbatim) across THREE files:

  real_data_prep.py:
  (A) AnnotationConfig: add `finngen_r13_path` field after `finngen_path` (L280).
  (B) annotation block: add a parallel R13 if/else after the R12 block (L587-599).
      R13 if-branch: FinnGenConnector(tsv_path=..., column_prefix="r13_").annotate(df).
      R13 else-branch: default-fill the PREFIXED names via finngen_columns("r13_")
      (NOT FINNGEN_COLUMNS, which is the unprefixed R12 triple) + r13 enrichment=1.0.

  run_phase2_eval.py:
  (C) argparse: add `--finngen-r13-path` mirroring `--finngen-path` (L133-137).
  (D) AnnotationConfig(...): add `finngen_r13_path=...` kwarg after finngen_path (L340).

  launch_run17_baseline.sh:
  (E) add R13 file pick + hard-fail guard mirroring R12 (L180-181), pointing at the
      verified-local data/external/finngen/finngen_R13_annotated_variants_v0.gz
      (correct-spelled 'finngen_' + '_v0', distinct from R12's typo 'finnge_' + '_v1').

ANCHOR-BASED, IDEMPOTENT, LF. Per-file guards (applies a file's edits or none).
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

RDP = Path("src/genomic_variant_classifier/data/real_data_prep.py")
RPE = Path("scripts/run_phase2_eval.py")
LAUNCH = Path("scripts/launch_run17_baseline.sh")

# ---- (A) AnnotationConfig field ----
A_OLD = '''    finngen_path: Optional[Path] = None  # FinnGen R10 annotated variants TSV'''
A_NEW = '''    finngen_path: Optional[Path] = None  # FinnGen R10 annotated variants TSV
    finngen_r13_path: Optional[Path] = None  # FinnGen R13 annotated variants (dual-release experiment)'''

# ---- (B) annotation block: append R13 if/else after R12's (anchor: R12 block end 593-599) ----
B_OLD = '''        else:
            from genomic_variant_classifier.data.finngen import FinnGenConnector, FINNGEN_COLUMNS

            for col in FINNGEN_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
            df["finngen_enrichment"] = 1.0

        return df'''
B_NEW = '''        else:
            from genomic_variant_classifier.data.finngen import FinnGenConnector, FINNGEN_COLUMNS

            for col in FINNGEN_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
            df["finngen_enrichment"] = 1.0

        # FinnGen R13 (dual-release experiment): independent second pass, prefixed columns
        if self.annotation_config.finngen_r13_path:
            from genomic_variant_classifier.data.finngen import FinnGenConnector

            finngen_r13 = FinnGenConnector(
                tsv_path=self.annotation_config.finngen_r13_path, column_prefix="r13_"
            )
            df = finngen_r13.annotate(df)
        else:
            from genomic_variant_classifier.data.finngen import finngen_columns

            for col in finngen_columns("r13_"):
                if col not in df.columns:
                    df[col] = 0.0
            df["finngen_r13_enrichment"] = 1.0

        return df'''

# ---- (C) argparse flag ----
C_OLD = '''    p.add_argument(
        "--finngen-path",
        default=None,'''
C_NEW = '''    p.add_argument(
        "--finngen-r13-path",
        default=None,
        help="FinnGen R13 annotated variants (gzipped) "
        "(data/external/finngen/finngen_R13_annotated_variants_v0.gz). "
        "Dual-release experiment: annotates finngen_r13_af_fin/af_nfsee/enrichment "
        "via an independent connector pass with column_prefix='r13_'.",
    )
    p.add_argument(
        "--finngen-path",
        default=None,'''

# ---- (D) AnnotationConfig kwarg ----
D_OLD = '''            finngen_path=Path(args.finngen_path) if args.finngen_path else None,'''
D_NEW = '''            finngen_path=Path(args.finngen_path) if args.finngen_path else None,
            finngen_r13_path=Path(args.finngen_r13_path) if args.finngen_r13_path else None,'''

# ---- (E) launcher R13 file pick + guard ----
E_OLD = '''FINNGEN_FILE="$DATA/external/finngen/finnge_R12_annotated_variants_v1.gz"  # registry typo 'finnge'
if [ -f "$FINNGEN_FILE" ]; then ARGS="$ARGS --finngen-path $FINNGEN_FILE"; echo "==> FinnGen wired: $FINNGEN_FILE" | tee -a "$LOG"; else echo "==> ABORT: FinnGen file missing: $FINNGEN_FILE" | tee -a "$LOG"; exit 7; fi'''
E_NEW = '''FINNGEN_FILE="$DATA/external/finngen/finnge_R12_annotated_variants_v1.gz"  # registry typo 'finnge'
if [ -f "$FINNGEN_FILE" ]; then ARGS="$ARGS --finngen-path $FINNGEN_FILE"; echo "==> FinnGen wired: $FINNGEN_FILE" | tee -a "$LOG"; else echo "==> ABORT: FinnGen file missing: $FINNGEN_FILE" | tee -a "$LOG"; exit 7; fi
FINNGEN_R13_FILE="$DATA/external/finngen/finngen_R13_annotated_variants_v0.gz"  # R13 dual-release (correct spelling, _v0)
if [ -f "$FINNGEN_R13_FILE" ]; then ARGS="$ARGS --finngen-r13-path $FINNGEN_R13_FILE"; echo "==> FinnGen R13 wired: $FINNGEN_R13_FILE" | tee -a "$LOG"; else echo "==> ABORT: FinnGen R13 file missing: $FINNGEN_R13_FILE" | tee -a "$LOG"; exit 7; fi'''

PY_MARK = "finngen_r13_path"
LAUNCH_MARK = "--finngen-r13-path"


def _do(path, edits, mark, label, is_py):
    src = path.read_text(encoding="utf-8")
    if mark in src:
        return ("idem", src, True)
    probs = []
    for nm, old, _new in edits:
        c = src.count(old)
        if c != 1:
            probs.append(f"  {label}/{nm}: anchor {c}x (need 1)")
    if probs:
        print(f"FAIL: {label}:\n" + "\n".join(probs)); return ("fail", src, False)
    return ("ok", src, False)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    for p in (RDP, RPE, LAUNCH):
        if not p.exists():
            print(f"FAIL: {p} missing."); return 2

    rdp_edits = [("field", A_OLD, A_NEW), ("block", B_OLD, B_NEW)]
    rpe_edits = [("argflag", C_OLD, C_NEW), ("kwarg", D_OLD, D_NEW)]
    launch_edits = [("guard", E_OLD, E_NEW)]

    s1, src1, idem1 = _do(RDP, rdp_edits, PY_MARK, "real_data_prep", True)
    s2, src2, idem2 = _do(RPE, rpe_edits, PY_MARK, "run_phase2_eval", True)
    s3, src3, idem3 = _do(LAUNCH, launch_edits, LAUNCH_MARK, "launch", False)
    if "fail" in (s1, s2, s3):
        print("RESULT: FAIL (anchor validation)"); return 3
    if ns.check:
        print(f"CHECK: real_data_prep={s1}, run_phase2_eval={s2}, launch={s3}.")
        print("RESULT: PASS (check)"); return 0

    def _write(path, status, src, edits, idem):
        if idem: return
        if status != "ok": return
        b = path.with_suffix(path.suffix + ".pre_stage4.bak")
        if not b.exists(): b.write_text(src, encoding="utf-8", newline="")
        new = src
        for _n, old, repl in edits:
            new = new.replace(old, repl, 1)
        path.write_text(new, encoding="utf-8", newline="\n")

    _write(RDP, s1, src1, rdp_edits, idem1)
    _write(RPE, s2, src2, rpe_edits, idem2)
    _write(LAUNCH, s3, src3, launch_edits, idem3)

    a1 = RDP.read_text(encoding="utf-8")
    a2 = RPE.read_text(encoding="utf-8")
    a3 = LAUNCH.read_text(encoding="utf-8")
    checks = {
        "rdp: field added": "finngen_r13_path: Optional[Path]" in a1,
        "rdp: r13 connector instantiated w/ prefix": 'column_prefix="r13_"' in a1,
        "rdp: r13 else uses finngen_columns(r13_)": 'finngen_columns("r13_")' in a1,
        "rdp: r13 enrichment default": 'df["finngen_r13_enrichment"] = 1.0' in a1,
        "rpe: argflag added": "--finngen-r13-path" in a2,
        "rpe: kwarg added": "finngen_r13_path=Path(args.finngen_r13_path)" in a2,
        "launch: r13 guard added": "--finngen-r13-path $FINNGEN_R13_FILE" in a3,
        "launch: r13 file correct spelling": "finngen_R13_annotated_variants_v0.gz" in a3,
        "launch: r13 hard-fail exit 7": a3.count("exit 7") >= 2,
    }
    try:
        ast.parse(a1); checks["rdp compiles"] = True
    except SyntaxError as e:
        checks["rdp compiles"] = False; print("  RDP SYNTAX:", e)
    try:
        ast.parse(a2); checks["rpe compiles"] = True
    except SyntaxError as e:
        checks["rpe compiles"] = False; print("  RPE SYNTAX:", e)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
