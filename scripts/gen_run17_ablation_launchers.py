#!/usr/bin/env python3
"""Generate the two FinnGen ablation launchers from launch_run17_baseline.sh.

Option A (three independent runs): the baseline launcher IS the "both" config and is
left untouched. This generator reads it and emits two near-identical copies that differ
ONLY in: (1) header line, (2) OUTDIR comment, (3) OUTDIR, (4) LOG, and (5) the FinnGen
flag block -- each ablation keeps ONE release's flag+hard-fail and replaces the other
release's two lines with an explicit EXCLUSION comment (no flag, no exit-7). The eval's
else-branch constant-fills the excluded release's 3 columns (0.0 / enrichment=1.0), so
the 91-feature contract holds.

Anchor-based: every transformed string is validated to occur EXACTLY ONCE in the
baseline before any output is written. bash -n syntax-gates each output (caller runs it).
Each output carries a GENERATED banner naming its baseline source + a drift warning.

Usage: python gen_run17_ablation_launchers.py <baseline.sh> <out_dir>
Emits: <out_dir>/launch_run17_r12only.sh  and  <out_dir>/launch_run17_r13only.sh
"""
from __future__ import annotations
import sys
from pathlib import Path

# --- Exact anchors (verbatim from the real baseline, MA1-MA4) ---
A_HEADER = "# launch_run17_baseline.sh -- Run 17 full multi-source GPU run (forked from launch_run15_baseline.sh)."
A_OUTDIR_COMMENT = "#   - OUTDIR pinned to outputs/run17_baseline/full."
A_OUTDIR = 'OUTDIR="$REPO/outputs/run17_baseline/full"'
A_LOG = "LOG=/workspace/run17_baseline_master.log"
A_R12_FILE = 'FINNGEN_FILE="$DATA/external/finngen/finnge_R12_annotated_variants_v1.gz"  # registry typo \'finnge\''
A_R12_IF = 'if [ -f "$FINNGEN_FILE" ]; then ARGS="$ARGS --finngen-path $FINNGEN_FILE"; echo "==> FinnGen wired: $FINNGEN_FILE" | tee -a "$LOG"; else echo "==> ABORT: FinnGen file missing: $FINNGEN_FILE" | tee -a "$LOG"; exit 7; fi'
A_R13_FILE = 'FINNGEN_R13_FILE="$DATA/external/finngen/finngen_R13_annotated_variants_v0.gz"  # R13 dual-release (correct spelling, _v0)'
A_R13_IF = 'if [ -f "$FINNGEN_R13_FILE" ]; then ARGS="$ARGS --finngen-r13-path $FINNGEN_R13_FILE"; echo "==> FinnGen R13 wired: $FINNGEN_R13_FILE" | tee -a "$LOG"; else echo "==> ABORT: FinnGen R13 file missing: $FINNGEN_R13_FILE" | tee -a "$LOG"; exit 7; fi'

ALL_ANCHORS = [A_HEADER, A_OUTDIR_COMMENT, A_OUTDIR, A_LOG, A_R12_FILE, A_R12_IF, A_R13_FILE, A_R13_IF]

R13_EXCLUSION = """# R13 INTENTIONALLY EXCLUDED (r12only ablation): no --finngen-r13-path passed.
# run_phase2_eval.py else-branch constant-fills finngen_r13_af_fin/af_nfsee (0.0) +
# finngen_r13_enrichment (1.0). The 91-feature contract holds; R13 columns carry no signal."""

R12_EXCLUSION = """# R12 INTENTIONALLY EXCLUDED (r13only ablation): no --finngen-path passed.
# run_phase2_eval.py else-branch constant-fills finngen_af_fin/finngen_af_nfsee (0.0) +
# finngen_enrichment (1.0). The 91-feature contract holds; R12 columns carry no signal."""


def banner(config: str, src_name: str) -> str:
    return (
        f"# [GENERATED from {src_name} by gen_run17_ablation_launchers.py -- DO NOT EDIT BY HAND]\n"
        f"# Config: {config}. If the baseline changes, RE-GENERATE (do not hand-patch) to avoid drift.\n"
    )


def make_variant(base: str, config: str, src_name: str) -> str:
    t = base
    if config == "r12only":
        new_header = "# launch_run17_r12only.sh -- Run 17 R12-only ablation (FinnGen R13 excluded; generated from baseline)."
        new_outdir_comment = "#   - OUTDIR pinned to outputs/run17_r12only/full."
        new_outdir = 'OUTDIR="$REPO/outputs/run17_r12only/full"'
        new_log = "LOG=/workspace/run17_r12only_master.log"
        # keep R12 (file + if); replace R13 (file + if) with exclusion comment
        t = t.replace(A_R13_FILE + "\n" + A_R13_IF, R13_EXCLUSION, 1)
    elif config == "r13only":
        new_header = "# launch_run17_r13only.sh -- Run 17 R13-only ablation (FinnGen R12 excluded; generated from baseline)."
        new_outdir_comment = "#   - OUTDIR pinned to outputs/run17_r13only/full."
        new_outdir = 'OUTDIR="$REPO/outputs/run17_r13only/full"'
        new_log = "LOG=/workspace/run17_r13only_master.log"
        # keep R13 (file + if); replace R12 (file + if) with exclusion comment
        t = t.replace(A_R12_FILE + "\n" + A_R12_IF, R12_EXCLUSION, 1)
    else:
        raise ValueError(config)

    t = t.replace(A_HEADER, new_header, 1)
    t = t.replace(A_OUTDIR_COMMENT, new_outdir_comment, 1)
    t = t.replace(A_OUTDIR, new_outdir, 1)
    t = t.replace(A_LOG, new_log, 1)

    # Insert the GENERATED banner right after the shebang line.
    lines = t.split("\n")
    assert lines[0].startswith("#!"), "expected shebang on line 1"
    out = lines[0] + "\n" + banner(config, src_name) + "\n".join(lines[1:])
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python gen_run17_ablation_launchers.py <baseline.sh> <out_dir>")
        return 2
    base_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    if not base_path.exists():
        print(f"ERROR: baseline {base_path} not found")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = base_path.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        print("ERROR: baseline has a UTF-8 BOM; expected BOM-free shell script.")
        return 2
    base = raw.decode("utf-8")

    # --- Validate EVERY anchor occurs exactly once BEFORE generating ---
    problems = []
    for a in ALL_ANCHORS:
        n = base.count(a)
        if n != 1:
            problems.append(f"  anchor occurs {n}x (expected 1): {a[:70]}...")
    # also validate the two-line R12 / R13 blocks join exactly
    for label, block in [("R12 block", A_R12_FILE + "\n" + A_R12_IF),
                         ("R13 block", A_R13_FILE + "\n" + A_R13_IF)]:
        n = base.count(block)
        if n != 1:
            problems.append(f"  {label} (file+if adjacency) occurs {n}x (expected 1)")
    if problems:
        print("ANCHOR VALIDATION FAILED -- no files written:")
        print("\n".join(problems))
        return 1

    results = {}
    for cfg, fname in [("r12only", "launch_run17_r12only.sh"),
                       ("r13only", "launch_run17_r13only.sh")]:
        variant = make_variant(base, cfg, base_path.name)
        # newline='' preserves LF; ensure no BOM
        (out_dir / fname).write_text(variant, encoding="utf-8", newline="")
        results[fname] = variant

    print("GENERATED:")
    for fname in results:
        print(f"  {out_dir / fname}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
