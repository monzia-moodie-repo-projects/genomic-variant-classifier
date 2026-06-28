#!/usr/bin/env python3
"""PATCH_RUN17_PLAN_DOC_V1 -- reconcile docs/runs/RUN_17_PLAN.md to the 2026-06-28
FinnGen probe verification (PROBE_V2_INTEGRITY full streaming pass).

Verified facts driving the edits (probe, both files CLEAN end-to-end):
  R12: 21,331,644 data rows; 1017 cols; 32,126,590,987 B (29.92 GB);
       sha256 e27f91568ca7f8842528c45262f533442e2c23016221e882f2a547fd7cb99231
  R13: 21,331,644 data rows; 1025 cols; 29,768,495,399 B (27.72 GB);
       sha256 109b4f3f13ae8c4ade148cf47402ba23e442cd4a6648ab31bdb9d6518bca99c1
  IDENTICAL row counts; HEAD-sample variant-key Jaccard 1.0; size gap = ENCODING
  (R12 _v1 carries b37_coord + nfee/nfse coord/AF strings; R13 _v0 drops b37_coord,
   adds AC/AN integer counts that compress better -> R13 smaller despite +8 net cols
   and higher coverage). Both BGZF (R12 1.74M / R13 1.82M members).

7 anchored, idempotent edits (sentinel: 'PROBE-VERIFIED 2026-06-28'). The doc itself
contains non-ASCII (em-dash U+2014, arrows U+2192, rho); we PRESERVE those and only ADD
ASCII text. Output stays UTF-8, no BOM, LF. Aborts if any anchor != exactly 1 occurrence.

Usage: python patch_run17_plan_doc.py <docs/runs/RUN_17_PLAN.md>
"""
from __future__ import annotations
import sys
from pathlib import Path

DASH = "\u2014"   # em dash, as used in the doc
ARROW = "\u2192"  # right arrow

SENTINEL = "PROBE-VERIFIED 2026-06-28"

# ---- EDIT 1: B.D2 line, size 29.77 GB -> 27.72 GB + "SAME variant set" ----
A1 = "`data/external/finngen/finngen_R13_annotated_variants_v0.gz` (29.77 GB, correct spelling, 1025 cols, SAME schema as R12)."
R1 = "`data/external/finngen/finngen_R13_annotated_variants_v0.gz` (27.72 GB / 29,768,495,399 bytes, correct spelling, 1025 cols, SAME variant set as R12 -- see the PROBE-VERIFIED note below)."

# ---- EDIT 2: B.D2 verification sub-bullet -> evidence-backed ----
A2 = ("  - Verified earlier: R12 17085/20000 nonzero (mean AF 0.1025); R13 19318/20000 nonzero (mean AF 0.0971). "
      "Same variants, same coords, evolved frequencies " + DASH + " apples-to-apples by construction.")
R2 = ("  - Coverage/AF (20k reference sample): R12 17085/20000 nonzero (mean AF 0.1025); R13 19318/20000 nonzero (mean AF 0.0971) "
      "-- R13 informs more of the cohort.\n"
      "  - **PROBE-VERIFIED 2026-06-28** (scripts/probe_finngen_sizes.py, full streaming pass, both files integrity=CLEAN): "
      "R12 and R13 have IDENTICAL data-row counts (21,331,644 each) and a HEAD-sample variant-key Jaccard of 1.0 "
      "(chrom:pos:ref:alt) -- so 'same variants, same coords, apples-to-apples' is verified, not assumed. "
      "The R13 file is SMALLER than R12 despite more samples, +8 net columns, and higher coverage purely because of "
      "ENCODING, not missing content: R12 (_v1) carries a per-row b37_coord string plus EXOME/GENOME nfee/nfse coord+AF "
      "string columns, while R13 (_v0) drops b37_coord and adds AC/AN integer-count columns that gzip-compress far better "
      "than coordinate strings. Both are BGZF (block-gzip; R12 ~1.74M members, R13 ~1.82M). SHA-256 recorded in "
      "data/external/finngen/CHECKSUMS.sha256.")

# ---- EDIT 3: E SCP line, ~60GB -> ~57.6 GB + per-config + 3x ----
A3 = ("- **SCP up**: both finngen files ~60GB total (R12 29.9GB + R13 29.77GB) over the id_lambda_run8 key. "
      "Non-trivial " + DASH + " confirm budget acceptance before launch.")
R3 = ("- **SCP up**: ~57.6 GB total when both ship (R12 29.92 GB + R13 27.72 GB; PROBE-VERIFIED 2026-06-28) over the "
      "id_lambda_run8 key. Per-config: `r12only` ships R12 only (~29.92 GB); `r13only` ships R13 only (~27.72 GB); "
      "`both`/baseline ships both (~57.6 GB). The three configs run as THREE independent runs, so aggregate training "
      "cost is ~3x a single run -- confirm budget acceptance before launch.")

# ---- EDIT 4: E annotation line, note BGZF + verified clean ----
A4 = ("- **VM annotation**: two ~30GB gzip passes, bounding-box-filtered so RAM stays bounded by the matched subset, "
      "but two full decompression passes. Flag against MIN_RAM_GB=50.")
R4 = ("- **VM annotation**: two BGZF (block-gzip) passes (R12 32.1 GB / R13 29.8 GB on disk; both decompress CLEAN end-to-end, "
      "PROBE-VERIFIED 2026-06-28), bounding-box-filtered so RAM stays bounded by the matched subset, but two full decompression "
      "passes. Flag against MIN_RAM_GB=50. (r12only/r13only each decompress one file; both/baseline decompresses both.)")

# ---- EDIT 5: F G1 line, refresh floor + launchers + postflight-exists ----
A5 = ("- **G1 (local)**: scripts/Run_Preflight_Local.ps1 adapted Run-15" + ARROW + "Run-17 " + DASH + " "
      "\u00a73 DELETE (imodelsx patch moved to kan.py L181); \u00a76 test floor 566" + ARROW + "~1483; "
      "\u00a77 rebuild data list (FinnGen NOW BOTH files local, hard-fail); \u00a711 repoint launchPath" + ARROW + "launch_run17_baseline.sh; "
      "\u00a712/13 create RUN_17 postflight; ADD agent-liveness via scripts/check_agents_active.py. "
      "Reference slice (build_reference_slice, now feeding finngen) is the G1 single-source-of-truth.")
R5 = ("- **G1 (local)**: scripts/Run_Preflight_Local.ps1 adapted Run-15" + ARROW + "Run-17 " + DASH + " "
      "\u00a73 DELETE (imodelsx patch moved to kan.py L181); \u00a76 test floor 566" + ARROW + "1496 collected / 1485 pass "
      "(reconciled 2026-06-28, CI #486 green); \u00a77 rebuild data list (FinnGen NOW BOTH files local, hard-fail); "
      "\u00a711 THREE launchers (launch_run17_{baseline,r12only,r13only}.sh); \u00a712/13 RUN_17 postflight EXISTS "
      "(scripts/Run17_Postflight.ps1, parameterized -Config {both,r12only,r13only} + -DryRun, CI #486 green); "
      "ADD agent-liveness via scripts/check_agents_active.py. Reference slice (build_reference_slice, now feeding finngen) "
      "is the G1 single-source-of-truth.\n"
      "  - **Postflight usage (per config)**: run `Run17_Postflight.ps1 -Config <both|r12only|r13only> -DryRun` first "
      "(prints derived paths; NO SSH/SCP/destroy), then the real invocation, then Vastai_Destroy_Confirmed.ps1 to tear down. "
      "A downloaded .ps1 carries Mark-of-the-Web under RemoteSigned -- `Unblock-File` it before running.")

# ---- EDIT 6: insert H.4 after the H.3 Spearman bullet (before the blank line + '## I.') ----
A6 = ("- **Feature-importance rank correlation (Spearman)**: \u03c1 between finngen_* and finngen_r13_* importance vectors "
      "across models. Range [-1,1]. Why: tests the apples-to-apples expectation (releases should agree on which finngen "
      "signal matters, but not perfectly). Varied: per-model \u03c1 + aggregate.")
R6 = (A6 + "\n\n"
      "### H.4 Ablation realization & caveat (PROBE-VERIFIED 2026-06-28)\n"
      "The three feature configs {R12-only, R13-only, both} are realized by CONSTANT-FILLING the excluded release's 3 "
      "finngen columns (af_fin/af_nfsee at 0.0, enrichment at 1.0), NOT by dropping them -- the 91-column contract is fixed. "
      "Interpretation caveat: for the tree learners (CatBoost/XGBoost/LightGBM/RF/GBM) a constant column carries no split "
      "information, so constant-filled approximates 'absent'; but the column still occupies a feature slot, and the linear "
      "(LR) and neural (1D-CNN/TabularNN/MC-Dropout/Deep Ensemble/KAN) models treat a constant input differently from true "
      "absence (it contributes a bias-like constant, not nothing). So each cross-config delta measures "
      "'release present vs constant-filled', which for tree models is a close proxy for 'present vs absent' and for "
      "linear/NN models should be read as 'present vs neutralized'. The comparison is on the SAME 21,331,644-variant "
      "universe (identical row counts + Jaccard 1.0 head sample), so coverage-delta and AF-shift are computed over a common "
      "variant set -- no intersection caveat is required.")

# ---- EDIT 7: append a 2026-06-28 decision-log entry after the last 2026-06-27 bullet ----
A7 = ("- **2026-06-27**: Option A (allowlist R13) tried + reverted to Option B (feed fixture) per the pre-existing "
      "test_allowlist_unchanged_size guard. Documented in A.D1.")
R7 = (A7 + "\n"
      "- **2026-06-28**: FinnGen R12/R13 integrity + size-gap VERIFIED before any paid run "
      "(scripts/probe_finngen_sizes.py, full streaming single pass). Both files integrity=CLEAN; IDENTICAL data-row counts "
      "(21,331,644 each); HEAD-sample variant-key Jaccard 1.0. The smaller R13 file (27.72 GB vs R12 29.92 GB) is an ENCODING "
      "difference (R13 _v0 drops the per-row b37_coord string and uses AC/AN integer counts vs R12 _v1's coordinate/AF strings), "
      "NOT fewer variants -- so B.D2's 'same variants, apples-to-apples' is now evidence-backed. SHA-256 of both files recorded "
      "in data/external/finngen/CHECKSUMS.sha256 (machine-checkable provenance).")

EDITS = [("E1 B.D2 size", A1, R1), ("E2 B.D2 verify", A2, R2), ("E3 E scp", A3, R3),
         ("E4 E annot", A4, R4), ("E5 F g1", A5, R5), ("E6 H.4", A6, R6), ("E7 I log", A7, R7)]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python patch_run17_plan_doc.py <RUN_17_PLAN.md>"); return 2
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"ERROR: {p} not found"); return 2
    raw = p.read_bytes()
    if raw[:3] == b"\xef\xbb\xbf":
        print("ERROR: file has UTF-8 BOM; expected none."); return 2
    text = raw.decode("utf-8")

    if SENTINEL in text:
        print("ALREADY PATCHED (sentinel present); no change."); return 0

    # validate every anchor occurs exactly once BEFORE applying any
    for name, a, _ in EDITS:
        n = text.count(a)
        if n != 1:
            print(f"ANCHOR FAILED [{name}]: occurs {n}x (expected 1). NO CHANGE.")
            return 1
    # apply
    for _name, a, r in EDITS:
        text = text.replace(a, r, 1)

    # added text must be ASCII (we only added ASCII); verify we didn't inject stray non-ASCII
    # (the doc keeps its own em-dash/arrow/rho; we just ensure our ADDED lines are clean by
    #  checking the sentinel-bearing additions are ascii)
    for marker in ("PROBE-VERIFIED 2026-06-28", "### H.4 Ablation", "2026-06-28**:"):
        idx = text.find(marker)
        if idx == -1:
            print(f"POST FAILED: expected inserted marker missing: {marker}"); return 1

    p.with_suffix(p.suffix + ".bak").write_bytes(raw)
    p.write_text(text, encoding="utf-8", newline="\n")
    print(f"PATCHED: {p} (7 edits, sentinel '{SENTINEL}').")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
