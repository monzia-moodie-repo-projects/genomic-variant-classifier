#!/usr/bin/env python3
"""
Read-only diagnostic for the Run 15 gene-level ablation state.

Consolidates three verifications that were pending, all with BOUNDED output
and NO file mutation:

  (1) ablation_results.parquet : shape, columns, dtypes, scalar-only tidy view,
      and best-effort per-arm AUROC (never dumps array-valued cells).
  (2) arm directories actually present under ablation_run15/.
  (3) each arm's eval_report.json : AUROC + any bootstrap CI keys.
  (4) ABLATION_n_pathogenic_in_gene.md : raw-byte check distinguishing
      genuine double-encoded mojibake from a PowerShell console display artifact.

Usage (PowerShell, from repo root):
    python inspect_ablation_state.py
Optional overrides:
    $env:ABL = "C:\\Projects\\genomic-variant-classifier\\outputs\\ablation_run15"
    $env:DOC = "C:\\Projects\\genomic-variant-classifier\\docs\\ABLATION_n_pathogenic_in_gene.md"

Everything is wrapped so one failing section never aborts the others, and
exceptions are PRINTED (never swallowed).
"""
import os
import sys
import json
import traceback

DEFAULT_ABL = r"C:\Projects\genomic-variant-classifier\outputs\ablation_run15"
DEFAULT_DOC = r"C:\Projects\genomic-variant-classifier\docs\ABLATION_n_pathogenic_in_gene.md"

ABL = os.environ.get("ABL", DEFAULT_ABL)
DOC = os.environ.get("DOC", DEFAULT_DOC)

# Known double-encoded sequences (file read as UTF-8) -> intended char.
# These appear ONLY when UTF-8 bytes were mis-decoded as cp1252 then re-saved.
MOJIBAKE = {
    "\u00e2\u2030\u00a4": "\u2264",   # 'â‰¤'  -> '<=' (U+2264)
    "\u00e2\u20ac\u201d": "\u2014",   # 'â€”'  -> em dash (U+2014)
    "\u00e2\u20ac\u2122": "\u2019",   # 'â€™' -> right single quote
    "\u00e2\u20ac\u0153": "\u201c",   # 'â€œ' -> left double quote
    "\u00e2\u20ac\u009d": "\u201d",   # 'â€\x9d' -> right double quote
    "\u00e2\u20ac\u02dc": "\u2018",   # 'â€˜' -> left single quote
    "\u00e2\u20ac\u00a6": "\u2026",   # 'â€¦' -> ellipsis
}


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def looks_arrayish(v):
    """True if a cell value is a list/tuple/ndarray (i.e. NOT a printable scalar)."""
    if isinstance(v, str):
        return False
    if isinstance(v, (list, tuple)):
        return True
    # numpy arrays / pandas extension arrays expose __len__ but are not scalars
    return hasattr(v, "__len__")


def inspect_parquet():
    section("(1) ablation_results.parquet  -- schema first, bounded")
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        print("  pandas import FAILED:", repr(e))
        return
    pq = os.path.join(ABL, "ablation_results.parquet")
    print("  path :", pq)
    print("  exists:", os.path.exists(pq))
    if not os.path.exists(pq):
        print("  -> aggregator parquet NOT FOUND (so any '3 rows' claim is unverified).")
        return
    try:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 32)
        d = pd.read_parquet(pq)
    except Exception:
        print("  read_parquet FAILED:")
        traceback.print_exc()
        return

    print("  SHAPE :", d.shape, "  (rows, cols)")
    print("  ROWS  :", len(d))
    print("  COLUMNS:", list(d.columns))
    print("  DTYPES:")
    for c in d.columns:
        print("    {:<28} {}".format(str(c), str(d.dtypes[c])))

    # Classify columns so we never print an array-valued cell.
    scalar_cols, array_cols = [], []
    for c in d.columns:
        try:
            any_arr = d[c].map(looks_arrayish).any()
        except Exception:
            any_arr = False
        (array_cols if any_arr else scalar_cols).append(c)
    print("  SCALAR COLS:", scalar_cols)
    print("  ARRAY  COLS:", array_cols, "(suppressed from the table below)")

    if scalar_cols:
        print("\n  --- scalar columns, first 20 rows ---")
        try:
            print(d[scalar_cols].head(20).to_string())
        except Exception:
            print("  tidy print FAILED:")
            traceback.print_exc()

    # Best-effort arm + AUROC summary without assuming exact names.
    arm_col = next((c for c in d.columns
                    if str(c).lower() in ("ablation", "arm", "run_id", "name", "variant")), None)
    auroc_cols = [c for c in scalar_cols if "auroc" in str(c).lower() or "auc" in str(c).lower()]
    print("\n  arm-name column detected :", arm_col)
    print("  AUROC scalar columns     :", auroc_cols)
    if arm_col and auroc_cols:
        try:
            print("\n  --- per-arm AUROC ---")
            print(d[[arm_col] + auroc_cols].to_string(index=False))
        except Exception:
            traceback.print_exc()
    else:
        print("  (could not resolve arm/AUROC columns -> read COLUMNS above and re-run focused.)")


def inspect_arm_dirs():
    section("(2) arm directories present under ablation_run15/")
    try:
        entries = sorted(
            name for name in os.listdir(ABL)
            if os.path.isdir(os.path.join(ABL, name))
        )
        print("  dirs:", entries if entries else "(none)")
        for expected in ("full", "no_gene_prevalence", "no_gene_level"):
            print("    {:<22} {}".format(expected, "PRESENT" if expected in entries else "MISSING"))
    except Exception:
        traceback.print_exc()


def inspect_eval_reports():
    section("(3) per-arm eval_report.json : AUROC + bootstrap CI")
    try:
        arms = sorted(
            name for name in os.listdir(ABL)
            if os.path.isdir(os.path.join(ABL, name))
        )
    except Exception:
        traceback.print_exc()
        return
    for arm in arms:
        er = os.path.join(ABL, arm, "eval_report.json")
        print("  [{}]".format(arm))
        if not os.path.exists(er):
            print("    eval_report.json: MISSING")
            continue
        try:
            with open(er, "r", encoding="utf-8") as fh:
                rep = json.load(fh)
        except Exception:
            print("    eval_report.json: read/parse FAILED")
            traceback.print_exc()
            continue
        keys = [k for k in rep
                if any(s in k.lower() for s in ("auroc", "auprc", "ci", "interval", "test", "ci_low", "ci_high"))]
        if not keys:
            print("    no auroc/ci-like top-level keys; top-level keys:", list(rep.keys())[:20])
        for k in keys:
            v = rep[k]
            # keep it short: don't print giant nested structures
            sv = v if isinstance(v, (int, float, str, bool, list)) and len(str(v)) <= 80 else "<{}>".format(type(v).__name__)
            print("    {:<24} {}".format(k, sv))


def inspect_mojibake():
    section("(4) ABLATION doc encoding : real corruption vs console display")
    print("  path :", DOC)
    if not os.path.exists(DOC):
        print("  NOT FOUND")
        return
    try:
        with open(DOC, "rb") as fh:
            raw = fh.read()
    except Exception:
        traceback.print_exc()
        return
    has_bom = raw[:3] == b"\xef\xbb\xbf"
    print("  size bytes :", len(raw))
    print("  UTF-8 BOM  :", has_bom)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print("  NOT valid UTF-8 (decode error) -> deeper encoding problem:", e)
        return

    clean_le = text.count("\u2264")   # correct '<='
    clean_em = text.count("\u2014")   # correct em dash
    print("  clean U+2264 ('<=')   count:", clean_le)
    print("  clean U+2014 (em dash) count:", clean_em)

    found = {seq: text.count(seq) for seq, _ in MOJIBAKE.items() if text.count(seq)}
    total_moji = sum(found.values())
    if found:
        print("  MOJIBAKE sequences present (file is genuinely double-encoded):")
        for seq, n in found.items():
            print("    {!r} x{}  -> intended {!r}".format(seq, n, MOJIBAKE[seq]))
    else:
        print("  MOJIBAKE sequences present: NONE")

    print("\n  VERDICT:")
    if total_moji == 0:
        print("    File bytes are correct UTF-8. The 'a-hat' rendering in PowerShell is a")
        print("    CONSOLE DISPLAY artifact (codepage != 65001). DO NOT 'repair' the file.")
        print("    For readable display in this session: run  chcp 65001  then re-open.")
    else:
        print("    File contains {} genuinely double-encoded sequence(s).".format(total_moji))
        print("    A TARGETED repair is warranted (replace only the sequences listed above).")
        print("    Run the separate repair script AFTER confirming; it backs up first.")


def main():
    print("ABL =", ABL)
    print("DOC =", DOC)
    for fn in (inspect_parquet, inspect_arm_dirs, inspect_eval_reports, inspect_mojibake):
        try:
            fn()
        except Exception:
            section("UNEXPECTED ERROR in " + fn.__name__)
            traceback.print_exc()
    print("\nDONE (read-only; nothing was modified).")


if __name__ == "__main__":
    main()
