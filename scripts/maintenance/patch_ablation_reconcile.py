#!/usr/bin/env python3
"""
Reconcile docs/ABLATION_n_pathogenic_in_gene.md with verified ground truth.

Every edit is a single-line, uniquely-anchored str.replace. The script:
  - reads the file as BYTES and decodes UTF-8 (preserves CRLF/LF; never adds a BOM),
  - PRE-CHECK: asserts each `old` anchor occurs EXACTLY once (else aborts, writes nothing);
    if every `old` is already absent and every `new` already present -> idempotent no-op,
  - backs up to <path>.bak_YYYYMMDD_HHMMSS before writing,
  - applies each replace exactly once,
  - POST-CHECK: asserts every `old` is gone and every `new` is present,
  - prints the changed lines for eyeballing.

Special characters are given as \\u escapes so this .py source stays pure ASCII.

Usage:
    python patch_ablation_reconcile.py [path-to-md]
Default path is the Windows repo location.
"""
import sys
import os
import shutil
import datetime

DEFAULT = r"C:\Projects\genomic-variant-classifier\docs\ABLATION_n_pathogenic_in_gene.md"

EN = "\u2013"   # en dash (CI ranges in the original/top region)
MINUS = "\u2212"  # minus sign (deltas in the original/top region)
EM = "\u2014"   # em dash

# (label, old, new) -- all single-line, all unique within the file.
EDITS = [
    ("E1 line8 de-stale 'untested'",
     "remains untested " + EM + " optional `no_gene_level` follow-up below.",
     "was subsequently tested too (the `no_gene_level` third arm below, 2026-06-07): "
     "likewise not load-bearing."),

    ("E2 line36 AUROC points+CIs -> eval_report 5dp",
     "| Test AUROC | 0.99820 [0.9980" + EN + "0.9984] | 0.99800 [0.9978" + EN + "0.9982] | " + MINUS + "0.0002 |",
     "| Test AUROC | 0.99817 [0.99797" + EN + "0.99836] | 0.99802 [0.99782" + EN + "0.99823] | " + MINUS + "0.0002 |"),

    ("E3 line70 caveat1 de-stale",
     "`gene_is_constrained`) tests this " + EM + " optional, since the",
     "`gene_is_constrained`) was tested in the third arm below (2026-06-07); even so, the"),

    ("E4 line83 optional-confirmation bullet -> DONE",
     "- **`no_gene_level`** (one-liner, harness already supports it):",
     "- **`no_gene_level`** (DONE 2026-06-07; see the Update section below " + EM +
     " was a one-liner the harness already supported):"),

    ("E5a line90 Artifacts add no_gene_level dir",
     "`outputs/ablation_run15/full/` and `outputs/ablation_run15/no_gene_prevalence/`",
     "`outputs/ablation_run15/full/`, `outputs/ablation_run15/no_gene_prevalence/`, "
     "and `outputs/ablation_run15/no_gene_level/`"),

    ("E5b line92 Artifacts 2 rows -> 3 rows",
     "models/); aggregator `outputs/ablation_run15/ablation_results.parquet` (2 rows).",
     "models/); aggregator `outputs/ablation_run15/ablation_results.parquet` (3 rows)."),

    ("E6a line95 datetime note past-tense",
     "`variant_ensemble.py:1344` uses deprecated `datetime.utcnow()` (DeprecationWarning) " + EM,
     "`variant_ensemble.py` previously used deprecated `datetime.utcnow()` (DeprecationWarning);"),

    ("E6b line96 datetime note RESOLVED",
     "switch to `datetime.now(datetime.UTC)`. Non-breaking; fix on next touch of that file.",
     "RESOLVED 2026-06-07 " + EM + " now `datetime.now(timezone.utc)` "
     "(added `from datetime import timezone`); full suite green."),

    ("E7h line110 add CI to header",
     "| Arm | Description | Test AUROC | Delta vs full |",
     "| Arm | Description | Test AUROC [95% CI] | Delta vs full |"),

    ("E7a line112 full -> 5dp + CI",
     "| `full` | all features | 0.9982 | -- |",
     "| `full` | all features | 0.99817 [0.99797-0.99836] | -- |"),

    ("E7b line113 prev -> 5dp + CI",
     "| `no_gene_prevalence` | zero `n_pathogenic_in_gene` only | 0.9980 | -0.0002 |",
     "| `no_gene_prevalence` | zero `n_pathogenic_in_gene` only | 0.99802 [0.99782-0.99823] | -0.0002 |"),

    ("E7c line114 level -> 5dp + CI + delta -0.0004 -> -0.0003",
     "| `no_gene_level` | zero all 4 gene-level features (incl. `gene_has_known_disease`) | 0.9978 | -0.0004 |",
     "| `no_gene_level` | zero all 4 gene-level features (incl. `gene_has_known_disease`) | 0.99783 [0.99761-0.99804] | -0.0003 |"),

    ("E8 line128 verdict 0.0004 -> 0.0003",
     "Removing the entire gene-level channel costs 0.0004 AUROC overall, and even the most affected subgroup",
     "Removing the entire gene-level channel costs 0.0003 AUROC overall (0.00034 at full precision), "
     "and even the most affected subgroup"),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        print("ABORT: file not found:", path)
        return 2

    raw = open(path, "rb").read()
    if raw[:3] == b"\xef\xbb\xbf":
        print("NOTE: file has a UTF-8 BOM; it will be preserved.")
        bom, body = raw[:3], raw[3:]
    else:
        bom, body = b"", raw
    text = body.decode("utf-8")

    # PRE-CHECK
    counts = [(label, text.count(old), old, new) for (label, old, new) in EDITS]
    bad = [(label, c) for (label, c, _, _) in counts if c != 1]
    if bad:
        already = all(text.count(old) == 0 and new in text for (_, old, new) in EDITS)
        if already:
            print("IDEMPOTENT NO-OP: every edit already applied; nothing to do.")
            return 0
        print("PRE-CHECK FAILED (no changes written). Anchors not matching exactly once:")
        for label, c in bad:
            print("  - {:<45} found {}".format(label, c))
        return 3

    # backup
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path + ".bak_" + stamp
    shutil.copy2(path, bak)
    print("backup written:", bak)

    # apply + record changed lines for the mini-diff
    before_lines = text.splitlines()
    for (label, old, new) in EDITS:
        text = text.replace(old, new, 1)
        print("applied:", label)
    after_lines = text.splitlines()

    # write back (bytes; preserves EOL of the body, re-attaches BOM if present)
    open(path, "wb").write(bom + text.encode("utf-8"))

    # POST-CHECK
    verify = open(path, "rb").read()
    if verify[:3] == b"\xef\xbb\xbf":
        verify = verify[3:]
    vt = verify.decode("utf-8")
    fails = []
    for (label, old, new) in EDITS:
        if old in vt:
            fails.append(label + " : OLD still present")
        if new not in vt:
            fails.append(label + " : NEW missing")
    if len(after_lines) != len(before_lines):
        fails.append("LINE COUNT changed {} -> {} (edits should be in-line)".format(
            len(before_lines), len(after_lines)))

    print("\nPOST-CHECK:", "PASS" if not fails else "FAIL")
    for f in fails:
        print("  -", f)

    # mini-diff: show lines that differ
    print("\nchanged lines:")
    for i, (b, a) in enumerate(zip(before_lines, after_lines), 1):
        if b != a:
            print("  {:>3} - {}".format(i, b))
            print("  {:>3} + {}".format(i, a))

    return 0 if not fails else 4


if __name__ == "__main__":
    sys.exit(main())
