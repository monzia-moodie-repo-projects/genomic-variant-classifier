#!/usr/bin/env python3
r"""patch_test_omim_genemap2_reconcile.py

Reconcile tests/unit/test_omim.py to the genemap2-driven OMIM contract (commit 3375965).
The connector is PROVEN correct (independent hand-count BB1; 15 new-contract tests pass BB2);
these are STALE tests. Six edits:

  1. Remove dead _GENEMAP2_CONTENT constant (lines ~47-55) + dead _write_genemap2 #1 (~56-61)
     -- referenced ONLY by each other (FF3); the ACTIVE _write_genemap2 is the later def.
  2. Remove dead _write_genemap2 #2 (~253-298) + its "Final corrected..." comment
     -- shadowed; Python used the last def. Leaves exactly ONE _write_genemap2 (the active one).
  3. test_annotate_dataframe_known_gene -> genemap2_path; TP53 omim_n_diseases == 1 (decision a).
  4. test_fetch_round_trip -> genemap2_path; BRCA1 omim_n_diseases == 1 (fetch routes via genemap2, DD3).
  5. test_parse_genemap2_autosomal_dominant_flags -> call the REAL method _parse_genemap2 (the
     _parse_genemap2_autosomal_dominant method NEVER EXISTED -- AttributeError, EE1/EE2); assert the
     real 4-col schema + verified values (n_diseases/molecular/AD all = the FF2-repro values).
  6. test_annotate_dataframe_uses_genemap2_autosomal_dominant -> TP53 omim_n_diseases 2 -> 1
     (active fixture has 1 TP53 phenotype entry).

Connector + launch are UNTOUCHED (both proven correct). ANCHOR-BASED, IDEMPOTENT, LF.
"""
from __future__ import annotations
import argparse, ast
from pathlib import Path

TARGET = Path("tests/unit/test_omim.py")
MARKER = "# genemap2-reconciled"  # idempotency sentinel (added to module docstring area)

# ---- Edit 3: known_gene -> genemap2 (exact anchor from CC3 155-160) ----
KNOWN_OLD = '''def test_annotate_dataframe_known_gene(tmp_path):
    path = _write_mim2gene(tmp_path)
    connector = OMIMConnector(mim2gene_path=path)
    df = _minimal_variant_df(gene_symbol="TP53")
    result = connector.annotate_dataframe(df)
    assert result["omim_n_diseases"].iloc[0] == 2'''
KNOWN_NEW = '''def test_annotate_dataframe_known_gene(tmp_path):
    # genemap2 is the disease-count source (mim2gene is NOT, post-3375965).
    path = _write_genemap2(tmp_path)
    connector = OMIMConnector(genemap2_path=path)
    df = _minimal_variant_df(gene_symbol="TP53")
    result = connector.annotate_dataframe(df)
    # TP53 has 1 phenotype entry in the genemap2 fixture.
    assert result["omim_n_diseases"].iloc[0] == 1'''

# ---- Edit 4: fetch_round_trip -> genemap2 (exact anchor from CC3 175-181) ----
FETCH_OLD = '''def test_fetch_round_trip(tmp_path):
    path = _write_mim2gene(tmp_path)
    connector = OMIMConnector(mim2gene_path=path)
    df = _minimal_variant_df(gene_symbol="BRCA1")
    result = connector.fetch(variant_df=df)
    assert "omim_n_diseases" in result.columns
    assert result["omim_n_diseases"].iloc[0] == 1'''
FETCH_NEW = '''def test_fetch_round_trip(tmp_path):
    # fetch() routes through genemap2 (same source as annotate_dataframe).
    path = _write_genemap2(tmp_path)
    connector = OMIMConnector(genemap2_path=path)
    df = _minimal_variant_df(gene_symbol="BRCA1")
    result = connector.fetch(variant_df=df)
    assert "omim_n_diseases" in result.columns
    # BRCA1 has 1 phenotype entry in the genemap2 fixture.
    assert result["omim_n_diseases"].iloc[0] == 1'''

# ---- Edit 5: rewrite the AttributeError test to the REAL method (exact anchor DD1 213-226) ----
AD_OLD = '''def test_parse_genemap2_autosomal_dominant_flags(tmp_path):
    path = _write_genemap2(tmp_path)
    connector = OMIMConnector()
    gene_table = connector._parse_genemap2_autosomal_dominant(path)

    assert set(gene_table.columns) == {"gene_symbol", "omim_is_autosomal_dominant"}

    tp53 = gene_table[gene_table["gene_symbol"] == "TP53"]
    brca1 = gene_table[gene_table["gene_symbol"] == "BRCA1"]
    brca2 = gene_table[gene_table["gene_symbol"] == "BRCA2"]

    assert tp53["omim_is_autosomal_dominant"].iloc[0] == 1
    assert brca1["omim_is_autosomal_dominant"].iloc[0] == 0
    assert brca2["omim_is_autosomal_dominant"].iloc[0] == 1'''
AD_NEW = '''def test_parse_genemap2_autosomal_dominant_flags(tmp_path):
    # _parse_genemap2 is the real method (the old _parse_genemap2_autosomal_dominant never existed).
    # It returns the full 4-column gene-level table.
    path = _write_genemap2(tmp_path)
    connector = OMIMConnector()
    gene_table = connector._parse_genemap2(path)

    assert set(gene_table.columns) == {
        "gene_symbol",
        "omim_n_diseases",
        "omim_n_diseases_molecular",
        "omim_is_autosomal_dominant",
    }

    tp53 = gene_table[gene_table["gene_symbol"] == "TP53"]
    brca1 = gene_table[gene_table["gene_symbol"] == "BRCA1"]
    brca2 = gene_table[gene_table["gene_symbol"] == "BRCA2"]

    # AD flag: TP53 dominant, BRCA1 recessive, BRCA2 dominant (per fixture Phenotypes strings).
    assert tp53["omim_is_autosomal_dominant"].iloc[0] == 1
    assert brca1["omim_is_autosomal_dominant"].iloc[0] == 0
    assert brca2["omim_is_autosomal_dominant"].iloc[0] == 1

    # Each fixture gene has exactly 1 phenotype entry, all molecular "(3)".
    assert tp53["omim_n_diseases"].iloc[0] == 1
    assert tp53["omim_n_diseases_molecular"].iloc[0] == 1
    assert brca1["omim_n_diseases"].iloc[0] == 1
    assert brca2["omim_n_diseases"].iloc[0] == 1'''

# ---- Edit 6: TP53 count 2 -> 1 in the big genemap2 annotate test (exact anchor DD1 238) ----
BIG_OLD = '''    assert result.loc[result["gene_symbol"] == "TP53", "omim_n_diseases"].iloc[0] == 2
    assert result.loc[result["gene_symbol"] == "TP53", "omim_is_autosomal_dominant"].iloc[0] == 1'''
BIG_NEW = '''    assert result.loc[result["gene_symbol"] == "TP53", "omim_n_diseases"].iloc[0] == 1
    assert result.loc[result["gene_symbol"] == "TP53", "omim_is_autosomal_dominant"].iloc[0] == 1'''

EDITS = [
    ("known_gene->genemap2", KNOWN_OLD, KNOWN_NEW),
    ("fetch->genemap2", FETCH_OLD, FETCH_NEW),
    ("AD-flags->real-method", AD_OLD, AD_NEW),
    ("big-test TP53 2->1", BIG_OLD, BIG_NEW),
]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")
    if MARKER in src:
        print("OK (idempotent): test_omim.py already reconciled."); return 0

    # Validate the 4 string-anchored edits occur exactly once.
    problems = []
    for name, old, _new in EDITS:
        c = src.count(old)
        if c != 1:
            problems.append(f"  {name}: anchor occurs {c}x (need 1)")
    # Validate the dead-def deletions are findable: exactly 3 _write_genemap2 defs present pre-patch.
    nwg = src.count("def _write_genemap2(")
    if nwg != 3:
        problems.append(f"  dead-def deletion: expected 3 _write_genemap2 defs, found {nwg}")
    if problems:
        print("FAIL: anchor validation:\n" + "\n".join(problems)); return 3
    if ns.check:
        print("CHECK: all 4 string anchors found once; 3 _write_genemap2 defs present."); print("RESULT: PASS (check)"); return 0

    backup = TARGET.with_suffix(".py.pre_genemap2_reconcile.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline=""); print(f"OK: backup -> {backup}")

    new = src
    # Apply the 4 string edits first.
    for name, old, repl in EDITS:
        new = new.replace(old, repl, 1)

    # Dead-def removal via AST: parse, find the FIRST TWO _write_genemap2 FunctionDefs (by line order),
    # plus the _GENEMAP2_CONTENT assignment, and delete their source line spans. Keep the THIRD def.
    tree = ast.parse(new)
    lines = new.splitlines(keepends=True)
    spans = []  # (start_lineno, end_lineno) 1-indexed inclusive, to delete

    wg_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_write_genemap2"]
    # delete all but the LAST (active) def
    for n in wg_defs[:-1]:
        spans.append((n.lineno, n.end_lineno))
    # delete _GENEMAP2_CONTENT assignment (module-level)
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == "_GENEMAP2_CONTENT":
                    spans.append((n.lineno, n.end_lineno))

    # Sort spans descending so deletions don't shift earlier line numbers.
    spans.sort(reverse=True)
    for start, end in spans:
        del lines[start-1:end]
        # also drop a trailing blank line or comment immediately after, if it's the dead "Final corrected" note
    new2 = "".join(lines)

    # Drop the orphan "# Final corrected tab-delimited genemap2 fixture override." comment if now dangling.
    new2 = new2.replace("# Final corrected tab-delimited genemap2 fixture override.\n", "")

    # Add idempotency marker near the top (after first docstring line set) -- append a comment after imports.
    # Simplest: add marker comment right before the first 'def _write' that remains.
    new2 = new2.replace("def _write_mim2gene(", MARKER + "\ndef _write_mim2gene(", 1)

    TARGET.write_text(new2, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = {
        "exactly 1 _write_genemap2 def": after.count("def _write_genemap2(") == 1,
        "_GENEMAP2_CONTENT removed": "_GENEMAP2_CONTENT" not in after,
        "_MIM2GENE_CONTENT kept": "_MIM2GENE_CONTENT" in after,
        "known_gene uses genemap2_path": "path = _write_genemap2(tmp_path)\n    connector = OMIMConnector(genemap2_path=path)\n    df = _minimal_variant_df(gene_symbol=\"TP53\")" in after,
        "AD test calls real _parse_genemap2": "connector._parse_genemap2(path)" in after,
        "no CALL to nonexistent method": "connector._parse_genemap2_autosomal_dominant(" not in after,
        "4-col schema asserted": '"omim_n_diseases_molecular",' in after,
        "marker present": MARKER in after,
    }
    try:
        ast.parse(after); checks["compiles"] = True
    except SyntaxError as e:
        checks["compiles"] = False; print("  SYNTAX ERROR:", e)
    for k, v in checks.items():
        print(f"  {'OK' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 5


if __name__ == "__main__":
    raise SystemExit(main())
