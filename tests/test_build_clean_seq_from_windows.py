"""tests/test_build_clean_seq_from_windows.py -- cover the repo-resident producer.

WHY THIS FILE EXISTS (2026-07-18)
---------------------------------
`scripts/build_clean_seq_from_windows.py` replaces `scripts/populate_fasta_seq.py` as the
producer of data/processed/clinvar_grch38_clean_seq.parquet. A retirement audit on
2026-07-18 established that the outgoing script was the SOLE repo-resident producer of an
artifact that 19 files consume, so retiring it without a tested replacement would leave a
534 MB artifact nothing in the repository could rebuild.

Coverage therefore exists BEFORE the old producer is deleted, not after.

WHAT IS PINNED HERE
-------------------
  * the join attaches windows AND provenance, and preserves cohort row count and order
  * expected counts are DERIVED from the inputs, never hardcoded -- a fixture whose
    numbers differ from the production cohort's 723/668/53/2 must still pass
  * the subset guard refuses rather than silently dropping rows
  * a failed post-check leaves the previous artifact untouched
  * --dry-run writes nothing
  * an existing backup is never clobbered
  * the module constructs NO placeholder literal, so it cannot become an offender in
    tests/unit/test_no_content_based_poly_detection.py
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "build_clean_seq_from_windows.py"

_SPEC = importlib.util.spec_from_file_location("build_clean_seq_from_windows", _SCRIPT)
bcs = importlib.util.module_from_spec(_SPEC)
sys.modules["build_clean_seq_from_windows"] = bcs
_SPEC.loader.exec_module(bcs)

WINDOW = 101


def _make(tmp_path, n=60, n_bad=9):
    """Cohort + windows fixture.

    The unusable counts here (9 = 5 + 3 + 1) deliberately do NOT match the production
    cohort's 723 = 668 + 53 + 2. If any expectation were hardcoded to the real numbers,
    every test below would fail.
    """
    proc = tmp_path / "data" / "processed"
    (proc / "seq_windows").mkdir(parents=True)
    ok = [True] * n
    reason = [""] * n
    for i in range(n_bad):
        ok[i] = False
        reason[i] = ("non_acgt_allele" if i < 5 else
                     "ref_mismatch" if i < 8 else "fetch_failed")
    good_ref = "ACGT" * 25 + "A"
    good_alt = "ACGT" * 25 + "G"
    bad = "N" * WINDOW
    win = pd.DataFrame({
        "chrom": ["1"] * n,
        "pos": list(range(1, n + 1)),
        "ref": ["A"] * n,
        "alt": ["G"] * n,
        "fasta_seq_ref": [good_ref if o else bad for o in ok],
        "fasta_seq_alt": [good_alt if o else bad for o in ok],
        "ok": ok,
        "reason": reason,
    })
    p_win = proc / "seq_windows" / "seq_windows.parquet"
    win.to_parquet(p_win, index=False)

    clean = win[["chrom", "pos", "ref", "alt"]].copy()
    clean["variant_id"] = ["v{}".format(i) for i in range(n)]
    p_clean = proc / "clinvar_grch38_clean.parquet"
    clean.to_parquet(p_clean, index=False)
    return p_clean, p_win, proc / "clinvar_grch38_clean_seq.parquet"


def _run(p_clean, p_win, p_out, *extra):
    return bcs.main(["--clean", str(p_clean), "--seq-windows", str(p_win),
                     "--out", str(p_out)] + list(extra))


def test_builds_and_attaches_provenance(tmp_path):
    c, w, o = _make(tmp_path)
    assert _run(c, w, o) == 0
    df = pd.read_parquet(o)
    assert len(df) == 60
    for col in ("fasta_seq_ref", "fasta_seq_alt", "ok", "reason"):
        assert col in df.columns
    assert "variant_id" in df.columns, "cohort columns must survive the join"
    assert int((~df["ok"].astype(bool)).sum()) == 9
    assert dict(df.loc[~df["ok"].astype(bool), "reason"].value_counts()) == {
        "non_acgt_allele": 5, "ref_mismatch": 3, "fetch_failed": 1}


def test_row_order_is_preserved(tmp_path):
    c, w, o = _make(tmp_path)
    assert _run(c, w, o) == 0
    assert list(pd.read_parquet(o)["variant_id"]) == list(pd.read_parquet(c)["variant_id"])


def test_expected_counts_are_derived_not_hardcoded(tmp_path):
    """Same code, different fixture -> the expectation must follow the input."""
    c, w, o = _make(tmp_path, n=40, n_bad=2)
    assert _run(c, w, o) == 0
    df = pd.read_parquet(o)
    assert len(df) == 40
    assert int((~df["ok"].astype(bool)).sum()) == 2


def test_refuses_when_cohort_is_not_a_subset(tmp_path):
    c, w, o = _make(tmp_path)
    d = pd.read_parquet(c)
    alien = d.head(3).copy()
    alien["pos"] = alien["pos"] + 10_000
    pd.concat([d, alien], ignore_index=True).to_parquet(c, index=False)
    assert _run(c, w, o) == 2
    assert not o.exists(), "nothing may be written when the join would drop rows"
    assert not o.with_suffix(".tmp.parquet").exists()


def test_missing_input_aborts(tmp_path):
    c, w, o = _make(tmp_path)
    assert _run(c.with_name("absent.parquet"), w, o) == 2
    assert _run(c, w.with_name("absent.parquet"), o) == 2
    assert not o.exists()


def test_dry_run_writes_nothing(tmp_path):
    c, w, o = _make(tmp_path)
    assert _run(c, w, o, "--dry-run") == 0
    assert not o.exists()
    assert not o.with_suffix(".tmp.parquet").exists()


def test_post_check_failure_leaves_previous_artifact_untouched(tmp_path):
    c, w, o = _make(tmp_path)
    assert _run(c, w, o) == 0
    before = o.read_bytes()

    win = pd.read_parquet(w)
    win.loc[20, "fasta_seq_ref"] = "ACGT"          # 4 characters, not 101
    win.to_parquet(w, index=False)

    assert _run(c, w, o, "--no-backup") == 1
    assert o.read_bytes() == before, "a failed post-check must not replace the artifact"
    assert o.with_suffix(".tmp.parquet").exists(), "candidate retained for inspection"


def test_existing_backup_is_never_clobbered(tmp_path):
    c, w, o = _make(tmp_path)
    assert _run(c, w, o) == 0
    bak = o.with_name(o.name + ".bak")
    bak.write_bytes(b"an older backup that must survive")
    assert _run(c, w, o) == 0
    assert bak.read_bytes() == b"an older backup that must survive"


def test_module_constructs_no_placeholder_literal():
    """Guards the design constraint permanently.

    tests/unit/test_no_content_based_poly_detection.py forbids live modules from building
    `"A" * n` or `"N" * n`, because content cannot distinguish a real homopolymer from a
    builder giving up. It would be absurd for the script that RESOLVES the two-builder
    defect to join that offender list, so the constraint is pinned here at its source.
    """
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
            continue
        for side in (node.left, node.right):
            if isinstance(side, ast.Constant) and isinstance(side.value, str) \
                    and side.value.upper() in {"A", "C", "G", "T", "N"}:
                offenders.append((node.lineno, side.value))
            name = getattr(side, "id", None) or getattr(side, "attr", None)
            if name in {"PAD_CHAR", "POLY", "PLACEHOLDER_BASE"}:
                offenders.append((node.lineno, name))
    assert not offenders, "placeholder literal constructed at {}".format(offenders)
