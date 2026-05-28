"""Smoke test for run_phase2_eval.py --unseen-gene-holdout flag (PM11b).

Verifies that:
1. The flag is registered in argparse.
2. Default is False when omitted.
3. action='store_true' rejects values.

Author: PM11b (2026-05-27)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def parse_args_fn():
    """Load scripts/run_phase2_eval.py and return its parse_args function."""
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "run_phase2_eval.py"
    spec = importlib.util.spec_from_file_location(
        "_test_run_phase2_eval_pm11b", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.parse_args


def test_unseen_gene_holdout_flag_present(parse_args_fn):
    """REGRESSION: --unseen-gene-holdout flag is registered."""
    args = parse_args_fn([
        "--clinvar", "ignored.parquet",
        "--unseen-gene-holdout",
    ])
    assert getattr(args, "unseen_gene_holdout", None) is True


def test_unseen_gene_holdout_flag_defaults_false(parse_args_fn):
    """REGRESSION: --unseen-gene-holdout defaults to False when omitted."""
    args = parse_args_fn([
        "--clinvar", "ignored.parquet",
    ])
    assert getattr(args, "unseen_gene_holdout", None) is False


def test_unseen_gene_holdout_flag_rejects_value(parse_args_fn):
    """action='store_true' rejects values; argparse calls sys.exit on error."""
    with pytest.raises(SystemExit):
        parse_args_fn([
            "--clinvar", "ignored.parquet",
            "--unseen-gene-holdout=somevalue",
        ])
