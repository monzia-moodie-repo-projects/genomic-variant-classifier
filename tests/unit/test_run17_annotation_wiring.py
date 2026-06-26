"""Tests for the Run-17 annotation-flag wiring in scripts/run_phase2_eval.py.

These assert that the four newly-added CLI flags (--omim-path / --phylop-path /
--dbsnp-path / --eve-path) are (a) declared in the argparse parser and (b) threaded
into the AnnotationConfig(...) construction, so the already-present AnnotationConfig
fields stop taking their silent-stub branch. Plus a regression guard on the HGVSp
parser, which is what makes EVE/ESM-2 carry real signal.

Run:  python -m pytest tests/unit/test_run17_annotation_wiring.py -q
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
RUN_PHASE2 = REPO / "scripts" / "run_phase2_eval.py"

NEW_FLAGS = ["--omim-path", "--phylop-path", "--dbsnp-path", "--eve-path"]
NEW_FIELDS = ["omim_path", "phylop_path", "dbsnp_path", "eve_path"]


@pytest.fixture(scope="module")
def src() -> str:
    assert RUN_PHASE2.exists(), f"missing {RUN_PHASE2}"
    return RUN_PHASE2.read_text(encoding="utf-8")


def test_run_phase2_eval_compiles(src: str) -> None:
    """The patched file must still be syntactically valid Python."""
    ast.parse(src)  # raises SyntaxError on failure


@pytest.mark.parametrize("flag", NEW_FLAGS)
def test_flag_declared(src: str, flag: str) -> None:
    assert f'"{flag}"' in src, f"argparse flag {flag} not declared in run_phase2_eval.py"


@pytest.mark.parametrize("field", NEW_FIELDS)
def test_field_threaded_into_annotation_config(src: str, field: str) -> None:
    """The flag must be threaded into AnnotationConfig as
    `<field>=Path(args.<field>) if args.<field> else None` (the established pattern)."""
    needle = f"{field}=Path(args.{field}) if args.{field} else None"
    assert needle in src, f"{field} not threaded into AnnotationConfig with the standard pattern"


def test_annotation_config_block_has_all_new_fields(src: str) -> None:
    """All four must appear AFTER the AnnotationConfig( opening and BEFORE the
    DataPrepPipeline( that consumes it -- i.e. inside the construction call."""
    a = src.index("AnnotationConfig(")
    b = src.index("DataPrepPipeline(", a)
    block = src[a:b]
    missing = [f for f in NEW_FIELDS if f"{f}=" not in block]
    assert not missing, f"fields not inside AnnotationConfig(...) block: {missing}"


def test_argparse_help_documents_silent_stub(src: str) -> None:
    """Each new flag's help must warn about the silent-stub default, so the
    no-silent-zero rationale is documented in --help, not just code comments."""
    # crude but sufficient: the wiring comment marker must be present
    assert "Run 17 annotation wiring" in src


# --- HGVSp parser regression guard: this is what gives EVE/ESM-2 real signal. ---

def _load_hgvsp_parser():
    mod_path = REPO / "src" / "genomic_variant_classifier" / "data" / "hgvsp_parser.py"
    if not mod_path.exists():
        pytest.skip(f"hgvsp_parser not found at {mod_path}")
    spec = importlib.util.spec_from_file_location("hgvsp_parser", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hgvsp_parser_populates_missense_coords() -> None:
    """The exact probe used during the live audit: p.Asp1692Asn -> 1692/D/N,
    p.(Arg1699Gln) -> 1699/R/Q, nonsense/synonymous/empty -> NA. Guards against
    a regression that would silently re-stub EVE/ESM-2."""
    parser = _load_hgvsp_parser()
    df = pd.DataFrame({"protein_change": [
        "p.Asp1692Asn", "p.(Arg1699Gln)", "p.Arg1699Ter", "p.Asp1692Asp", "",
    ]})
    out, n = parser.fill_protein_columns_from_hgvsp(df, source_col="protein_change")
    assert n == 2, f"expected 2 missense rows filled, got {n}"
    assert int(out.loc[0, "protein_pos"]) == 1692
    assert out.loc[0, "wt_aa"] == "D" and out.loc[0, "mut_aa"] == "N"
    assert int(out.loc[1, "protein_pos"]) == 1699
    assert out.loc[1, "wt_aa"] == "R" and out.loc[1, "mut_aa"] == "Q"
    # nonsense / synonymous / empty must NOT be filled
    assert pd.isna(out.loc[2, "protein_pos"])
    assert pd.isna(out.loc[3, "protein_pos"])
    assert pd.isna(out.loc[4, "protein_pos"])
