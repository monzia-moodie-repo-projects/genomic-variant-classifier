"""Stage 4 wiring test: FinnGen R13 dual-release path reaches the pipeline.

The R13 annotation logic lives inline at the tail of a large DataPrepPipeline
annotation method that first parses gnomAD loci and builds join keys, so it cannot
be invoked in isolation on a bare frame. These tests therefore validate the wiring
at the source level (like the run_phase2_eval check) plus the one behavioral path
that IS isolable (the AnnotationConfig field). Together they guard:
  - AnnotationConfig exposes finngen_r13_path (default None) -- behavioral
  - run_phase2_eval declares --finngen-r13-path and routes it into AnnotationConfig
  - real_data_prep runs an independent R13 connector pass (column_prefix="r13_")
    and its else-branch default-fills the PREFIXED names via finngen_columns("r13_")
    (NOT the unprefixed FINNGEN_COLUMNS) with finngen_r13_enrichment = 1.0.
"""
from __future__ import annotations

from pathlib import Path

_RDP = Path("src/genomic_variant_classifier/data/real_data_prep.py")
_RPE = Path("scripts/run_phase2_eval.py")


def test_annotation_config_has_finngen_r13_path():
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig
    cfg = AnnotationConfig(finngen_r13_path="/some/r13.gz")
    assert cfg.finngen_r13_path == "/some/r13.gz"
    assert AnnotationConfig().finngen_r13_path is None


def test_run_phase2_eval_declares_and_routes_finngen_r13():
    src = _RPE.read_text(encoding="utf-8")
    assert '"--finngen-r13-path"' in src, "--finngen-r13-path flag not declared"
    assert "finngen_r13_path=Path(args.finngen_r13_path)" in src, (
        "args.finngen_r13_path not routed into AnnotationConfig"
    )


def test_real_data_prep_runs_independent_r13_connector_pass():
    src = _RDP.read_text(encoding="utf-8")
    assert "if self.annotation_config.finngen_r13_path:" in src, (
        "R13 connector branch not gated on finngen_r13_path"
    )
    assert 'column_prefix="r13_"' in src, (
        "R13 connector not instantiated with column_prefix='r13_'"
    )


def test_real_data_prep_r13_else_fills_prefixed_names():
    src = _RDP.read_text(encoding="utf-8")
    assert 'finngen_columns("r13_")' in src, (
        "R13 else-branch must use finngen_columns('r13_'), not unprefixed FINNGEN_COLUMNS"
    )
    assert 'df["finngen_r13_enrichment"] = 1.0' in src, (
        "R13 enrichment must default to 1.0 in the else-branch"
    )