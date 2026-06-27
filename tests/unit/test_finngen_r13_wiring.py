"""Stage 4 wiring test: FinnGen R13 dual-release path reaches the pipeline.

Validates (without the ~30GB file, and without importing heavy deps) that:
  - AnnotationConfig exposes finngen_r13_path (default None)
  - run_phase2_eval.py declares --finngen-r13-path AND routes args.finngen_r13_path
    into AnnotationConfig (source-level check; no execution of the heavy module)
  - the annotation block runs an independent R13 connector pass (column_prefix="r13_")
    producing finngen_r13_* columns, and its else-branch default-fills the PREFIXED
    names (not the unprefixed R12 FINNGEN_COLUMNS).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

_RPE = Path("scripts/run_phase2_eval.py")


def test_annotation_config_has_finngen_r13_path():
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig
    cfg = AnnotationConfig(finngen_r13_path="/some/r13.gz")
    assert cfg.finngen_r13_path == "/some/r13.gz"
    assert AnnotationConfig().finngen_r13_path is None


def test_run_phase2_eval_declares_and_routes_finngen_r13():
    # Source-level: the flag is declared AND the kwarg is wired into AnnotationConfig.
    # (Avoids importing run_phase2_eval, which pulls torch/xgboost/etc.)
    src = _RPE.read_text(encoding="utf-8")
    assert '"--finngen-r13-path"' in src, "--finngen-r13-path flag not declared"
    assert "finngen_r13_path=Path(args.finngen_r13_path)" in src, (
        "args.finngen_r13_path not routed into AnnotationConfig"
    )


def test_r13_else_branch_fills_prefixed_names_not_unprefixed():
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig, Pipeline
    p = Pipeline(AnnotationConfig(finngen_r13_path=None))
    out = p._annotate_finngen(pd.DataFrame({"variant_id": ["v0", "v1"]}))
    for c in ("finngen_r13_af_fin", "finngen_r13_af_nfsee", "finngen_r13_enrichment"):
        assert c in out.columns, f"{c} missing from R13 else-branch default-fill"
    assert (out["finngen_r13_af_fin"] == 0.0).all()
    assert (out["finngen_r13_enrichment"] == 1.0).all()


def test_r13_connector_pass_annotates_prefixed_columns():
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig, Pipeline
    p = Pipeline(AnnotationConfig(finngen_r13_path="/fake/r13.gz"))
    out = p._annotate_finngen(pd.DataFrame({"variant_id": ["v0", "v1"]}))
    assert {"finngen_r13_af_fin", "finngen_r13_af_nfsee", "finngen_r13_enrichment"} <= set(out.columns)
