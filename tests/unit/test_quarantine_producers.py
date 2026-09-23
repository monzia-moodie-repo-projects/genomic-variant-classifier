"""Containment of the two quarantined structural producers and of step 14 (2026-09-22).

Basis: quarantine_policy.py; docs/CONTAINMENT_2026-07-24.md section 4; the September 2026 ruling
that BOTH producers are blocked. Each refusal test also proves the refusal touched nothing: the
side effects a producer would perform are replaced by a function that fails the test if reached.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from genomic_variant_classifier.containment import ContainmentError
from genomic_variant_classifier.data import alphafold as af
from genomic_variant_classifier.data import database_connectors as dbc
from genomic_variant_classifier.data import real_data_prep as rdp
from genomic_variant_classifier.pipelines import protein_pipeline as pp
from genomic_variant_classifier.quarantine_policy import (
    BLOCKED_PRODUCERS, QUARANTINED_FEATURES, STRUCTURAL_CONFIG_FIELDS,
)


def _forbidden(*args, **kwargs):
    raise AssertionError("a quarantined producer reached a side effect before refusing")


_MISSENSE = pd.DataFrame({"gene_symbol": ["TP53"], "protein_change": ["p.Arg175His"],
                          "is_missense": [1], "protein_pos": [175], "wt_aa": ["R"]})


# ---------------------------------------------------------------------------
# Policy coherence
# ---------------------------------------------------------------------------
def test_policy_lists_are_unique_nonempty_strings():
    for values in (QUARANTINED_FEATURES, BLOCKED_PRODUCERS, STRUCTURAL_CONFIG_FIELDS):
        assert values and all(isinstance(v, str) and v for v in values)
        assert len(values) == len(set(values))


def test_each_producer_identity_is_blocked():
    # Guards renaming drift: a producer whose constant stopped matching the policy would run.
    assert pp._PRODUCER_ID in BLOCKED_PRODUCERS
    assert af._PRODUCER_ID in BLOCKED_PRODUCERS


def test_structural_config_fields_are_real_annotation_config_fields():
    # If a field were renamed, getattr(..., None) would silently check nothing.
    fields = {f.name for f in dataclasses.fields(rdp.AnnotationConfig)}
    assert set(STRUCTURAL_CONFIG_FIELDS) <= fields


# ---------------------------------------------------------------------------
# ProteinStructurePipeline
# ---------------------------------------------------------------------------
def test_protein_pipeline_constructor_refuses_before_any_side_effect(tmp_path):
    target = tmp_path / "alphafold_cache"
    with patch.object(Path, "mkdir", _forbidden), \
         patch.object(pp, "_UniProtMapper", _forbidden), \
         patch.object(pp.requests, "get", _forbidden):
        with pytest.raises(ContainmentError):
            pp.ProteinStructurePipeline(cache_dir=target)
    assert not target.exists()


def test_protein_pipeline_default_constructor_refuses_before_any_side_effect():
    with patch.object(Path, "mkdir", _forbidden), \
         patch.object(pp, "_UniProtMapper", _forbidden):
        with pytest.raises(ContainmentError):
            pp.ProteinStructurePipeline()


def test_protein_pipeline_instance_without_init_refuses():
    instance = object.__new__(pp.ProteinStructurePipeline)
    with patch.object(pp.requests, "get", _forbidden):
        with pytest.raises(ContainmentError):
            instance.annotate_dataframe(_MISSENSE.copy())


def test_get_alphafold_features_wrapper_refuses():
    with patch.object(Path, "mkdir", _forbidden), patch.object(pp.requests, "get", _forbidden):
        with pytest.raises(ContainmentError):
            pp.get_alphafold_features("P04637", 175)


# ---------------------------------------------------------------------------
# AlphaFoldConnector
# ---------------------------------------------------------------------------
def test_alphafold_connector_constructor_refuses_before_session(tmp_path):
    with patch.object(dbc.requests, "Session", _forbidden):
        with pytest.raises(ContainmentError):
            af.AlphaFoldConnector(parquet_path=tmp_path / "af.parquet",
                                  uniprot_index_path=tmp_path / "up.parquet")


@pytest.mark.parametrize("method", ["annotate_dataframe", "fetch"])
def test_alphafold_connector_instance_without_init_refuses(method):
    instance = object.__new__(af.AlphaFoldConnector)
    with patch.object(pd, "read_parquet", _forbidden):
        with pytest.raises(ContainmentError):
            getattr(instance, method)(_MISSENSE.copy())


# ---------------------------------------------------------------------------
# real_data_prep: configuration refusal and step 14
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field", STRUCTURAL_CONFIG_FIELDS)
def test_data_prep_refuses_a_config_requesting_structural_features(tmp_path, field):
    ac = rdp.AnnotationConfig(**{field: tmp_path / "requested"})
    with pytest.raises(ContainmentError, match=field):
        rdp.DataPrepPipeline(config=rdp.DataPrepConfig(output_dir=tmp_path / "splits"),
                             annotation_config=ac)


def test_data_prep_default_config_constructs(tmp_path):
    rdp.DataPrepPipeline(config=rdp.DataPrepConfig(output_dir=tmp_path / "splits"))


def test_config_changed_after_construction_is_still_refused(tmp_path):
    pipeline = rdp.DataPrepPipeline(config=rdp.DataPrepConfig(output_dir=tmp_path / "splits"))
    pipeline.annotation_config.alphafold_path = tmp_path / "late"
    with pytest.raises(ContainmentError):
        rdp._refuse_structural_config(pipeline.annotation_config)


def _annotate_scores_tree() -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(rdp.DataPrepPipeline._annotate_scores))
    return ast.parse(source).body[0]


def test_step_14_constructs_neither_producer():
    names = {n.id for n in ast.walk(_annotate_scores_tree()) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(_annotate_scores_tree()) if isinstance(n, ast.Attribute)}
    assert not (names & set(BLOCKED_PRODUCERS))


def test_step_14_calls_the_refusal_helper():
    calls = [n for n in ast.walk(_annotate_scores_tree())
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_refuse_structural_config"]
    assert calls


def test_real_data_prep_imports_neither_producer_module():
    tree = ast.parse(inspect.getsource(rdp))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert "genomic_variant_classifier.pipelines.protein_pipeline" not in modules
    assert "genomic_variant_classifier.data.alphafold" not in modules
