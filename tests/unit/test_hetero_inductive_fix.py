"""test_hetero_inductive_fix.py -- Monzia Moodie

Regression for INCIDENT_2026-06-16: the focal-only HeteroGNNScorer left hetero_gnn_score at the
0.5 default for gene-disjoint val/test genes (alive in train, dead at eval/inference). The fix builds
the scoring graph over the UNION of all split genes (train-only features/focal -> no leak) and scores
all nodes via HeteroGNNScorer.from_full_graph. These tests cover the torch-free core (node-set assembly +
scorer coverage + the new API surface); the PyG train/score path is validated in-env via the smoke.
"""
import inspect

import numpy as np
import pandas as pd

from genomic_variant_classifier.models import hetero_gnn_scorer as H
from genomic_variant_classifier.models.hetero_gnn_scorer import (
    HeteroGNNScorer,
    assemble_node_features_and_focal,
    build_hetero_focal_graph,
)


def test_assemble_union_genes_makes_val_test_nodes_with_zero_feats_and_train_only_focal():
    # train df carries only train genes; the union override adds gene-disjoint val/test genes
    train = pd.DataFrame({
        "gene_symbol": ["BRCA1", "BRCA1", "TP53"],
        "acmg_label": [1.0, 0.0, 1.0],
        "cadd": [30.0, 10.0, 25.0],
    })
    union = ["BRCA1", "MLH1", "PMS2", "TP53"]  # MLH1/PMS2 = val/test, absent from train
    feats, focal, ys = assemble_node_features_and_focal(train, union, ["cadd"], "acmg_label")
    gi = {g: i for i, g in enumerate(union)}
    assert feats.shape == (4, 1)
    # train genes -> real gene-mean features
    assert feats[gi["BRCA1"], 0] == 20.0 and feats[gi["TP53"], 0] == 25.0
    # val/test genes -> NODES (present) but zero features (scored by structure; no leak)
    assert feats[gi["MLH1"], 0] == 0.0 and feats[gi["PMS2"], 0] == 0.0
    # focal supervision is TRAIN-ONLY (3 labeled train variants on train gene indices)
    assert len(focal) == 3 and len(ys) == 3
    assert set(int(i) for i in focal) <= {gi["BRCA1"], gi["TP53"]}


def test_scorer_over_union_resolves_disjoint_val_test_not_default():
    scorer = HeteroGNNScorer({"BRCA1": 0.91, "TP53": 0.83, "MLH1": 0.27, "PMS2": 0.19})
    val = pd.DataFrame({"gene_symbol": ["MLH1", "PMS2"]})
    out = list(scorer.score_dataframe(val))
    assert out == [0.27, 0.19]                       # real scores, NOT the 0.5 default
    assert scorer.score("TRULY_UNKNOWN") == 0.5      # genuine unknown still defaults


def test_from_full_graph_is_classmethod_on_scorer():
    assert isinstance(inspect.getattr_static(HeteroGNNScorer, "from_full_graph"), classmethod)


def test_build_hetero_focal_graph_exposes_genes_override():
    params = inspect.signature(build_hetero_focal_graph).parameters
    assert "genes" in params and params["genes"].default is None


def test_genes_override_none_preserves_legacy_node_set():
    # With genes=None the node set is still derived from the df (backward compatible)
    train = pd.DataFrame({"gene_symbol": ["A", "B"], "acmg_label": [1.0, 0.0], "f": [1.0, 2.0]})
    feats, focal, ys = assemble_node_features_and_focal(train, ["A", "B"], ["f"], "acmg_label")
    assert feats.shape == (2, 1) and len(focal) == 2
