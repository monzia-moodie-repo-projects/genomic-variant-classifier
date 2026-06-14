"""test_hetero_gnn_scorer.py  --  Monzia Moodie

Heterogeneous GNN trainer + scorer. The torch-free assembly core runs anywhere;
the full build -> train -> score path runs under .venv312 (torch_geometric).
"""
import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.models.hetero_gnn_scorer import (
    assemble_node_features_and_focal,
)


def test_assemble_node_features_and_focal():
    df = pd.DataFrame({
        "gene_symbol": ["A", "A", "B", "C", "C", "Z"],
        "feat1":       [1.0, 3.0, 5.0, 7.0, np.nan, 9.0],
        "feat2":       [10., 10., 20., 30., 30., 99.],
        "acmg_label":  [1, 1, 0, 1, np.nan, 1],
    })
    feats, focal, y = assemble_node_features_and_focal(
        df, ["A", "B", "C"], ["feat1", "feat2"], "acmg_label")
    assert feats.shape == (3, 2) and feats.dtype == np.float32
    assert abs(feats[0, 0] - 2.0) < 1e-6        # A mean(1,3)
    assert abs(feats[2, 0] - 7.0) < 1e-6        # C mean ignores NaN
    assert focal.tolist() == [0, 0, 1, 2]       # Z + NaN-label dropped
    assert y.tolist() == [1.0, 1.0, 0.0, 1.0]
    feats2, _, _ = assemble_node_features_and_focal(
        df, ["A", "B", "C"], ["feat1", "NOPE"], "acmg_label")
    assert (feats2[:, 1] == 0).all()            # missing column -> 0


def test_build_train_score_pipeline():
    pytest.importorskip("torch_geometric")
    from genomic_variant_classifier.models.hetero_gnn_scorer import (
        HeteroGNNScorer, HeteroGNNTrainer, build_hetero_focal_graph,
    )
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(8)]
    df = pd.DataFrame([
        {"gene_symbol": genes[i % 8], "f1": float(rng.normal()),
         "f2": float(rng.normal()), "acmg_label": int(i % 8 < 4)}
        for i in range(60)
    ])
    string_edges = [("G0", "G1"), ("G1", "G2"), ("G3", "G4")]
    kg = {"shares_pathway": [("G2", "G3"), ("G5", "G6")],
          "shares_disease": [("G6", "G7")]}
    fg = build_hetero_focal_graph(df, string_edges, kg, ["f1", "f2"],
                                  label_col="acmg_label")
    assert fg.relations == ["interacts_with", "shares_pathway", "shares_disease"]
    assert fg.focal_idx.numel() == 60

    trainer = HeteroGNNTrainer(in_dim=2, relations=fg.relations, hidden=8,
                               n_layers=2, epochs=5)
    loss = trainer.train(fg)
    assert np.isfinite(loss)

    scores = trainer.score_all_nodes(fg)
    assert scores.shape[0] == 8
    assert np.isfinite(scores).all()
    assert (scores >= 0).all() and (scores <= 1).all()       # sigmoid range

    scorer = HeteroGNNScorer.from_trained(trainer, fg)
    assert set(scorer.gene_scores) == set(genes)
    assert scorer.score("G0") == scorer.gene_scores["G0"]
    assert scorer.score("UNKNOWN_GENE") == HeteroGNNScorer.DEFAULT_SCORE
    s = scorer.score_dataframe(pd.DataFrame({"gene_symbol": ["G0", "UNKNOWN"]}))
    assert s.iloc[1] == HeteroGNNScorer.DEFAULT_SCORE


def test_empty_focal_does_not_crash():
    pytest.importorskip("torch_geometric")
    from genomic_variant_classifier.models.hetero_gnn_scorer import (
        HeteroGNNScorer, HeteroGNNTrainer, build_hetero_focal_graph,
    )
    df = pd.DataFrame({"gene_symbol": ["A", "B", "C"], "f1": [0.1, 0.2, 0.3],
                       "acmg_label": [np.nan, np.nan, np.nan]})   # nothing labeled
    fg = build_hetero_focal_graph(df, [("A", "B")], {"shares_pathway": []},
                                  ["f1"], label_col="acmg_label")
    trainer = HeteroGNNTrainer(in_dim=1, relations=fg.relations, hidden=4, epochs=3)
    trainer.train(fg)                                            # warns, no crash
    scorer = HeteroGNNScorer.from_trained(trainer, fg)
    assert len(scorer.gene_scores) == 3                          # still scores all nodes
