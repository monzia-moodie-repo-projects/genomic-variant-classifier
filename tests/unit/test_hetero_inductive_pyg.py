"""test_hetero_inductive_pyg.py -- Monzia Moodie

Torch-gated regression for INCIDENT_2026-06-16 (hetero_gnn_score inert in val/test). The torch-free
core lives in test_hetero_inductive_fix.py; THIS file runs the real PyG HeteroConv forward to prove the
property end-to-end: with the genes= union override, HeteroGNNScorer.from_full_graph scores
gene-disjoint val/test genes (they are graph nodes and land in the score map), while the legacy
train-only node set leaves them at the 0.5 default. Skips where torch_geometric is absent.
"""
import numpy as np
import pandas as pd
import pytest


def _train_only_df(seed, train_genes, n=40):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([
        {"gene_symbol": train_genes[i % len(train_genes)],
         "f1": float(rng.normal()), "f2": float(rng.normal()),
         "acmg_label": int(i % len(train_genes) < len(train_genes) // 2)}
        for i in range(n)
    ])


def test_from_full_graph_covers_gene_disjoint_val_test():
    pytest.importorskip("torch_geometric")
    from genomic_variant_classifier.models.hetero_gnn_scorer import (
        HeteroGNNScorer, HeteroGNNTrainer, build_hetero_focal_graph,
    )
    train_genes = ["T0", "T1", "T2", "T3"]
    valtest_genes = ["V0", "V1", "E0", "E1"]          # gene-disjoint val + test genes
    union = train_genes + valtest_genes
    df = _train_only_df(0, train_genes)               # focal supervision TRAIN-only
    string_edges = [("T0", "T1"), ("T1", "T2"), ("T2", "V0"),
                    ("V0", "V1"), ("T3", "E0"), ("E0", "E1")]
    kg = {"shares_pathway": [("V1", "E0"), ("T0", "E1")]}

    fg = build_hetero_focal_graph(df, string_edges, kg, ["f1", "f2"],
                                  label_col="acmg_label", genes=union)
    # the union genes (incl. gene-disjoint val/test) are real graph nodes
    assert set(fg.node_genes) == set(union)
    assert fg.focal_idx.numel() == 40                 # focal supervision stayed train-only

    trainer = HeteroGNNTrainer(in_dim=2, relations=fg.relations, hidden=8, n_layers=2, epochs=5)
    assert np.isfinite(trainer.train(fg))
    scorer = HeteroGNNScorer.from_full_graph(trainer, fg)

    # THE fix: val/test genes are in the score map (not the 0.5 default), finite, in range
    assert set(valtest_genes) <= set(scorer.gene_scores)
    vt = [scorer.score(g) for g in valtest_genes]
    assert all(np.isfinite(s) and 0.0 <= s <= 1.0 for s in vt)
    assert not all(s == HeteroGNNScorer.DEFAULT_SCORE for s in vt)   # not inert
    # whole-graph non-degeneracy + genuine unknown still defaults
    alls = [scorer.score(g) for g in union]
    assert len(set(np.round(alls, 6))) > 1
    assert scorer.score("NOT_IN_GRAPH") == HeteroGNNScorer.DEFAULT_SCORE


def test_train_only_node_set_leaves_val_test_inert():
    # Characterizes the bug: legacy train-only node set (genes=None) -> val/test fall to 0.5.
    pytest.importorskip("torch_geometric")
    from genomic_variant_classifier.models.hetero_gnn_scorer import (
        HeteroGNNScorer, HeteroGNNTrainer, build_hetero_focal_graph,
    )
    train_genes = ["T0", "T1", "T2", "T3"]
    df = _train_only_df(1, train_genes)
    fg = build_hetero_focal_graph(df, [("T0", "T1"), ("T1", "T2")],
                                  {"shares_pathway": []}, ["f1", "f2"],
                                  label_col="acmg_label")            # genes=None -> train-only
    assert set(fg.node_genes) == set(train_genes)                    # val/test NOT nodes
    trainer = HeteroGNNTrainer(in_dim=2, relations=fg.relations, hidden=4, epochs=3)
    trainer.train(fg)
    scorer = HeteroGNNScorer.from_full_graph(trainer, fg)
    assert scorer.score("V0") == HeteroGNNScorer.DEFAULT_SCORE       # the inert behaviour
    assert scorer.score("E0") == HeteroGNNScorer.DEFAULT_SCORE
