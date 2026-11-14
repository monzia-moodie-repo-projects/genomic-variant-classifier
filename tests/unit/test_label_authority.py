"""Stage 1D-1b.3 (set "ar"): authorized labels -- the authority derives the fit's scope from canonical gene identity, the
fold-scoped store records and refuses reads, and the required non-interference test (ruling 7a2a6ade section 13) shows the
legacy graph builders take whatever labels the caller hands them while the authority's masked labels make them
independent of the held-out labels. Owner rulings f2ec92cd (Unit 1D-1b.3) and 7a2a6ade (section 13).
Author: Monzia Moodie."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import labels as L

REPO = Path(__file__).resolve().parents[2]
SITES = REPO / "docs/census/LABEL_SITES_1D1b3.json"


def _records():
    # gene HGNC:9 carries rows in BOTH partitions (two symbols of one gene, GENE-SYMBOL-SPLIT-IDENTITY-1)
    return [L.LabelRecord("r1", "HGNC:1", 1, "train"), L.LabelRecord("r2", "HGNC:1", 0, "train"),
            L.LabelRecord("r3", "HGNC:2", 1, "train"), L.LabelRecord("r4", "HGNC:9", 0, "train"),
            L.LabelRecord("r5", "HGNC:3", 1, "heldout"), L.LabelRecord("r6", "HGNC:9", 1, "heldout")]


def test_the_scope_is_derived_from_canonical_genes_and_shared_genes_are_refused():
    a = L.LabelAuthority(_records())
    auth = a.authorize()
    assert (auth.authorized_rows, auth.authorized_genes, auth.heldout_rows, auth.heldout_genes) == (3, 2, 2, 2)
    assert auth.refused_training_rows == 1 and auth.refused_genes == ("HGNC:9",)
    store = a.store_for_fit()
    assert sorted(store.authorized_rows()) == ["r1", "r2", "r3"]
    assert [store.read(r) for r in ("r1", "r2", "r3")] == [1, 0, 1]
    assert store.accesses == (("r1", "HGNC:1"), ("r2", "HGNC:1"), ("r3", "HGNC:2"))
    for row in ("r4", "r5", "r6"):                                     # refused or held out: not even present
        with pytest.raises(KeyError):
            store.read(row)


def test_the_store_refuses_out_of_scope_reads_and_cannot_be_copied_out():
    s = L.FoldScopedLabelStore({"x": ("HGNC:1", 1), "y": ("HGNC:2", 0)}, frozenset({"HGNC:1"}))
    assert s.read("x") == 1
    with pytest.raises(PermissionError, match="unauthorized"):
        s.read("y")
    assert s.accesses == (("x", "HGNC:1"),)
    with pytest.raises(TypeError, match="serialised"):
        pickle.dumps(s)
    with pytest.raises(TypeError, match="serialised"):
        pickle.dumps(L.LabelAuthority(_records()))
    with pytest.raises(ValueError, match="empty authorization"):
        L.FoldScopedLabelStore({}, frozenset())
    with pytest.raises(AttributeError):
        s._extra = 1                                                    # no instance dictionary to smuggle data in


@pytest.mark.parametrize("bad", [dict(label=True), dict(label=2), dict(partition="val"), dict(gene_id="")])
def test_label_records_are_validated(bad):
    base = dict(row_id="r", gene_id="HGNC:1", label=1, partition="train")
    base.update(bad)
    with pytest.raises(ValueError):
        L.LabelRecord(**base)
    with pytest.raises(ValueError, match="twice"):
        L.LabelAuthority([L.LabelRecord("r", "HGNC:1", 1, "train"), L.LabelRecord("r", "HGNC:2", 0, "heldout")])


def test_masked_labels_keep_only_authorized_training_labels():
    a = L.LabelAuthority(_records())
    got = a.masked_labels(["r1", "r2", "r3", "r4", "r5", "r6", "unknown"])
    assert got[:3].tolist() == [1.0, 0.0, 1.0] and np.isnan(got[3:]).all()


def test_the_flip_changes_only_labelled_heldout_rows():
    y = np.array([1.0, 0.0, np.nan, 1.0])
    h = np.array([False, True, True, True])
    assert np.array_equal(L.flip_heldout(y, h), np.array([1.0, 1.0, np.nan, 0.0]), equal_nan=True)


def test_output_digest_is_exact():
    a = {"x": np.array([1.0, 2.0]), "y": [1, "a"]}
    assert L.output_digest(a) == L.output_digest({"y": [1, "a"], "x": np.array([1.0, 2.0])})
    assert L.output_digest(a) != L.output_digest({"x": np.array([1.0, 2.0000001]), "y": [1, "a"]})
    assert L.output_digest(np.array([1, 2], np.int32)) != L.output_digest(np.array([1, 2], np.int64))


# ------------------------------------------------------------------------------------------- the required test
def _frame():
    genes = ["G1", "G1", "G2", "G2", "G3", "G3", "G4", "G4"]
    return pd.DataFrame({"variant_id": [f"v{i}" for i in range(8)], "gene_symbol": genes,
                         "f1": np.arange(8, dtype=float), "f2": np.linspace(0, 1, 8)})


HELDOUT = np.array([False, False, False, False, True, True, True, True])
LABELS = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])


def _assemble(labels: np.ndarray):
    from genomic_variant_classifier.models.hetero_gnn_scorer import assemble_node_features_and_focal

    df = _frame()
    df["acmg_label"] = labels
    return assemble_node_features_and_focal(df, ["G1", "G2", "G3", "G4"], ["f1", "f2"], "acmg_label")


def test_the_legacy_hetero_builder_fails_non_interference_when_handed_raw_labels():
    r = L.label_noninterference(_assemble, LABELS, HELDOUT)
    assert not r.holds and r.heldout_labeled_rows == 4


def test_the_authoritys_masked_labels_make_the_hetero_builder_non_interfering():
    recs = [L.LabelRecord(f"v{i}", f"HGNC:{g[1]}", int(LABELS[i]), "heldout" if HELDOUT[i] else "train")
            for i, g in enumerate(_frame()["gene_symbol"])]
    a = L.LabelAuthority(recs)

    def build(labels: np.ndarray):
        # The producer never sees `labels`: the authority supplies the only labels it gets.
        return _assemble(a.masked_labels([f"v{i}" for i in range(8)]))

    r = L.label_noninterference(build, LABELS, HELDOUT)
    assert r.holds and r.digest_original == r.digest_altered
    feats, focal, y = build(LABELS)
    assert focal.tolist() == [0, 0, 1, 1] and y.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_the_legacy_homogeneous_builder_fails_and_then_passes_the_same_test():
    nx = pytest.importorskip("networkx")
    pytest.importorskip("torch_geometric")
    from genomic_variant_classifier.models.gnn import build_pyg_dataset

    graph = nx.Graph()
    graph.add_edge("G1", "G2", experimental=0.9, database=0.0, coexpression=0.1)
    graph.add_edge("G3", "G4", experimental=0.8, database=0.0, coexpression=0.0)

    def legacy(labels: np.ndarray):
        df = _frame()
        df["acmg_label"] = labels
        ds = build_pyg_dataset(df, graph, ["f1", "f2"])
        return ds.x, ds.edge_index, ds.edge_attr, ds.focal_idx, ds.y, ds.variant_ids

    assert not L.label_noninterference(legacy, LABELS, HELDOUT).holds
    masked = np.where(HELDOUT, np.nan, LABELS)
    assert L.label_noninterference(lambda _l: legacy(masked), LABELS, HELDOUT).holds


def test_a_vacuous_test_is_refused():
    with pytest.raises(ValueError, match="vacuous"):
        L.label_noninterference(lambda y: y, np.array([1.0, np.nan]), np.array([False, True]))


# ------------------------------------------------------------------------------------------- static discovery
def test_every_label_site_is_reviewed_and_no_review_is_stale():
    raw = SITES.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"}\n")
    table = json.loads(raw)
    assert table["schema"] == "gvc.label-sites.1d1b3/1"
    L.check_label_sites(L.discover_label_sites(REPO), table["sites"])


def test_discovery_finds_a_new_reader_and_the_check_refuses_it(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "m.py").write_bytes(b"def f(df, label_col='y'):\n    return df\n"
                                            b"class K:\n    def g(self):\n        return {'acmg_label': 1}\n")
    found = L.discover_label_sites(tmp_path)
    assert found == {"src/m.py": ["constant:K.g", "parameter:f"]}
    with pytest.raises(ValueError, match="unreviewed"):
        L.check_label_sites(found, {})
    with pytest.raises(ValueError, match="differ"):
        L.check_label_sites(found, {"src/m.py": {"markers": ["parameter:f"], "roles": ["fitter"], "reason": "x"}})
    with pytest.raises(ValueError, match="reviewed but absent"):
        L.check_label_sites(found, {"src/m.py": {"markers": found["src/m.py"], "roles": ["fitter"], "reason": "x"},
                                    "src/gone.py": {"markers": ["parameter:g"], "roles": ["fitter"], "reason": "x"}})
    with pytest.raises(ValueError, match="roles"):
        L.check_label_sites(found, {"src/m.py": {"markers": found["src/m.py"], "roles": ["owner"], "reason": "x"}})
    L.check_label_sites(found, {"src/m.py": {"markers": found["src/m.py"], "roles": ["fitter"], "reason": "x"}})
