"""Stage 1D-1b.3 (set "ar"), repaired in stage 1D-1b.4 (set "as"): authorized training labels -- the fit's scope is derived
from the split's components (population.authorize_fit), the training labels hold authorized native integers only and record
every read with its outcome (LABEL-TYPE-IDENTITY-1 and LABEL-DENIED-READ-AUDIT-1, repaired), and the required
non-interference test (ruling 7a2a6ade section 13) shows the legacy graph builders take whatever labels the caller hands them
while the masked training labels make them independent of the held-out labels. Owner rulings f2ec92cd (Unit 1D-1b.3),
7a2a6ade (section 13) and 68ca322f. Author: Monzia Moodie."""
from __future__ import annotations

import copy
import json
import pickle
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genomic_variant_classifier.information_sources import labels as L
from genomic_variant_classifier.information_sources import population as P

REPO = Path(__file__).resolve().parents[2]
SITES = REPO / "docs/census/LABEL_SITES_1D1b3.json"
SCOPE = {"model": "m", "population": "p", "split": "s"}


def _rows():
    # HGNC:9 carries rows in BOTH partitions; r3 is joined to the held-out gene HGNC:9 only through r4's gene list.
    t, h = P.Partition.TRAIN, P.Partition.TEST

    def r(i, genes, p):
        return P.PopulationRow(i, f"v_{i}", P.VariantIdentity.VRS_VERIFIED, frozenset(genes), (), p)
    return [r("r1", ["HGNC:1"], t), r("r2", ["HGNC:1"], t), r("r3", ["HGNC:2"], t), r("r4", ["HGNC:2", "HGNC:9"], t),
            r("r5", ["HGNC:3"], h), r("r6", ["HGNC:9"], P.Partition.VALIDATION)]


SOURCE = {"r1": 1, "r2": 0, "r3": 1, "r4": 0, "r5": 1, "r6": 1}


def _labels(source=SOURCE):
    auth = P.authorize_fit(_rows(), P.FitAuthority.CONDITIONAL_COMPONENT)
    return auth, L.TrainingLabels(auth, source, fit_scope=SCOPE)


def test_the_scope_comes_from_components_and_only_authorized_labels_are_held():
    auth, t = _labels()
    assert auth.authorized_rows == {"r1", "r2"}                          # r3 and r4 share a component with held-out r6
    assert t.authorized_rows == ("r1", "r2") and [t.read("r1"), t.read("r2")] == [1, 0]
    assert set(t._labels) == {"r1", "r2"}                                # no held-out or refused value is stored


def test_every_read_is_recorded_with_its_outcome():
    """LABEL-DENIED-READ-AUDIT-1 (repaired): a refused training row and a held-out row are DENIED, an unknown row is
    UNKNOWN; all three raise LabelAccessError and all three are in the receipt."""
    _, t = _labels()
    assert t.read("r1") == 1
    for row in ("r3", "r6", "nope", 5):
        with pytest.raises(L.LabelAccessError):
            t.read(row)
    rec = t.receipt()
    assert (rec["authorized"], rec["denied"], rec["unknown"]) == (1, 2, 2)
    before = rec["events_digest"]
    with pytest.raises(PermissionError):                                 # LabelAccessError is a PermissionError
        t.read("r5")
    assert t.receipt()["events_digest"] != before and t.receipt()["denied"] == 3


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.0, np.float64(1.0), np.int64(1), Fraction(1), 2, -1])
def test_a_label_is_a_native_int_zero_or_one(value):
    """LABEL-TYPE-IDENTITY-1 (repaired): every value equal to 1 that is not the native int 1 is refused at the adapter."""
    with pytest.raises(TypeError, match="native int"):
        _labels(dict(SOURCE, r1=value))


def test_held_out_values_are_never_validated_or_kept():
    _, t = _labels(dict(SOURCE, r6=np.float64(1.0), r5="pathogenic"))   # held-out values are not even read
    assert set(t._labels) == {"r1", "r2"}


def test_the_source_must_cover_every_authorized_row_and_the_scope_must_be_complete():
    auth = P.authorize_fit(_rows(), P.FitAuthority.CONDITIONAL_COMPONENT)
    with pytest.raises(ValueError, match="lacks"):
        L.TrainingLabels(auth, {"r1": 1}, fit_scope=SCOPE)
    for scope in ({"model": "m", "population": "p"}, dict(SCOPE, split=""), dict(SCOPE, extra="x")):
        with pytest.raises(ValueError, match="fit_scope"):
            L.TrainingLabels(auth, SOURCE, fit_scope=scope)
    with pytest.raises(TypeError, match="FitAuthorization"):
        L.TrainingLabels({"r1"}, SOURCE, fit_scope=SCOPE)


def test_the_digest_binds_authorization_scope_and_labels():
    _, a = _labels()
    _, b = _labels(dict(SOURCE, r1=0))
    auth = P.authorize_fit(_rows(), P.FitAuthority.CONDITIONAL_COMPONENT)
    c = L.TrainingLabels(auth, SOURCE, fit_scope=dict(SCOPE, model="other"))
    assert len({a.digest, b.digest, c.digest}) == 3


def test_training_labels_cannot_be_copied_or_serialised():
    _, t = _labels()
    for attempt in (lambda: pickle.dumps(t), lambda: copy.copy(t), lambda: copy.deepcopy(t)):
        with pytest.raises(TypeError, match="cannot be serialised or copied"):
            attempt()
    with pytest.raises(AttributeError):
        t._extra = 1                                                    # no instance dictionary to smuggle data in


def test_masked_labels_keep_only_authorized_labels_and_the_bulk_read_is_recorded():
    _, t = _labels()
    got = t.masked(["r1", "r2", "r3", "r4", "r5", "r6", "unknown"])
    assert got[:2].tolist() == [1.0, 0.0] and np.isnan(got[2:]).all() and not got.flags.writeable
    rec = t.receipt()
    assert (rec["masked_calls"], rec["masked_authorized"], rec["masked_denied"], rec["masked_unknown"]) == (1, 2, 4, 1)
    assert (rec["authorized"], rec["denied"], rec["unknown"]) == (0, 0, 0)


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


def test_the_masked_training_labels_make_the_hetero_builder_non_interfering():
    rows = [P.PopulationRow(f"v{i}", f"var{i}", P.VariantIdentity.VRS_VERIFIED, frozenset({f"HGNC:{g[1]}"}), (),
                            P.Partition.TEST if HELDOUT[i] else P.Partition.TRAIN)
            for i, g in enumerate(_frame()["gene_symbol"])]
    auth = P.authorize_fit(rows, P.FitAuthority.STRICT, P.SplitProtocol((P.Partition.TRAIN, P.Partition.TEST)))
    t = L.TrainingLabels(auth, {f"v{i}": int(LABELS[i]) for i in range(8)}, fit_scope=SCOPE)

    def build(labels: np.ndarray):
        # The producer never sees `labels`: the training labels supply the only labels it gets.
        return _assemble(t.masked([f"v{i}" for i in range(8)]))

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
