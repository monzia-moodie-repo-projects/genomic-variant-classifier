"""Executable witnesses of registered defects (owner ruling 68ca322f, 2026-11-10; set "ar" revision B onward).

Each test PASSES while its defect still exists, so the registration in the census ledger is evidence, not prose. A stage that
repairs a defect will see its witness FAIL: the repair must then remove the witness and the registration together, in the
same change. Set "as" (stage 1D-1b.4) repaired the four defects set "ar" registered with witnesses -- LABEL-TYPE-IDENTITY-1
and LABEL-DENIED-READ-AUDIT-1 (labels.TrainingLabels), GRAPH-MAPPING-RECEIPT-AUTHORITY-1 (graph_identity.accept_graph
re-derives from the pinned files) and GRAPH-MUTABLE-MATERIALIZATION-1 (bytes-backed read-only buffers, read-only mappings)
-- and their witnesses now FAIL: they were removed with their registrations, and their inverted forms are tests in
test_label_authority.py and test_graph_trust_boundary.py. The two defects below are in production code (models/gnn.py,
data/real_data_prep.py); registering them changes no behaviour. Author: Monzia Moodie."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
GNN = REPO / "src/genomic_variant_classifier/models/gnn.py"
RDP = REPO / "src/genomic_variant_classifier/data/real_data_prep.py"


def _registered(defect_id: str) -> None:
    from genomic_variant_classifier.information_sources import census as C

    ids = {d["id"] for d in C.strict_load(REPO / C.LEDGER_PATH)["registered_defects"]}
    assert defect_id in ids, f"{defect_id} is witnessed here but not registered in the census ledger"


def _function(path: Path, name: str, cls: str | None = None) -> ast.FunctionDef:
    tree = ast.parse(path.read_bytes())
    body = tree.body
    if cls is not None:
        body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_witness_gnn_early_stop_row_split_1():
    """GNN-EARLY-STOP-ROW-SPLIT-1: train_gnn_pipeline early-stops on a ROW-level split of the training rows
    (train_test_split over np.arange(n), stratified by label, no groups), so rows of one gene sit on both sides of the
    checkpoint split. Executed here on 4 genes of 10 rows each: every gene lands on both sides."""
    _registered("GNN-EARLY-STOP-ROW-SPLIT-1")
    fn = _function(GNN, "train_gnn_pipeline")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "train_test_split"]
    assert len(calls) == 1
    call = calls[0]
    assert ast.unparse(call.args[0]) == "np.arange(n)"
    assert {k.arg for k in call.keywords} == {"test_size", "random_state", "stratify"}       # no groups
    from sklearn.model_selection import train_test_split

    genes = np.repeat(np.arange(4), 10)
    y = np.tile(np.array([0, 1] * 5), 4)
    tr, va = train_test_split(np.arange(40), test_size=0.15, random_state=42, stratify=y)
    assert set(genes[tr]) & set(genes[va])


def test_witness_gene_field_prose_1():
    """GENE-FIELD-PROSE-1: the legacy split's group is the raw gene field, so a ClinVar description ("covers 26 genes,
    ...") is one 'gene' group and the genes it covers bind nothing; the shadow resolver flags it as prose and leaves the row
    gene-incomplete."""
    _registered("GENE-FIELD-PROSE-1")
    tree = ast.parse(RDP.read_bytes())
    groups = [ast.unparse(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "groups"]
    assert groups.count("df[self.config.group_column].fillna('unknown').reset_index(drop=True)") == 2
    from genomic_variant_classifier.information_sources import graph_identity as G
    from genomic_variant_classifier.information_sources import population as P

    hgnc = G.HgncIndex({"AAA": "HGNC:1"}, {}, {}, "a" * 64, 1)
    got = P.resolve_gene_field("covers 26 genes, none of which curated to show dosage sensitivity", hgnc)
    assert got.prose and not got.gene_ids
