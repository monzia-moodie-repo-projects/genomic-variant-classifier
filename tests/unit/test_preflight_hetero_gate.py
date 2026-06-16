"""test_preflight_hetero_gate.py -- Monzia Moodie

Run-17 no-defer: --hetero-gnn + --kg-edges reactome:<gmt> must appear in the emitted launch command AND be
enforced by --check, so hetero_gnn_score can never be silently left at its 0.5 default. Mirrors kg_gate.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

import preflight_gate as gate
import preflight_run17 as P


def _ns(cmd):
    return gate._parse_candidate(cmd)


def test_emit_includes_hetero_and_kg_edges():
    cmd = P.emit_command("kg.parquet", "outputs/run17", None)
    assert "--hetero-gnn" in cmd
    assert "--kg-edges reactome:data/external/reactome/ReactomePathways.gmt" in cmd


def test_emit_defer_still_includes_hetero():
    # deferring af_1kg (no --kg parquet) does NOT defer the hetero-GNN
    cmd = P.emit_command(None, "outputs/run17", None)
    assert "--kg " not in cmd            # no af_1kg parquet flag
    assert "--hetero-gnn" in cmd and "--kg-edges reactome:" in cmd


def test_hetero_gate_fails_when_hetero_missing(tmp_path):
    r = tmp_path / "external" / "reactome"
    r.mkdir(parents=True)
    (r / "ReactomePathways.gmt").write_text("x")
    ns = _ns(f"python scripts/run_phase2_eval.py --kg-edges reactome:{(r / 'ReactomePathways.gmt').as_posix()}")
    rows = P.hetero_gate(ns, str(tmp_path))
    assert any(lv == "FAIL" and "hetero-gnn" in m for lv, m in rows)


def test_hetero_gate_fails_when_kg_edges_missing(tmp_path):
    ns = _ns("python scripts/run_phase2_eval.py --hetero-gnn")
    rows = P.hetero_gate(ns, str(tmp_path))
    assert any(lv == "FAIL" and "kg-edges" in m for lv, m in rows)


def test_hetero_gate_fails_when_reactome_path_absent(tmp_path):
    ns = _ns(f"python scripts/run_phase2_eval.py --hetero-gnn --kg-edges reactome:{(tmp_path / 'nope.gmt').as_posix()}")
    rows = P.hetero_gate(ns, str(tmp_path))
    assert any(lv == "FAIL" and "not found" in m for lv, m in rows)


def test_hetero_gate_ok_when_both_present(tmp_path):
    r = tmp_path / "external" / "reactome"
    r.mkdir(parents=True)
    (r / "ReactomePathways.gmt").write_text("x")
    ns = _ns(f"python scripts/run_phase2_eval.py --hetero-gnn --kg-edges reactome:{(r / 'ReactomePathways.gmt').as_posix()}")
    rows = P.hetero_gate(ns, str(tmp_path))
    assert any(lv == "OK" for lv, m in rows) and not any(lv == "FAIL" for lv, m in rows)
