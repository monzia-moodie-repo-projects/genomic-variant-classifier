"""Stage 1D-1b.1 (set "ap"): typed observations and the shadow admission report derived from the merged census.
Owner ruling f2ec92cd (2026-11-09), section XIII. Author: Monzia Moodie."""
from __future__ import annotations

import ast
import copy
import fractions
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from genomic_variant_classifier.information_sources import admission as A
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import consumers as K
from genomic_variant_classifier.information_sources import observation as O

REPO = Path(__file__).resolve().parents[2]
SHA = "a" * 64
S = O.ObservationState


@pytest.fixture(scope="module")
def derived():
    return A.build(REPO)                      # verifies the census with fresh measurement first


@pytest.fixture(scope="module")
def ledger():
    return C.strict_load(REPO / C.LEDGER_PATH)


def row(report, name):
    return next(r for r in report["rows"] if r["name"] == name)


# ------------------------------------------------------------------------------------------------- the committed report
def test_the_committed_report_is_exactly_the_fresh_derivation(derived):
    raw = (REPO / A.REPORT_PATH).read_bytes()
    assert A.report_bytes(derived) == raw
    assert b"\r" not in raw and raw.endswith(b"}\n")


def test_the_report_is_bound_to_the_census_it_was_derived_from(derived, ledger):
    raw = (REPO / C.LEDGER_PATH).read_bytes()
    assert derived["census"]["ledger_sha256"] == C.sha256_bytes(raw)
    assert derived["census"]["semantic_digest"] == C.semantic_contract_digest(ledger)
    assert derived["census"]["contract_digest"] == C.source_contract(REPO).contract_digest


def test_every_census_name_appears_once_in_census_order(derived, ledger):
    names = [e["name"] for e in ledger["entries"]] + [s["name"] for s in ledger["information_sources"]]
    assert [r["name"] for r in derived["rows"]] == names
    assert {r["disposition"] for r in derived["rows"]} <= {d.value for d in A.REVIEWED}


def test_admitted_observed_and_available_are_reported_separately(derived):
    for r in derived["rows"]:
        assert set(r["admission"]) == set(A.REGIMES)
        assert r["observation"] == A.OBSERVATION_NOT_RECORDED          # no producer records it yet: unknown, not observed
        assert all(A.SURFACE_CLASS[s] == "predictive" for s in r["available_to_fitted_model"])
        assert all(A.SURFACE_CLASS[s] != "predictive" for s in r["computed_or_stored_only"])
    assert row(derived, "af_raw")["admission"] == {"strict": "admitted", "conditional": "admitted"}
    assert "tabular_fit" in row(derived, "af_raw")["available_to_fitted_model"]


def test_strict_admission_equals_the_census_gate(derived, ledger):
    report = C.verify_census(REPO, ledger)
    admitted = [r["name"] for r in derived["rows"] if r["admission"]["strict"] == "admitted"]
    assert admitted == list(report.strictly_admitted)
    assert derived["counts"]["strict"]["transitive_only"] == 0


def test_the_gnn_scores_are_not_admissible_under_any_regime_through_their_dependencies(derived):
    for name in ("gnn_score", "hetero_gnn_score"):
        r = row(derived, name)
        assert r["disposition"] == "conditional_only" and r["admission"]["conditional"] == "not_admitted"
        v = next(v for v in derived["violations"] if v["regime"] == "conditional" and v["name"] == name)
        assert v["reason"].startswith("depends on entries not admitted")
        assert "n_pathogenic_in_gene" in v["reason"] and "string_relation:combined_score_threshold" in v["reason"]
    assert derived["counts"]["conditional"]["transitive_only"] == 2


def test_every_unadmitted_name_that_reaches_a_model_is_a_violation_with_its_identity(derived):
    for regime in A.REGIMES:
        expected = sorted(r["name"] for r in derived["rows"]
                          if r["admission"][regime] == "not_admitted" and r["available_to_fitted_model"])
        got = sorted(v["name"] for v in derived["violations"] if v["regime"] == regime)
        assert got == expected and len(got) == derived["counts"][regime]["violations"]
    v = next(v for v in derived["violations"] if v["name"] == "kg_relation:clingen" and v["regime"] == "strict")
    assert v["kind"] == "graph_relation" and v["predictive_surfaces"] == ["hetero_gnn_relations"]


def test_registered_defects_are_attached_to_the_names_they_affect(derived):
    assert row(derived, "eve_score")["registered_defects"] == ["EVE-COVERAGE-IDENTITY-1", "MISSINGNESS-SEMANTICS-1"]
    assert row(derived, "codon_position")["registered_defects"] == ["CODON-POSITION-SEMANTICS-1"]


# ------------------------------------------------------------------------------------------------------ the derivation
def test_effective_admission_is_the_greatest_fixed_point():
    Dd = A.D
    disp = {"a": Dd.ADMIT_STRICT, "b": Dd.ADMIT_STRICT, "c": Dd.ADMIT_STRICT, "x": Dd.EXCLUDE_STRICT,
            "p": Dd.ADMIT_STRICT, "q": Dd.ADMIT_STRICT}
    deps = {"a": ("b",), "b": ("c",), "c": ("x",), "p": ("q",), "q": ("p",)}       # a chain to x; a mutual cycle
    out = A.effective_admission(disp, deps, A.REGIMES["strict"])
    assert out["c"] == ("x",) and out["b"] == ("c",) and out["a"] == ("b",)           # propagates along the chain
    assert out["p"] == () and out["q"] == ()                                          # an admitted cycle stays admitted
    with pytest.raises(ValueError, match="unknown entries"):
        A.effective_admission(disp, {"a": ("nowhere",)}, A.REGIMES["strict"])


def test_surface_classes_cover_exactly_the_measured_consumers():
    A.surface_classes_cover(K.SURFACES, K.GRAPH_CONSUMERS)
    with pytest.raises(ValueError, match=r"missing \['new_surface'\]"):
        A.surface_classes_cover((*K.SURFACES, "new_surface"), K.GRAPH_CONSUMERS)


def test_an_unreviewed_or_duplicated_name_refuses(ledger, derived):
    meta = {"ledger_sha256": SHA, "contract_digest": SHA}
    bad = copy.deepcopy(ledger)
    bad["entries"][0]["disposition"] = "pending_review"
    with pytest.raises(ValueError, match="unreviewed disposition"):
        A.shadow_report(bad, **meta)
    bad = copy.deepcopy(ledger)
    bad["information_sources"][0]["name"] = bad["entries"][0]["name"]
    with pytest.raises(ValueError, match="named twice"):
        A.shadow_report(bad, **meta)
    with pytest.raises(ValueError, match="SHA-256"):
        A.shadow_report(ledger, ledger_sha256="x", contract_digest=SHA)


def test_no_report_is_derived_from_a_ledger_the_census_gate_refuses(ledger, tmp_path, monkeypatch):
    bad = copy.deepcopy(ledger)
    next(e for e in bad["entries"] if e["name"] == "eve_score")["disposition"] = "admit_strict"   # breaks a pinned ruling
    path = tmp_path / "ledger.json"
    path.write_bytes(C.ledger_bytes(bad))                       # canonical in form: only the gate can refuse it
    monkeypatch.setattr(C, "LEDGER_PATH", str(path))            # an absolute path: repo / LEDGER_PATH is this file
    with pytest.raises(ValueError, match="pinned ruling"):
        A.build(REPO)


def test_a_consumer_outside_the_classification_refuses(ledger):
    bad = copy.deepcopy(ledger)
    bad["entries"][0]["consumers"].append("somewhere_new")
    with pytest.raises(ValueError, match="unclassified surfaces"):
        A.shadow_report(bad, ledger_sha256=SHA, contract_digest=SHA)


def test_the_command_line_checks_byte_for_byte_and_never_overwrites(tmp_path, capsys):
    good = tmp_path / "report.json"
    assert A.main(["--repo", str(REPO), "--out", str(good)]) == 0
    assert good.read_bytes() == (REPO / A.REPORT_PATH).read_bytes()
    assert A.main(["--repo", str(REPO), "--check", str(good)]) == 0
    assert "SHADOW" in capsys.readouterr().out
    tampered = json.loads(good.read_text())
    tampered["counts"]["strict"]["violations"] = 0
    good.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    assert A.main(["--repo", str(REPO), "--check", str(good)]) == 1
    with pytest.raises(SystemExit, match="overwrite"):
        A.main(["--repo", str(REPO), "--out", str(good)])


# ------------------------------------------------------------------------------------------------------ shadow mode
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_bytes(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return found


def test_shadow_mode_no_production_code_imports_the_new_modules():
    package = REPO / "src/genomic_variant_classifier/information_sources"
    new = {"genomic_variant_classifier.information_sources.admission",
           "genomic_variant_classifier.information_sources.observation"}
    users = []
    for root in ("src", "scripts"):
        for path in sorted((REPO / root).rglob("*.py")):
            if path.parent == package:
                continue
            if _imports(path) & new:
                users.append(path.relative_to(REPO).as_posix())
    assert users == []


def test_the_new_modules_are_not_consumers_of_the_feature_contract():
    sites = K.discover_sites(REPO)
    for name in ("admission.py", "observation.py"):
        assert f"src/genomic_variant_classifier/information_sources/{name}" not in sites   # the census stays unchanged


def test_the_contract_modules_import_without_the_model_training_stack():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'numpy', 'pandas', 'torch', 'sklearn', 'xgboost', 'lightgbm', 'catboost',\n"
        "                                  'networkx', 'torch_geometric', 'scipy'}:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from genomic_variant_classifier.information_sources import observation as O, admission as A\n"
        "o = O.observed(0.5, 'a' * 64)\n"
        "assert o.value == 0.5 and A.effective_admission({'a': A.D.ADMIT_STRICT}, {}, A.REGIMES['strict']) == {'a': ()}\n"
        "print('ok')\n")
    env = {"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


# -------------------------------------------------------------------------------------------------- typed observations
def test_an_observed_value_is_a_finite_real_number_never_a_boolean():
    for flag in (True, False, np.bool_(True)):
        with pytest.raises(ValueError, match="not a boolean"):
            O.observed(flag, SHA)
    for bad in (float("nan"), float("inf"), np.float64("-inf")):
        with pytest.raises(ValueError, match="finite"):
            O.observed(bad, SHA)
    with pytest.raises(ValueError, match="real number"):
        O.observed("0.5", SHA)
    with pytest.raises(ValueError, match="real number"):
        O.observed(complex(1, 0), SHA)


def test_numeric_types_are_normalised_so_equal_measurements_are_equal():
    a, b, c = O.observed(np.float32(0.25), SHA), O.observed(fractions.Fraction(1, 4), SHA), O.observed(0.25, SHA)
    assert a == b == c and type(a.value) is float
    assert O.observed(np.int64(0), SHA).value == 0.0 and O.observed(0, SHA).is_observed   # a genuine zero is observed


def test_an_absent_observation_carries_no_sentinel():
    for state in S:
        if state is S.OBSERVED:
            continue
        for sentinel in (0.0, 0.5, -1, 0):
            with pytest.raises(ValueError, match="no value"):
                O.NumericObservation(state, sentinel, None if state is S.SOURCE_ABSENT else SHA)
    with pytest.raises(ValueError, match="other than OBSERVED"):
        O.absent(S.OBSERVED, SHA)


def test_the_state_decides_whether_a_source_must_be_named():
    assert set(O.SOURCE_RULE) == set(S)
    for state in (S.OBSERVED, S.NOT_IN_SOURCE, S.UNSUPPORTED_VARIANT):
        with pytest.raises(ValueError, match="must name its source"):
            O.NumericObservation(state, 1.0 if state is S.OBSERVED else None, None)
    with pytest.raises(ValueError, match="no source"):
        O.absent(S.SOURCE_ABSENT, SHA)
    assert O.absent(S.SOURCE_ABSENT).source_id is None
    for state in (S.BAD_MAPPING, S.COMPUTATION_FAILED):
        assert O.absent(state).source_id is None and O.absent(state, SHA).source_id == SHA
    for bad in ("src", "A" * 64, "a" * 63):
        with pytest.raises(ValueError, match="SHA-256"):
            O.absent(S.NOT_IN_SOURCE, bad)
    with pytest.raises(ValueError, match="ObservationState"):
        O.NumericObservation("observed", 1.0, SHA)
