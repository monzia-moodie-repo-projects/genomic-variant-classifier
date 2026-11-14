"""Stages 1D-1b.1 (set "ap") and 1D-1b.2 (set "aq"): the shadow admission report derived from the merged census.
Owner rulings f2ec92cd (section XIII) and 7f4b69d0 (sections 2.2 and 3). Author: Monzia Moodie."""
from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources import admission as A
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import consumers as K

REPO = Path(__file__).resolve().parents[2]
SHA = "a" * 64


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


def test_admitted_observed_and_exposed_are_reported_separately(derived):
    for r in derived["rows"]:
        assert set(r["admission"]) == set(A.REGIMES)
        exposure = r["exposure"]
        assert list(exposure) == [e.value for e in A.InputExposure]
        assert exposure["declared"] is True
        assert all(A.SURFACE_CLASS[s] == "predictive" for s in exposure["statically_reachable"])
        for level in ("materialized", "fitted_input", "inference_input"):      # need execution evidence: not claimed
            assert exposure[level] == A.NOT_ESTABLISHED
        assert all(A.SURFACE_CLASS[s] != "predictive" for s in r["computed_or_stored_only"])
        expected = (f"shadow_materializer:{A.SHADOW_MATERIALIZERS[r['name']]}" if r["name"] in A.SHADOW_MATERIALIZERS
                    else A.OBSERVATION_NOT_RECORDED)
        assert r["observation"] == expected
    assert row(derived, "af_raw")["admission"] == {"strict": "admitted", "conditional": "admitted"}
    assert "tabular_fit" in row(derived, "af_raw")["exposure"]["statically_reachable"]
    assert "available_to_fitted_model" not in json.dumps(derived)          # renamed: static exposure is not fitted input


def test_the_shadow_materialisers_named_exist_and_are_not_production():
    tree = ast.parse((REPO / "src/genomic_variant_classifier/information_sources/materialization.py").read_text())
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for feature, symbol in A.SHADOW_MATERIALIZERS.items():
        path, qual = symbol.split("::")
        assert path == "src/genomic_variant_classifier/information_sources/materialization.py" and qual in defined, feature


def test_the_report_carries_the_lineage_facts(derived):
    lin = derived["lineage"]
    assert (lin["identities"], lin["dependency_edges"], lin["leaves"]) == (102, 212, 85)
    assert lin["cyclic_components"] == [] and lin["ungrounded_leaves"] == []
    assert lin["registered_datasets"] == len(C.strict_load(REPO / C.LEDGER_PATH)["datasets"])


def test_strict_admission_equals_the_census_gate(derived, ledger):
    report = C.verify_census(REPO, ledger)
    admitted = [r["name"] for r in derived["rows"] if r["admission"]["strict"] == "admitted"]
    assert admitted == list(report.strictly_admitted)
    assert derived["counts"]["strict"]["transitive_only"] == 0
    assert (derived["counts"]["strict"]["admitted"], derived["counts"]["conditional"]["admitted"]) == (55, 65)


def _greatest_fixed_point(dispositions, depends_on, admitted):
    """The 1D-1b.1 algorithm, kept here to show the least fixed point equals it on the acyclic census."""
    ok = {n for n, d in dispositions.items() if d in admitted}
    changed = True
    while changed:
        changed = False
        for name in sorted(ok):
            if any(dep not in ok for dep in depends_on.get(name, ())):
                ok.discard(name)
                changed = True
    return {n: tuple(sorted(d for d in depends_on.get(n, ()) if d not in ok)) for n in dispositions}


def test_the_least_fixed_point_equals_the_greatest_on_the_acyclic_census(ledger):
    records = [*ledger["entries"], *ledger["information_sources"]]
    disp = {r["name"]: A.D(r["disposition"]) for r in records}
    deps = {r["name"]: tuple(r.get("depends_on", ())) for r in records}
    for regime, admitted in A.REGIMES.items():
        assert A.effective_admission(disp, deps, admitted) == _greatest_fixed_point(disp, deps, admitted), regime


def test_the_gnn_scores_are_not_admissible_under_any_regime_through_their_dependencies(derived):
    for name in ("gnn_score", "hetero_gnn_score"):
        r = row(derived, name)
        assert r["disposition"] == "conditional_only" and r["admission"]["conditional"] == "not_admitted"
        v = next(v for v in derived["structural_exposure_risks"] if v["regime"] == "conditional" and v["name"] == name)
        assert v["reason"].startswith("depends on entries not admitted")
        assert "n_pathogenic_in_gene" in v["reason"] and "string_relation:combined_score_threshold" in v["reason"]
    assert derived["counts"]["conditional"]["transitive_only"] == 2


def test_every_unadmitted_name_that_reaches_a_model_is_a_violation_with_its_identity(derived):
    for regime in A.REGIMES:
        expected = sorted(r["name"] for r in derived["rows"]
                          if r["admission"][regime] == "not_admitted" and r["exposure"]["statically_reachable"])
        got = sorted(v["name"] for v in derived["structural_exposure_risks"] if v["regime"] == regime)
        assert got == expected and len(got) == derived["counts"][regime]["structural_exposure_risks"]
    assert (derived["counts"]["strict"]["structural_exposure_risks"],
            derived["counts"]["conditional"]["structural_exposure_risks"]) == (47, 37)
    assert "not a demonstration that a particular fit consumed it" in derived["structural_exposure_risk_meaning"]
    v = next(v for v in derived["structural_exposure_risks"] if v["name"] == "kg_relation:clingen" and v["regime"] == "strict")
    assert v["kind"] == "graph_relation" and v["predictive_surfaces"] == ["hetero_gnn_relations"]


def test_registered_defects_are_attached_to_the_names_they_affect(derived):
    assert row(derived, "eve_score")["registered_defects"] == ["EVE-COVERAGE-IDENTITY-1",
                                                               "EVE-DUPLICATE-KEY-MULTIPLIES-ROWS-1",
                                                               "MISSINGNESS-SEMANTICS-1"]
    assert row(derived, "revel_pathogenic")["registered_defects"] == ["ABSENCE-AS-EVIDENCE-1"]
    assert "GNOMAD-CHRX-LOCUS-PARSE-1" in row(derived, "af_is_ultra_rare")["registered_defects"]
    assert row(derived, "codon_position")["registered_defects"] == ["CODON-POSITION-SEMANTICS-1"]


# ------------------------------------------------------------------------------------------------------ the derivation
def test_effective_admission_propagates_and_refuses_every_cycle():
    Dd = A.D
    disp = {"a": Dd.ADMIT_STRICT, "b": Dd.ADMIT_STRICT, "c": Dd.ADMIT_STRICT, "x": Dd.EXCLUDE_STRICT}
    deps = {"a": ("b",), "b": ("c",), "c": ("x",)}                                 # a chain to an excluded source
    out = A.effective_admission(disp, deps, A.REGIMES["strict"])
    assert out["c"] == ("x",) and out["b"] == ("c",) and out["a"] == ("b",)
    # ADMISSION-CYCLIC-LINEAGE-1: the 1D-1b.1 greatest fixed point admitted p <-> q and a self-loop; now refused.
    cyc = {"p": Dd.ADMIT_STRICT, "q": Dd.ADMIT_STRICT, "s": Dd.ADMIT_STRICT}
    with pytest.raises(ValueError, match="cyclic information lineage.*p <-> q; s"):
        A.effective_admission(cyc, {"p": ("q",), "q": ("p",), "s": ("s",)}, A.REGIMES["strict"])
    assert _greatest_fixed_point(cyc, {"p": ("q",), "q": ("p",)}, A.REGIMES["strict"]) == {"p": (), "q": (), "s": ()}
    with pytest.raises(ValueError, match="unknown entries"):
        A.effective_admission(disp, {"a": ("nowhere",)}, A.REGIMES["strict"])


def test_a_ledger_with_an_ungrounded_leaf_produces_no_report(ledger):
    bad = copy.deepcopy(ledger)
    src = next(s for s in bad["information_sources"] if s["name"] == "kg_relation:go")
    src["source_ids"] = [s for s in src["source_ids"] if not s.startswith("dataset:")]
    with pytest.raises(ValueError, match=r"lineage leaves name no registered dataset: \['kg_relation:go'\]"):
        A.shadow_report(bad, ledger_sha256=SHA, contract_digest=SHA)


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
    tampered["counts"]["strict"]["structural_exposure_risks"] = 0
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


NEW_MODULES = ("admission", "observation", "lineage", "columns", "materialization", "absence_census", "graph_identity",
               "labels", "population", "claims")
#: Measurement scripts that may import the shadow modules: they fit no model and write no production artefact.
DIAGNOSTIC_IMPORTERS = {"scripts/measure_1d1b2_pilots.py", "scripts/measure_1d1b3_graph_identity.py",
                        "scripts/measure_1d1b4_population.py"}


def test_shadow_mode_no_production_code_imports_the_new_modules():
    package = REPO / "src/genomic_variant_classifier/information_sources"
    new = {f"genomic_variant_classifier.information_sources.{m}" for m in NEW_MODULES}
    users = []
    for root in ("src", "scripts"):
        for path in sorted((REPO / root).rglob("*.py")):
            if path.parent == package:
                continue
            if _imports(path) & new:
                users.append(path.relative_to(REPO).as_posix())
    assert set(users) == DIAGNOSTIC_IMPORTERS


def test_only_the_absence_census_consumes_the_feature_contract_and_it_is_classified_diagnostic(ledger):
    sites = K.discover_sites(REPO)
    base = "src/genomic_variant_classifier/information_sources/"
    for name in ("admission", "observation", "lineage", "columns", "materialization", "graph_identity", "labels",
                 "population", "claims"):
        assert f"{base}{name}.py" not in sites
    for site in (f"{base}absence_census.py", "scripts/measure_1d1b2_pilots.py"):
        assert ledger["consumer_sites"][site]["classes"] == ["diagnostic"], site


def test_the_contract_modules_import_without_the_model_training_stack():
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'numpy', 'pandas', 'torch', 'sklearn', 'xgboost', 'lightgbm', 'catboost',\n"
        "                                  'networkx', 'torch_geometric', 'scipy'}:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from genomic_variant_classifier.information_sources import observation as O, admission as A, lineage as L\n"
        "o = O.observed('f', 'r', 0.5, O.SourceIdentity('dataset:eve', 'a' * 64))\n"
        "assert o.value == 0.5 and A.effective_admission({'a': A.D.ADMIT_STRICT}, {}, A.REGIMES['strict']) == {'a': ()}\n"
        "print('ok')\n")
    env = {"PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
