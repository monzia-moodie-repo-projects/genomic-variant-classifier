"""Stage 1D-1b.2 (set "aq"): lineage integrity -- ADMISSION-CYCLIC-LINEAGE-1 refused by two independent mechanisms.
Owner ruling 7f4b69d0 (2026-11-09), section 3. Author: Monzia Moodie."""
from __future__ import annotations

from pathlib import Path

import pytest

from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import lineage as L

REPO = Path(__file__).resolve().parents[2]


def test_every_cyclic_component_is_named_at_once_self_loops_included():
    deps = {"a": ("b",), "b": ("a",), "c": ("c",), "d": ("e",), "e": (), "f": ("g",), "g": ("h",), "h": ("f",)}
    assert L.cyclic_components(deps) == [["a", "b"], ["c"], ["f", "g", "h"]]
    with pytest.raises(ValueError, match=r"a <-> b; c; f <-> g <-> h"):
        L.require_acyclic(deps)
    assert L.cyclic_components({"x": (), "y": ("x",)}) == []


def test_undeclared_or_repeated_dependencies_refuse():
    with pytest.raises(ValueError, match="undeclared"):
        L.cyclic_components({"x": ("nowhere",)})
    with pytest.raises(ValueError, match="listed twice"):
        L.cyclic_components({"x": ("y", "y"), "y": ()})


def test_the_structural_check_has_no_recursion_limit():
    deps = {f"n{i:05d}": (f"n{i + 1:05d}",) for i in range(5000)}
    deps["n05000"] = ()
    assert L.cyclic_components(deps) == []
    order = L.dependency_order(deps)
    assert order[0] == "n05000" and order[-1] == "n00000" and len(order) == 5001


def test_dependency_order_is_deterministic_and_dependencies_first():
    deps = {"z": (), "a": ("z",), "m": ("a", "z"), "b": ()}
    assert L.dependency_order(deps) == ("b", "z", "a", "m")
    with pytest.raises(ValueError, match="cyclic information lineage"):
        L.dependency_order({"p": ("q",), "q": ("p",)})


def test_the_least_fixed_point_never_admits_a_cycle_even_without_the_structural_check():
    deps = {"a": ("b",), "b": ("a",), "self": ("self",), "src": (), "ok": ("src",), "down": ("a",)}
    adm = L.least_fixed_point({k: True for k in deps}, deps)
    assert adm == {"a": False, "b": False, "self": False, "src": True, "ok": True, "down": False}


def test_least_and_greatest_fixed_points_agree_on_an_acyclic_graph():
    deps = {"s1": (), "s2": (), "x": ("s1", "s2"), "y": ("x",), "z": ("s2",)}
    disp = {"s1": True, "s2": False, "x": True, "y": True, "z": True}
    assert L.least_fixed_point(disp, deps) == {"s1": True, "s2": False, "x": False, "y": False, "z": False}
    with pytest.raises(ValueError, match="same identities"):
        L.least_fixed_point({"s1": True}, deps)


def test_transitive_lineage_and_grounding():
    deps = {"leaf": (), "mid": ("leaf",), "top": ("mid", "leaf"), "loose": ()}
    assert L.ancestors("top", deps) == ("leaf", "mid") and L.ancestors("leaf", deps) == ()
    src = {"leaf": ["dataset:eve", "code:x.py::f"], "loose": ["dataset:not_registered"], "mid": [], "top": []}
    assert L.ungrounded_leaves(src, deps, ["eve", "gnomad_af"]) == ["loose"]
    with pytest.raises(ValueError, match="unknown identity"):
        L.ancestors("nowhere", deps)


def test_the_census_lineage_is_acyclic_and_grounded():
    ledger = C.strict_load(REPO / C.LEDGER_PATH)
    deps = {e["name"]: tuple(e["depends_on"]) for e in ledger["entries"]}
    deps.update({s["name"]: () for s in ledger["information_sources"]})
    assert L.cyclic_components(deps) == []
    sources = {r["name"]: r["source_ids"] for r in [*ledger["entries"], *ledger["information_sources"]]}
    assert L.ungrounded_leaves(sources, deps, ledger["datasets"]) == []
    assert len(deps) == 102 and sum(len(v) for v in deps.values()) == 212
    assert sum(1 for v in deps.values() if not v) == 85


def test_the_census_gate_refuses_a_cycle_and_an_ungrounded_leaf():
    import copy

    ledger = C.strict_load(REPO / C.LEDGER_PATH)
    report = C.verify_census(REPO, ledger)                # the committed ledger passes
    assert report.semantic_digest
    cyclic = copy.deepcopy(ledger)
    by_name = {e["name"]: e for e in cyclic["entries"]}
    by_name["af_raw"]["depends_on"] = ["af_log10"]          # af_log10 already depends on af_raw: a 2-cycle
    with pytest.raises(ValueError, match="cyclic information lineage.*af_log10 <-> af_raw"):
        C.verify_census(REPO, cyclic)
    loose = copy.deepcopy(ledger)
    entry = next(e for e in loose["entries"] if e["name"] == "af_raw")
    entry["source_ids"] = [s for s in entry["source_ids"] if not s.startswith("dataset:")]
    with pytest.raises(ValueError, match=r"lineage leaves name no registered dataset: \['af_raw'\]"):
        C.verify_census(REPO, loose)
