"""Build planning (owner rulings 2026-10-08 to 2026-10-10): the owner's reference tests VERBATIM from the ruling generations (graph 41474c47
L267-312; dependencies a4a5a4f9 L228-275 with Artifact -> DependencyArtifact; routes ce2726fc L214-265), admit_build's refusal table, and a FROZEN
real-data regression: the LinkingTo graph of the 83 admitted source archives and the 76 admitted binaries (acquisition run 2026-10-06 08:11:12 UTC).

Author: Monzia Moodie
"""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from genomic_variant_classifier.environment_qualification.build_plan import (
    BuildExpectation, BuildReceipt, DependencyArtifact, DependencyExpectation, DependencyObservation, admit_build, admit_dependencies,
    installation_order, rebuild_closure, select_routes)
from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError


# ------------------------------------------------------------------ graph (verbatim)

def test_transitive_rebuild():
    graph = {
        "Provider": set(),
        "Middle": {"Provider"},
        "Consumer": {"Middle"},
        "Independent": set(),
    }
    selected = rebuild_closure(
        graph,
        seeds={"Provider"},
        planned=set(graph),
        runtime_packages=set(),
    )
    assert set(selected) == {"Provider", "Middle", "Consumer"}
    assert selected["Consumer"] == (
        "Provider", "Middle", "Consumer"
    )


def test_dependency_first_order():
    graph = {
        "Provider": set(),
        "Middle": {"Provider"},
        "Consumer": {"Middle"},
    }
    order = installation_order(
        graph,
        planned=set(graph),
        runtime_packages=set(),
    )
    assert order.index("Provider") < order.index("Middle")
    assert order.index("Middle") < order.index("Consumer")


def test_missing_metadata_is_not_an_empty_dependency_set():
    try:
        rebuild_closure(
            {"Provider": set()},
            seeds={"Provider"},
            planned={"Provider", "Consumer"},
            runtime_packages=set(),
        )
    except ValueError as error:
        assert str(error).startswith("linking_to.coverage:")
    else:
        raise AssertionError("Incomplete metadata was accepted")


# ------------------------------------------------------------------ dependencies (verbatim; Artifact -> DependencyArtifact)



def test_modified_installed_dependency_is_refused():
    artifact = DependencyArtifact("S4Vectors", "0.50.1", "a" * 64)
    expected = DependencyExpectation(
        artifact=artifact,
        producer_receipt_sha256="b" * 64,
        installation_record_sha256="c" * 64,
        installed_tree_sha256="d" * 64,
        installed_location="C:/qualification/library/S4Vectors",
    )
    observed = DependencyObservation(
        package="S4Vectors",
        version="0.50.1",
        archive_sha256="a" * 64,
        producer_receipt_sha256="b" * 64,
        installation_record_sha256="c" * 64,
        installed_tree_sha256="e" * 64,  # Changed files.
        installed_location="C:/qualification/library/S4Vectors",
    )

    with pytest.raises(
        AdmissionError, match=r"^dependency\.installed_tree_mismatch$"
    ):
        admit_dependencies((expected,), (observed,))


def test_matching_archive_does_not_excuse_wrong_build_record():
    artifact = DependencyArtifact("S4Vectors", "0.50.1", "a" * 64)
    expected = DependencyExpectation(
        artifact, "b" * 64, "c" * 64, "d" * 64,
        "C:/qualification/library/S4Vectors",
    )
    observed = DependencyObservation(
        "S4Vectors", "0.50.1", "a" * 64,
        "b" * 64, "c" * 64, "d" * 64,
        "C:/qualification/library/S4Vectors",
    )

    admit_dependencies((expected,), (observed,))

    wrong = replace(observed, producer_receipt_sha256="f" * 64)
    with pytest.raises(
        AdmissionError, match=r"^dependency\.producer_receipt_mismatch$"
    ):
        admit_dependencies((expected,), (wrong,))


# ------------------------------------------------------------------ routes (verbatim)



def example():
    return dict(
        packages=frozenset({"A", "B", "C", "renv", "Matrix"}),
        runtime=frozenset({"Matrix"}),
        bootstrap=frozenset({"renv"}),
        source_available=frozenset({"A", "B", "C"}),
        binary_available=frozenset({"B", "C"}),
        linking_to={
            "A": frozenset(),
            "B": frozenset({"A"}),
            "C": frozenset({"B"}),
            "renv": frozenset(),
            "Matrix": frozenset(),
        },
        policy_roots=frozenset(),
    )


def test_missing_binary_triggers_transitive_rebuild():
    routes = select_routes(**example())
    assert routes == {
        "A": "local_build",
        "B": "local_build",
        "C": "local_build",
        "Matrix": "runtime",
        "renv": "bootstrap",
    }


def test_missing_source_is_not_silently_replaced_by_binary():
    inputs = example()
    inputs["source_available"] = frozenset({"A", "B"})

    with pytest.raises(
        ValueError,
        match=r"^plan\.required_source_unavailable$",
    ):
        select_routes(**inputs)


def test_complete_binary_coverage_avoids_unnecessary_builds():
    inputs = example()
    inputs["binary_available"] = frozenset({"A", "B", "C"})

    routes = select_routes(**inputs)
    assert all(
        routes[name] == "upstream_binary"
        for name in ("A", "B", "C")
    )


# ------------------------------------------------------------------ admit_build: the owner's refusal table (ruling 41474c47 L634-641)

def _expectation():
    return BuildExpectation("IRanges", "2.46.0", "a" * 64, "b" * 64, "c" * 64, "d" * 64, (("S4Vectors", "e" * 64),))


def _receipt(**changes):
    base = BuildReceipt("IRanges", "2.46.0", "a" * 64, "b" * 64, "c" * 64, "d" * 64, (("S4Vectors", "e" * 64),), 0, "f" * 64, "9" * 64)
    return replace(base, **changes)


def _admit(receipt, **independent):
    kw = dict(inspected_package="IRanges", inspected_version="2.46.0", inspected_binary_sha256="f" * 64, admitted_inspection_record_sha256="9" * 64)
    kw.update(independent)
    return admit_build(_expectation(), receipt, **kw)


def test_a_consistent_build_is_admitted():
    assert _admit(_receipt()) == "f" * 64


@pytest.mark.parametrize("receipt, independent, reason", [
    (dict(source_sha256="0" * 64), {}, "build.mismatch:source_sha256"),
    (dict(dependency_identities=(("S4Vectors", "1" * 64),)), {}, "build.dependency_mismatch"),
    (dict(exit_code=False), {}, "build.unsuccessful"),
    ({}, dict(inspected_version="2.46.1"), "build.output_identity_mismatch"),
    ({}, dict(inspected_binary_sha256="2" * 64), "build.output_bytes_mismatch"),
    (dict(dependency_identities=(("S4Vectors", "e" * 64), ("S4Vectors", "e" * 64))), {}, "build.observed_dependencies.duplicate:S4Vectors"),
])
def test_admit_build_refusals(receipt, independent, reason):
    with pytest.raises(ValueError) as error:
        _admit(_receipt(**receipt), **independent)
    assert str(error.value) == reason


# ------------------------------------------------------------------ the frozen REAL data: routes from the admitted artifacts

REAL = json.loads('{"binaries":["BH","Biobase","BiocFileCache","BiocGenerics","BiocIO","BiocManager","BiocParallel","BiocVersion","DBI","DelayedArray","GenomicAlignments","GenomicRanges","IRanges","MatrixGenerics","R.methodsS3","R.oo","R.utils","R6","RCurl","Rhtslib","Rsamtools","Seqinfo","SummarizedExperiment","XML","XVector","abind","askpass","bit","bit64","bitops","blob","cachem","cli","cpp11","crayon","curl","data.table","dplyr","fastmap","filelock","formatR","futile.logger","futile.options","generics","glue","httr","httr2","jsonlite","lambda.r","lifecycle","magrittr","matrixStats","memoise","mime","openssl","pillar","pkgconfig","purrr","rappdirs","recount3","restfulr","rjson","rlang","rtracklayer","sessioninfo","snow","stringi","stringr","sys","tibble","tidyr","tidyselect","utf8","vctrs","withr","yaml"],"linking_to":{"BH":[],"Biobase":[],"BiocFileCache":[],"BiocGenerics":[],"BiocIO":[],"BiocManager":[],"BiocParallel":["BH","cpp11"],"BiocVersion":[],"Biostrings":["IRanges","S4Vectors","XVector"],"DBI":[],"DelayedArray":[],"GenomicAlignments":["IRanges","S4Vectors"],"GenomicRanges":[],"IRanges":["S4Vectors"],"MatrixGenerics":[],"R.methodsS3":[],"R.oo":[],"R.utils":[],"R6":[],"RCurl":[],"RSQLite":["cpp11"],"Rhtslib":[],"Rsamtools":["Biostrings","IRanges","Rhtslib","S4Vectors","XVector"],"S4Arrays":["S4Vectors"],"S4Vectors":[],"Seqinfo":[],"SparseArray":["IRanges","S4Vectors","XVector"],"SummarizedExperiment":[],"XML":[],"XVector":["IRanges","S4Vectors"],"abind":[],"askpass":[],"bit":[],"bit64":[],"bitops":[],"blob":[],"cachem":[],"cigarillo":["IRanges","S4Vectors"],"cli":[],"cpp11":[],"crayon":[],"curl":[],"data.table":[],"dbplyr":[],"dplyr":[],"fastmap":[],"filelock":[],"formatR":[],"futile.logger":[],"futile.options":[],"generics":[],"glue":[],"httr":[],"httr2":[],"jsonlite":[],"lambda.r":[],"lifecycle":[],"magrittr":[],"matrixStats":[],"memoise":[],"mime":[],"openssl":[],"pillar":[],"pkgconfig":[],"purrr":["cli"],"rappdirs":[],"recount3":[],"restfulr":[],"rjson":[],"rlang":[],"rtracklayer":["IRanges","S4Vectors","XVector"],"sessioninfo":[],"snow":[],"stringi":[],"stringr":[],"sys":[],"tibble":[],"tidyr":["cpp11"],"tidyselect":[],"utf8":[],"vctrs":[],"withr":[],"yaml":[]}}')
RUNTIME = frozenset({"Matrix", "codetools", "lattice"})
BOOTSTRAP = frozenset({"renv"})


def test_the_admitted_artifacts_route_to_exactly_the_predicted_twelve_local_builds():
    linking = {p: frozenset(v) for p, v in REAL["linking_to"].items()}
    for p in RUNTIME | BOOTSTRAP:
        linking[p] = frozenset()
    routes = select_routes(packages=frozenset(linking), runtime=RUNTIME, bootstrap=BOOTSTRAP, source_available=frozenset(REAL["linking_to"]),
                           binary_available=frozenset(REAL["binaries"]), linking_to=linking, policy_roots=frozenset())
    counts = {r: sum(1 for v in routes.values() if v == r) for r in ("upstream_binary", "local_build", "runtime", "bootstrap")}
    assert counts == {"upstream_binary": 71, "local_build": 12, "runtime": 3, "bootstrap": 1}
    assert {p for p, r in routes.items() if r == "local_build"} == {"Biostrings", "GenomicAlignments", "IRanges", "RSQLite", "Rsamtools", "S4Arrays",
                                                                     "S4Vectors", "SparseArray", "XVector", "cigarillo", "dbplyr", "rtracklayer"}


def test_the_real_rebuild_closure_explains_every_dependent():
    linking = {p: set(v) for p, v in REAL["linking_to"].items()}
    reasons = rebuild_closure(linking, seeds={"Biostrings", "S4Arrays", "S4Vectors", "SparseArray", "cigarillo", "RSQLite", "dbplyr"},
                              planned=set(linking), runtime_packages=set())
    assert reasons["Rsamtools"] == ("Biostrings", "Rsamtools") and reasons["IRanges"] == ("S4Vectors", "IRanges") and len(reasons) == 12
