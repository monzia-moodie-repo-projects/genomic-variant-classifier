"""Stages 1D-1b.1 / 1D-1b.2: semantic admission derived from the merged census, reported in SHADOW MODE.

Owner rulings f2ec92cd (section XIII) and 7f4b69d0 (sections 2.2, 3 and 19). Things that are not interchangeable are kept
apart:

    scientifically admitted   the reviewed disposition in docs/census/FEATURE_CENSUS_1D1a.json, under a named regime;
    observation               whether a producer records, per row, that a value was measured (two shadow materialisers do
                              in 1D-1b.2: information_sources/materialization.py; no production producer does yet);
    exposure                  DECLARED (in the feature contract or the measured graph inventory) and STATICALLY_REACHABLE
                              (the predictive surfaces the census measured from the code as consuming the name) are
                              established here; MATERIALIZED, FITTED_INPUT and INFERENCE_INPUT need execution evidence
                              (materialisation receipts, fitted-model artefacts) and are reported NOT ESTABLISHED.

The report lists all of them for every feature and every non-tabular information source, and records as a STRUCTURAL
EXPOSURE RISK every name that is statically reachable by a predictive surface without being admitted under the regime --
with its exact identity, its disposition and the reason. That is a structural statement about the code, not a claim that
the name was consumed by a particular fit (ruling 7f4b69d0, section 2.2). Nothing is enforced and nothing is changed: no
model, matrix, split or prediction is read or written. Stage 1D-1c turns the risks into refusals.

* ONE authority. The disposition vocabulary is the census's own ``Disposition`` (minus ``pending_review``, which the
  census refuses); the report is computed from the census only after the census gate (``verify_census``) has re-measured
  the repository, so a stale or edited ledger cannot produce a report.
* Lineage is valid BEFORE admission is computed (ADMISSION-CYCLIC-LINEAGE-1, lineage.py): every dependency declared, no
  cycle (every cyclic component named), every leaf grounded in a registered dataset. Admission is then the LEAST fixed
  point, evaluated bottom-up in dependency order: a name is admitted iff its disposition is admitted by the regime AND
  every dependency was admitted first. A cycle can therefore never admit itself, even past the structural check. On the
  acyclic census this equals the 1D-1b.1 result exactly (55 strict, 65 conditional; tested).
* Admission is TRANSITIVE under every regime: under the conditional regime ``gnn_score`` is ``conditional_only`` but its
  measured dependencies include the retired ``n_pathogenic_in_gene`` and the excluded STRING combined-score relation.

The module imports only the standard library and the census and lineage modules (standard-library at import time);
verifying the census, which the command line does, runs the repository's feature builder and therefore needs pandas.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from enum import Enum

from genomic_variant_classifier.information_sources import census as _census
from genomic_variant_classifier.information_sources import lineage as _lineage

logger = logging.getLogger(__name__)

SCHEMA = "gvc.admission-shadow.1d1b2/1"
REPORT_PATH = "docs/census/ADMISSION_SHADOW_1D1b2.json"
D = _census.Disposition

#: The reviewed dispositions: the census vocabulary without pending_review, which the census gate refuses.
REVIEWED: tuple[D, ...] = tuple(d for d in D if d is not D.PENDING_REVIEW)

#: Scientific regimes and the dispositions each admits. "strict" is the leakage-free baseline's regime (1D-2);
#: "conditional" additionally admits the conditional_only entries, whose conditions 1D-1b.4 will derive per claim.
REGIMES: Mapping[str, frozenset[D]] = {
    "strict": frozenset({D.ADMIT_STRICT}),
    "conditional": frozenset({D.ADMIT_STRICT, D.CONDITIONAL_ONLY}),
}

#: Every consumer the census measures, classified once. PREDICTIVE: the name's values shape a fitted model or its output.
#: NOT predictive: the name is computed or stored, but no model is fitted on it there.
SURFACE_CLASS: Mapping[str, str] = {
    "feature_engineering": "computed",
    "split_persistence": "stored",
    "export": "stored",
    "tabular_fit": "predictive",
    "scaler_fit": "predictive",
    "gnn_node_attributes": "predictive",
    "hetero_gnn_node_attributes": "predictive",
    "inference": "predictive",
    "calibration": "predictive",
    "evaluation": "predictive",
    "continual_training": "predictive",
    "gnn_edges": "predictive",
    "gnn_edge_attributes": "predictive",
    "gnn_supervision": "predictive",
    "hetero_gnn_relations": "predictive",
    "hetero_gnn_supervision": "predictive",
}

#: No production producer records an observation state yet.
OBSERVATION_NOT_RECORDED = "not_recorded_by_producer"
#: The features a SHADOW materialiser (information_sources/materialization.py) observes per row, with its function. A test
#: checks each function exists there; the materialisers are not wired into production.
SHADOW_MATERIALIZERS: Mapping[str, str] = {
    "af_raw": "src/genomic_variant_classifier/information_sources/materialization.py::materialize_gnomad_af",
    "eve_score": "src/genomic_variant_classifier/information_sources/materialization.py::materialize_eve",
}
NOT_ESTABLISHED = "not_established"


class InputExposure(str, Enum):
    """Ruling 7f4b69d0, section 2.2: what is known about a name reaching a model, weakest to strongest evidence."""
    DECLARED = "declared"                          # in the feature contract or the measured graph inventory
    STATICALLY_REACHABLE = "statically_reachable"  # a predictive surface's code consumes it (census measurement)
    MATERIALIZED = "materialized"                  # a materialisation receipt records it for a population
    FITTED_INPUT = "fitted_input"                  # a fitted model artefact records it as an input
    INFERENCE_INPUT = "inference_input"            # an inference record shows it consumed


#: The evidence each exposure level needs, and whether this report can establish it.
EXPOSURE_EVIDENCE: Mapping[InputExposure, str] = {
    InputExposure.DECLARED: "the census ledger (features: the contract; sources: the measured graph inventory)",
    InputExposure.STATICALLY_REACHABLE: "the census's measured consumer surfaces classified predictive",
    InputExposure.MATERIALIZED: "a materialisation receipt for a named population (none in production; the 1D-1b.2 "
                                "pilot receipts are shadow measurements, docs/census/PILOTS_1D1b2.json)",
    InputExposure.FITTED_INPUT: "a fitted model artefact's recorded input columns (the model execution authority)",
    InputExposure.INFERENCE_INPUT: "an inference record of the rendered input columns",
}


def surface_classes_cover(surfaces: Sequence[str], graph_consumers: Sequence[str]) -> None:
    """The classification must cover exactly the census's surfaces and graph consumers -- no more, no fewer."""
    expected = set(surfaces) | set(graph_consumers)
    if set(SURFACE_CLASS) != expected:
        raise ValueError(f"surface classes must cover exactly the measured consumers: missing "
                         f"{sorted(expected - set(SURFACE_CLASS))}, extra {sorted(set(SURFACE_CLASS) - expected)}")


def effective_admission(dispositions: Mapping[str, D], depends_on: Mapping[str, Sequence[str]],
                        admitted: frozenset[D]) -> dict[str, tuple[str, ...]]:
    """Least fixed point of "admitted iff the disposition is admitted and every dependency is admitted FIRST", after the
    lineage is shown acyclic (refused otherwise, every cyclic component named).

    Returns {name: blocking dependencies}; a name is effectively admitted iff its tuple is empty AND its own disposition is
    admitted. A blocking dependency is a DIRECT dependency that is not effectively admitted."""
    unknown = sorted({d for deps in depends_on.values() for d in deps} - set(dispositions))
    if unknown:
        raise ValueError(f"dependencies name unknown entries: {unknown}")
    deps = {n: tuple(depends_on.get(n, ())) for n in dispositions}
    _lineage.require_acyclic(deps)
    ok = _lineage.least_fixed_point({n: d in admitted for n, d in dispositions.items()}, deps)
    return {n: tuple(sorted(d for d in deps[n] if not ok[d])) for n in dispositions}


def lineage_summary(records: Sequence[Mapping[str, Any]], datasets: Sequence[str]) -> dict[str, Any]:
    """The lineage graph's structural facts; refuses a cycle or an ungrounded leaf (ADMISSION-CYCLIC-LINEAGE-1)."""
    deps = {r["name"]: tuple(r.get("depends_on", ())) for r in records}
    _lineage.require_acyclic(deps)
    ungrounded = _lineage.ungrounded_leaves({r["name"]: r["source_ids"] for r in records}, deps, datasets)
    if ungrounded:
        raise ValueError(f"lineage leaves name no registered dataset: {ungrounded}")
    order = _lineage.dependency_order(deps)
    return {"identities": len(deps), "dependency_edges": sum(len(v) for v in deps.values()),
            "leaves": sum(1 for v in deps.values() if not v), "cyclic_components": [], "ungrounded_leaves": [],
            "registered_datasets": len(datasets),
            "dependency_order_sha256": _census.sha256_bytes(_census.canonical_bytes(list(order)))}


@dataclass(frozen=True)
class ExposureRisk:
    regime: str
    name: str
    kind: str
    disposition: str
    reason: str
    predictive_surfaces: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"regime": self.regime, "name": self.name, "kind": self.kind, "disposition": self.disposition,
                "reason": self.reason, "predictive_surfaces": list(self.predictive_surfaces)}


def shadow_report(ledger: Mapping[str, Any], *, ledger_sha256: str, contract_digest: str) -> dict[str, Any]:
    """The shadow admission report of a VERIFIED census ledger (call ``build`` to verify first)."""
    from genomic_variant_classifier.information_sources import consumers as _consumers

    surface_classes_cover(_consumers.SURFACES, _consumers.GRAPH_CONSUMERS)
    if not _census.HEX64.match(ledger_sha256) or not _census.HEX64.match(contract_digest):
        raise ValueError("the ledger and contract identities must be SHA-256 digests")
    records: list[tuple[str, Mapping[str, Any]]] = [("feature", e) for e in ledger["entries"]]
    records += [(s["kind"], s) for s in ledger["information_sources"]]
    dispositions: dict[str, D] = {}
    for _kind, r in records:
        disp = D(r["disposition"])
        if disp not in REVIEWED:
            raise ValueError(f"{r['name']}: unreviewed disposition {disp.value}")
        if r["name"] in dispositions:
            raise ValueError(f"{r['name']}: named twice in the census")
        dispositions[r["name"]] = disp
    lineage = lineage_summary([r for _k, r in records], sorted(ledger["datasets"]))
    depends_on = {r["name"]: tuple(r.get("depends_on", ())) for _k, r in records}
    defects: dict[str, list[str]] = {}
    for d in ledger["registered_defects"]:
        for name in d["affects"]:
            defects.setdefault(name, []).append(d["id"])

    blocking = {regime: effective_admission(dispositions, depends_on, admitted) for regime, admitted in REGIMES.items()}
    rows: list[dict[str, Any]] = []
    risks: list[ExposureRisk] = []
    for kind, r in records:
        name, disp = r["name"], dispositions[r["name"]]
        consumers_ = sorted(r["consumers"])
        unknown = [c for c in consumers_ if c not in SURFACE_CLASS]
        if unknown:
            raise ValueError(f"{name}: consumed by unclassified surfaces {unknown}")
        predictive = tuple(c for c in consumers_ if SURFACE_CLASS[c] == "predictive")
        admission: dict[str, str] = {}
        for regime, admitted in REGIMES.items():
            blockers = blocking[regime][name]
            if disp not in admitted:
                status, reason = "not_admitted", f"disposition {disp.value} is not admitted under the {regime} regime"
            elif blockers:
                status, reason = "not_admitted", f"depends on entries not admitted under the {regime} regime: " + \
                    ", ".join(blockers)
            else:
                status, reason = "admitted", ""
            admission[regime] = status
            if status == "not_admitted" and predictive:
                risks.append(ExposureRisk(regime, name, kind, disp.value, reason, predictive))
        materializer = SHADOW_MATERIALIZERS.get(name)
        rows.append({
            "name": name, "kind": kind, "disposition": disp.value, "admission": admission,
            "observation": f"shadow_materializer:{materializer}" if materializer else OBSERVATION_NOT_RECORDED,
            "exposure": {InputExposure.DECLARED.value: True,
                         InputExposure.STATICALLY_REACHABLE.value: list(predictive),
                         InputExposure.MATERIALIZED.value: NOT_ESTABLISHED,
                         InputExposure.FITTED_INPUT.value: NOT_ESTABLISHED,
                         InputExposure.INFERENCE_INPUT.value: NOT_ESTABLISHED},
            "computed_or_stored_only": [c for c in consumers_ if SURFACE_CLASS[c] != "predictive"],
            "registered_defects": sorted(defects.get(name, [])),
        })

    counts: dict[str, dict[str, int]] = {}
    for regime in REGIMES:
        admitted_n = sum(1 for row in rows if row["admission"][regime] == "admitted")
        regime_risks = [v for v in risks if v.regime == regime]
        counts[regime] = {
            "admitted": admitted_n, "not_admitted": len(rows) - admitted_n,
            "structural_exposure_risks": len(regime_risks),
            "risks_by_disposition": dict(sorted(
                (k, sum(1 for v in regime_risks if v.disposition == k))
                for k in {v.disposition for v in regime_risks})),
            "transitive_only": sum(1 for v in regime_risks if v.reason.startswith("depends on")),
        }
    return {
        "schema": SCHEMA,
        "mode": "shadow: computed from the verified census; no model, matrix, split or prediction is read or changed",
        "census": {"path": _census.LEDGER_PATH, "ledger_sha256": ledger_sha256, "contract_digest": contract_digest,
                   "semantic_digest": _census.semantic_contract_digest(ledger), "schema": ledger["schema"]},
        "regimes": {k: sorted(d.value for d in v) for k, v in REGIMES.items()},
        "surface_classes": dict(sorted(SURFACE_CLASS.items())),
        "observation_recording": {
            "state": OBSERVATION_NOT_RECORDED,
            "shadow_materializers": dict(sorted(SHADOW_MATERIALIZERS.items())),
            "meaning": "no production producer records whether a value was measured or filled; every feature's "
                       "observation is therefore unknown, not observed. Two shadow materialisers (EVE, gnomAD allele "
                       "frequency) record it per row; they are not wired into production (stage 1D-1c)."},
        "exposure_levels": {e.value: EXPOSURE_EVIDENCE[e] for e in InputExposure},
        "lineage": lineage,
        "counts": counts,
        "rows": rows,
        "structural_exposure_risks": [v.as_dict() for v in risks],
        "structural_exposure_risk_meaning": (
            "a name that is not admitted under the regime and is STATICALLY REACHABLE by a predictive surface of the "
            "existing code; not a demonstration that a particular fit consumed it (that needs FITTED_INPUT evidence)"),
        "scientific_truth": _census.SCIENTIFIC_TRUTH_LINE,
    }


def build(repo: Path) -> dict[str, Any]:
    """Verify the census against THIS repository (fresh measurement), then derive the shadow report from it."""
    path = repo / _census.LEDGER_PATH
    raw = path.read_bytes()
    ledger = _census.strict_loads(raw.decode("ascii"))
    if _census.ledger_bytes(ledger) != raw:
        raise ValueError("the census ledger is not in its canonical committed form")
    _census.verify_census(repo, ledger)
    contract = _census.source_contract(repo)
    return shadow_report(ledger, ledger_sha256=_census.sha256_bytes(raw), contract_digest=contract.contract_digest)


def report_bytes(report: Mapping[str, Any]) -> bytes:
    return _census.ledger_bytes(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 1D-1b.2 shadow admission report (reads the census; changes nothing).")
    parser.add_argument("--repo", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", type=Path, help="write the report here (refuses to overwrite)")
    group.add_argument("--check", type=Path, help="exit 1 unless this file is byte-identical to a fresh derivation")
    args = parser.parse_args(argv)
    report = build(args.repo.resolve())
    data = report_bytes(report)
    if args.out is not None:
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite {args.out}")
        args.out.write_bytes(data)
    elif args.check.read_bytes() != data:
        print(f"{args.check}: DIFFERS from the report derived from the census now")
        return 1
    for regime, c in report["counts"].items():
        print(f"{regime} regime: {c['admitted']} admitted, {c['not_admitted']} not admitted; "
              f"{c['structural_exposure_risks']} statically reachable by a predictive surface unadmitted "
              f"({c['transitive_only']} only through a dependency)")
    lin = report["lineage"]
    print(f"lineage: {lin['identities']} identities, {lin['dependency_edges']} edges, {lin['leaves']} leaves; acyclic; "
          f"every leaf grounded in one of {lin['registered_datasets']} registered datasets")
    shadow = sum(1 for r in report["rows"] if r["observation"] != OBSERVATION_NOT_RECORDED)
    print(f"observation: {OBSERVATION_NOT_RECORDED} for {len(report['rows']) - shadow} names; {shadow} observed by a "
          "shadow materialiser only")
    print("mode: SHADOW -- nothing enforced, nothing changed")
    print(report["scientific_truth"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
