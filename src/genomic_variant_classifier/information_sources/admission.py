"""Stage 1D-1b.1: semantic admission derived from the merged census, reported in SHADOW MODE.

Owner ruling f2ec92cd (2026-11-09), section XIII. Three things are kept apart, because they are not interchangeable:

    scientifically admitted       the reviewed disposition in docs/census/FEATURE_CENSUS_1D1a.json, under a named regime;
    actually observed             whether the producer recorded that a value was measured (none does yet: 1D-1b.2);
    available to the fitted model the PREDICTIVE surfaces the census measured from the code as consuming the name.

The report lists, for every feature and every non-tabular information source, all three, and records as a VIOLATION
every name that reaches a predictive surface without being admitted under the regime -- with its exact identity, its
disposition and the reason. Nothing is enforced and nothing is changed: no model, matrix, split or prediction is read or
written (shadow mode). Stage 1D-1c turns the violations into refusals.

Two refinements over the ruling's sketch, each tested:

* ONE authority. The disposition vocabulary is the census's own ``Disposition`` (minus ``pending_review``, which the
  census refuses), not a second hand-maintained enum; the report is computed from the census only after the census gate
  (``verify_census``) has re-measured the repository, so a stale or edited ledger cannot produce a report.
* Admission is TRANSITIVE under every regime. The census enforced monotone admission only for ``admit_strict``. Under the
  conditional regime ``gnn_score`` is ``conditional_only`` but its measured dependencies include the retired
  ``n_pathogenic_in_gene`` and the excluded STRING combined-score relation, so it is not effectively admissible there
  either. Effective admission is the greatest fixed point: a name is admitted iff its disposition is admitted by the
  regime AND every dependency is effectively admitted.

The module imports only the standard library and the census module (itself standard-library at import time); verifying
the census, which the command line does, runs the repository's feature builder and therefore needs pandas.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from genomic_variant_classifier.information_sources import census as _census

logger = logging.getLogger(__name__)

SCHEMA = "gvc.admission-shadow.1d1b1/1"
REPORT_PATH = "docs/census/ADMISSION_SHADOW_1D1b1.json"
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

#: What 1D-1b.1 can say about observation: no producer records an observation state yet (that is 1D-1b.2's work).
OBSERVATION_NOT_RECORDED = "not_recorded_by_producer"


def surface_classes_cover(surfaces: Sequence[str], graph_consumers: Sequence[str]) -> None:
    """The classification must cover exactly the census's surfaces and graph consumers -- no more, no fewer."""
    expected = set(surfaces) | set(graph_consumers)
    if set(SURFACE_CLASS) != expected:
        raise ValueError(f"surface classes must cover exactly the measured consumers: missing "
                         f"{sorted(expected - set(SURFACE_CLASS))}, extra {sorted(set(SURFACE_CLASS) - expected)}")


def effective_admission(dispositions: Mapping[str, D], depends_on: Mapping[str, Sequence[str]],
                        admitted: frozenset[D]) -> dict[str, tuple[str, ...]]:
    """Greatest fixed point of "admitted iff the disposition is admitted and every dependency is admitted".

    Returns {name: blocking dependencies}; a name is effectively admitted iff its tuple is empty AND its own disposition is
    admitted. A blocking dependency is a DIRECT dependency that is not effectively admitted."""
    unknown = sorted({d for deps in depends_on.values() for d in deps} - set(dispositions))
    if unknown:
        raise ValueError(f"dependencies name unknown entries: {unknown}")
    ok = {n for n, d in dispositions.items() if d in admitted}
    changed = True
    while changed:
        changed = False
        for name in sorted(ok):
            if any(dep not in ok for dep in depends_on.get(name, ())):
                ok.discard(name)
                changed = True
    return {n: tuple(sorted(d for d in depends_on.get(n, ()) if d not in ok)) for n in dispositions}


@dataclass(frozen=True)
class Violation:
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
    depends_on = {r["name"]: tuple(r.get("depends_on", ())) for _k, r in records}
    defects: dict[str, list[str]] = {}
    for d in ledger["registered_defects"]:
        for name in d["affects"]:
            defects.setdefault(name, []).append(d["id"])

    blocking = {regime: effective_admission(dispositions, depends_on, admitted) for regime, admitted in REGIMES.items()}
    rows: list[dict[str, Any]] = []
    violations: list[Violation] = []
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
                violations.append(Violation(regime, name, kind, disp.value, reason, predictive))
        rows.append({
            "name": name, "kind": kind, "disposition": disp.value, "admission": admission,
            "observation": OBSERVATION_NOT_RECORDED,
            "available_to_fitted_model": list(predictive),
            "computed_or_stored_only": [c for c in consumers_ if SURFACE_CLASS[c] != "predictive"],
            "registered_defects": sorted(defects.get(name, [])),
        })

    counts: dict[str, dict[str, int]] = {}
    for regime in REGIMES:
        admitted_n = sum(1 for row in rows if row["admission"][regime] == "admitted")
        regime_violations = [v for v in violations if v.regime == regime]
        counts[regime] = {
            "admitted": admitted_n, "not_admitted": len(rows) - admitted_n,
            "violations": len(regime_violations),
            "violations_by_disposition": dict(sorted(
                (k, sum(1 for v in regime_violations if v.disposition == k))
                for k in {v.disposition for v in regime_violations})),
            "transitive_only": sum(1 for v in regime_violations if v.reason.startswith("depends on")),
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
            "meaning": "no producer records whether a value was measured or filled; every feature's observation is "
                       "therefore unknown, not observed (stage 1D-1b.2 makes producers emit "
                       "information_sources.observation.NumericObservation)"},
        "counts": counts,
        "rows": rows,
        "violations": [v.as_dict() for v in violations],
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
    parser = argparse.ArgumentParser(description="Stage 1D-1b.1 shadow admission report (reads the census; changes nothing).")
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
        print(f"{regime} regime: {c['admitted']} admitted, {c['not_admitted']} not admitted; {c['violations']} reach a "
              f"predictive surface unadmitted ({c['transitive_only']} only through a dependency)")
    print(f"observation: {OBSERVATION_NOT_RECORDED} for all {len(report['rows'])} names")
    print("mode: SHADOW -- nothing enforced, nothing changed")
    print(report["scientific_truth"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
