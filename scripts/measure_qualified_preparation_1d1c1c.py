"""Set "av" (stage 1D-1c.1C): the qualified preparation measured on the REAL files -- the record
docs/census/QUALIFIED_PREPARATION_1D1c1C.json. Owner rulings aa999b96 (revised section 10) and ff77015c (sections 11
and 13). Author: Monzia Moodie.

The population is the sealed pilot's (scripts/measure_1d1b2_pilots.py::load_population, refused unless its version-1
identity is the pilot record's). The three sources are DECLARED by the committed allele-frequency ledger
(docs/census/AF_REPAIR_1D1c1.json, its ``inputs``): a file whose bytes differ is refused by name. The reference status
is materialised from the evidence for exactly this population; the census must equal the ledger's pre-registered
expectation (1,028,224 + 459,512 + 500 + 1,778 = 1,490,014; 1,487,736 sequence-resolved); the selection identity and the
reference result digest must equal the ledger's. No annotation source is read: every non-allele-frequency feature is the
builder's default, and the record says so -- its subject is the allele-frequency stage. No model is fitted; scientific
training stays refused. Changes nothing in production.

  python scripts/measure_qualified_preparation_1d1c1c.py --clinvar data/processed/clinvar_grch38_clean.parquet \\
      --gnomad data/processed/gnomad_v4_exomes.parquet --reference-evidence data/processed/clinvar_grch38_clean_seq.parquet \\
      --scratch SCRATCH --out docs/census/QUALIFIED_PREPARATION_1D1c1C.json

``--check`` re-measures and compares with the committed record byte for byte instead of writing (a difference is named
field by field, and with ``--measured-out PATH`` the measured record is kept there); ``--store DIR`` also writes the
content-addressed canonical matrix there (``<identity>/<payload>.parquet``). The record holds no machine-dependent bits:
af_log10 is certified, not sealed (qualified_preparation.certify_log10; CANONICAL-IDENTITY-ENVIRONMENT-1), and the
matrix's representation (its exact bits) is printed, not recorded.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SCHEMA = "gvc.qualified-preparation-record.1d1c1c/1"
SCRIPT = "scripts/measure_qualified_preparation_1d1c1c.py"
AF_LEDGER = "docs/census/AF_REPAIR_1D1c1.json"
PILOT_POPULATION_V1 = "d4085699adaf257ad032da6eb1284d5ea8d05d220defd7dc7144bfa2485d79f5"


def _pilots(repo: Path):
    spec = importlib.util.spec_from_file_location("measure_1d1b2_pilots", repo / "scripts" / "measure_1d1b2_pilots.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_1d1b2_pilots"] = mod
    spec.loader.exec_module(mod)
    return mod


def measure(args: argparse.Namespace) -> tuple[dict, object]:
    import numpy as np

    from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline
    from genomic_variant_classifier.information_sources import allele_frequency as A
    from genomic_variant_classifier.information_sources import census as C
    from genomic_variant_classifier.information_sources import materialization as M
    from genomic_variant_classifier.information_sources import qualified_preparation as Q
    from genomic_variant_classifier.information_sources import reference_status as R

    repo = Path(__file__).resolve().parents[1]
    registry = sorted(C.strict_load(repo / C.LEDGER_PATH)["datasets"])
    ledger_raw = (repo / AF_LEDGER).read_bytes()
    ledger = C.strict_loads(ledger_raw.decode("ascii"))
    declared = {Q.COHORT: ledger["inputs"]["cohort"]["sha256"], A.PRIMARY: ledger["inputs"]["gnomad"]["sha256"],
                R.EVIDENCE_DATASET: ledger["inputs"]["reference_evidence"]["sha256"]}

    cohort = M.read_verified_bytes(args.clinvar, expected_sha256=declared[Q.COHORT])
    pilots = _pilots(repo)
    frame, _unread = pilots.load_population(DataPrepPipeline(DataPrepConfig(output_dir=args.scratch)), cohort.value)
    cohort_sha, cohort_bytes = cohort.sha256, cohort.n_bytes
    del cohort
    import pyarrow as pa

    pa.default_memory_pool().release_unused()     # the cohort read's Arrow buffers, freed before the builder runs
    frame = frame.reset_index(drop=True)
    population = M.population_from_frame(frame, pilots.POPULATION_ID)
    if population.sha256_v1 != PILOT_POPULATION_V1:
        raise SystemExit(f"the population is not the pilot's: version-1 identity {population.sha256_v1}")
    cohort_af = int(frame["allele_freq"].notna().sum()) if "allele_freq" in frame.columns else None
    gnomad = M.read_verified_bytes(args.gnomad, expected_sha256=declared[A.PRIMARY])
    evidence = M.read_verified_bytes(args.reference_evidence, expected_sha256=declared[R.EVIDENCE_DATASET])
    status = R.materialize_reference_status(population, evidence, registered_datasets=registry)
    evidence_bytes = evidence.n_bytes
    del evidence
    policy = A.AFSelectionPolicy(unknown_base_treatment=A.UnknownBaseTreatment.EXCLUDE_FROM_SEQUENCE_RESOLVED_PRIMARY)
    cm = Q.prepare_qualified(frame, population, mode=Q.PreparationMode.QUALIFIED, cohort_sha256=cohort_sha,
                             gnomad=gnomad, reference=status, reference_pin=R.GRCH38_PRIMARY_ASSEMBLY, policy=policy,
                             registered_datasets=registry, declared_sources=declared, expected_census=ledger["expected"])
    receipt = cm.receipt
    if receipt["selection"]["identity"] != ledger["selection_identity"]:
        raise SystemExit("the selection identity differs from the allele-frequency ledger's")
    if status.result_sha256 != ledger["reference_evidence"]["rows_sha256"]:
        raise SystemExit("the reference result digest differs from the allele-frequency ledger's")
    verdicts = status.identity_payload()["validation_result"]["verdicts"]
    if verdicts != ledger["reference_evidence"]["verdicts"]:
        raise SystemExit("the reference verdict counts differ from the allele-frequency ledger's")
    # Chunked building against ONE call of the builder on the first rows (the whole cohort in one call needs about
    # 4.5 GB above its input), bit for bit.
    from genomic_variant_classifier.models.feature_semantics import FeatureSemantics
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    head_n = min(args.compare_rows, len(population))
    readable = Q.builder_input_names()
    head = frame.iloc[:head_n][[c for c in frame.columns if c in readable and c not in ("label", "allele_freq")]]
    head = head.assign(allele_freq=cm.features["af_raw"].to_numpy()[:head_n])
    one_call = engineer_features(head, semantics=FeatureSemantics.V2_OBSERVED_AF)
    for c in cm.features.columns:
        a = one_call[c].to_numpy(dtype=np.float64)
        b = cm.features[c].to_numpy()[:head_n]
        if not (np.array_equal(np.isnan(a), np.isnan(b)) and
                np.array_equal(a[~np.isnan(a)].view(np.uint64), b[~np.isnan(b)].view(np.uint64))):
            raise SystemExit(f"{c}: chunked building differs from one call on the first {head_n} rows")
    del head, one_call
    sel = cm.selection == A.SELECTION_STATES.index("selected")
    family = {}
    for name in ("af_raw", "af_log10", "af_is_observed_zero", "af_is_ultra_rare", "af_is_rare", "af_is_common"):
        v = cm.features[name].to_numpy()
        family[name] = {"missing": int(np.isnan(v).sum()), "missing_equals_unselected": bool((np.isnan(v) == ~sel).all()),
                        "ones": int((v == 1.0).sum()) if name != "af_raw" and name != "af_log10" else None}
    record = {
        "schema": SCHEMA,
        "inputs": {"cohort": {"logical_source": Q.COHORT, "sha256": cohort_sha, "bytes": cohort_bytes,
                              "allele_freq_non_missing_in_population": cohort_af},
                   "gnomad": {"logical_source": A.PRIMARY, "sha256": gnomad.sha256, "bytes": gnomad.n_bytes},
                   "reference_evidence": {"logical_source": R.EVIDENCE_DATASET, "sha256": status.source_sha256,
                                          "bytes": evidence_bytes},
                   "declared_by": {"record": AF_LEDGER, "sha256": C.sha256_bytes(ledger_raw)}},
        "population": {"population_id": population.population_id, "rows": len(population),
                       "sha256_v1": population.sha256_v1, "sha256_v2": population.sha256},
        "reference_status": {"identity": status.identity, "payload": status.identity_payload()},
        "canonical_matrix": {"identity": cm.identity, "receipt": dict(receipt),
                             "exact_columns": Q.exact_column_digests(cm.features)},
        "reconciliation": {"census": receipt["selection"]["census"], "expected": dict(ledger["expected"]),
                           "selection_identity_equals_af_ledger": True, "reference_digest_equals_af_ledger": True},
        "af_family": family,
        "chunking": {"rows_per_chunk": 262_144, "row_local_proof_sha256": Q.row_local_proof(),
                     "compared_with_one_call": {"rows": head_n, "columns": len(cm.features.columns), "identical": True}},
        "scope": ("allele-frequency stage only: no annotation source is read, so every other feature is the builder's "
                  "default; no model is fitted; scientific training is refused"),
        "procedure": {SCRIPT + "::measure": C.symbol_closure_digest(repo, SCRIPT + "::measure")},
    }
    return record, cm


def _differences(committed, measured, path: str = "") -> list[str]:
    """The paths at which two records differ (so a mismatch names its field, never only a digest)."""
    if isinstance(committed, dict) and isinstance(measured, dict):
        out = []
        for k in sorted(set(committed) | set(measured)):
            if k not in committed or k not in measured:
                out.append(f"{path}/{k}: present in only one record")
            else:
                out += _differences(committed[k], measured[k], f"{path}/{k}")
        return out
    return [] if committed == measured else [f"{path}: committed {str(committed)[:100]!r}, measured {str(measured)[:100]!r}"]


def main(argv=None) -> int:
    from genomic_variant_classifier.information_sources import census as C
    from genomic_variant_classifier.information_sources import qualified_preparation as Q

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clinvar", type=Path, required=True)
    ap.add_argument("--gnomad", type=Path, required=True)
    ap.add_argument("--reference-evidence", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True, help="an empty directory for DataPrepPipeline's output_dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="compare with --out byte for byte instead of writing")
    ap.add_argument("--store", type=Path, help="also write the content-addressed canonical matrix here")
    ap.add_argument("--measured-out", type=Path, help="with --check: where a DIFFERING measured record is written")
    ap.add_argument("--compare-rows", type=int, default=300_000, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.measured_out is not None and (not a.check or a.measured_out.exists()):
        raise SystemExit("--measured-out goes with --check, to a path that does not exist yet")
    record, cm = measure(a)
    raw = C.ledger_bytes(record)
    worst = Q.certify_log10(cm.features["af_raw"].to_numpy(), cm.features["af_log10"].to_numpy())
    print(f"af_log10 on this machine: at most {worst:.3f} ULP from log10 to {Q.REFERENCE_PRECISION_BITS} bits "
          f"(registered bound {Q.LOG10_ULP_BOUND}); its bits are bound by the certificate, not sealed; this machine's "
          f"representation {cm.representation}")
    if a.store is not None:
        stored = Q.write_canonical(cm, a.store)
        print(f"canonical matrix stored: identity {stored.identity}, payload {stored.payload_sha256} in {a.store}")
    if a.check:
        committed = a.out.read_bytes()
        if committed != raw:
            print(f"DIFFERS: {a.out} (committed sha256 {C.sha256_bytes(committed)}, measured {C.sha256_bytes(raw)})")
            for line in _differences(C.strict_loads(committed.decode("ascii")), C.strict_loads(raw.decode("ascii")))[:40]:
                print(f"  differs at {line}")
            if a.measured_out is not None:
                a.measured_out.write_bytes(raw)
                print(f"measured record written: {a.measured_out}")
            return 1
        print(f"IDENTICAL: {a.out} sha256 {C.sha256_bytes(raw)}")
        return 0
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    a.out.write_bytes(raw)
    print(f"wrote {a.out} sha256 {C.sha256_bytes(raw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
