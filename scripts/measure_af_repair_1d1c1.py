"""Stage 1D-1c.1B: the allele-frequency repair measured on the REAL files -- the difference ledger
docs/census/AF_REPAIR_1D1c1.json. Owner rulings 482cd291, 4bef3910 and b937681e (2026-11-14). Author: Monzia Moodie.

The population is the one the sealed 1D-1b.2 pilot measured: DataPrepPipeline._load_and_label on the verified ClinVar
cohort bytes, through the pilot's own load_population (one authority), so its version-1 identity must equal
docs/census/PILOTS_1D1b2.json's population digest -- refused otherwise. Every input is read through
materialization.read_verified_bytes and recorded by digest; the record binds the code by semantic-closure digest. Labels
and split assignments are read only for the descriptive rare-flag table, after materialisation, which never sees them (the
record carries a label-permutation check). No model is fitted. Changes nothing in production.

  python scripts/measure_af_repair_1d1c1.py --clinvar data/processed/clinvar_grch38_clean.parquet \\
      --gnomad data/processed/gnomad_v4_exomes.parquet --reference-evidence data/processed/clinvar_grch38_clean_seq.parquet \\
      --splits outputs/run15_rerun_report/full/splits --out docs/census/AF_REPAIR_1D1c1.json

``--check`` re-measures and compares with the committed record byte for byte instead of writing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = "gvc.af-repair-ledger.1d1c1/1"
SCRIPT = "scripts/measure_af_repair_1d1c1.py"
AF = "src/genomic_variant_classifier/information_sources/allele_frequency.py"
PROCEDURE = (f"{SCRIPT}::measure", f"{AF}::read_source_tables", f"{AF}::materialize_allele_frequency",
             f"{AF}::select_strict", f"{AF}::audit_binary_flag", "scripts/measure_1d1b2_pilots.py::load_population")
#: Pre-registered from the owner's rulings (b937681e sections 4 and 21; 4bef3910 part two section 3). A difference is
#: investigated, never absorbed by changing these numbers.
EXPECTED = {"full": 1_490_014, "primary_domain": 1_488_236, "sequence_resolved_primary": 1_487_736, "selected": 1_028_224,
            "eligible_unselected": 459_512, "unknown_base": 500, "outside_domain": 1_778, "selected_reported_zero": 42_571}
PILOT_POPULATION_V1 = "d4085699adaf257ad032da6eb1284d5ea8d05d220defd7dc7144bfa2485d79f5"


def _round(x: float) -> float:
    """Floats in the record carry 12 significant digits, so a last-unit difference between platforms cannot change bytes."""
    return float(f"{x:.12g}")


def _pilots(repo: Path):
    spec = importlib.util.spec_from_file_location("measure_1d1b2_pilots", repo / "scripts" / "measure_1d1b2_pilots.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["measure_1d1b2_pilots"] = mod
    spec.loader.exec_module(mod)
    return mod


def measure(args: argparse.Namespace) -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline
    from genomic_variant_classifier.information_sources import allele_frequency as A
    from genomic_variant_classifier.information_sources import census as C
    from genomic_variant_classifier.information_sources import columns as K
    from genomic_variant_classifier.information_sources import materialization as M

    repo = Path(__file__).resolve().parents[1]
    registry = sorted(C.strict_load(repo / C.LEDGER_PATH)["datasets"])
    inputs: dict = {}

    def verified(label: str, logical: str, path: Path) -> M.Parsed:
        M.check_registered(logical, registry)
        parsed = M.read_verified_bytes(path)
        inputs[label] = {"logical_source": logical, "file": path.name, "bytes": parsed.n_bytes, "sha256": parsed.sha256}
        return parsed

    # 1. The population, exactly as the pilot built it.
    cohort = verified("cohort", "dataset:clinvar_cohort", args.clinvar)
    pilots = _pilots(repo)
    pipe = DataPrepPipeline(DataPrepConfig(output_dir=args.scratch))
    frame, unread = pilots.load_population(pipe, cohort.value)
    del cohort
    inputs["cohort"]["columns_not_read"] = unread
    frame = frame.reset_index(drop=True)
    population = M.population_from_frame(frame, pilots.POPULATION_ID)
    if population.sha256_v1 != PILOT_POPULATION_V1:
        raise SystemExit(f"the population is not the pilot's: version-1 identity {population.sha256_v1}")
    labels = frame["label"].to_numpy(dtype=np.int64)
    identity_frame = frame[["variant_id", "chrom", "pos", "ref", "alt"]].copy()
    legacy_in = frame[["variant_id", "chrom", "pos", "ref", "alt", "allele_freq"]].copy()
    del frame

    # 2. The two logical sources from one verified read, and the strict selection.
    gn = verified("gnomad", "dataset:gnomad_af", args.gnomad)
    inputs["gnomad"]["logical_sources"] = [A.PRIMARY, A.YMT_API]
    mat = A.materialize_allele_frequency(identity_frame, population, gn, registered_datasets=registry)

    # 3. Reference identity: the recorded GRCh38 window check, for every population row.
    # The table covers all of ClinVar (4,399,089 rows): it stays in Arrow form, is refused if any identity repeats in the WHOLE
    # table, and only the population's rows become Python objects (measured 2026-11-15: converting the whole table to pandas
    # peaked 1.9 GB above the baseline, the largest step of this script).
    ev = verified("reference_evidence", "dataset:grch38_reference_check", args.reference_evidence)
    table = pq.read_table(pa.BufferReader(ev.value), columns=["variant_id", "ok", "reason"])
    del ev
    if pc.count_distinct(table["variant_id"], mode="all").as_py() != table.num_rows:
        raise SystemExit("the reference evidence names a variant twice")
    wanted = pa.array(population.row_ids, type=table.schema.field("variant_id").type)
    evid = table.filter(pc.is_in(table["variant_id"], value_set=wanted)).to_pandas()
    del table, wanted
    evid = evid.set_index("variant_id").reindex(pd.Index(population.row_ids))
    if evid["ok"].isna().any():
        raise SystemExit(f"{int(evid['ok'].isna().sum())} population rows have no reference evidence")
    ref_ok = evid["ok"].to_numpy(dtype=bool)
    reasons = evid["reason"].fillna("").astype(str)
    evidence_rows = C.sha256_bytes(C.canonical_bytes([[bool(o), r] for o, r in zip(ref_ok.tolist(), reasons.tolist())]))
    policy = A.AFSelectionPolicy(unknown_base_treatment=A.UnknownBaseTreatment.EXCLUDE_FROM_SEQUENCE_RESOLVED_PRIMARY)
    sel = A.select_strict(mat, policy, reference_verified=ref_ok, reference_evidence_sha256=evidence_rows)
    census = sel.census()
    if census != EXPECTED:
        raise SystemExit(f"the census differs from the pre-registered expectation: {census} != {EXPECTED}")
    arms = {}
    for arm in A.Arm:
        s = A.select_strict(mat, A.AFSelectionPolicy(unknown_base_treatment=policy.unknown_base_treatment, arm=arm),
                            reference_verified=ref_ok, reference_evidence_sha256=evidence_rows)
        arms[arm.value] = {"identity": s.identity, "census": s.census(),
                           "selection": {A.SELECTION_STATES[c]: int((s.selection == c).sum()) for c in np.unique(s.selection)}}

    # 4. Label-blind: the materialiser never receives labels; a permuted-label frame gives the same identities.
    perm = identity_frame.assign(label=np.random.default_rng(20261115).permutation(labels))
    mat2 = A.materialize_allele_frequency(perm, population, gn, registered_datasets=registry)
    label_blind = all(a.sha256 == b.sha256 for a, b in zip(mat.columns(), mat2.columns())) and mat.receipt == mat2.receipt
    if not label_blind:
        raise SystemExit("the materialisation depends on labels")
    del mat2, perm

    # 5. The legacy join (the CURRENT DataPrepPipeline._join_gnomad, through the pilot's legacy_af_values) on the same rows.
    legacy = M.legacy_af_values(legacy_in, args.gnomad)
    if inputs["gnomad"]["sha256"] != M.read_verified_bytes(args.gnomad).sha256:
        raise SystemExit("the gnomAD file changed during the measurement")
    chrom = mat.chromosome
    cls = np.full(len(population), "", dtype=object)
    leg_obs = ~np.isnan(legacy)
    new_sel = sel.selection == A.SELECTION_STATES.index("selected")
    same = leg_obs & new_sel & (legacy == sel.af_model)
    cls[same] = "unchanged_value"
    # The CURRENT join also parses the API identifiers ('gnomad:MT:...' -> MT): it selects Y and MT values that the strict
    # policy withdraws (GNOMAD-YMT-STATISTIC-MIXING-1). Every other difference stops the measurement.
    out_dom = sel.claim == A.CLAIM_STATES.index("outside_domain")
    for c in ("Y", "MT"):
        cls[leg_obs & ~same & out_dom & (chrom == c)] = f"withdrawn_policy_out_of_scope:{c}"
    cls[leg_obs & ~same & (cls == "")] = "legacy_value_changed_or_withdrawn"
    st = [K.STATE_CODES[c].value for c in mat.primary.states]
    det = [mat.primary.detail_vocabulary[d] for d in mat.primary.details]
    for i in np.flatnonzero(~leg_obs):
        if new_sel[i]:
            cls[i] = "recovered_x" if chrom[i] == "X" else "recovered_other"
        else:
            cls[i] = f"still_unselected:{A.SELECTION_STATES[sel.selection[i]]}:{st[i]}:{det[i]}"
    if (cls == "").any():
        raise SystemExit("unclassified rows")
    legacy_classes = {k: int(v) for k, v in zip(*np.unique(cls.astype(str), return_counts=True))}
    if any(k.startswith(("legacy_value_changed", "recovered_other")) for k in legacy_classes):
        raise SystemExit(f"unexpected legacy differences: {legacy_classes}")

    # 6. Descriptive rare-flag table (labels and splits read only here).
    split = np.full(len(population), "", dtype=object)
    for name in ("train", "val", "test"):
        ids = pd.read_parquet(args.splits / f"meta_{name}.parquet", columns=["variant_id"])["variant_id"]
        pos = pd.Index(population.row_ids).get_indexer(ids)
        if (pos < 0).any() or (split[pos] != "").any():
            raise SystemExit(f"split {name}: rows outside the population or in two splits")
        split[pos] = name
    if (split == "").any():
        raise SystemExit("population rows in no split")
    for name in ("train", "val", "test"):
        inputs[f"split_{name}"] = {"file": f"meta_{name}.parquet",
                                   "sha256": M.read_verified_bytes(args.splits / f"meta_{name}.parquet").sha256}
    p_obs = mat.primary.states == K.CODE[K.ObservationState.OBSERVED]
    flags = {
        "unknown_base": sel.claim == A.CLAIM_STATES.index("incomplete_allele"),
        "outside_domain_y_mt": sel.claim == A.CLAIM_STATES.index("outside_domain"),
        "record_without_value": mat.primary.details == A.PRIMARY_DETAILS.index("record_without_value"),
        "no_row_in_extracted_table": mat.primary.details == A.PRIMARY_DETAILS.index("no_row_in_extracted_table"),
        "selected_reported_zero": new_sel & (sel.af_model == 0.0),
        "selected_nonzero": new_sel & (sel.af_model > 0.0),
        "mt_homoplasmic_zero": mat.mt_api.details == A.API_DETAILS.index("zero_ac_hom_or_missing_ac_hom"),
    }
    P, N = int(labels.sum()), int((1 - labels).sum())
    rare = {}
    for name, m in flags.items():
        d = A.audit_binary_flag(n_positive=P, n_negative=N, flagged_positive=int((m & (labels == 1)).sum()),
                                flagged_negative=int((m & (labels == 0)).sum()))
        d = {k: (_round(v) if isinstance(v, float) else [_round(x) for x in v] if isinstance(v, list) else v)
             for k, v in d.items()}
        d["by_split"] = {s: {"n": int((m & (split == s)).sum()), "pathogenic": int((m & (split == s) & (labels == 1)).sum())}
                         for s in ("train", "val", "test")}
        rare[name] = d

    code = {sym: C.symbol_closure_digest(repo, sym) for sym in PROCEDURE}
    return {
        "schema": SCHEMA, "inputs": inputs, "procedure": code,
        "population": {"population_id": population.population_id, "rows": len(population),
                       "sha256_v1": population.sha256_v1, "sha256_v2": population.sha256,
                       "pilot_population_v1": PILOT_POPULATION_V1, "pathogenic": P, "benign": N},
        "statistics": {k: dict(v) for k, v in A.STATISTICS.items()},
        "receipt": json.loads(json.dumps(mat.receipt)),
        "reference_evidence": {"rows_sha256": evidence_rows,
                               "verdicts": {f"{bool(o)}:{r}": int(n) for (o, r), n in
                                            pd.Series(list(zip(ref_ok.tolist(), reasons.tolist()))).value_counts().items()}},
        "policy": policy.as_dict(), "selection_identity": sel.identity, "census": census, "expected": EXPECTED,
        "arms": arms, "label_blind": label_blind, "legacy_comparison": legacy_classes, "rare_flags": rare,
    }


def main(argv=None) -> int:
    from genomic_variant_classifier.information_sources import census as C

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clinvar", type=Path, required=True)
    ap.add_argument("--gnomad", type=Path, required=True)
    ap.add_argument("--reference-evidence", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True, help="an empty directory for DataPrepPipeline's output_dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="compare with --out byte for byte instead of writing")
    a = ap.parse_args(argv)
    raw = C.ledger_bytes(measure(a))
    if a.check:
        committed = a.out.read_bytes()
        if committed != raw:
            print(f"DIFFERS: {a.out} (committed sha256 {C.sha256_bytes(committed)}, measured {C.sha256_bytes(raw)})")
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
