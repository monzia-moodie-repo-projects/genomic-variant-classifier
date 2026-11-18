"""Synthetic sources for the qualified preparation (set "av"): a labelled cohort frame, a gnomAD extract in the two
grammars, the reference-check evidence, and a small FASTA with its pin. Every case the strict policy distinguishes is
present: an observed value, an observed zero, a record without value, a row absent from the extract, X (recovered by
the strict policy), Y and MT (outside its domain) and an allele with an unknown base.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from genomic_variant_classifier.information_sources import allele_frequency as A
from genomic_variant_classifier.information_sources import census as C
from genomic_variant_classifier.information_sources import materialization as M
from genomic_variant_classifier.information_sources import reference_status as R

REPO = Path(__file__).resolve().parents[2]
POPULATION_ID = "synthetic:qualified-preparation"
REGISTRY = tuple(sorted(C.strict_load(REPO / C.LEDGER_PATH)["datasets"]))
POLICY = A.AFSelectionPolicy(unknown_base_treatment=A.UnknownBaseTreatment.EXCLUDE_FROM_SEQUENCE_RESOLVED_PRIMARY)


def _parquet(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), buf)
    return buf.getvalue()


def cohort_frame(n: int = 320, seed: int = 0) -> pd.DataFrame:
    """The labelled cohort frame: chromosome 1 SNVs, then the special rows at fixed positions."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n - 6):
        ref, alt = ("A", "G") if i % 2 else ("C", "T")
        rows.append(("1", 1000 + 10 * i, ref, alt))
    rows += [("X", 5000, "G", "A"), ("Y", 2800000, "C", "T"), ("MT", 150, "T", "C"),
             ("2", 7000, "AN", "A"), ("2", 7100, "GT", "G"), ("3", 9000, "A", "AT")]
    f = pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])
    f["variant_id"] = "clinvar:" + f["chrom"] + ":" + f["pos"].astype(str) + ":" + f["ref"] + ":" + f["alt"]
    f["gene_symbol"] = [f"G{i % 23}" for i in range(len(f))]
    f["consequence"] = rng.choice(["missense_variant", "synonymous_variant", "stop_gained"], len(f))
    f["label"] = (rng.random(len(f)) < 0.4).astype(int)
    # Non-allele-frequency annotation, as the existing annotation steps would supply it (never an AF source).
    f["cadd_phred"] = rng.normal(15, 5, len(f))
    f["revel_score"] = rng.random(len(f))
    f["gene_constraint_oe"] = rng.random(len(f))
    f["allele_freq"] = None
    f["source_db"] = "clinvar"
    return f[["variant_id", "chrom", "pos", "ref", "alt", "gene_symbol", "consequence", "label", "allele_freq",
              "source_db", "cadd_phred", "revel_score", "gene_constraint_oe"]]


def gnomad_frame(cohort: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Records for most chromosome-1 rows (one observed zero, one record without value, every 9th row absent), X, Y
    and MT records, and an extra record no cohort row requests."""
    rng = np.random.default_rng(seed)
    ids, afs = [], []
    chr1 = cohort[cohort["chrom"] == "1"].reset_index(drop=True)
    for i, r in chr1.iterrows():
        if i % 9 == 4:
            continue                                                   # no row in the extract
        key = f"{r.chrom}:{r.pos}:{r.ref}:{r.alt}"
        ids.append(key)
        afs.append(0.0 if i == 0 else np.nan if i == 1 else float(np.round(rng.beta(0.4, 40), 6)))
    ids += ["X:5000:G:A", "gnomad:Y:2800000:C:T", "gnomad:MT:150:T:C", "7:123:A:C"]
    afs += [0.0123, 0.2, 0.0, 0.5]
    return pd.DataFrame({"variant_id": ids, "allele_freq": np.asarray(afs, dtype=np.float64)})


def evidence_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    """The reference-check verdicts: ok for every fully specified allele, non_acgt_allele for the unknown base; plus an
    unrelated row (the table covers more than the population)."""
    ok = ~cohort["ref"].str.contains("N") & ~cohort["alt"].str.contains("N")
    reason = np.where(ok, "", "non_acgt_allele")
    out = pd.DataFrame({"variant_id": cohort["variant_id"], "ok": ok.to_numpy(), "reason": reason})
    return pd.concat([out, pd.DataFrame({"variant_id": ["clinvar:9:1:A:C"], "ok": [True], "reason": [""]})],
                     ignore_index=True)


def write_sources(tmp: Path, cohort: pd.DataFrame) -> dict[str, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    paths = {"gnomad": tmp / "gnomad.parquet", "evidence": tmp / "evidence.parquet", "cohort": tmp / "cohort.parquet"}
    paths["gnomad"].write_bytes(_parquet(gnomad_frame(cohort)))
    paths["evidence"].write_bytes(_parquet(evidence_frame(cohort)))
    paths["cohort"].write_bytes(_parquet(cohort.astype({"allele_freq": "float64"})))
    return paths


def prepared(tmp: Path, cohort: pd.DataFrame | None = None, policy: A.AFSelectionPolicy = POLICY, **kw):
    """Run the qualified preparation on the synthetic sources; returns (canonical matrix, inputs)."""
    from genomic_variant_classifier.information_sources import qualified_preparation as Q

    cohort = cohort_frame() if cohort is None else cohort
    paths = write_sources(tmp, cohort)
    population = M.population_from_frame(cohort, POPULATION_ID)
    gnomad = M.read_verified_bytes(paths["gnomad"])
    evidence = M.read_verified_bytes(paths["evidence"])
    cohort_sha = M.read_verified_bytes(paths["cohort"]).sha256
    status = R.materialize_reference_status(population, evidence, registered_datasets=REGISTRY)
    declared = {Q.COHORT: cohort_sha, A.PRIMARY: gnomad.sha256, R.EVIDENCE_DATASET: evidence.sha256}
    args = dict(mode=Q.PreparationMode.QUALIFIED, cohort_sha256=cohort_sha, gnomad=gnomad, reference=status,
                reference_pin=R.GRCH38_PRIMARY_ASSEMBLY, policy=policy, registered_datasets=REGISTRY,
                declared_sources=declared)
    args.update(kw)
    cm = Q.prepare_qualified(cohort, population, **args)
    return cm, {"population": population, "gnomad": gnomad, "evidence": evidence, "status": status,
                "declared": declared, "cohort": cohort, "paths": paths, "args": args}


def fasta(tmp: Path, contigs: dict[str, str]) -> tuple[Path, R.ReferencePin]:
    """A FASTA with a faidx index (60 bases per line) and the pin of exactly its bytes."""
    path = tmp / "ref.fa"
    lines, fai, offset = [], [], 0
    for name, seq in contigs.items():
        header = f">{name}\n"
        offset += len(header)
        body = "".join(seq[i:i + 60] + "\n" for i in range(0, len(seq), 60))
        fai.append(f"{name}\t{len(seq)}\t{offset}\t60\t61\n")
        lines.append(header + body)
        offset += len(body)
    path.write_text("".join(lines), encoding="ascii", newline="\n")
    Path(str(path) + ".fai").write_text("".join(fai), encoding="ascii", newline="\n")
    raw, idx = path.read_bytes(), Path(str(path) + ".fai").read_bytes()
    pin = R.ReferencePin("synthetic", "synthetic test sequence", hashlib.sha256(raw).hexdigest(), len(raw),
                         hashlib.sha256(idx).hexdigest(), hashlib.sha256(idx + str(len(raw)).encode()).hexdigest())
    return path, pin
