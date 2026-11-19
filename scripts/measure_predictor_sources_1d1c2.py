"""Stage 1D-1c.2A (set "aw"): the computational-predictor source audit, measured on the REAL files (owner ruling e755519d,
2026-11-17/18). Author: Monzia Moodie.

Five subcommands; each writes one JSON part and refuses to overwrite; ``collect`` verifies the parts and writes the record.

  index-defaults  the full dbNSFP index (85.3 M rows): how many rows hold each predictor's connector default EXACTLY,
                  against the neighbouring values, and the joint pattern. Imports only numpy and pyarrow, so it runs on the
                  machine that holds the 939 MB file. The bytes are read once; the digest covers exactly the bytes parsed.
  pilot           the stage 1D-1b.2 pilot population (rebuilt by the pilot's own load_population and verified against its
                  recorded identity) joined to the ClinVar dbNSFP index through the CURRENT connector with its fill
                  replaced by NaN, exactly as the pilot measured REVEL: rows "observed" by the join that hold the default
                  exactly, by label and by consequence.
  mane            GENCODE v50 MANE Select and Ensembl-canonical transcripts per gene name, and how many missense rows of
                  the pilot population name a gene with exactly one.
  revel           the publisher's REVEL v1.3 archive (Zenodo 10.5281/zenodo.7072866): a full, validated scan of the
                  transcript-aware member (every row; schema, coordinates, scores, transcript syntax refused on any
                  deviation), the source census, and -- at the pilot population's alleles -- the comparison with the
                  historical index and the transcript evidence. Labels are used ONLY to report counts after every
                  selection is made (label-blind; owner ruling e755519d, section 9).
  collect         the four parts into docs/census/PREDICTOR_SOURCES_1D1c2.json, refusing unless every part's inputs and
                  step-code digests are those recorded and consistent.

Changes nothing in production and fits nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = "gvc.predictor-sources.1d1c2/1"
SCRIPT = "scripts/measure_predictor_sources_1d1c2.py"
PILOT = "scripts/measure_1d1b2_pilots.py"
#: The connector's fills (data/dbnsfp.py DEFAULT_*), which the index build applied BEFORE caching.
DEFAULTS = {"revel_score": 0.5, "sift_score": 0.5, "polyphen2_score": 0.5, "cadd_phred": 15.0}
#: The neighbourhood step of each score's stored grid (REVEL, SIFT, PolyPhen-2: 0.001; CADD PHRED: 0.01).
STEP = {"revel_score": 0.001, "sift_score": 0.001, "polyphen2_score": 0.001, "cadd_phred": 0.01}
REVEL_COLUMNS = ("chr", "hg19_pos", "grch38_pos", "ref", "alt", "aaref", "aaalt", "REVEL", "Ensembl_transcriptid")
REVEL_MEMBER = "revel_with_transcript_ids"
ZENODO_MD5 = "3ea2bc33e6b5455fc7e9899da863b5fe"
CHROMS = tuple(str(i) for i in range(1, 23)) + ("X", "Y")
ENST_UNVERSIONED = re.compile(r"ENST\d{11}\Z")
ENST_VERSIONED = re.compile(r"ENST\d{11}\.\d+\Z")
AMINO = frozenset("ACDEFGHIKLMNPQRSTVWY")
PROCEDURE = (f"{SCRIPT}::index_defaults", f"{SCRIPT}::pilot_defaults", f"{SCRIPT}::mane_coverage", f"{SCRIPT}::revel_source",
             f"{SCRIPT}::read_revel_member", f"{SCRIPT}::scan_revel", f"{SCRIPT}::clinvar_declared", f"{SCRIPT}::select_scalar", f"{SCRIPT}::_population_and_join", f"{SCRIPT}::_load_frame", f"{SCRIPT}::collect",
             f"{PILOT}::load_population")


def _refuse_existing(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")


def _read_all(path: Path) -> tuple[bytes, str]:
    """The file's bytes and their SHA-256: everything parsed afterwards is parsed from these bytes."""
    raw = path.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest()


def _report_run(start: float) -> None:
    """Time, peak memory and environment go to the console, never into a part: a part must be byte-reproducible."""
    import platform

    try:
        import resource
        peak = f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB peak"
    except ImportError:                                   # Windows has no resource module
        peak = "peak memory not measured on this platform"
    print(f"{round(time.monotonic() - start, 1)} s; {peak}; Python {sys.version.split()[0]} on {platform.system()}")


def _canonical(record: dict) -> bytes:
    """A part's bytes: ASCII JSON, one-space indent, sorted keys, LF line ends -- the same bytes on every platform. The record
    binds each part's SHA-256, so a part written in text mode (CR LF line ends on Windows) changes the record (measured
    2026-11-18: the owner's V1 re-measured every value identically and differed only in the parts' line ends)."""
    return (json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 6)


def _write(path: Path, record: dict) -> None:
    path.write_bytes(_canonical(record))
    print(f"wrote {path}")


def _read_part(path: Path) -> tuple[dict, bytes]:
    """A part and its bytes, refused unless the bytes are exactly the part's canonical serialisation."""
    raw = path.read_bytes()
    record = json.loads(raw.decode("ascii"))
    if raw != _canonical(record):
        cr = raw.count(b"\r")
        raise SystemExit(f"{path.name} is not a canonical part ({cr} carriage returns; its bytes differ from its own "
                         f"canonical serialisation): measure it again with this script")
    return record, raw


# ------------------------------------------------------------------------------------------------------ index-defaults
def index_defaults(index: Path, out: Path) -> int:
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    _refuse_existing(out)
    start = time.monotonic()
    raw, sha = _read_all(index)
    f = pq.ParquetFile(pa.BufferReader(raw))
    per = {c: {"rows": 0, "null": 0, "exactly_default": 0, "neighbours": {}} for c in DEFAULTS}
    joint = {"rows": 0, "revel_sift_polyphen2_all_default": 0, "all_four_default": 0,
             "revel_default_sift_and_polyphen2_not": 0, "sift_default_revel_not": 0, "polyphen2_default_revel_not": 0}
    for batch in f.iter_batches(batch_size=4_000_000, columns=list(DEFAULTS)):
        at = {}
        for c, d in DEFAULTS.items():
            a = batch.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
            per[c]["rows"] += len(a)
            per[c]["null"] += int(np.isnan(a).sum())
            at[c] = a == d
            per[c]["exactly_default"] += int(at[c].sum())
            grid = np.round(a / STEP[c])
            for k in (-3, -2, -1, 1, 2, 3):
                key = f"{d + k * STEP[c]:.3f}"
                per[c]["neighbours"][key] = per[c]["neighbours"].get(key, 0) + int((grid == round(d / STEP[c]) + k).sum())
        rsp = at["revel_score"] & at["sift_score"] & at["polyphen2_score"]
        joint["rows"] += len(batch)
        joint["revel_sift_polyphen2_all_default"] += int(rsp.sum())
        joint["all_four_default"] += int((rsp & at["cadd_phred"]).sum())
        joint["revel_default_sift_and_polyphen2_not"] += int((at["revel_score"] & ~at["sift_score"]
                                                              & ~at["polyphen2_score"]).sum())
        joint["sift_default_revel_not"] += int((at["sift_score"] & ~at["revel_score"]).sum())
        joint["polyphen2_default_revel_not"] += int((at["polyphen2_score"] & ~at["revel_score"]).sum())
    for c in DEFAULTS:
        nb = per[c]["neighbours"]
        per[c]["neighbour_mean"] = round(sum(nb.values()) / len(nb), 1)
        per[c]["excess_over_neighbour_mean"] = round(per[c]["exactly_default"] - per[c]["neighbour_mean"], 1)
    _write(out, {"part": "index-defaults", "input": {"file": index.name, "bytes": len(raw), "sha256": sha,
                                                     "rows": f.metadata.num_rows},
                 "per_predictor": per, "joint": joint})
    _report_run(start)
    return 0


# ---------------------------------------------------------------------------------------------------------------- pilot
def _pilot_module(repo: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("measure_1d1b2_pilots", repo / PILOT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _population_and_join(repo: Path, cohort: Path, index: Path, scratch: Path):
    """The pilot population (verified against PILOTS_1D1b2.json) and the current connector's join with NaN fills."""
    import numpy as np

    from genomic_variant_classifier.data.dbnsfp import DbNSFPConnector
    from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline
    from genomic_variant_classifier.information_sources import materialization as M

    pilot = _pilot_module(repo)
    recorded = json.loads((repo / "docs/census/PILOTS_1D1b2.json").read_text(encoding="utf-8"))
    cohort_bytes = M.read_verified_bytes(cohort)
    if cohort_bytes.sha256 != recorded["inputs"]["cohort"]["sha256"]:
        raise SystemExit("the cohort is not the one the pilot measured")
    index_bytes = M.read_verified_bytes(index)
    if index_bytes.sha256 != recorded["inputs"]["dbnsfp"]["sha256"]:
        raise SystemExit("the dbNSFP index is not the one the pilot measured")
    frame, _ = pilot.load_population(DataPrepPipeline(DataPrepConfig(output_dir=scratch)), cohort_bytes.value)
    population = M.population_from_frame(frame, pilot.POPULATION_ID)
    if population.sha256_v1 != recorded["population"]["sha256"]:
        raise SystemExit("the rebuilt population differs from the pilot's (identity version 1)")
    nan = float("nan")
    joined = DbNSFPConnector(dbnsfp_file=index).annotate_dataframe(
        frame[["chrom", "pos", "ref", "alt"]].copy(), revel_score=nan, sift_score=nan, polyphen2_score=nan, cadd_phred=nan)
    if len(joined) != len(frame):
        raise SystemExit("the connector join changed the row count")
    values = {c: joined[c].to_numpy(dtype=np.float64) for c in DEFAULTS}
    inputs = {"cohort": {"file": cohort.name, "bytes": cohort_bytes.n_bytes, "sha256": cohort_bytes.sha256},
              "dbnsfp_clinvar_index": {"file": index.name, "bytes": index_bytes.n_bytes, "sha256": index_bytes.sha256}}
    return frame, population, values, inputs, pilot


FRAME_COLUMNS = ("variant_id", "source_id", "chrom", "pos", "ref", "alt", "gene_symbol", "consequence", "label",
                 "transcript_id", "protein_change")


def pilot_defaults(repo: Path, cohort: Path, index: Path, scratch: Path, out: Path, out_frame: Path) -> int:
    import numpy as np

    start = time.monotonic()
    _refuse_existing(out)
    _refuse_existing(out_frame)
    frame, population, values, inputs, pilot = _population_and_join(repo, cohort, index, scratch)
    compact = frame[list(FRAME_COLUMNS)].copy()
    for c, v in values.items():
        compact[f"joined_{c}"] = v
    compact.to_parquet(out_frame, index=False)
    frame_bytes, frame_sha = _read_all(out_frame)
    label = frame["label"].to_numpy(dtype=int)
    consequence = frame["consequence"].fillna("<none>").astype(str).to_numpy()
    missense = frame["consequence"].fillna("").str.contains("missense", case=False).to_numpy()
    per = {}
    at_default = {}
    for c, d in DEFAULTS.items():
        v = values[c]
        obs = np.isfinite(v)
        at = obs & (v == d)
        at_default[c] = at
        grid = np.round(v[obs] / STEP[c])
        per[c] = {"joined_finite": int(obs.sum()), "exactly_default": int(at.sum()),
                  "neighbours": {f"{d + k * STEP[c]:.3f}": int((grid == round(d / STEP[c]) + k).sum())
                                 for k in (-3, -2, -1, 1, 2, 3)},
                  "exactly_default_missense": int((at & missense).sum()),
                  "exactly_default_by_label": pilot._by_label(at, label),
                  "exactly_default_by_consequence": dict(Counter(consequence[at]).most_common()),
                  # Association BETWEEN consequence classes or WITHIN them (owner ruling ace92e09, section 15): the
                  # indicator's AUROC among the joined rows, overall and within each of the cohort's ClinVar consequence
                  # classes holding at least 1,000 joined rows (None where one label class is absent).
                  "exactly_default_auroc_among_joined_rows": _round(pilot._auroc(at[obs], label[obs])),
                  "exactly_default_auroc_within_consequence_among_joined_rows": {
                      k: _round(pilot._auroc(at[obs & (consequence == k)], label[obs & (consequence == k)]))
                      for k, n in sorted(Counter(consequence[obs]).items()) if n >= 1000}}
    r = values["revel_score"]
    robs = np.isfinite(r)
    rdef = at_default["revel_score"]
    revel = {
        "pilot_recorded_observed": int(robs.sum()),
        "join_unobserved": int((~robs).sum()),
        "observed_not_at_default": int((robs & ~rdef).sum()),
        "observed_not_at_default_ge_0_5": int((robs & ~rdef & (r >= 0.5)).sum()),
        "unobserved_or_at_default": int((~robs | rdef).sum()),
        "legacy_revel_pathogenic_without_a_non_default_score": int((~robs | rdef).sum()),
        "note": "a value exactly 0.5 in the index is a genuine score OR the index build's fill; the publisher source "
                "(part revel) separates the two at these alleles",
    }
    _write(out, {"part": "pilot", "inputs": inputs,
                 "frame": {"file": out_frame.name, "bytes": len(frame_bytes), "sha256": frame_sha,
                           "columns": list(FRAME_COLUMNS) + [f"joined_{c}" for c in values]},
                 "population": {"population_id": population.population_id, "rows": len(population),
                                "sha256_v1": population.sha256_v1, "sha256_v2": population.sha256,
                                "pathogenic": int((label == 1).sum()), "benign": int((label == 0).sum()),
                                "missense": int(missense.sum())},
                 "per_predictor": per, "revel": revel,
                 "joint": {"revel_sift_polyphen2_all_default": int((rdef & at_default["sift_score"]
                                                                    & at_default["polyphen2_score"]).sum())}})
    _report_run(start)
    return 0


def _load_frame(pilot_part: Path, frame_path: Path):
    """The population frame the pilot part wrote, verified by the part's digest and by the population identity."""
    import numpy as np
    import pandas as pd

    from genomic_variant_classifier.information_sources import materialization as M

    part, _ = _read_part(pilot_part)
    raw, sha = _read_all(frame_path)
    if part.get("part") != "pilot" or sha != part["frame"]["sha256"]:
        raise SystemExit("the population frame is not the one the pilot part recorded")
    frame = pd.read_parquet(io.BytesIO(raw))
    del raw
    population = M.population_from_frame(frame, part["population"]["population_id"])
    if population.sha256_v1 != part["population"]["sha256_v1"]:
        raise SystemExit("the population frame does not reproduce the pilot population")
    values = {c: frame[f"joined_{c}"].to_numpy(dtype=np.float64) for c in DEFAULTS}
    inputs = {**part["inputs"], "pilot_frame": {"file": frame_path.name, "sha256": sha}}
    return frame, population, values, inputs


# ----------------------------------------------------------------------------------------------------------------- mane
def _gtf_transcripts(raw_gz: bytes) -> dict:
    """Per gene name: its gene identifiers, MANE Select and Ensembl-canonical transcripts; and, per chromosome (no "chr"
    prefix), the coding segments of every MANE Select transcript as (start, end, transcript stable identifier, gene name),
    sorted, so a GRCh38 position can be resolved to the MANE coding sequences that cover it (``mane_cds_covering``)."""
    import gzip

    attr = re.compile(r'(\w+) "([^"]*)"')
    out: dict = {"mane": {}, "canonical": {}, "gene_ids": {}, "mane_cds": {}, "sequences": set()}
    with gzip.open(io.BytesIO(raw_gz), "rt", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) != 9:
                raise SystemExit("GTF record without nine fields")
            out["sequences"].add(p[0])
            if p[2] == "CDS" and 'tag "MANE_Select"' in p[8]:
                d = {k: v for k, v in attr.findall(p[8]) if k != "tag"}
                out["mane_cds"].setdefault(p[0].removeprefix("chr"), []).append(
                    (int(p[3]), int(p[4]), d["transcript_id"].split(".", 1)[0], d["gene_name"]))
                continue
            if p[2] != "transcript":
                continue
            d: dict[str, str] = {}
            tags: list[str] = []
            for k, v in attr.findall(p[8]):
                if k == "tag":
                    tags.append(v)
                else:
                    d[k] = v
            g = d["gene_name"]
            out["gene_ids"].setdefault(g, set()).add(d["gene_id"])
            if "MANE_Select" in tags:
                out["mane"].setdefault(g, set()).add(d["transcript_id"])
            if "Ensembl_canonical" in tags:
                out["canonical"].setdefault(g, set()).add(d["transcript_id"])
    for c, segs in out["mane_cds"].items():
        segs.sort()
        out["mane_cds"][c] = ([s[0] for s in segs], segs, max(s[1] - s[0] for s in segs))
    return out


def mane_cds_covering(tx: dict, chrom: str, pos: int) -> dict[str, str]:
    """{MANE Select transcript stable identifier: gene name} for every MANE Select coding segment that covers the GRCh38
    position: the gene resolved from the LOCUS, not from any annotation of the variant."""
    import bisect

    if chrom not in tx["mane_cds"]:
        return {}
    starts, segs, longest = tx["mane_cds"][chrom]
    j = bisect.bisect_right(starts, pos) - 1
    hit: dict[str, str] = {}
    while j >= 0 and starts[j] >= pos - longest:
        s, e, t, g = segs[j]
        if s <= pos <= e:
            hit[t] = g
        j -= 1
    return hit


def stable_mane_of_gene(tx: dict, gene: str) -> set[str]:
    """The MANE Select stable identifiers of the cohort's gene_symbol (a ClinVar-derived field), when it names one GENCODE
    v50 gene; else the empty set."""
    if not gene or re.search(r"[;,|]", gene) or len(tx["gene_ids"].get(gene, ())) != 1:
        return set()
    return {t.split(".", 1)[0] for t in tx["mane"].get(gene, set())}


def mane_coverage(pilot_part: Path, frame_path: Path, gtf: Path, gtf_sha256: str, out: Path) -> int:
    import numpy as np

    _refuse_existing(out)
    raw, sha = _read_all(gtf)
    if sha != gtf_sha256:
        raise SystemExit(f"GENCODE GTF has SHA-256 {sha}, not the declared {gtf_sha256}")
    tx = _gtf_transcripts(raw)
    frame, population, _values, inputs = _load_frame(pilot_part, frame_path)
    missense = (frame["consequence"] == "missense_variant").to_numpy()
    genes = frame["gene_symbol"].fillna("").astype(str).to_numpy()
    def classify(g: str) -> str:
        if not g:
            return "no_gene_symbol"
        if re.search(r"[;,|]", g):
            return "several_symbols"
        if g not in tx["gene_ids"]:
            return "symbol_not_in_gencode_v50"
        if len(tx["gene_ids"][g]) > 1:
            return "symbol_names_several_gencode_genes"
        n = len(tx["mane"].get(g, ()))
        return "one_mane_select" if n == 1 else ("no_mane_select" if n == 0 else "several_mane_select")
    cls = Counter(classify(g) for g in genes[missense])
    # The cohort-gene MANE policy's dependence on the cohort's gene_symbol, measured against the LOCUS: which MANE Select
    # coding sequences cover each single-nucleotide missense position (owner ruling ace92e09, sections 4 and 8).
    chrom = frame["chrom"].astype(str).str.replace(r"^chr", "", regex=True).to_numpy()
    snv = ((frame["ref"].astype(str).str.len() == 1) & (frame["alt"].astype(str).str.len() == 1)).to_numpy()
    pos = frame["pos"].to_numpy()
    locus = Counter()
    for i in np.flatnonzero(missense & snv):
        cov = set(mane_cds_covering(tx, chrom[i], int(pos[i])))
        own = stable_mane_of_gene(tx, genes[i])
        if not cov:
            locus["no_mane_coding_sequence_covers_the_position"] += 1
        elif len(cov) == 1:
            locus["one_mane_coding_sequence_" + ("of_the_cohort_gene" if cov == own else "of_another_gene")] += 1
        else:
            locus["several_mane_coding_sequences_" + ("cohort_gene_among_them" if own and own <= cov
                                                      else "cohort_gene_not_among_them")] += 1
    _write(out, {"part": "mane", "inputs": {**inputs, "gencode_v50_gtf": {"file": gtf.name, "bytes": len(raw),
                                                                          "sha256": sha}},
                 "population_sha256_v1": population.sha256_v1,
                 "gencode": {"mane_select_transcripts": sum(len(v) for v in tx["mane"].values()),
                             "mane_select_transcripts_with_coding_sequence": len({s[2] for v in tx["mane_cds"].values()
                                                                                   for s in v[1]}),
                             "genes_with_mane_select": len(tx["mane"]),
                             "genes_with_several_mane_select": sum(len(v) > 1 for v in tx["mane"].values()),
                             "mane_select_versioned": all("." in t for v in tx["mane"].values() for t in v),
                             "sequences": sorted(tx["sequences"])},
                 "pilot_missense_rows": int(missense.sum()),
                 "pilot_missense_rows_by_gene_resolution": dict(sorted(cls.items())),
                 "pilot_missense_snv_rows_by_locus_mane": dict(sorted(locus.items())),
                 "cohort_transcript_id_null_rows": int(frame["transcript_id"].isna().sum()),
                 "cohort_protein_change_null_rows": int(frame["protein_change"].isna().sum())})
    return 0


# ---------------------------------------------------------------------------------------------------------------- revel
def read_revel_member(raw_zip: bytes, member: str):
    """Every row of the publisher's CSV member, validated; yields pyarrow record batches of strings (exact tokens kept).
    Refuses: a member that is missing or duplicated, a header other than the nine publisher columns, a row with another
    number of fields (pyarrow's parser), a CRC mismatch (zipfile, at the end of the member)."""
    import zipfile

    import pyarrow as pa
    import pyarrow.csv as pacsv

    z = zipfile.ZipFile(io.BytesIO(raw_zip))
    names = [i.filename for i in z.infolist() if not i.is_dir()]
    if names.count(member) != 1:
        raise SystemExit(f"archive members {names}: expected exactly one {member!r}")
    with z.open(member) as fh:
        head = fh.peek(4096)[:4096].split(b"\n", 1)[0].rstrip(b"\r")
        if head.decode("utf-8-sig") != ",".join(REVEL_COLUMNS):
            raise SystemExit(f"unexpected REVEL header: {head[:200]!r}")
        reader = pacsv.open_csv(
            fh, read_options=pacsv.ReadOptions(block_size=16 << 20),      # 16 MB: peak 3.5 GB (64 MB: 5.7 GB), same output
            parse_options=pacsv.ParseOptions(delimiter=","),
            convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in REVEL_COLUMNS},
                                                 strings_can_be_null=False, quoted_strings_can_be_null=False))
        if tuple(reader.schema.names) != REVEL_COLUMNS:
            raise SystemExit(f"unexpected REVEL columns {reader.schema.names}")
        for batch in reader:
            yield batch


def scan_revel(raw_zip: bytes, wanted: set[str], wanted_grch37: set[str] = frozenset()):
    """A full validated scan of the publisher member: the source census, every row whose GRCh38 allele is in ``wanted``
    (row number, score token or None, amino-acid change, transcript tuple), and -- for the rows whose GRCh38 position the
    publisher left unresolved -- every one whose GRCh37 allele (chr:hg19_pos:ref:alt) is in ``wanted_grch37`` (its score
    token or None). Refuses on the first deviation."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    census = Counter()
    chrom_rows = Counter()
    tx_count_hist = Counter()
    decimals = Counter()
    bad = Counter()
    matched: dict[str, list[tuple[int, str | None, str, str, tuple[str, ...]]]] = {}
    unresolved_at_grch37: dict[str, list[str | None]] = {}
    wanted_array = pa.array(sorted(wanted), type=pa.string())
    wanted37_array = pa.array(sorted(wanted_grch37), type=pa.string())
    score_re = r"^(0(\.[0-9]+)?|1(\.0+)?)$"
    row_number = 1                                    # the header is row 1
    for batch in read_revel_member(raw_zip, REVEL_MEMBER):
        n = batch.num_rows
        first = row_number + 1
        col = {c: batch.column(c) for c in REVEL_COLUMNS}

        def refuse_unless(mask, what: str) -> None:
            ok = pc.fill_null(mask, False)
            if not pc.all(ok).as_py():
                i = pc.index(ok, False).as_py()
                raise SystemExit(f"row {first + i}: {what}: " + ",".join(str(col[c][i]) for c in REVEL_COLUMNS))
        refuse_unless(pc.is_in(col["chr"], value_set=pa.array(CHROMS)), "unexpected chromosome")
        refuse_unless(pc.match_substring_regex(col["hg19_pos"], r"^[1-9][0-9]*$"), "invalid hg19_pos")
        for c in ("ref", "alt"):
            refuse_unless(pc.match_substring_regex(col[c], r"^[ACGT]$"), f"{c} is not one of A, C, G, T")
        refuse_unless(pc.not_equal(col["ref"], col["alt"]), "ref equals alt")
        missing_score = pc.is_in(col["REVEL"], value_set=pa.array([".", ""]))
        refuse_unless(pc.or_(missing_score, pc.match_substring_regex(col["REVEL"], score_re)),
                      "REVEL score is not a decimal in [0, 1]")
        unresolved = pc.is_in(col["grch38_pos"], value_set=pa.array([".", ""]))
        refuse_unless(pc.or_(unresolved, pc.match_substring_regex(col["grch38_pos"], r"^[1-9][0-9]*$")),
                      "invalid grch38_pos")
        for c in ("aaref", "aaalt"):
            bad[f"{c}_not_one_of_20"] += int(pc.sum(pc.invert(pc.is_in(col[c], value_set=pa.array(sorted(AMINO))))).as_py() or 0)
        census["score_missing"] += int(pc.sum(missing_score).as_py() or 0)
        census["grch38_unresolved"] += int(pc.sum(unresolved).as_py() or 0)
        scored = pc.filter(col["REVEL"], pc.invert(missing_score))
        census["score_exactly_0_5"] += int(pc.sum(pc.equal(pc.cast(scored, pa.float64()), 0.5)).as_py() or 0)
        point = np.asarray(pc.find_substring(scored, "."))             # -1 when the token has no decimal point
        places = np.where(point < 0, 0, np.asarray(pc.utf8_length(scored)) - point - 1)
        for k, v in zip(*np.unique(places, return_counts=True)):
            decimals[int(k)] += int(v)
        for item in pc.value_counts(col["chr"]).to_pylist():
            chrom_rows[item["values"]] += int(item["counts"])
        no_tx = pc.is_in(col["Ensembl_transcriptid"], value_set=pa.array([".", ""]))
        census["no_transcript"] += int(pc.sum(no_tx).as_py() or 0)
        lists = pc.split_pattern(col["Ensembl_transcriptid"], ";")
        lengths = np.asarray(pc.list_value_length(lists))
        lengths = np.where(np.asarray(no_tx), 0, lengths)
        for k, v in zip(*np.unique(lengths, return_counts=True)):
            tx_count_hist[int(k)] += int(v)
        flat = pc.list_flatten(pc.filter(lists, pc.invert(no_tx)))
        unv = pc.match_substring_regex(flat, r"^ENST[0-9]{11}$")
        ver = pc.match_substring_regex(flat, r"^ENST[0-9]{11}\.[0-9]+$")
        if not pc.all(pc.or_(unv, ver)).as_py():
            badid = pc.filter(flat, pc.invert(pc.or_(unv, ver)))[0]
            raise SystemExit(f"transcript identifier {badid} is not ENST + 11 digits (rows {first}..{first + n - 1})")
        census["transcript_ids_unversioned"] += int(pc.sum(unv).as_py() or 0)
        census["transcript_ids_versioned"] += int(pc.sum(ver).as_py() or 0)
        parents = pc.list_parent_indices(pc.filter(lists, pc.invert(no_tx)))
        dup = pa.table({"p": parents, "t": flat}).group_by(["p", "t"]).aggregate([("t", "count")])
        if pc.max(dup["t_count"]).as_py() not in (None, 1):
            raise SystemExit(f"a row lists one transcript twice (rows {first}..{first + n - 1})")
        keys = pc.binary_join_element_wise(col["chr"], col["grch38_pos"], col["ref"], col["alt"], ":")
        hit = pc.and_(pc.invert(unresolved), pc.is_in(keys, value_set=wanted_array))
        for i in np.flatnonzero(np.asarray(hit)):
            s = col["REVEL"][i].as_py()
            t = col["Ensembl_transcriptid"][i].as_py()
            matched.setdefault(keys[i].as_py(), []).append(
                (first + int(i), None if s in (".", "") else s, col["aaref"][i].as_py(), col["aaalt"][i].as_py(),
                 () if t in (".", "") else tuple(t.split(";"))))
        if len(wanted37_array):
            keys37 = pc.binary_join_element_wise(col["chr"], col["hg19_pos"], col["ref"], col["alt"], ":")
            for i in np.flatnonzero(np.asarray(pc.and_(unresolved, pc.is_in(keys37, value_set=wanted37_array)))):
                s = col["REVEL"][i].as_py()
                unresolved_at_grch37.setdefault(keys37[i].as_py(), []).append(None if s in (".", "") else s)
        row_number += n
    census["rows"] = row_number - 1
    return census, chrom_rows, tx_count_hist, decimals, bad, matched, unresolved_at_grch37


THREE_LETTER = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
                "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
                "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V"}
CLINVAR_NAME = re.compile(r"^(N[MR]_\d+\.\d+)\(([^)]+)\):(c\.[^ ]+)(?: \((p\.[^)]+)\))?$")
MISSENSE_HGVSP = re.compile(r"^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$")


def clinvar_declared(raw_gz: bytes, frame):
    """ClinVar's own declaration for each population row, from variant_summary's Name: the RefSeq transcript (with its
    version) and, for a single-residue substitution, the reference and alternate amino acids. A row is matched to the ONE
    GRCh38 record of its VariationID (the cohort's source_id) and its allele (PositionVCF, ReferenceAlleleVCF,
    AlternateAlleleVCF); a key with more than one record is not matched. Also each row's GRCh37 allele
    (chr:pos:ref:alt), from the ONE GRCh37 record of its VariationID (none where it has several). Returns (status per row,
    transcript per row, (aaref, aaalt) per row, GRCh37 allele per row, census). The Name is the variant's nomenclature
    (ClinVar's calculated preferred name), not its classification."""
    import gzip

    import pandas as pd

    cols = ["Name", "Assembly", "Chromosome", "VariationID", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]
    wanted = set(frame["source_id"].astype(str))
    parts, parts37, n_all = [], [], 0
    with gzip.open(io.BytesIO(raw_gz)) as fh:
        for ch in pd.read_csv(fh, sep="\t", usecols=cols, dtype=str, chunksize=400_000, low_memory=False):
            n_all += len(ch)
            mine = ch["VariationID"].isin(wanted)
            parts.append(ch[(ch["Assembly"] == "GRCh38") & mine])
            parts37.append(ch[(ch["Assembly"] == "GRCh37") & mine])
    vs = pd.concat(parts, ignore_index=True)
    g37 = pd.concat(parts37, ignore_index=True)
    several37 = g37["VariationID"].duplicated(keep=False)
    one37 = g37[~several37]
    key37 = dict(zip(one37["VariationID"], one37["Chromosome"] + ":" + one37["PositionVCF"] + ":"
                     + one37["ReferenceAlleleVCF"] + ":" + one37["AlternateAlleleVCF"]))
    grch37 = [key37.get(v) for v in frame["source_id"].astype(str)]
    vs["key"] = (vs["VariationID"] + "|" + vs["Chromosome"] + ":" + vs["PositionVCF"] + ":" + vs["ReferenceAlleleVCF"]
                 + ":" + vs["AlternateAlleleVCF"])
    duplicated = vs["key"].duplicated(keep=False)
    by_key = vs[~duplicated].set_index("key")["Name"]
    key = (frame["source_id"].astype(str) + "|" + frame["chrom"].astype(str) + ":" + frame["pos"].astype(str) + ":"
           + frame["ref"].astype(str) + ":" + frame["alt"].astype(str))
    names = key.map(by_key).to_numpy(dtype=object)
    status, transcript, substitution = [], [], []
    for n in names:
        if not isinstance(n, str):
            status.append("no_unique_grch38_record"); transcript.append(None); substitution.append(None); continue
        m = CLINVAR_NAME.match(n)
        if not m:
            status.append("name_not_transcript_hgvs"); transcript.append(None); substitution.append(None); continue
        transcript.append(m.group(1))
        q = MISSENSE_HGVSP.match(m.group(4) or "")
        if not q or q.group(1) not in THREE_LETTER or q.group(3) not in THREE_LETTER or q.group(1) == q.group(3):
            status.append("no_single_residue_substitution"); substitution.append(None); continue
        status.append("declared"); substitution.append((THREE_LETTER[q.group(1)], THREE_LETTER[q.group(3)]))
    census = {"rows": n_all, "grch38_records_of_population_variants": int(len(vs)),
              "keys_with_several_records": int(vs.loc[duplicated, "key"].nunique()),
              "grch37_records_of_population_variants": int(len(g37)),
              "variation_ids_with_one_grch37_record": len(key37),
              "variation_ids_with_several_grch37_records": int(g37.loc[several37, "VariationID"].nunique())}
    return status, transcript, substitution, grch37, census


CANDIDATE = "candidate"
#: What a disagreement among applicable rows is under each selection: rows naming ONE transcript that disagree are a
#: source conflict; rows sharing only an amino-acid pair are an unresolved context (owner ruling ace92e09, section 8).
MANE_AMBIGUOUS = "source_conflict_same_transcript"
CLINVAR_AMBIGUOUS = "context_unresolved_same_amino_acid_pair"
NO_PUBLISHER_SCORE = ("hist_absent_pub_absent", "hist_0_5_pub_absent__fabricated", "hist_value_pub_absent")
DOMAINS = ("not_single_nucleotide", "single_nucleotide_missense", "single_nucleotide_other_consequence")


def select_scalar(rows, keep, ambiguous: str) -> str:
    """The state of one prediction unit under a declared selection: ``keep`` picks the publisher rows that apply. Never
    a maximum, never the first row: several applicable rows must agree on the score AND the amino-acid change, else the
    state is ``ambiguous``, named by the caller for what its selection can establish. A selected score is a CANDIDATE:
    no selection here verifies the transcript version, the coding sequence or the residue (owner ruling ace92e09); a
    qualified transcript context is stage 1D-1c.2B's."""
    if not rows:
        return "not_in_publisher_grch38_view"
    hit = [r for r in rows if keep(r)]
    if not hit:
        return "no_applicable_source_row"
    if len({r[1] for r in hit}) > 1 or len({(r[2], r[3]) for r in hit}) > 1:
        return ambiguous
    return "applicable_row_without_value" if hit[0][1] is None else CANDIDATE


def domain_of(key: str | None, missense: bool) -> str:
    """REVEL's domain by the cohort's own fields: REVEL scores single-nucleotide missense variants only. Applicability
    here is the cohort's ClinVar consequence; per-transcript applicability is stage 1D-1c.2B's (ruling ace92e09, sec. 11)."""
    if key is None:
        return DOMAINS[0]
    return DOMAINS[1] if missense else DOMAINS[2]


def absence_semantics(compare: Counter, by_domain: dict, rescued: dict, rescued_scored: int) -> dict:
    """The historical REVEL flag against the publisher (owner ruling ace92e09, section 3): every quantity is DERIVED from
    the eight-category partition (historical_index_against_publisher), never entered, and the record is refused unless
    they reconcile. A default-triggered positive is a row whose index value is absent or exactly 0.5 (both make
    revel_pathogenic 1)."""
    c = {k: int(compare.get(k, 0)) for k in ("hist_absent_pub_absent", "hist_absent_pub_present",
                                              "hist_0_5_pub_absent__fabricated", "hist_0_5_pub_has_0_5__genuine",
                                              "hist_0_5_pub_other_values", "hist_value_pub_absent",
                                              "hist_value_equals_pub_max", "hist_value_differs_from_pub_max")}
    if set(compare) - set(c):
        raise SystemExit(f"unexpected partition categories {sorted(set(compare) - set(c))}")
    unobserved = c["hist_absent_pub_absent"] + c["hist_absent_pub_present"]
    at_half = c["hist_0_5_pub_absent__fabricated"] + c["hist_0_5_pub_has_0_5__genuine"] + c["hist_0_5_pub_other_values"]
    unsupported = unobserved + at_half - c["hist_0_5_pub_has_0_5__genuine"]
    no_publisher = c["hist_absent_pub_absent"] + c["hist_0_5_pub_absent__fabricated"]
    publisher_present = c["hist_absent_pub_present"] + c["hist_0_5_pub_other_values"]
    no_pub_domain = {d: sum(by_domain[k].get(d, 0) for k in ("hist_absent_pub_absent", "hist_0_5_pub_absent__fabricated"))
                     for d in DOMAINS}
    for k, v in by_domain.items():
        if sum(v.values()) != c[k]:
            raise SystemExit(f"{k}: the domain partition sums to {sum(v.values())}, not {c[k]}")
    if no_publisher + publisher_present != unsupported or sum(no_pub_domain.values()) != no_publisher:
        raise SystemExit("the absence semantics do not reconcile with the partition")
    return {
        "pilot_unobserved": unobserved, "pilot_observed_at_0_5": at_half,
        "of_which_publisher_has_0_5_at_allele": c["hist_0_5_pub_has_0_5__genuine"],
        "default_triggered_positives": unobserved + at_half,
        "default_triggered_without_index_score_or_without_a_publisher_0_5_at_the_allele": unsupported,
        "of_which_no_publisher_score_in_the_grch38_view": no_publisher,
        "of_which_a_publisher_score_exists_at_the_allele": publisher_present,
        "no_publisher_score_in_the_grch38_view_by_domain": no_pub_domain,
        "no_publisher_score_in_the_grch38_view_but_a_publisher_row_at_the_grch37_allele_with_unresolved_grch38": rescued,
        "of_which_the_grch37_row_has_a_score": rescued_scored,
        "nondefault_index_value_without_publisher_score_positivity_not_recorded": c["hist_value_pub_absent"],
        "definitions": {
            "default_triggered_without_index_score_or_without_a_publisher_0_5_at_the_allele":
                "revel_pathogenic = 1 from no index score, or from an index 0.5 where the publisher has no 0.5 at the "
                "allele; NOT the number without a publisher score (some have one)",
            "of_which_no_publisher_score_in_the_grch38_view":
                "no publisher row with a score at the GRCh38 allele, in the completely scanned GRCh38 view; not a "
                "claim about the GRCh37 source (see the GRCh37 rows) and, outside REVEL's domain, not a missing value",
            "of_which_publisher_has_0_5_at_allele":
                "a publisher 0.5 at the allele; whether it belongs to the transcript the index value described is "
                "unverified",
            "domain": "REVEL scores single-nucleotide missense variants; the domain is read from the cohort's ClinVar "
                      "consequence (one consequence per row, not per transcript)"},
    }


def revel_source(pilot_part: Path, frame_path: Path, gtf: Path, gtf_sha256: str, archive: Path,
                 archive_sha256: str, summary: Path, summary_sha256: str, out: Path) -> int:
    import numpy as np

    _refuse_existing(out)
    start = time.monotonic()
    # Memory order (measured): the declared substitutions and the GTF are built and their bytes released BEFORE the
    # 667 MB archive is read, so the three large inputs are never held together.
    gtf_raw, gtf_sha = _read_all(gtf)
    if gtf_sha != gtf_sha256:
        raise SystemExit("GENCODE GTF differs from the declared digest")
    tx = _gtf_transcripts(gtf_raw)
    del gtf_raw
    frame, population, values, inputs = _load_frame(pilot_part, frame_path)
    summary_raw, summary_sha = _read_all(summary)
    if summary_sha != summary_sha256:
        raise SystemExit(f"ClinVar variant_summary has SHA-256 {summary_sha}, not the declared {summary_sha256}")
    cv_status, cv_transcript, cv_substitution, cv_grch37, cv_census = clinvar_declared(summary_raw, frame)
    n_summary = len(summary_raw)
    del summary_raw
    raw, sha = _read_all(archive)
    if sha != archive_sha256:
        raise SystemExit(f"REVEL archive has SHA-256 {sha}, not the declared {archive_sha256}")
    md5 = hashlib.md5(raw).hexdigest()
    if md5 != ZENODO_MD5:
        raise SystemExit(f"REVEL archive MD5 {md5} differs from Zenodo's published {ZENODO_MD5}")

    # The population's SNV alleles, keyed exactly as the publisher's GRCh38 view will be, and their ClinVar GRCh37 alleles.
    chrom = frame["chrom"].astype(str).str.replace(r"^chr", "", regex=True).to_numpy()
    pos = frame["pos"].to_numpy(dtype=np.int64)
    ref = frame["ref"].astype(str).str.upper().to_numpy()
    alt = frame["alt"].astype(str).str.upper().to_numpy()
    key_of_row = [f"{c}:{p}:{r}:{a}" if len(r) == 1 and len(a) == 1 else None
                  for c, p, r, a in zip(chrom, pos, ref, alt)]
    key37_of_row = [k37 if k is not None else None for k, k37 in zip(key_of_row, cv_grch37)]
    wanted = {k for k in key_of_row if k is not None}
    wanted37 = {k for k in key37_of_row if k is not None}

    census, chrom_rows, tx_count_hist, decimals, bad, matched, unresolved37 = scan_revel(raw, wanted, wanted37)
    scan_seconds = round(time.monotonic() - start, 1)

    # ---- at the population's alleles (label-blind until the end) --------------------------------------------------
    hist = values["revel_score"]
    hist_obs = np.isfinite(hist)
    hist_half = hist_obs & (hist == 0.5)
    genes = frame["gene_symbol"].fillna("").astype(str).to_numpy()
    missense = (frame["consequence"] == "missense_variant").to_numpy()
    label = frame["label"].to_numpy(dtype=int)
    allele = Counter()
    compare = Counter()
    by_domain: dict[str, Counter] = {}
    rescued = {k: Counter() for k in NO_PUBLISHER_SCORE}
    rescued_scored = 0
    state = Counter()
    state_of_row = np.empty(len(frame), dtype=object)
    cv_state = Counter()
    cv_state_of_row = np.empty(len(frame), dtype=object)
    agree = Counter()
    disagreement = Counter()
    for i, k in enumerate(key_of_row):
        rows = matched.get(k, []) if k is not None else []
        scores = {r[1] for r in rows}
        observed_scores = {float(s) for s in scores if s is not None}
        if rows:
            allele["alleles_with_rows"] += 1
            allele[f"rows_per_allele_{min(len(rows), 5)}{'+' if len(rows) >= 5 else ''}"] += 1
            if len(observed_scores) > 1:
                allele["alleles_with_several_distinct_scores"] += 1
            if None in scores:
                allele["alleles_with_a_row_without_score"] += 1
            if len({(r[2], r[3]) for r in rows}) > 1:
                allele["alleles_with_several_amino_acid_changes"] += 1
        # Historical (allele-level max of dbNSFP, then the 0.5 fill) against the publisher's rows.
        if not hist_obs[i]:
            cat = "hist_absent_pub_absent" if not observed_scores else "hist_absent_pub_present"
        elif hist_half[i]:
            cat = ("hist_0_5_pub_absent__fabricated" if not observed_scores else
                   "hist_0_5_pub_has_0_5__genuine" if 0.5 in observed_scores else "hist_0_5_pub_other_values")
        else:
            cat = ("hist_value_pub_absent" if not observed_scores else
                   "hist_value_equals_pub_max" if abs(max(observed_scores) - hist[i]) < 5e-7
                   else "hist_value_differs_from_pub_max")
        compare[cat] += 1
        dom = domain_of(k, bool(missense[i]))
        by_domain.setdefault(cat, Counter())[dom] += 1
        if cat in NO_PUBLISHER_SCORE and key37_of_row[i] in unresolved37:
            rescued[cat][dom] += 1
            rescued_scored += any(s is not None for s in unresolved37[key37_of_row[i]])
        # Two declared, label-blind CANDIDATE selections (owner ruling ace92e09): A, the GENCODE v50 MANE Select
        # transcript of the cohort's gene_symbol (a ClinVar-derived field), matched by stable identifier to the publisher's
        # transcript list; B, ClinVar's preferred name (variant_summary Name), matched to the publisher's amino-acid pair.
        # Applicability = the cohort's ClinVar consequence.
        if not missense[i]:
            s_ = t_ = "not_applicable_consequence"
        elif k is None:
            s_ = t_ = "not_applicable_not_snv"
        else:
            own = stable_mane_of_gene(tx, genes[i])
            if len(own) != 1:
                s_ = "transcript_unresolved"
            else:
                stable = next(iter(own))
                s_ = select_scalar(rows, lambda r, _s=stable: any(x.split(".", 1)[0] == _s for x in r[4]), MANE_AMBIGUOUS)
            if cv_status[i] != "declared":
                t_ = f"clinvar_{cv_status[i]}"
            else:
                t_ = select_scalar(rows, lambda r, _d=cv_substitution[i]: (r[2], r[3]) == _d, CLINVAR_AMBIGUOUS)
        if s_ == t_ == CANDIDATE:
            stable = next(iter(stable_mane_of_gene(tx, genes[i])))
            mane_rows = [r for r in rows if any(x.split(".", 1)[0] == stable for x in r[4])]
            cv_rows = [r for r in rows if (r[2], r[3]) == cv_substitution[i]]
            if mane_rows[0][1] == cv_rows[0][1]:
                agree["policy_agreement_same_score"] += 1
            else:
                agree["policy_disagreement_different_score"] += 1
                disagreement["rows"] += 1
                disagreement["amino_acid_change_differs"] += (mane_rows[0][2], mane_rows[0][3]) != cv_substitution[i]
                disagreement["mane_transcript_named_by_a_clinvar_selected_row"] += any(
                    stable in {x.split(".", 1)[0] for x in r[4]} for r in cv_rows)
                disagreement["crosses_0_5"] += (float(mane_rows[0][1]) >= 0.5) != (float(cv_rows[0][1]) >= 0.5)
        else:
            agree[f"mane={s_}|clinvar={t_}"] += 1
        cv_state[t_] += 1
        cv_state_of_row[i] = t_
        state[s_] += 1
        state_of_row[i] = s_
    def by_label(states, names):
        return {s_: {"pathogenic": int(((states == s_) & (label == 1)).sum()),
                     "benign": int(((states == s_) & (label == 0)).sum())} for s_ in sorted(names)}
    _write(out, {"part": "revel",
                 "inputs": {**inputs,
                            "revel_archive": {"file": archive.name, "bytes": len(raw), "sha256": sha, "md5": md5,
                                              "zenodo_record": "10.5281/zenodo.7072866", "member": REVEL_MEMBER},
                            "gencode_v50_gtf": {"file": gtf.name, "sha256": gtf_sha},
                            "clinvar_variant_summary": {"file": summary.name, "bytes": n_summary, "sha256": summary_sha}},
                 "population_sha256_v1": population.sha256_v1,
                 "source_census": dict(sorted(census.items())), "rows_by_chromosome": dict(chrom_rows),
                 "transcripts_per_row": {str(k): v for k, v in sorted(tx_count_hist.items())},
                 "score_decimal_places": {str(k): v for k, v in sorted(decimals.items())},
                 "nonfatal_anomalies": dict(bad),
                 "at_population_alleles": dict(sorted(allele.items())),
                 "historical_index_against_publisher": dict(sorted(compare.items())),
                 "historical_index_against_publisher_by_domain": {k: dict(sorted(v.items()))
                                                                  for k, v in sorted(by_domain.items())},
                 "publisher_grch37_rows_with_unresolved_grch38_at_population_alleles": len(unresolved37),
                 "selection_mane_select": {
                     "policy": "CANDIDATE selection, not a qualified transcript context: the GENCODE v50 MANE Select "
                               "transcript of the cohort's gene_symbol -- a ClinVar-derived field, so the selection is "
                               "not independent of the ClinVar record (part mane, pilot_missense_snv_rows_by_locus_mane, "
                               "measures the dependence) -- matched by stable identifier to the publisher's unversioned "
                               "transcript list, with no version, coding-sequence or residue check; applicability = the "
                               "cohort's ClinVar consequence. Applicable rows that disagree are a source conflict.",
                     "states": dict(sorted(state.items())),
                     "by_label_reported_after_selection": by_label(state_of_row, state)},
                 "selection_clinvar_declared_substitution": {
                     "policy": "CANDIDATE selection for clinical-context reconstruction, not a qualified transcript "
                               "context and not a prospective selector: the variant_summary Name (ClinVar's calculated "
                               "preferred name) of the row's unique GRCh38 record; the publisher rows at the GRCh38 allele "
                               "whose (aaref, aaalt) equal the declared single-residue substitution -- the allele and the "
                               "amino-acid pair only: the RefSeq accession and the residue position are parsed but not "
                               "matched (the publisher has neither). Applicable rows that disagree are an unresolved "
                               "context, not a source conflict.",
                     "clinvar_summary": cv_census,
                     "states": dict(sorted(cv_state.items())),
                     "by_label_reported_after_selection": by_label(cv_state_of_row, cv_state)},
                 "selection_agreement": dict(sorted(agree.items())),
                 "policy_disagreements": {
                     "note": "the two policies select different publisher rows; a different score under a different "
                             "selection is a policy disagreement, not a source conflict (owner ruling ace92e09, section 8)",
                     **{k: int(v) for k, v in sorted(disagreement.items())}},
                 "absence_as_evidence_correction": absence_semantics(
                     compare, by_domain, {k: dict(sorted(v.items())) for k, v in rescued.items()}, rescued_scored),
                 })
    print(f"scan {scan_seconds} s")
    _report_run(start)
    return 0


# -------------------------------------------------------------------------------------------------------------- collect
def collect(repo: Path, parts: list[Path], out: Path, check: bool = False) -> int:
    """Write the record, or with ``check`` compare what would be written with the committed file byte for byte (writing
    nothing) and print IDENTICAL or DIFFERS with every differing top-level field."""
    from genomic_variant_classifier.information_sources import census as C

    if not check:
        _refuse_existing(out)
    loaded = {}
    for p in parts:
        d, raw = _read_part(p)
        if d["part"] in loaded:
            raise SystemExit(f"two {d['part']} parts")
        loaded[d["part"]] = {"record": d, "sha256": hashlib.sha256(raw).hexdigest(), "file": p.name}
    if sorted(loaded) != ["index-defaults", "mane", "pilot", "revel"]:
        raise SystemExit(f"parts {sorted(loaded)}: need index-defaults, mane, pilot and revel")
    pops = {loaded[k]["record"].get("population_sha256_v1") or loaded[k]["record"]["population"]["sha256_v1"]
            for k in ("pilot", "mane", "revel")}
    if len(pops) != 1:
        raise SystemExit("the parts measured different populations")
    cohorts = {loaded[k]["record"]["inputs"]["cohort"]["sha256"] for k in ("pilot", "mane", "revel")}
    if len(cohorts) != 1:
        raise SystemExit("the parts read different cohorts")
    frame = loaded["pilot"]["record"]["frame"]["sha256"]
    if any(loaded[k]["record"]["inputs"]["pilot_frame"]["sha256"] != frame for k in ("mane", "revel")):
        raise SystemExit("the mane or revel part read another population frame than the pilot part wrote")
    record = {"schema": SCHEMA,
              "mode": "shadow: measured on the real source files; nothing in production changed; nothing fitted",
              "parts": {k: {"file": v["file"], "sha256": v["sha256"]} for k, v in sorted(loaded.items())},
              **{k.replace("-", "_"): {kk: vv for kk, vv in v["record"].items() if kk != "part"}
                 for k, v in sorted(loaded.items())},
              "procedure": {sym: C.symbol_closure_digest(repo, sym) for sym in PROCEDURE}}
    data = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    if check:
        committed = out.read_bytes()
        if committed == data:
            print(f"IDENTICAL: {out.name} sha256 {hashlib.sha256(data).hexdigest()}")
            return 0
        old_record = json.loads(committed.decode("ascii"))
        diff = sorted(k for k in set(old_record) | set(record) if old_record.get(k) != record.get(k))
        print(f"DIFFERS: {out.name}: measured sha256 {hashlib.sha256(data).hexdigest()}, committed "
              f"{hashlib.sha256(committed).hexdigest()}; fields {diff}")
        return 1
    out.write_bytes(data)
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index-defaults")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for name in ("pilot", "mane", "revel"):
        p = sub.add_parser(name)
        for a in (("cohort", "index", "scratch", "out", "out_frame") if name == "pilot" else
                  ("pilot_part", "frame", "out")):
            p.add_argument(f"--{a.replace('_', '-')}", type=Path, required=True)
        if name in ("mane", "revel"):
            p.add_argument("--gtf", type=Path, required=True)
            p.add_argument("--gtf-sha256", required=True)
        if name == "revel":
            p.add_argument("--archive", type=Path, required=True)
            p.add_argument("--archive-sha256", required=True)
            p.add_argument("--clinvar-summary", type=Path, required=True)
            p.add_argument("--clinvar-summary-sha256", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--parts", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--check", action="store_true", help="compare with --out byte for byte; write nothing")
    a = ap.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if a.cmd == "index-defaults":
        return index_defaults(a.index, a.out)
    if a.cmd == "pilot":
        return pilot_defaults(repo, a.cohort, a.index, a.scratch, a.out, a.out_frame)
    if a.cmd == "mane":
        return mane_coverage(a.pilot_part, a.frame, a.gtf, a.gtf_sha256, a.out)
    if a.cmd == "revel":
        return revel_source(a.pilot_part, a.frame, a.gtf, a.gtf_sha256, a.archive, a.archive_sha256,
                            a.clinvar_summary, a.clinvar_summary_sha256, a.out)
    return collect(repo, a.parts, a.out, a.check)


if __name__ == "__main__":
    sys.exit(main())
