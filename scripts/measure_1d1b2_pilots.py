"""Stage 1D-1b.2 pilots on REAL source joins: EVE and gnomAD allele frequency (owner ruling 7f4b69d0, sections 5 and 19),
and the extent of ABSENCE-AS-EVIDENCE-1 on the Run 15 cohort. Author: Monzia Moodie.

Measured on the real files, never on the scaled Run 15 matrices (the EVE finding showed why). Three subcommands:

  eve-parse    on the machine that holds EVE (10.6 GB, 3,211 files): every *.csv parsed ONCE through a VerifiedReader
               (materialization.parse_eve_file; the digest covers exactly the bytes parsed), and the CURRENT connector's
               parse (EVEConnector._parse_single_csv) of the same file compared with it. Resumable: --max-seconds bounds
               one call; finished files are recorded in --work and skipped.
  eve-collect  the per-file results into one table (parquet) and one manifest (JSON), refusing unless the manifest covers
               the directory's *.csv files exactly.
  run          the joins, on the cohort the production pipeline builds (DataPrepPipeline._load_and_label, review tier <= 3,
               conflicting removed), with protein coordinates from the AlphaMissense index cache production reads.
               Writes the measurement record (--out, refusing to overwrite).

Every input file is read through materialization.read_verified_bytes, and its digest is recorded; the record is bound to
the code by each materialiser's semantic-closure digest (a test re-measures them). Changes nothing in production.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = "gvc.pilot-measurements.1d1b2/1"
POPULATION_ID = "clinvar_grch38_clean:labelled:review_tier_le_3:non_conflicting"
SCRIPT = "scripts/measure_1d1b2_pilots.py"
MAT = "src/genomic_variant_classifier/information_sources/materialization.py"
#: The code each measurement step executed; the record binds their semantic-closure digests (tests re-measure them).
PROCEDURE = (f"{SCRIPT}::eve_parse", f"{SCRIPT}::eve_collect", f"{SCRIPT}::population_step", f"{SCRIPT}::load_population",
             f"{SCRIPT}::eve_join", f"{SCRIPT}::load_eve_table", f"{SCRIPT}::run", f"{MAT}::parse_verified",
             f"{MAT}::parse_eve_file", f"{MAT}::entry_map_from_index", f"{MAT}::read_verified_bytes",
             f"{MAT}::materialize_eve", f"{MAT}::materialize_gnomad_af", f"{MAT}::legacy_eve_values",
             f"{MAT}::legacy_af_values")


# ------------------------------------------------------------------------------------------------------ eve-parse
def eve_parse(eve_dir: Path, uniprot: Path, work: Path, max_seconds: float) -> int:
    import pandas as pd

    from genomic_variant_classifier.data.eve import EVEConnector, load_eve_entry_map
    from genomic_variant_classifier.information_sources import materialization as M

    work.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    uni = M.read_verified_bytes(uniprot)
    entry_map = M.entry_map_from_index(uni.value)
    legacy_map = load_eve_entry_map(uniprot)
    if entry_map != legacy_map:
        raise SystemExit("the entry map built from the verified bytes differs from the legacy loader's")
    (work / "entry_map.json").write_text(json.dumps({"sha256": uni.sha256, "bytes": uni.n_bytes,
                                                     "entries": len(entry_map), "equal_to_legacy_loader": True}),
                                         encoding="utf-8")
    names = sorted(p.name for p in eve_dir.glob("*.csv"))
    (work / "listing.json").write_text(json.dumps(names), encoding="utf-8")
    done_path = work / "files.jsonl"
    done = set()
    if done_path.exists():
        done = {json.loads(line)["name"] for line in done_path.read_text(encoding="utf-8").splitlines() if line}
    connector = EVEConnector(eve_path=None)
    batch, records = [], []
    for name in names:
        if name in done:
            continue
        if time.monotonic() - start > max_seconds:
            break
        stem = Path(name).stem
        parsed = M.parse_verified(eve_dir / name, lambda s, _stem=stem: M.parse_eve_file(s, _stem, entry_map))
        rows, resolved, ok = parsed.value
        legacy = connector._parse_single_csv(eve_dir / name, entry_map)
        mine = rows.dropna(subset=["eve_score"]).copy()
        mine["eve_score"] = mine["eve_score"].clip(0.0, 1.0)
        a = mine[["gene_symbol", "aa_change", "eve_score"]].reset_index(drop=True)
        b = legacy[["gene_symbol", "aa_change", "eve_score"]].reset_index(drop=True)
        equal = bool(len(a) == len(b) and (a.astype(str) == b.astype(str)).all().all()
                     and ((a["eve_score"].to_numpy() == b["eve_score"].to_numpy()).all()))
        rows = rows.copy()
        rows["file"] = name
        batch.append(rows)
        records.append({"name": name, "sha256": parsed.sha256, "bytes": parsed.n_bytes, "rows": int(len(rows)),
                        "scored": int(rows["eve_score"].notna().sum()), "resolved_by_entry_map": bool(resolved),
                        "has_required_columns": bool(ok), "legacy_parse_equal": equal})
    if batch:
        part = work / f"rows_{len(list(work.glob('rows_*.parquet'))):05d}.parquet"
        pd.concat(batch, ignore_index=True).to_parquet(part, index=False)
        with done_path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    total_done = len(done) + len(records)
    print(f"eve-parse: {len(records)} files this call; {total_done}/{len(names)} done; "
          f"{sum(not r['legacy_parse_equal'] for r in records)} legacy-parse differences this call")
    return 0 if total_done == len(names) else 3


def eve_collect(work: Path, table: Path, manifest: Path) -> int:
    import pandas as pd

    names = json.loads((work / "listing.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (work / "files.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if sorted(r["name"] for r in records) != sorted(names) or len(records) != len(names):
        raise SystemExit("the per-file records do not cover the directory listing exactly once")
    parts = sorted(work.glob("rows_*.parquet"))
    rows = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    per_file = rows.groupby("file").size()
    for r in records:
        if int(per_file.get(r["name"], 0)) != r["rows"]:
            raise SystemExit(f"{r['name']}: table rows differ from the recorded count")
    rows.to_parquet(table, index=False)
    manifest.write_text(json.dumps({"files": sorted(records, key=lambda r: r["name"]),
                                    "entry_map": json.loads((work / "entry_map.json").read_text(encoding="utf-8"))},
                                   sort_keys=True), encoding="utf-8")
    print(f"eve-collect: {len(records)} files, {len(rows)} substitution rows -> {table}")
    return 0


# --------------------------------------------------------------------------------------------------------- run
def _auroc(indicator, label) -> float | None:
    """AUROC of a 0/1 indicator for the label (ties: mid-rank). None when one class is absent."""
    import numpy as np

    y = np.asarray(label, dtype=int)
    x = np.asarray(indicator, dtype=float)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if not pos or not neg:
        return None
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def _by_label(mask, label) -> dict:
    import numpy as np

    y = np.asarray(label, dtype=int)
    m = np.asarray(mask, dtype=bool)
    out = {}
    for name, cls in (("pathogenic", 1), ("benign", 0)):
        n = int((y == cls).sum())
        k = int((m & (y == cls)).sum())
        out[name] = {"rows": n, "count": k, "fraction": round(k / n, 6) if n else None}
    out["auroc_of_the_indicator_for_the_label"] = None if _auroc(m, y) is None else round(_auroc(m, y), 6)
    return out


UNREAD_COHORT_COLUMNS = ("metadata",)


def progress(step: str, *detail) -> None:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    print(f"[pilot] {step} {' '.join(str(d) for d in detail)} (peak {peak} MB)", file=sys.stderr, flush=True)


def load_population(pipe, cohort_bytes: bytes):
    """DataPrepPipeline._load_and_label on the verified cohort bytes, without the per-row ``metadata`` struct (review
    status and rsID as Python dicts: 2.4 GB of the 4.0 GB frame for 4,399,089 rows). _load_and_label never reads it -- its
    selection reads clinical_sig, ReviewStatus, ref, alt and variant_id -- so the population (rows and order) is the one
    production builds; tests/unit/test_pilot_record.py shows the two selections equal on a fixture."""
    from unittest import mock

    import pandas as pd
    import pyarrow.parquet as pq

    names = pq.ParquetFile(io.BytesIO(cohort_bytes)).schema_arrow.names
    keep = [c for c in names if c not in UNREAD_COHORT_COLUMNS]
    original = pd.read_parquet

    def without_unread(path, *a, **k):
        k.setdefault("columns", keep)
        return original(path, *a, **k)

    with mock.patch.object(pd, "read_parquet", without_unread):
        frame = pipe._load_and_label(io.BytesIO(cohort_bytes))
    return frame, sorted(set(names) - set(keep))


POPULATION_COLUMNS = ("variant_id", "chrom", "pos", "ref", "alt", "gene_symbol", "consequence", "protein_change",
                      "allele_freq", "label", "protein_pos", "wt_aa", "mut_aa")


def population_step(args: argparse.Namespace) -> int:
    """The population as production builds it (verified cohort bytes), with protein keys exactly as production reads them
    (the AlphaMissense index cache, from verified bytes), written as one parquet frame and its identity."""
    import pandas as pd

    from genomic_variant_classifier.data.hgvsp_parser import fill_protein_columns_from_hgvsp
    from genomic_variant_classifier.data.protein_coords import ProteinCoordConnector
    from genomic_variant_classifier.data.real_data_prep import DataPrepConfig, DataPrepPipeline
    from genomic_variant_classifier.information_sources import census as C
    from genomic_variant_classifier.information_sources import materialization as M

    for out in (args.out_frame, args.out_meta):
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    repo = Path(__file__).resolve().parents[1]
    registry = sorted(C.strict_load(repo / C.LEDGER_PATH)["datasets"])
    inputs = {}
    M.check_registered("dataset:clinvar_cohort", registry)
    cohort = M.read_verified_bytes(args.clinvar)
    inputs["cohort"] = {"logical_source": "dataset:clinvar_cohort", "file": args.clinvar.name, "bytes": cohort.n_bytes,
                        "sha256": cohort.sha256}
    pipe = DataPrepPipeline(DataPrepConfig(output_dir=args.scratch / "splits"))
    frame, unread = load_population(pipe, cohort.value)
    del cohort
    inputs["cohort"]["columns_not_read"] = unread
    progress("population built", len(frame))
    M.check_registered("dataset:protein_coordinates", registry)
    am = M.read_verified_bytes(args.am_index)
    inputs["protein_coordinates"] = {"logical_source": "dataset:protein_coordinates", "file": args.am_index.name,
                                     "bytes": am.n_bytes, "sha256": am.sha256}
    pc = ProteinCoordConnector(alphamissense_file=None, cache_dir=args.am_index.parent)
    pc._index = pd.read_parquet(io.BytesIO(am.value))
    keyed = pc.annotate_dataframe(frame)
    keyed["is_missense"] = keyed["consequence"].fillna("").str.contains("missense", case=False).astype(int)
    keyed, n_hgvsp = fill_protein_columns_from_hgvsp(keyed, source_col="protein_change")
    progress("protein coordinates joined", int(keyed["protein_pos"].notna().sum()))
    keyed[list(POPULATION_COLUMNS)].to_parquet(args.out_frame, index=False)
    digest = M.read_verified_bytes(args.out_frame).sha256
    args.out_meta.write_text(json.dumps({"frame_sha256": digest, "rows": int(len(keyed)), "inputs": inputs,
                                         "step_code": C.symbol_closure_digest(repo, f"{SCRIPT}::population_step"),
                                         "hgvsp_fallback_coordinates": int(n_hgvsp)}, sort_keys=True),
                             encoding="utf-8")
    print(f"population: {len(keyed):,} rows -> {args.out_frame} (sha256 {digest})")
    return 0


def load_eve_table(raw: bytes):
    """The transported EVE table with gene and substitution as categoricals built from the parquet dictionary encoding
    (44.5 million rows: object strings would need about 8 GB)."""
    import gc

    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(raw), columns=["gene_symbol", "aa_change", "eve_score"])
    cols = {}
    for c in ("gene_symbol", "aa_change"):
        enc = pc.dictionary_encode(table[c]).combine_chunks()
        cols[c] = pd.Categorical.from_codes(enc.indices.to_numpy(zero_copy_only=False).astype(np.int32),
                                            categories=pd.Index(enc.dictionary.to_pylist(), dtype=object))
    cols["eve_score"] = table["eve_score"].to_numpy()
    del table
    gc.collect()
    return pd.DataFrame(cols, copy=False)


def eve_join(args: argparse.Namespace) -> int:
    """EVE materialised for the population in its own process, with the legacy comparison; writes arrays and receipt."""
    import numpy as np
    import pandas as pd

    from genomic_variant_classifier.information_sources import census as C
    from genomic_variant_classifier.information_sources import materialization as M

    for out in (args.out_npz, args.out_json):
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
    repo = Path(__file__).resolve().parents[1]
    registry = sorted(C.strict_load(repo / C.LEDGER_PATH)["datasets"])
    meta = json.loads(args.population_meta.read_text(encoding="utf-8"))
    pop_bytes = M.read_verified_bytes(args.population)
    if pop_bytes.sha256 != meta["frame_sha256"]:
        raise SystemExit("the population frame is not the one the population step wrote")
    keyed = pd.read_parquet(io.BytesIO(pop_bytes.value), columns=["variant_id", "gene_symbol", "consequence",
                                                                  "protein_change", "protein_pos", "wt_aa", "mut_aa"])
    del pop_bytes
    population = M.population_from_frame(keyed, POPULATION_ID)
    manifest = json.loads(args.eve_manifest.read_text(encoding="utf-8"))
    table_bytes = M.read_verified_bytes(args.eve_table)
    if table_bytes.sha256 != args.eve_table_sha256:
        raise SystemExit("the EVE table is not the one produced on the source machine")
    table_sha256, n_table_bytes = table_bytes.sha256, table_bytes.n_bytes
    rows = load_eve_table(table_bytes.value)
    del table_bytes
    progress("EVE table loaded", len(rows))
    files = manifest["files"]
    content, mf = M.directory_manifest((f["name"], f["sha256"], f["bytes"]) for f in files)
    source = M.EveSourceTable(rows, content, tuple(mf),
                              sum(f["resolved_by_entry_map"] for f in files if f["has_required_columns"]),
                              sum(not f["resolved_by_entry_map"] for f in files if f["has_required_columns"]),
                              tuple(f["name"] for f in files if not f["has_required_columns"]))
    M.check_registered("dataset:uniprot_index", registry)
    eve = M.materialize_eve(keyed, population, source, registered_datasets=registry,
                            entry_map_input={"logical_source": "dataset:uniprot_index", **manifest["entry_map"]})
    progress("EVE materialised", int(eve.column.observed_mask().sum()))
    lookup = M.legacy_lookup_from_source(source, eve.source_rows_relevant)
    ambiguous = eve.column.details == M.EVE_DETAILS.index("ambiguous_source_record")
    cmp: dict = {"rows_matching_a_key_scored_more_than_once": int(ambiguous.sum())}
    try:
        legacy_all = M.legacy_eve_values(keyed, lookup)
        cmp.update(M.compare_legacy(M.engineer_fill(eve.legacy_projection, 0.5), legacy_all))
    except ValueError as exc:                     # the legacy join multiplied rows: measured, not hidden
        cmp["legacy_join_on_the_whole_population"] = str(exc)
        keep = ~ambiguous
        sub = keyed[keep].reset_index(drop=True)
        legacy_sub = M.legacy_eve_values(sub, lookup)
        cmp["on_rows_not_matching_a_duplicated_key"] = M.compare_legacy(
            M.engineer_fill(eve.legacy_projection[keep], 0.5), legacy_sub)
        cmp["identical"] = False
    progress("EVE compared with the legacy join", cmp.get("identical"))
    buf = io.BytesIO()
    np.savez(buf, values=eve.column.values, states=eve.column.states, details=eve.column.details,
             legacy_projection=eve.legacy_projection)
    args.out_npz.write_bytes(buf.getvalue())
    files_ok = sum(f["legacy_parse_equal"] for f in files)
    args.out_json.write_text(json.dumps({
        "npz_sha256": M.read_verified_bytes(args.out_npz).sha256,
        "step_code": C.symbol_closure_digest(repo, f"{SCRIPT}::eve_join"),
        "receipt": eve.receipt.as_dict(), "legacy_equivalence": cmp,
        "source": {"logical_source": "dataset:eve", "directory_manifest_sha256": content, "files": len(files),
                   "bytes": sum(f["bytes"] for f in files), "table_sha256": table_sha256, "table_bytes": n_table_bytes,
                   "parsed_on": "the owner's machine, through VerifiedReader (one pass per file)",
                   "legacy_parse_equal_files": files_ok, "substitution_rows": int(len(rows)),
                   "scored_rows": int(np.isfinite(rows["eve_score"].to_numpy()).sum()),
                   "files_resolved_by_entry_map": source.files_resolved_by_map,
                   "files_by_filename_prefix_fallback": source.files_by_fallback,
                   "fallback_files": sorted(f["name"] for f in files if not f["resolved_by_entry_map"])},
    }, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"eve-join: observed {int(eve.column.observed_mask().sum()):,} of {len(population):,}")
    return 0


def by_chromosome(chrom, observed) -> dict:
    """Rows and observed rows by chromosome class (autosome, X, Y, MT, other), to locate where observation fails."""
    import numpy as np

    c = np.asarray(chrom.astype(str))
    classes = np.where(np.char.isdigit(c.astype(str)), "autosome", np.where(np.isin(c, ["X", "Y", "MT"]), c, "other"))
    obs = np.asarray(observed, dtype=bool)
    return {k: {"rows": int((classes == k).sum()), "observed": int(((classes == k) & obs).sum())}
            for k in ("autosome", "X", "Y", "MT", "other") if (classes == k).any()}


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from genomic_variant_classifier.data.dbnsfp import DbNSFPConnector
    from genomic_variant_classifier.information_sources import census as C
    from genomic_variant_classifier.information_sources import columns as K
    from genomic_variant_classifier.information_sources import observation as O
    from genomic_variant_classifier.information_sources import materialization as M
    from genomic_variant_classifier.models.variant_ensemble import engineer_features

    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    repo = Path(__file__).resolve().parents[1]
    registry = sorted(C.strict_load(repo / C.LEDGER_PATH)["datasets"])
    inputs = {}

    def verified(label: str, logical: str, path: Path) -> M.Parsed:
        M.check_registered(logical, registry)
        parsed = M.read_verified_bytes(path)
        inputs[label] = {"logical_source": logical, "file": path.name, "bytes": parsed.n_bytes, "sha256": parsed.sha256}
        return parsed

    # 1-2. The population and its protein keys, built in a separate process (population), verified here.
    meta = json.loads(args.population_meta.read_text(encoding="utf-8"))
    eve_meta = json.loads(args.eve_json.read_text(encoding="utf-8"))
    for step, recorded in (("population_step", meta.get("step_code")), ("eve_join", eve_meta.get("step_code"))):
        if recorded != C.symbol_closure_digest(repo, f"{SCRIPT}::{step}"):
            raise SystemExit(f"{step} ran with code that differs from the code now; re-run it")
    pop_bytes = M.read_verified_bytes(args.population)
    if pop_bytes.sha256 != meta["frame_sha256"]:
        raise SystemExit("the population frame is not the one the population step wrote")
    inputs.update(meta["inputs"])
    keyed = pd.read_parquet(io.BytesIO(pop_bytes.value))
    del pop_bytes
    frame = keyed
    n_hgvsp = meta["hgvsp_fallback_coordinates"]
    population = M.population_from_frame(frame, POPULATION_ID)
    label = frame["label"].to_numpy(dtype=int)
    progress("population loaded", len(frame))

    # 3. EVE: joined in its own process (eve-join), loaded here and re-verified by the column's identity.
    eve_arrays = M.read_verified_bytes(args.eve_npz)
    if eve_arrays.sha256 != eve_meta["npz_sha256"]:
        raise SystemExit("the EVE column arrays are not the ones eve-join wrote")
    with np.load(io.BytesIO(eve_arrays.value)) as z:
        eve_column = K.ObservationColumn(
            "eve_score", O.SourceIdentity("dataset:eve", eve_meta["receipt"]["content_sha256"]), population,
            z["values"], z["states"], z["details"], tuple(M.EVE_DETAILS))
        eve_legacy_projection = z["legacy_projection"]
    if eve_column.sha256 != eve_meta["receipt"]["column_sha256"]:
        raise SystemExit("the EVE column does not reproduce its receipt's identity")
    eve_obs = eve_column.observed_mask()
    eve_vals = eve_column.values
    progress("EVE column loaded and re-verified", int(eve_obs.sum()))

    # 4. gnomAD allele frequency.
    gn = verified("gnomad_af", "dataset:gnomad_af", args.gnomad)
    af = M.materialize_gnomad_af(frame, population, gn, registered_datasets=registry)
    legacy_af = M.legacy_af_values(frame, args.gnomad)
    af_cmp = M.compare_legacy(af.legacy_projection, legacy_af)
    progress("allele frequency materialised and compared", af_cmp["identical"])
    af_obs = af.column.observed_mask()
    af_vals = af.column.values
    af_filled = np.isnan(legacy_af)                 # the legacy join produced no value: engineer_features fills 0.0

    # 5. REVEL through the CURRENT dbNSFP connector, its fill replaced by NaN so an unobserved score stays visible.
    verified("dbnsfp", "dataset:dbnsfp", args.dbnsfp)          # identity recorded; the connector reads the path itself
    revel_frame = DbNSFPConnector(dbnsfp_file=args.dbnsfp).annotate_dataframe(
        frame[["chrom", "pos", "ref", "alt"]].copy(), revel_score=float("nan"), sift_score=float("nan"),
        polyphen2_score=float("nan"), cadd_phred=float("nan"))
    revel = revel_frame["revel_score"].to_numpy(dtype=np.float64)
    revel_obs = np.isfinite(revel)
    progress("REVEL measured", int(revel_obs.sum()))

    # 6. The legacy features a model sees, from the CURRENT builder on the legacy joined values.
    legacy_in = frame[["ref", "alt", "consequence"]].copy()
    legacy_in["allele_freq"] = legacy_af
    legacy_in["revel_score"] = M.engineer_fill(revel, 0.5)
    legacy_in["eve_score"] = M.engineer_fill(eve_legacy_projection, 0.5)
    for c in ("sift_score", "polyphen2_score", "cadd_phred"):
        legacy_in[c] = revel_frame[c].to_numpy(dtype=np.float64)
    feats = engineer_features(legacy_in)
    missense = frame["consequence"].fillna("").str.contains("missense", case=False).to_numpy()

    def i(x) -> int:
        return int(np.asarray(x).sum())

    record = {
        "schema": SCHEMA,
        "mode": "shadow: measured on the real source files; nothing in production changed",
        "inputs": dict(sorted(inputs.items())),
        "eve_source": eve_meta["source"],
        "population": {"population_id": POPULATION_ID, "rows": len(population), "sha256": population.sha256,
                       "pathogenic": i(label == 1), "benign": i(label == 0), "missense": i(missense),
                       "built_by": "DataPrepPipeline._load_and_label (DataPrepConfig defaults: review tier <= 3, "
                                   "conflicting removed)", "hgvsp_fallback_coordinates": int(n_hgvsp)},
        "eve": {
            "receipt": eve_meta["receipt"],
            "legacy_equivalence": eve_meta["legacy_equivalence"],
            "observed": i(eve_obs), "observed_exactly_0_5": i(eve_obs & (eve_vals == 0.5)),
            "legacy_coverage_count_score_ne_0_5": i(M.engineer_fill(eve_legacy_projection, 0.5) != 0.5),
            "fabricated_0_5": i(~eve_obs),
            "unobserved_by_label": _by_label(~eve_obs, label),
            "unobserved_among_missense_by_label": _by_label(~eve_obs[missense], label[missense]),
        },
        "allele_frequency": {
            "receipt": af.receipt.as_dict(),
            "legacy_equivalence": af_cmp,
            "observed": i(af_obs), "observed_exactly_zero": i(af_obs & (af_vals == 0.0)),
            "observed_below_1e-4": i(af_obs & (af_vals < 1e-4)),
            "unobserved": i(~af_obs),
            "legacy_value_filled": i(af_filled),
            "unobserved_but_legacy_kept_a_first_record": i(~af_obs & ~af_filled),
            "legacy_af_is_ultra_rare": i(feats["af_is_ultra_rare"] == 1),
            "legacy_af_is_ultra_rare_set_by_absence": i((feats["af_is_ultra_rare"] == 1) & af_filled),
            "legacy_af_is_absent": i(feats["af_is_absent"] == 1),
            "legacy_af_is_absent_set_by_absence": i((feats["af_is_absent"] == 1) & af_filled),
            "legacy_af_is_absent_observed_zero": i((feats["af_is_absent"] == 1) & af_obs & (af_vals == 0.0)),
            "unobserved_by_label": _by_label(~af_obs, label),
            "by_chromosome_class": by_chromosome(frame["chrom"], af_obs),
        },
        "revel": {
            "measured_through": "DbNSFPConnector.annotate_dataframe with the REVEL fill replaced by NaN (a measurement; "
                                "REVEL is not a pilot producer)",
            "observed": i(revel_obs), "unobserved": i(~revel_obs),
            "unobserved_missense": i(~revel_obs & missense), "unobserved_non_missense": i(~revel_obs & ~missense),
            "legacy_revel_pathogenic": i(feats["revel_pathogenic"] == 1),
            "legacy_revel_pathogenic_set_by_absence": i((feats["revel_pathogenic"] == 1) & ~revel_obs),
            "observed_revel_pathogenic": i(revel_obs & (revel >= 0.5)),
            "unobserved_by_label": _by_label(~revel_obs, label),
            "legacy_revel_pathogenic_by_label": _by_label(feats["revel_pathogenic"].to_numpy() == 1, label),
        },
        "is_snv": {"rows_with_absent_ref_or_alt": i(frame["ref"].isna() | frame["alt"].isna())},
        "procedure": {sym: C.symbol_closure_digest(repo, sym) for sym in PROCEDURE},
        "environments": {"eve_parse": args.eve_parse_environment.read_text(encoding="utf-8").strip(),
                         "joins": f"Python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}"},
    }
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
                        encoding="ascii")
    print(json.dumps({k: record[k] for k in ("population",)}, indent=1))
    print(f"EVE observed {record['eve']['observed']:,}; AF observed {record['allele_frequency']['observed']:,}; "
          f"REVEL observed {record['revel']['observed']:,} of {len(population):,}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("eve-parse")
    p.add_argument("--eve-dir", type=Path, required=True)
    p.add_argument("--uniprot", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=150.0)
    p = sub.add_parser("eve-collect")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--table", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p = sub.add_parser("population")
    for name in ("clinvar", "am-index", "out-frame", "out-meta", "scratch"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p = sub.add_parser("eve-join")
    for name in ("population", "population-meta", "eve-table", "eve-manifest", "out-npz", "out-json"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--eve-table-sha256", required=True)
    p = sub.add_parser("run")
    for name in ("population", "population-meta", "gnomad", "dbnsfp", "eve-npz", "eve-json", "eve-parse-environment",
                 "out"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "eve-parse":
        return eve_parse(a.eve_dir, a.uniprot, a.work, a.max_seconds)
    if a.cmd == "eve-collect":
        return eve_collect(a.work, a.table, a.manifest)
    if a.cmd == "population":
        return population_step(a)
    if a.cmd == "eve-join":
        return eve_join(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
