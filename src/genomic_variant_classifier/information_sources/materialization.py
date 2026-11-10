"""Stage 1D-1b.2: source materialisation -- bytes bound to the parse, receipts measured by the materialiser, and the two
representative pilot producers (EVE and gnomAD allele frequency). SHADOW MODE: nothing in production imports this module.

Owner ruling 7f4b69d0 (2026-11-09), sections 4.1, 5 and 19.

* ``VerifiedReader`` hashes exactly the bytes a parser consumes, in the same single pass, and ``parse_verified`` refuses
  unless the parser consumed the whole file (no buffered or unread remainder) and, when a pin is given, the digest equals
  it. Hashing a file and reopening it to parse proves nothing about the parse (the file can change in between); here there
  is one open and one stream. A columnar format that needs random access (parquet) is read into memory through the same
  reader and parsed from those bytes (``read_verified_bytes``).
* ``MaterializationReceipt`` is written by the materialiser itself from what it measured: the logical source, the digest of
  the bytes parsed, the release (only when the source states it), the parser's identity (the semantic-closure digest of the
  materialising function, measured from the code), the population's order-binding identity, the per-state and per-detail
  row counts (they sum to the population), and the materialised column's identity. Every field is derived from the
  column and the code (``make_receipt``); none is a caller-supplied string.
* ``materialize_eve`` (EVE: a genuine score of 0.5 is not "absent") and ``materialize_gnomad_af`` (allele frequency: an
  observed 0 is not "unavailable") are the pilots. Each reproduces its legacy connector's join EXACTLY -- the same keys, the
  same gene resolution, the same parse -- and records, per row, why a value is or is not observed. Each also returns the
  legacy projection (the value the current code writes), and ``legacy_*`` functions run the CURRENT connector code on the
  same input, so the shadow comparison "old numerical values unchanged" is measured, never assumed.

numpy and pandas are required.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from genomic_variant_classifier.information_sources.columns import CODE, ObservationColumn, Population
from genomic_variant_classifier.information_sources.observation import HEX64, ObservationState, SourceIdentity

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA = "gvc.materialization-receipt.1d1b2/1"
S = ObservationState
_BLOCK = 1 << 20


# ------------------------------------------------------------------------------------------- bytes bound to the parse
class VerifiedReader(io.RawIOBase):
    """A raw binary stream that hashes exactly what is read through it. ``finish()`` refuses unless the underlying file is
    at end of file, and, with a pin, unless the digest equals the pin."""

    def __init__(self, raw: io.BufferedIOBase | io.RawIOBase, expected_sha256: str | None = None):
        super().__init__()
        if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not HEX64.match(expected_sha256)):
            raise ValueError("pin a lower-case SHA-256, or None to measure")
        self._raw, self._pin = raw, expected_sha256
        self._h = hashlib.sha256()
        self._n = 0
        self._eof = False
        self.verified_sha256: str | None = None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:      # type: ignore[override]
        chunk = self._raw.read(len(b))
        if not chunk:
            self._eof = True
            return 0
        n = len(chunk)
        b[:n] = chunk
        self._h.update(chunk)
        self._n += n
        return n

    @property
    def bytes_read(self) -> int:
        return self._n

    def finish(self) -> str:
        """Verify and return the digest of the bytes read. Refuses on an unread remainder or a pin mismatch."""
        if not self._eof and self._raw.read(1):
            raise ValueError("the parser did not consume the whole source; the digest would not cover it")
        self._eof = True
        digest = self._h.hexdigest()
        if self._pin is not None and digest != self._pin:
            raise ValueError(f"the bytes parsed have SHA-256 {digest}, not the pinned {self._pin}")
        self.verified_sha256 = digest
        return digest

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


@dataclass(frozen=True)
class Parsed:
    value: Any
    sha256: str
    n_bytes: int


def parse_verified(path: Path, parser: Callable[[io.BufferedReader], Any], *,
                   expected_sha256: str | None = None) -> Parsed:
    """Open ``path`` once and give ``parser`` a buffered stream whose every byte is hashed. Refuses unless the parser left
    nothing unread -- neither in the buffer nor in the file -- so the digest covers exactly the bytes parsed."""
    reader = VerifiedReader(open(path, "rb"), expected_sha256)
    buffered = io.BufferedReader(reader, buffer_size=_BLOCK)
    try:
        value = parser(buffered)
        if buffered.read(1):
            raise ValueError(f"{path}: the parser stopped before end of file; the digest would not match the parse")
        digest = reader.finish()
        return Parsed(value, digest, reader.bytes_read)
    finally:
        buffered.close()


def read_verified_bytes(path: Path, *, expected_sha256: str | None = None) -> Parsed:
    """The whole file through one VerifiedReader: parse from ``Parsed.value`` (bytes) and the parse IS of the hashed bytes."""
    def take_all(stream: io.BufferedReader) -> bytes:
        return stream.read()
    return parse_verified(path, take_all, expected_sha256=expected_sha256)


def directory_manifest(entries: Iterable[tuple[str, str, int]]) -> tuple[str, list[dict[str, Any]]]:
    """A directory source's content identity: SHA-256 of its canonical manifest (relative name, file SHA-256, size)."""
    rows = [{"name": n, "sha256": h, "bytes": b} for n, h, b in sorted(entries)]
    if len({r["name"] for r in rows}) != len(rows):
        raise ValueError("a manifest names each file once")
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), rows


# ----------------------------------------------------------------------------------------------------------- receipts
@dataclass(frozen=True)
class MaterializationReceipt:
    feature: str
    logical_source: str
    content_sha256: str | None
    release: str | None
    parser: str                      # 'path::qualname' of the materialising function
    parser_sha256: str               # its semantic-closure digest, measured from the code
    population_id: str
    population_sha256: str
    n_population: int
    state_counts: Mapping[str, int]
    detail_counts: Mapping[str, Mapping[str, int]]
    column_sha256: str
    inputs: Mapping[str, Mapping[str, Any]]     # other materialised inputs (mapping tables), by logical id
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"schema": RECEIPT_SCHEMA, "feature": self.feature, "logical_source": self.logical_source,
                "content_sha256": self.content_sha256, "release": self.release, "parser": self.parser,
                "parser_sha256": self.parser_sha256, "population_id": self.population_id,
                "population_sha256": self.population_sha256, "n_population": self.n_population,
                "state_counts": dict(self.state_counts),
                "detail_counts": {k: dict(v) for k, v in sorted(self.detail_counts.items())},
                "column_sha256": self.column_sha256, "inputs": {k: dict(v) for k, v in sorted(self.inputs.items())},
                "notes": list(self.notes)}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parser_identity(qualname: str) -> tuple[str, str]:
    """('path::qualname', semantic-closure digest) of a function in THIS module, measured from its source now."""
    from genomic_variant_classifier.information_sources import census as _census

    symbol = f"src/genomic_variant_classifier/information_sources/materialization.py::{qualname}"
    return symbol, _census.symbol_closure_digest(_repo_root(), symbol)


def make_receipt(column: ObservationColumn, *, qualname: str, inputs: Mapping[str, Mapping[str, Any]] | None = None,
                 notes: Sequence[str] = ()) -> MaterializationReceipt:
    symbol, digest = parser_identity(qualname)
    counts = column.counts()
    if sum(counts.values()) != len(column.population):
        raise AssertionError("receipt counts do not reconcile with the population")
    return MaterializationReceipt(
        feature=column.feature, logical_source=column.source.logical_id, content_sha256=column.source.content_sha256,
        release=column.source.release, parser=symbol, parser_sha256=digest,
        population_id=column.population.population_id, population_sha256=column.population.sha256,
        n_population=len(column.population), state_counts=counts, detail_counts=column.detail_counts(),
        column_sha256=column.sha256, inputs=dict(inputs or {}), notes=tuple(notes))


def check_registered(logical_id: str, registered_datasets: Iterable[str]) -> None:
    """A materialised source must be a census dataset (one authority for logical identities; no free strings)."""
    if not logical_id.startswith("dataset:") or logical_id[len("dataset:"):] not in set(registered_datasets):
        raise ValueError(f"{logical_id}: not a dataset registered in the census")


# ----------------------------------------------------------------------------------------------------------- results
@dataclass(frozen=True)
class Materialized:
    column: ObservationColumn
    receipt: MaterializationReceipt
    legacy_projection: np.ndarray        # the value the CURRENT code would write for each row (before engineer_features)
    source_rows_relevant: np.ndarray | None = None   # EVE: source rows that can change the legacy join for this population


def _build_column(feature: str, source: SourceIdentity, population: Population, values: np.ndarray,
                  states: np.ndarray, details: np.ndarray, vocabulary: Sequence[str]) -> ObservationColumn:
    return ObservationColumn(feature, source, population, values.astype(np.float64), states.astype(np.uint8),
                             details.astype(np.uint16), tuple(vocabulary))


def population_from_frame(frame: pd.DataFrame, population_id: str, id_column: str = "variant_id") -> Population:
    ids = frame[id_column]
    if ids.isna().any():
        raise ValueError(f"{int(ids.isna().sum())} rows have no {id_column}")
    return Population(population_id, tuple(str(v) for v in ids.tolist()))


# =============================================================================================================== EVE
EVE_DETAILS: tuple[str, ...] = ("", "not_a_missense_substitution", "missense_without_protein_key", "gene_not_modeled",
                                "substitution_not_in_source", "substitution_unscored", "ambiguous_source_record")
_EVE_REQUIRED = {"position", "wt_aa", "mt_aa", "EVE_scores_ASM"}


@dataclass(frozen=True)
class EveSourceTable:
    """Every substitution row of every EVE file, scored or not, keyed as the legacy connector keys them."""
    rows: pd.DataFrame                       # gene_symbol, aa_change, eve_score (NaN = unscored), file
    content_sha256: str
    manifest: tuple[dict[str, Any], ...]
    files_resolved_by_map: int
    files_by_fallback: int
    files_without_required_columns: tuple[str, ...]


def entry_map_from_index(raw_parquet: bytes) -> dict[str, str]:
    """{ENTRY_NAME: HGNC} from the UniProt index's verified bytes, by data.eve.load_eve_entry_map's rule (a test proves the
    two agree): upper-cased, stripped, first occurrence kept, empty values skipped. {} without the two columns."""
    df = pd.read_parquet(io.BytesIO(raw_parquet))
    if "entry_name" not in df.columns or "gene_symbol" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for en, g in zip(df["entry_name"].astype(str), df["gene_symbol"].astype(str)):
        en, g = en.strip().upper(), g.strip().upper()
        if en and g and en not in out:
            out[en] = g
    return out


def parse_eve_file(stream: io.BufferedReader, stem: str, entry_map: Mapping[str, str]) -> tuple[pd.DataFrame, bool, bool]:
    """One EVE per-protein CSV, parsed as EVEConnector._parse_single_csv parses it, but KEEPING unscored substitutions.
    Returns (rows, resolved_via_map, has_required_columns)."""
    from genomic_variant_classifier.data.eve import resolve_eve_gene

    raw = pd.read_csv(stream, dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    if not _EVE_REQUIRED.issubset(raw.columns):
        return pd.DataFrame(columns=["gene_symbol", "aa_change", "eve_score"]), False, False
    raw["position"] = pd.to_numeric(raw["position"], errors="coerce")
    raw["EVE_scores_ASM"] = pd.to_numeric(raw["EVE_scores_ASM"], errors="coerce")
    raw = raw.dropna(subset=["position"])                     # legacy also drops unscored rows; we keep them (NaN score)
    if "mutations_protein_name" in raw.columns:
        raw["gene_symbol"] = raw["mutations_protein_name"].astype(str).str.split("_").str[0]
        resolved = True
    else:
        hgnc, resolved = resolve_eve_gene(stem, dict(entry_map))
        raw["gene_symbol"] = hgnc
    raw["aa_change"] = raw["wt_aa"].str.strip() + raw["position"].astype(int).astype(str) + raw["mt_aa"].str.strip()
    out = raw[["gene_symbol", "aa_change", "EVE_scores_ASM"]].rename(columns={"EVE_scores_ASM": "eve_score"})
    return out.reset_index(drop=True), resolved, True


def read_eve_source(directory: Path, entry_map: Mapping[str, str], *,
                    file_results: Mapping[str, tuple[pd.DataFrame, str, int, bool, bool]] | None = None
                    ) -> EveSourceTable:
    """Parse every ``*.csv`` in ``directory`` (the legacy connector's glob) through a VerifiedReader.

    ``file_results`` lets a caller that processed files in batches supply {name: (rows, sha256, bytes, resolved, ok)}; the
    set of names must equal the directory's ``*.csv`` files exactly."""
    names = sorted(p.name for p in directory.glob("*.csv"))
    if not names:
        raise ValueError(f"{directory}: no EVE CSV files")
    if file_results is not None and sorted(file_results) != names:
        raise ValueError("batched EVE results do not cover the directory's files exactly")
    parts, entries, missing_cols = [], [], []
    by_map = by_fallback = 0
    for name in names:
        if file_results is not None:
            rows, digest, n_bytes, resolved, ok = file_results[name]
        else:
            stem = Path(name).stem
            parsed = parse_verified(directory / name, lambda s, _stem=stem: parse_eve_file(s, _stem, entry_map))
            (rows, resolved, ok), digest, n_bytes = parsed.value, parsed.sha256, parsed.n_bytes
        entries.append((name, digest, n_bytes))
        if not ok:
            missing_cols.append(name)
            continue
        by_map += int(resolved)
        by_fallback += int(not resolved)
        rows = rows.copy()
        rows["file"] = name
        parts.append(rows)
    content, manifest = directory_manifest(entries)
    table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["gene_symbol", "aa_change", "eve_score", "file"])
    return EveSourceTable(table, content, tuple(manifest), by_map, by_fallback, tuple(missing_cols))


def legacy_lookup_from_source(table: EveSourceTable, rows: np.ndarray | None = None) -> pd.DataFrame:
    """The lookup the legacy connector builds from the same files (unscored rows dropped, score clipped to [0, 1]).

    ``rows`` (Materialized.source_rows_relevant) restricts it to the rows that can change the legacy join for one
    population: every scored row whose (gene, substitution) the population requests, plus one scored row of every scored
    gene, so the legacy gene-resolution set is unchanged. The legacy join on the restricted lookup equals the join on the
    full lookup row for row (rows matching no population key never join; the gene set is preserved); a test shows it."""
    src = table.rows if rows is None else table.rows.iloc[rows]
    lk = src.dropna(subset=["eve_score"])[["gene_symbol", "aa_change", "eve_score"]].copy()
    lk["eve_score"] = lk["eve_score"].clip(0.0, 1.0)
    return lk.reset_index(drop=True)


def _eve_variant_keys(frame: pd.DataFrame) -> list[str | None]:
    """The variant-side key exactly as EVEConnector._annotate builds it: the coordinate triple first, then HGVSp."""
    from genomic_variant_classifier.data.eve import _hgvsp_to_eve_key

    def triple(w: object, p: object, m: object) -> str | None:
        if w is None or m is None or pd.isna(p):
            return None
        ws, ms = str(w).strip(), str(m).strip()
        if not ws or not ms:
            return None
        try:
            return f"{ws}{int(p)}{ms}"
        except (TypeError, ValueError):
            return None

    if {"wt_aa", "protein_pos", "mut_aa"}.issubset(frame.columns):
        trip = [triple(w, p, m) for w, p, m in zip(frame["wt_aa"], frame["protein_pos"], frame["mut_aa"])]
    else:
        trip = [None] * len(frame)
    pc = frame.get("protein_change", pd.Series([""] * len(frame), index=frame.index)).fillna("")
    hg = pc.map(_hgvsp_to_eve_key).tolist()
    return [t if t is not None else h for t, h in zip(trip, hg)]


def materialize_eve(frame: pd.DataFrame, population: Population, source: EveSourceTable | None, *,
                    registered_datasets: Iterable[str], release: str | None = None,
                    entry_map_input: Mapping[str, Any] | None = None) -> Materialized:
    """EVE observations for ``population`` (rows of ``frame`` in population order). ``source`` None = not configured."""
    from genomic_variant_classifier.data.gene_symbols import gene_symbol_candidates, normalize_gene_symbol

    check_registered("dataset:eve", registered_datasets)
    n = len(population)
    if len(frame) != n or [str(v) for v in frame["variant_id"].tolist()] != list(population.row_ids):
        raise ValueError("the frame's rows are not the population's rows in population order")
    values = np.full(n, np.nan)
    states = np.empty(n, np.uint8)
    details = np.zeros(n, np.uint16)
    det = {d: i for i, d in enumerate(EVE_DETAILS)}
    legacy = np.full(n, 0.5)
    notes: list[str] = []
    if source is None:
        states[:] = CODE[S.SOURCE_ABSENT]
        column = _build_column("eve_score", SourceIdentity("dataset:eve", None, release), population, values, states,
                               details, EVE_DETAILS)
        return Materialized(column, make_receipt(column, qualname="materialize_eve", notes=("source not configured",)),
                            legacy)

    # The source side, by normalised gene, without materialising one Python object per source row (44.5 million rows in
    # the real EVE release): genes are categorical, and per-gene slices are compared with the keys the population needs.
    gcat = pd.Categorical(source.rows["gene_symbol"])
    norm_names = [normalize_gene_symbol(c) for c in gcat.categories]
    norm_unique = sorted(set(norm_names))
    norm_index = {g: i for i, g in enumerate(norm_unique)}
    cat_to_norm = np.array([norm_index[g] for g in norm_names], dtype=np.int64)
    codes = np.where(gcat.codes >= 0, cat_to_norm[np.maximum(gcat.codes, 0)], -1)
    valid = codes >= 0
    if "" in norm_index:
        valid &= codes != norm_index[""]
    score = source.rows["eve_score"].to_numpy(dtype=np.float64)
    scored_mask = valid & ~np.isnan(score)
    out_of_range = int(((score[scored_mask] < 0) | (score[scored_mask] > 1)).sum())
    if out_of_range:
        raise ValueError(f"{out_of_range} EVE scores lie outside [0, 1]; the legacy connector would clip them silently")
    modeled = {norm_unique[c] for c in np.unique(codes[valid])}            # every gene with a file, scored rows or not
    legacy_genes = {norm_unique[c] for c in np.unique(codes[scored_mask])}  # the legacy lookup's genes (scored rows)
    aa_all = source.rows["aa_change"].to_numpy(dtype=object)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(norm_unique) + 1))
    keys = _eve_variant_keys(frame)
    genes_raw = frame.get("gene_symbol", pd.Series([""] * n, index=frame.index)).fillna("")

    def resolve(raw: object) -> str:                       # EVEConnector._annotate._resolve_gene, unchanged
        for cand in gene_symbol_candidates(raw):
            if cand in legacy_genes:
                return cand
        return normalize_gene_symbol(raw)

    genes = [resolve(g) for g in genes_raw.tolist()]
    missense = frame.get("consequence", pd.Series([""] * n, index=frame.index)).fillna("").str.contains(
        "missense", case=False).to_numpy()
    needed: dict[str, set[str]] = {}
    for key, gene in zip(keys, genes):
        if key is not None and gene in modeled:
            needed.setdefault(gene, set()).add(key)
    score_of: dict[tuple[str, str], tuple[float, float, int]] = {}
    unscored: set[tuple[str, str]] = set()
    n_dup_keys = n_ambiguous_keys = 0
    lookup_rows: list[np.ndarray] = []                      # the legacy lookup's rows that can affect this population
    for gene, gi in norm_index.items():
        if gene == "":
            continue
        rows_g = order[bounds[gi]:bounds[gi + 1]]
        aa_g, sc_g = aa_all[rows_g], score[rows_g]
        has = ~np.isnan(sc_g)
        if has.any():                                         # one row keeps the gene in the legacy gene set
            lookup_rows.append(rows_g[np.flatnonzero(has)[:1]])
        uniq, inv, cnt = np.unique(aa_g[has].astype(str), return_inverse=True, return_counts=True)
        if (cnt > 1).any():                                   # a key scored more than once in the source
            lo_all = np.full(len(uniq), np.inf)
            hi_all = np.full(len(uniq), -np.inf)
            np.minimum.at(lo_all, inv, sc_g[has])
            np.maximum.at(hi_all, inv, sc_g[has])
            n_dup_keys += int((cnt > 1).sum())
            n_ambiguous_keys += int(((cnt > 1) & (lo_all != hi_all)).sum())
        want = needed.get(gene)
        if not want:
            continue
        hit = np.isin(aa_g.astype(str), np.array(sorted(want), dtype=str))
        lookup_rows.append(rows_g[hit & has])
        for aa, sc in zip(aa_g[hit & has].astype(str), sc_g[hit & has]):
            lo, hi, size = score_of.get((gene, aa), (np.inf, -np.inf, 0))
            score_of[(gene, aa)] = (min(lo, float(sc)), max(hi, float(sc)), size + 1)
        unscored.update((gene, aa) for aa in aa_g[hit & ~has].astype(str))
    for i, (key, gene) in enumerate(zip(keys, genes)):
        if key is None:
            if missense[i]:
                states[i], details[i] = CODE[S.BAD_MAPPING], det["missense_without_protein_key"]
            else:
                states[i], details[i] = CODE[S.UNSUPPORTED_VARIANT], det["not_a_missense_substitution"]
            continue
        k = (gene, key)
        if gene not in modeled:
            states[i], details[i] = CODE[S.NOT_IN_SOURCE], det["gene_not_modeled"]
        elif k in score_of:
            lo, hi, size = score_of[k]
            if lo != hi:
                states[i], details[i] = CODE[S.BAD_MAPPING], det["ambiguous_source_record"]
            else:
                states[i], values[i] = CODE[S.OBSERVED], lo
                legacy[i] = lo
                if size > 1:
                    details[i] = det["ambiguous_source_record"]   # duplicated but identical: observed, flagged
        elif k in unscored:
            states[i], details[i] = CODE[S.NOT_IN_SOURCE], det["substitution_unscored"]
        else:
            states[i], details[i] = CODE[S.NOT_IN_SOURCE], det["substitution_not_in_source"]
    if n_dup_keys:
        notes.append(f"{n_dup_keys} (gene, substitution) keys are scored more than once in the source "
                     f"({n_ambiguous_keys} with differing scores): the legacy left join multiplies every cohort row "
                     "that matches one")
    if source.files_by_fallback:
        notes.append(f"{source.files_by_fallback} files did not resolve entry name -> HGNC and are keyed by the legacy "
                     "filename-prefix fallback")
    if source.files_without_required_columns:
        notes.append(f"{len(source.files_without_required_columns)} files lack the required columns and are skipped, as "
                     "the legacy connector skips them")
    column = _build_column("eve_score", SourceIdentity("dataset:eve", source.content_sha256, release), population,
                           values, states, details, EVE_DETAILS)
    inputs = {"dataset:eve#files": {"n_files": len(source.manifest), "resolved_by_entry_map": source.files_resolved_by_map,
                                    "by_fallback": source.files_by_fallback,
                                    "without_required_columns": len(source.files_without_required_columns),
                                    "substitution_rows": int(len(score)), "scored_rows": int(scored_mask.sum()),
                                    "modeled_genes": len(modeled), "keys_scored_more_than_once": n_dup_keys,
                                    "keys_with_differing_scores": n_ambiguous_keys}}
    if entry_map_input:
        inputs["entry_map"] = dict(entry_map_input)
    relevant = np.unique(np.concatenate(lookup_rows)) if lookup_rows else np.zeros(0, np.int64)
    return Materialized(column, make_receipt(column, qualname="materialize_eve", inputs=inputs, notes=notes), legacy,
                        relevant)


def legacy_eve_values(frame: pd.DataFrame, lookup: pd.DataFrame) -> np.ndarray:
    """The CURRENT connector's join (EVEConnector._annotate) on the same rows and the same lookup."""
    from genomic_variant_classifier.data.eve import EVEConnector

    out = EVEConnector(eve_path=None)._annotate(frame, lookup)
    if len(out) != len(frame):
        raise ValueError(f"the legacy EVE join returned {len(out)} rows for {len(frame)}: duplicated source keys "
                         "multiplied cohort rows")
    return out["eve_score"].to_numpy(dtype=np.float64)


# ===================================================================================================== allele frequency
AF_DETAILS: tuple[str, ...] = ("", "locus_not_in_source", "record_without_value", "ambiguous_source_record",
                               "unmappable_locus")


def _parse_gnomad_locus(vid: object) -> tuple[str, str, str, str] | None:
    """DataPrepPipeline._join_gnomad's variant_id parse, unchanged."""
    parts = str(vid).split(":")
    if not parts[0].replace("X", "").replace("Y", "").replace("M", "").isdigit():
        parts = parts[1:]
    if len(parts) < 4:
        return None
    return parts[0], parts[1], parts[2], parts[3]


def gnomad_source_table(raw_parquet: bytes) -> pd.DataFrame:
    """The gnomAD AF table keyed as the legacy join keys it, ALL records kept (duplicates are resolved by the materialiser,
    not by keep-first)."""
    g = pd.read_parquet(io.BytesIO(raw_parquet), columns=["variant_id", "allele_freq"])
    loci = g["variant_id"].map(_parse_gnomad_locus)
    ok = loci.notna()
    g = g[ok].copy()
    parts = pd.DataFrame(loci[ok].tolist(), index=g.index, columns=["_chrom", "_pos", "_ref", "_alt"])
    g = pd.concat([g, parts], axis=1)
    g["_pos"] = pd.to_numeric(g["_pos"], errors="coerce").fillna(0).astype(int)
    g["allele_freq"] = pd.to_numeric(g["allele_freq"], errors="coerce")
    g.attrs["unparsable_variant_ids"] = int((~ok).sum())
    return g[["_chrom", "_pos", "_ref", "_alt", "allele_freq"]].reset_index(drop=True)


def materialize_gnomad_af(frame: pd.DataFrame, population: Population, source: Parsed | None, *,
                          registered_datasets: Iterable[str], release: str | None = None) -> Materialized:
    """gnomAD allele-frequency observations. ``source``: the verified parquet bytes (read_verified_bytes) or None."""
    check_registered("dataset:gnomad_af", registered_datasets)
    n = len(population)
    if len(frame) != n or [str(v) for v in frame["variant_id"].tolist()] != list(population.row_ids):
        raise ValueError("the frame's rows are not the population's rows in population order")
    values = np.full(n, np.nan)
    states = np.empty(n, np.uint8)
    details = np.zeros(n, np.uint16)
    det = {d: i for i, d in enumerate(AF_DETAILS)}
    cohort_af = pd.to_numeric(frame.get("allele_freq", pd.Series(np.nan, index=frame.index)),
                              errors="coerce").to_numpy(dtype=np.float64)
    notes: list[str] = []
    if int(np.isfinite(cohort_af).sum()):
        notes.append(f"{int(np.isfinite(cohort_af).sum())} cohort rows carry their own allele_freq, which the legacy "
                     "join keeps in preference to gnomAD")
    if source is None:
        states[:] = CODE[S.SOURCE_ABSENT]
        column = _build_column("af_raw", SourceIdentity("dataset:gnomad_af", None, release), population, values,
                               states, details, AF_DETAILS)
        legacy = cohort_af.copy()
        return Materialized(column, make_receipt(column, qualname="materialize_gnomad_af",
                                                 notes=("source not configured", *notes)), legacy)

    g = gnomad_source_table(source.value)
    unparsable = g.attrs.get("unparsable_variant_ids", 0)
    agg = g.groupby(["_chrom", "_pos", "_ref", "_alt"], sort=False)["allele_freq"].agg(
        n="size", n_value="count", lo="min", hi="max").reset_index()
    left = pd.DataFrame({
        "_chrom": frame["chrom"].astype(str).to_numpy(),
        "_pos_raw": pd.to_numeric(frame["pos"], errors="coerce").to_numpy(),
        "_ref": frame["ref"].fillna("").astype(str).to_numpy(),
        "_alt": frame["alt"].fillna("").astype(str).to_numpy(),
    })
    unmappable = np.isnan(left["_pos_raw"].to_numpy(dtype=np.float64))
    left["_pos"] = pd.Series(left["_pos_raw"]).fillna(0).astype(int)
    merged = left.merge(agg, on=["_chrom", "_pos", "_ref", "_alt"], how="left", validate="many_to_one")
    if len(merged) != n:
        raise AssertionError("an aggregated many-to-one join changed the row count")
    found = merged["n"].notna().to_numpy()
    n_value = merged["n_value"].fillna(0).to_numpy()
    lo, hi = merged["lo"].to_numpy(dtype=np.float64), merged["hi"].to_numpy(dtype=np.float64)
    ambiguous = found & (n_value > 0) & (lo != hi)
    no_value = found & (n_value == 0)
    observed = found & (n_value > 0) & ~ambiguous
    states[:] = CODE[S.NOT_IN_SOURCE]
    details[:] = det["locus_not_in_source"]
    states[no_value], details[no_value] = CODE[S.NOT_IN_SOURCE], det["record_without_value"]
    states[ambiguous], details[ambiguous] = CODE[S.BAD_MAPPING], det["ambiguous_source_record"]
    states[observed], details[observed] = CODE[S.OBSERVED], 0
    values[observed] = lo[observed]
    states[unmappable], details[unmappable] = CODE[S.BAD_MAPPING], det["unmappable_locus"]
    values[unmappable] = np.nan
    if (values[observed & ~unmappable] < 0).any():
        raise ValueError("a negative allele frequency in the source")
    # The legacy join keeps the FIRST record of a duplicated locus; its allele_freq is the projection, ambiguous or not.
    first = g.drop_duplicates(subset=["_chrom", "_pos", "_ref", "_alt"])
    lmerged = left.merge(first, on=["_chrom", "_pos", "_ref", "_alt"], how="left")
    legacy = pd.Series(cohort_af).fillna(pd.Series(lmerged["allele_freq"].to_numpy(dtype=np.float64))).to_numpy()
    if unparsable:
        notes.append(f"{unparsable} source variant_ids do not parse to a locus and are dropped, as the legacy join drops them")
    n_amb_loci = int(((agg["n_value"] > 0) & (agg["lo"] != agg["hi"])).sum())
    n_dup_loci = int((agg["n"] > 1).sum())
    if n_dup_loci:
        notes.append(f"{n_dup_loci} source loci occur more than once ({n_amb_loci} with differing frequencies); the "
                     "legacy join keeps the first")
    column = _build_column("af_raw", SourceIdentity("dataset:gnomad_af", source.sha256, release), population, values,
                           states, details, AF_DETAILS)
    inputs = {"dataset:gnomad_af#records": {"n_records": int(len(g) + unparsable), "unparsable_variant_ids": int(unparsable),
                                            "distinct_loci": int(len(agg)), "duplicated_loci": n_dup_loci,
                                            "ambiguous_loci": n_amb_loci, "bytes": source.n_bytes}}
    return Materialized(column, make_receipt(column, qualname="materialize_gnomad_af", inputs=inputs, notes=notes),
                        legacy)


def legacy_af_values(frame: pd.DataFrame, gnomad_path: Path) -> np.ndarray:
    """The CURRENT join (DataPrepPipeline._join_gnomad, no 1000 Genomes path) on the same rows: its allele_freq."""
    from genomic_variant_classifier.data.real_data_prep import DataPrepPipeline

    pipe = DataPrepPipeline.__new__(DataPrepPipeline)          # the join needs only annotation_config.finngen paths
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig

    pipe.annotation_config = AnnotationConfig()
    out = pipe._join_gnomad(frame.copy(), str(gnomad_path), kg_path=None)
    if len(out) != len(frame):
        raise ValueError(f"the legacy gnomAD join returned {len(out)} rows for {len(frame)}")
    return pd.to_numeric(out["allele_freq"], errors="coerce").to_numpy(dtype=np.float64)


# ------------------------------------------------------------------------------------------------- shadow comparison
def compare_legacy(projection: np.ndarray, legacy: np.ndarray) -> dict[str, Any]:
    """Row-by-row equality of the materialiser's legacy projection with the current code's output (NaN equals NaN)."""
    if projection.shape != legacy.shape:
        raise ValueError("projection and legacy output differ in length")
    both_nan = np.isnan(projection) & np.isnan(legacy)
    equal = both_nan | (projection == legacy)
    return {"rows": int(len(legacy)), "equal": int(equal.sum()), "differ": int((~equal).sum()),
            "identical": bool(equal.all())}


def engineer_fill(values: np.ndarray, fill: float) -> np.ndarray:
    """What engineer_features does to a raw input column: NaN -> the fill value."""
    out = values.copy()
    out[np.isnan(out)] = fill
    return out


__all__ = ["AF_DETAILS", "EVE_DETAILS", "EveSourceTable", "MaterializationReceipt", "Materialized", "Parsed",
           "VerifiedReader", "check_registered", "compare_legacy", "directory_manifest", "engineer_fill",
           "gnomad_source_table", "legacy_af_values", "legacy_eve_values", "legacy_lookup_from_source", "make_receipt",
           "materialize_eve", "materialize_gnomad_af", "parse_eve_file", "parse_verified", "parser_identity",
           "population_from_frame", "read_eve_source", "read_verified_bytes"]
