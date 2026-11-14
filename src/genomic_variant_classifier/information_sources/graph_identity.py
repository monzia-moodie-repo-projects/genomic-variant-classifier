"""Stage 1D-1b.3: source-bound graph identity (SHADOW MODE: nothing in production imports this module).

Owner rulings f2ec92cd (Unit 1D-1b.3 and the disk-backed exact-pair approach, section III), 7f4b69d0 (sections 7-10) and the
owner's integrity reference (GVC_1D1B_SourceBound_STRING_SplitIsolation_GeneMetrics, 2026-11-09). The legacy graph code
(models/gnn.py, data/kg_edges.py, models/hetero_gnn_scorer.py) accepts a graph by its cache FILENAME, reduces STRING's two
directed records of a pair by overwriting one with the other, keeps unmapped protein ids as nodes, prunes pathway sets by the
size they have inside a cohort and merges sources into one relation. This module states what a graph IS and measures it:

* ``read_string_pairs``: one pass over the bytes through ``materialization.VerifiedReader`` (the digest covers exactly what
  was parsed), every record validated, the two directions of every pair matched EXACTLY in disk-backed hash partitions
  (memory bounded by the largest partition, not by the source); an asymmetric record, a contradictory direction, a duplicate
  in either direction or a self-pair is refused, never reduced.
* ``classify_threshold``: the four-way threshold status of every pair under the DECLARED quantization model, compared in
  integer thousandths, with whether the published score passes (ruling 7f4b69d0, sections 7-8); a comparison whose float
  value lies within floating-point error of the threshold is decided in exact rational arithmetic, and counted.
* ``HgncIndex`` / ``map_string_proteins``: STRING preferred names are not canonical identity (section 9.1). Each protein keeps
  its STRING id, preferred name, HGNC id, mapping status and mapping source; the collision report is measured AFTER
  canonicalisation, and the one rule that resolves a collision (approved-symbol precedence) is declared, not silent.
* ``GraphPolicy`` / ``materialize_string_graph`` / ``accept_graph``: the requested graph's identity and the materialised
  graph's identity (section 9: source, mapping, node universe, relation policy, edge content, schema), every digest computed
  from content by the producer; ``accept_graph`` recomputes them from the graph itself and refuses any mismatch.
* ``legacy_string_cache_name``: the legacy cache key, read from the code, for the measurement that it cannot tell policies
  apart.
* ``read_gene_sets`` / ``project_pathways`` / ``induced_subgraph_consistency``: pathway relations keyed by (source, relation)
  and (source, pathway id), weighted 1/(|S_s| - 1) with the GLOBAL source-defined |S_s| (section 10), and the check that
  growing the node universe never removes or reweights an existing edge -- which the legacy projection fails.
* ``hypergraph_matvec``: the factorised operator B D B^T x - diag(B D B^T) x of section 10.1, for the equivalence test only;
  the hypergraph model itself is stage 1D-3.

The graph this module builds is a research object of declared semantics. Whether its sources are admissible is the census's
decision (``census_admission``), and STRING scores are predicted functional association, not physical interaction.
numpy is required.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import io
import logging
import math
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from genomic_variant_classifier.information_sources.census import HEX64, canonical_bytes, sha256_bytes
from genomic_variant_classifier.information_sources.materialization import Parsed, parse_verified

logger = logging.getLogger(__name__)

GRAPH_SCHEMA_VERSION = "gvc.graph-materialization.1d1b3/1"
STRING_HEADER: tuple[str, ...] = ("protein1", "protein2", "neighborhood", "fusion", "cooccurence", "coexpression",
                                  "experimental", "database", "textmining", "combined_score")
STRING_CHANNELS: tuple[str, ...] = STRING_HEADER[2:9]
#: STRING's prior probability of a random association (string-db.org/help/scores): removed from each channel before the
#: channels are combined and added back once afterwards.
STRING_PRIOR = 0.041
#: Channels that carry curated pathway or literature knowledge; a curation-reduced graph (G1) may not select on them.
STRING_CURATED_CHANNELS: frozenset[str] = frozenset({"database", "textmining"})
STRING_INFO_HEADER: tuple[str, ...] = ("#string_protein_id", "preferred_name", "protein_size", "annotation")
#: One record of a partition file: canonical low and high protein index, direction (0: file order was low->high), the seven
#: channels and the combined score, little-endian.
_RECORD = np.dtype([("lo", "<i4"), ("hi", "<i4"), ("rev", "u1"), ("v", "<u2", (8,))])
#: Thousandths. A float reconstruction this close to a decision boundary (the threshold, or an integer for floor) is
#: decided again in EXACT rational arithmetic; the float error of the seven-channel combination is of order 1e-12 thousandths.
_FLOAT_GUARD = 1e-9
_PRIOR_Q = Fraction(41, 1000)


def _refuse_carriage_return(line: bytes | str, path: Path, lineno: int) -> None:
    """A pinned source is parsed exactly as published (every file measured here has LF line endings). A carriage return
    (a CRLF file, e.g. one rewritten by a Windows text-mode write) is refused by name: left in place it would silently
    corrupt the last field of every line -- the last member of every gene set, the last column of every row."""
    if (b"\r" if isinstance(line, bytes) else "\r") in line:
        raise ValueError(f"{path} line {lineno}: carriage return (CRLF line endings); a pinned source is read exactly "
                         "as published, nothing is stripped")


# =================================================================================================== STRING exact pairs
@dataclass(frozen=True)
class StringPairTable:
    """Every undirected STRING pair once, in canonical order: ``a < b`` index into ``proteins`` (sorted), rows sorted by
    (a, b). ``channels`` holds the seven channel integers (thousandths) in ``STRING_CHANNELS`` order, ``combined`` the
    published combined score. ``content_sha256`` is computed from those arrays; ``source_sha256`` from the bytes parsed."""

    proteins: tuple[str, ...]
    a: np.ndarray
    b: np.ndarray
    channels: np.ndarray
    combined: np.ndarray
    source_sha256: str
    n_bytes: int
    n_records: int
    partitions: int
    max_partition_records: int
    scratch_bytes: int
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.a)
        if not (len(self.b) == len(self.combined) == n and self.channels.shape == (n, len(STRING_CHANNELS))):
            raise ValueError("pair-table arrays disagree in length")
        if self.n_records != 2 * n:
            raise ValueError("every pair must come from exactly two directed records")
        object.__setattr__(self, "content_sha256", pair_table_digest(self.proteins, self.a, self.b, self.channels,
                                                                     self.combined))

    def __len__(self) -> int:
        return len(self.a)


def pair_table_digest(proteins: Sequence[str], a: np.ndarray, b: np.ndarray, channels: np.ndarray,
                      combined: np.ndarray) -> str:
    """SHA-256 of the canonical pair table: a header naming the schema, channels, protein list and pair count, then the
    fixed-width little-endian arrays, hashed in place (no concatenated copy)."""
    if list(proteins) != sorted(set(proteins)):
        raise ValueError("protein list must be sorted and unique")
    if len(a):
        key = a.astype(np.int64) * len(proteins) + b
        if not (np.all(a < b) and bool((key[1:] > key[:-1]).all())):
            raise ValueError("pairs must be canonical (a < b) and strictly sorted")
        del key
    head = canonical_bytes({"schema": "gvc.string-pair-table/1", "channels": list(STRING_CHANNELS),
                            "proteins_sha256": sha256_bytes("\n".join(proteins).encode("ascii")), "pairs": int(len(a))})
    h = hashlib.sha256(head)
    for arr, dt in ((a, "<i4"), (b, "<i4"), (channels, "<u2"), (combined, "<u2")):
        h.update(memoryview(np.ascontiguousarray(arr, dt)).cast("B"))
    return h.hexdigest()


_SCORES = re.compile(rb"(?:0|[1-9][0-9]{0,3})(?: (?:0|[1-9][0-9]{0,3})){7}\Z")


def _split_string_record(line: bytes, lineno: int) -> tuple[bytes, bytes, bytes]:
    """(protein1, protein2, the eight score fields) of one record, refusing any deviation from the documented form:
    exactly ten single-space-separated fields, nonempty ASCII identifiers, two different proteins, canonical decimal
    integers (no sign, no leading zero, at most four digits; the [0, 1000] range is checked on the converted chunk)."""
    if b"\r" in line:
        raise ValueError(f"STRING record {lineno}: carriage return (CRLF line endings); a pinned source is read exactly "
                         "as published, nothing is stripped")
    parts = line.rstrip(b"\n").split(b" ", 2)
    if len(parts) != 3 or not _SCORES.match(parts[2]):
        raise ValueError(f"STRING record {lineno}: not two identifiers and eight canonical integer scores, "
                         "single-space separated")
    p1, p2 = parts[0], parts[1]
    if not p1 or not p2 or not p1.isascii() or not p2.isascii():
        raise ValueError(f"STRING record {lineno}: empty or non-ASCII protein identifier")
    if p1 == p2:
        raise ValueError(f"STRING record {lineno}: a protein paired with itself ({p1.decode()})")
    return p1, p2, parts[2]


def _check_partition(recs: np.ndarray, proteins: Sequence[bytes]) -> np.ndarray:
    """Exact pairing inside one partition: every key exactly twice, once per direction, equal scores. Returns the records
    of the pairs (one per pair, direction dropped)."""
    if len(recs) == 0:
        return recs
    order = np.lexsort((recs["rev"], recs["hi"], recs["lo"]))
    r = recs[order]
    key = r["lo"].astype(np.int64) * len(proteins) + r["hi"]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    counts = np.diff(np.r_[starts, len(r)])

    def name(i: int) -> str:
        return f"{proteins[r['lo'][i]].decode()} -- {proteins[r['hi'][i]].decode()}"

    if (counts == 1).any():
        raise ValueError(f"STRING pair without its reciprocal record: {name(int(starts[counts == 1][0]))}")
    if (counts > 2).any():
        raise ValueError(f"STRING pair recorded more than twice: {name(int(starts[counts > 2][0]))}")
    first, second = r[0::2], r[1::2]
    same_direction = first["rev"] == second["rev"]
    if same_direction.any():
        raise ValueError(f"STRING pair duplicated in one direction: {name(2 * int(np.flatnonzero(same_direction)[0]))}")
    differ = (first["v"] != second["v"]).any(axis=1)
    if differ.any():
        raise ValueError("STRING pair whose two directions carry different scores: "
                         f"{name(2 * int(np.flatnonzero(differ)[0]))}")
    return first


def read_string_pairs(path: Path, *, expected_sha256: str | None = None, scratch_dir: Path | None = None,
                      partitions: int = 64, chunk_records: int = 1 << 18) -> StringPairTable:
    """Read a STRING ``protein.links.detailed`` file (gzip or plain) in ONE verified pass and return every undirected pair
    once. Records are spooled to ``partitions`` scratch files by a hash of the canonical pair; each partition is then
    checked exactly (``_check_partition``). Scratch is deleted on success and on failure."""
    if not 1 <= partitions <= 4096 or chunk_records < 1:
        raise ValueError("partitions must be in [1, 4096] and chunk_records positive")
    index: dict[bytes, int] = {}
    names: list[bytes] = []
    n_records = 0
    with tempfile.TemporaryDirectory(prefix="gvc_string_pairs_", dir=scratch_dir) as temp:
        spool = [Path(temp) / f"part_{i:04d}.bin" for i in range(partitions)]
        handles = [p.open("wb") for p in spool]
        try:
            def intern(p: bytes) -> int:
                i = index.get(p)
                if i is None:
                    i = index[p] = len(names)
                    names.append(p)
                return i

            def flush(p1s: list[bytes], p2s: list[bytes], scores: list[bytes], first_line: int) -> None:
                if not p1s:
                    return
                ia = np.fromiter((intern(p) for p in p1s), np.int64, len(p1s))
                ib = np.fromiter((intern(p) for p in p2s), np.int64, len(p2s))
                swap = np.fromiter((x > y for x, y in zip(p1s, p2s)), bool, len(p1s))
                vals = np.array(b" ".join(scores).split(b" "), dtype="S4").astype(np.uint16).reshape(-1, 8)
                over = (vals > 1000).any(axis=1)
                if over.any():
                    raise ValueError(f"STRING record {first_line + int(np.flatnonzero(over)[0])}: a score above 1000")
                recs = np.empty(len(p1s), _RECORD)
                recs["lo"] = np.where(swap, ib, ia)
                recs["hi"] = np.where(swap, ia, ib)
                recs["rev"] = swap
                recs["v"] = vals
                part = (recs["lo"].astype(np.int64) * 1_000_003 + recs["hi"]) % partitions
                order = np.argsort(part, kind="stable")
                bounds = np.searchsorted(part[order], np.arange(partitions + 1))
                for k in range(partitions):
                    if bounds[k + 1] > bounds[k]:
                        handles[k].write(recs[order[bounds[k]:bounds[k + 1]]].tobytes())

            def parse(stream: io.BufferedReader) -> int:
                nonlocal n_records
                head = stream.peek(2)[:2]
                text = gzip.GzipFile(fileobj=stream, mode="rb") if head == b"\x1f\x8b" else stream
                first_line = text.readline()
                _refuse_carriage_return(first_line, path, 1)
                header = first_line.rstrip(b"\n").split(b" ")
                if tuple(h.decode("ascii", "replace") for h in header) != STRING_HEADER:
                    raise ValueError(f"{path}: header is not the STRING detailed-links header {' '.join(STRING_HEADER)}")
                p1s: list[bytes] = []
                p2s: list[bytes] = []
                scores: list[bytes] = []
                first = 2
                for lineno, line in enumerate(text, start=2):
                    p1, p2, s = _split_string_record(line, lineno)
                    p1s.append(p1)
                    p2s.append(p2)
                    scores.append(s)
                    if len(p1s) >= chunk_records:
                        flush(p1s, p2s, scores, first)
                        n_records += len(p1s)
                        first = lineno + 1
                        p1s, p2s, scores = [], [], []
                flush(p1s, p2s, scores, first)
                n_records += len(p1s)
                if text is not stream:
                    text.close()
                return n_records

            parsed: Parsed = parse_verified(Path(path), parse, expected_sha256=expected_sha256)
        finally:
            for h in handles:
                h.close()
        scratch = sum(p.stat().st_size for p in spool)
        if scratch != n_records * _RECORD.itemsize:
            raise AssertionError("partition files do not hold every record exactly once")
        if not names:
            raise ValueError(f"{path}: no STRING records")
        taxa = {p.split(b".", 1)[0] for p in names}
        if len(taxa) != 1 or any(b"." not in p for p in names):
            raise ValueError(f"{path}: protein identifiers name {len(taxa)} taxa or lack the 'taxon.' prefix")
        pieces = []
        largest = 0
        for p in spool:
            recs = np.fromfile(p, dtype=_RECORD)
            largest = max(largest, len(recs))
            pieces.append(_check_partition(recs, names))
            p.unlink()
    pairs = np.concatenate(pieces) if pieces else np.empty(0, _RECORD)
    del pieces
    sorted_names = sorted(names)
    rank = np.empty(len(names), np.int32)
    rank[np.argsort(np.array(names, dtype=object))] = np.arange(len(names), dtype=np.int32)
    a, b = rank[pairs["lo"]], rank[pairs["hi"]]
    if not np.all(a < b):
        raise AssertionError("canonical orientation disagrees with protein rank")
    order = np.lexsort((b, a))
    a, b = a[order], b[order]
    vals = pairs["v"][order]
    del pairs, order
    channels, combined = np.ascontiguousarray(vals[:, :7]), np.ascontiguousarray(vals[:, 7])
    del vals
    return StringPairTable(tuple(n.decode("ascii") for n in sorted_names), a, b, channels, combined, parsed.sha256,
                           parsed.n_bytes, int(n_records), partitions, int(largest), int(scratch))


def read_string_info(path: Path, *, expected_sha256: str | None = None) -> tuple[dict[str, str], str]:
    """STRING ``protein.info``: {protein id: preferred name} and the SHA-256 of the bytes parsed. Every row must have the
    four documented fields; a repeated protein id is refused."""
    def parse(stream: io.BufferedReader) -> dict[str, str]:
        head = stream.peek(2)[:2]
        text = gzip.GzipFile(fileobj=stream, mode="rb") if head == b"\x1f\x8b" else stream
        first_line = text.readline()
        _refuse_carriage_return(first_line, path, 1)
        header = tuple(first_line.rstrip(b"\n").decode("ascii").split("\t"))
        if header != STRING_INFO_HEADER:
            raise ValueError(f"{path}: header {header} is not {STRING_INFO_HEADER}")
        out: dict[str, str] = {}
        for lineno, line in enumerate(text, start=2):
            _refuse_carriage_return(line, path, lineno)
            parts = line.rstrip(b"\n").decode("utf-8").split("\t")
            if len(parts) != len(STRING_INFO_HEADER) or not parts[0] or not parts[1]:
                raise ValueError(f"{path} row {lineno}: expected {len(STRING_INFO_HEADER)} fields with id and name")
            if parts[0] in out:
                raise ValueError(f"{path} row {lineno}: protein {parts[0]} listed twice")
            out[parts[0]] = parts[1]
        if text is not stream:
            text.close()
        return out
    parsed = parse_verified(Path(path), parse, expected_sha256=expected_sha256)
    return parsed.value, parsed.sha256


# ======================================================================================== reconstruction and threshold
class QuantizationModel(str, Enum):
    """How STRING's exported integers relate to its internal scores. ROUNDED_CHANNELS_FLOORED_COMBINED is the model
    measured as best supported among the four tested (ruling 7f4b69d0, section 6): it is a measured model, not a fact
    STRING documents."""
    ROUNDED_CHANNELS_FLOORED_COMBINED = "rounded_channels_floored_combined"


def combine_channels(milli: np.ndarray) -> np.ndarray:
    """STRING's combination of channel probabilities (thousandths in, probability out), applied in column order: prior
    removed per channel, independent combination, prior added back."""
    s = np.asarray(milli, dtype=np.float64) / 1000.0
    if s.ndim != 2:
        raise ValueError("channel matrix must be two-dimensional")
    adj = np.clip((s - STRING_PRIOR) / (1.0 - STRING_PRIOR), 0.0, None)
    p = np.ones(s.shape[0])
    for j in range(s.shape[1]):
        p = p * (1.0 - adj[:, j])
    return (1.0 - p) + p * STRING_PRIOR


def combine_exact(probabilities: Sequence[Fraction]) -> Fraction:
    """``combine_channels`` for one pair in exact rational arithmetic (STRING_PRIOR = 41/1000 exactly)."""
    p = Fraction(1)
    for s in probabilities:
        p *= 1 - max(Fraction(0), (s - _PRIOR_Q) / (1 - _PRIOR_Q))
    return (1 - p) + p * _PRIOR_Q


def _exact_milli(row: Sequence[int], offset_half_milli: int) -> Fraction:
    """1000 x the exact combination of one row of channel integers moved by ``offset_half_milli`` half-thousandths
    (-1: lower bound, 0: the stored values, +1: upper bound), each channel clipped to [0, 1]."""
    vals = [min(Fraction(1), max(Fraction(0), Fraction(2 * int(k) + offset_half_milli, 2000))) for k in row]
    return combine_exact(vals) * 1000


def _ge_exactly(approx: np.ndarray, rows: np.ndarray, offset: int, target: int) -> tuple[np.ndarray, int]:
    """approx >= target, with every value within the float guard of ``target`` decided exactly. Returns the decision and
    how many pairs needed the exact decision."""
    out = approx >= target
    near = np.flatnonzero(np.abs(approx - target) < _FLOAT_GUARD)
    for i in near:
        out[i] = _exact_milli(rows[i], offset) >= target
    return out, int(len(near))


def _floor_exactly(approx: np.ndarray, rows: np.ndarray, offset: int) -> tuple[np.ndarray, int]:
    """floor(approx) with every value within the float guard of an integer floored exactly."""
    out = np.floor(approx)
    near = np.flatnonzero(np.abs(approx - np.round(approx)) < _FLOAT_GUARD)
    for i in near:
        out[i] = math.floor(_exact_milli(rows[i], offset))
    return out, int(len(near))


def channel_bounds(milli: np.ndarray, model: QuantizationModel) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds of the combined probability over every true channel vector consistent with the stored
    integers under ``model`` (the combination is nondecreasing in every channel)."""
    if model is not QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED:
        raise ValueError(f"unsupported quantization model {model}")
    k = np.asarray(milli, dtype=np.float64)
    return combine_channels(np.clip(k - 0.5, 0, 1000)), combine_channels(np.clip(k + 0.5, 0, 1000))


def model_consistency(table: StringPairTable, model: QuantizationModel) -> dict[str, Any]:
    """How many published combined scores are attainable under ``model`` from the stored channels (all seven): stored
    combined = floor(1000 S) for some S in [lower, upper]. Floors within the float guard of an integer are exact."""
    lo, hi = channel_bounds(table.channels, model)
    pl, n_lo = _floor_exactly(lo * 1000, table.channels, -1)
    ph, n_hi = _floor_exactly(hi * 1000, table.channels, +1)
    p = table.combined.astype(np.float64)
    miss = np.where(p < pl, pl - p, np.where(p > ph, p - ph, 0))
    return {"model": model.value, "pairs": int(len(p)), "inconsistent_pairs": int((miss > 0).sum()),
            "consistent_fraction": float(1 - (miss > 0).mean()) if len(p) else 1.0,
            "max_miss_thousandths": int(miss.max()) if len(p) else 0,
            "bounds_floored_exactly": n_lo + n_hi}


class ThresholdStatus(str, Enum):
    SELECTED_ROBUST = "selected_robust"
    SELECTED_SENSITIVE = "selected_sensitive"
    EXCLUDED_SENSITIVE = "excluded_sensitive"
    EXCLUDED_ROBUST = "excluded_robust"


_STATUS_CODE = {s: i for i, s in enumerate(ThresholdStatus)}


@dataclass(frozen=True)
class ThresholdClassification:
    """Per pair: the four-way status (codes index ``ThresholdStatus``) and whether the published combined score passes.
    ``selected`` = reconstruction >= threshold AND published >= threshold (G1 is nested in G2, ruling 7f4b69d0 sec. 8)."""

    channels: tuple[str, ...]
    threshold_milli: int
    model: QuantizationModel
    status: np.ndarray
    published_passes: np.ndarray
    decided_exactly: Mapping[str, int]

    @property
    def selected(self) -> np.ndarray:
        return (self.status == _STATUS_CODE[ThresholdStatus.SELECTED_ROBUST]) | (
            self.status == _STATUS_CODE[ThresholdStatus.SELECTED_SENSITIVE])

    def crosstab(self) -> dict[str, int]:
        out = {}
        for s in ThresholdStatus:
            m = self.status == _STATUS_CODE[s]
            out[f"{s.value}|published_passes"] = int((m & self.published_passes).sum())
            out[f"{s.value}|published_fails"] = int((m & ~self.published_passes).sum())
        return out


def check_threshold_milli(threshold_milli: int) -> int:
    if isinstance(threshold_milli, bool) or not isinstance(threshold_milli, int) or not 1 <= threshold_milli <= 1000:
        raise ValueError("the threshold is an integer number of thousandths in [1, 1000]")
    return threshold_milli


def classify_threshold(table: StringPairTable, channels: Sequence[str], threshold_milli: int,
                       model: QuantizationModel = QuantizationModel.ROUNDED_CHANNELS_FLOORED_COMBINED
                       ) -> ThresholdClassification:
    """Ruling 7f4b69d0 section 7, with section 8's nesting: reconstruct the restricted-channel score, bound it under the
    declared model, and classify every pair. Every comparison whose float value lies within the guard of the threshold is
    decided in exact rational arithmetic (a single channel stored as 700 reconstructs to exactly 0.700, which floats
    place on either side), and how many were is recorded."""
    check_threshold_milli(threshold_milli)
    cols = _channel_columns(channels)
    sub = table.channels[:, cols]
    rec = combine_channels(sub) * 1000
    lo, hi = (x * 1000 for x in channel_bounds(sub, model))
    rec_ge, n_rec = _ge_exactly(rec, sub, 0, threshold_milli)
    lo_ge, n_lo = _ge_exactly(lo, sub, -1, threshold_milli)
    hi_ge, n_hi = _ge_exactly(hi, sub, +1, threshold_milli)
    published = table.combined.astype(np.int64) >= threshold_milli
    selected = rec_ge & published
    sensitive = ~lo_ge & hi_ge
    status = np.full(len(table), _STATUS_CODE[ThresholdStatus.EXCLUDED_ROBUST], np.uint8)
    status[~selected & sensitive] = _STATUS_CODE[ThresholdStatus.EXCLUDED_SENSITIVE]
    status[selected & ~sensitive] = _STATUS_CODE[ThresholdStatus.SELECTED_ROBUST]
    status[selected & sensitive] = _STATUS_CODE[ThresholdStatus.SELECTED_SENSITIVE]
    return ThresholdClassification(tuple(channels), threshold_milli, model, status, published,
                                   {"reconstruction": n_rec, "lower_bound": n_lo, "upper_bound": n_hi})


def _channel_columns(channels: Sequence[str]) -> list[int]:
    if not channels or len(set(channels)) != len(channels) or any(c not in STRING_CHANNELS for c in channels):
        raise ValueError(f"channels must be distinct members of {STRING_CHANNELS}")
    return [STRING_CHANNELS.index(c) for c in channels]


# ============================================================================================ canonical gene identity
HGNC_ID = re.compile(r"HGNC:[1-9][0-9]*\Z")
HGNC_REQUIRED = ("hgnc_id", "symbol", "status", "prev_symbol", "alias_symbol")


class MappingStatus(str, Enum):
    APPROVED_SYMBOL = "approved_symbol"
    PREVIOUS_SYMBOL_UNIQUE = "previous_symbol_unique"
    PREVIOUS_SYMBOL_AMBIGUOUS = "previous_symbol_ambiguous"
    UNRESOLVED = "unresolved"
    COLLISION_SUPERSEDED = "collision_superseded"     # its HGNC id is claimed by a protein named by the approved symbol
    COLLISION_AMBIGUOUS = "collision_ambiguous"       # its HGNC id is claimed only by several previous-symbol proteins


MAPPED = frozenset({MappingStatus.APPROVED_SYMBOL, MappingStatus.PREVIOUS_SYMBOL_UNIQUE})


@dataclass(frozen=True)
class HgncIndex:
    """The approved-gene table of one HGNC release: approved symbol -> id, previous symbol -> ids, alias -> ids. Symbols
    are matched EXACTLY (HGNC symbols are case-sensitive in form, e.g. C10orf113)."""

    approved: Mapping[str, str]
    previous: Mapping[str, frozenset[str]]
    alias: Mapping[str, frozenset[str]]
    source_sha256: str
    n_rows: int

    def resolve(self, symbol: str) -> tuple[str | None, MappingStatus]:
        if symbol in self.approved:
            return self.approved[symbol], MappingStatus.APPROVED_SYMBOL
        ids = self.previous.get(symbol)
        if ids:
            return (next(iter(ids)), MappingStatus.PREVIOUS_SYMBOL_UNIQUE) if len(ids) == 1 else (
                None, MappingStatus.PREVIOUS_SYMBOL_AMBIGUOUS)
        return None, MappingStatus.UNRESOLVED


def read_hgnc(path: Path, *, expected_sha256: str | None = None) -> HgncIndex:
    """hgnc_complete_set.txt (tab-separated, one header row) in one verified pass. Only Approved rows are indexed; a
    repeated HGNC id or approved symbol is refused."""
    def parse(stream: io.BufferedReader) -> HgncIndex:
        lines = io.TextIOWrapper(stream, encoding="utf-8", newline="\n")
        first_line = lines.readline()
        _refuse_carriage_return(first_line, path, 1)
        header = first_line.rstrip("\n").split("\t")
        missing = [c for c in HGNC_REQUIRED if c not in header]
        if missing or len(set(header)) != len(header):
            raise ValueError(f"{path}: HGNC header lacks {missing} or repeats a column")
        col = {c: header.index(c) for c in HGNC_REQUIRED}
        approved: dict[str, str] = {}
        ids: set[str] = set()
        prev: dict[str, set[str]] = {}
        alias: dict[str, set[str]] = {}
        n = 0
        for lineno, line in enumerate(lines, start=2):
            _refuse_carriage_return(line, path, lineno)
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                raise ValueError(f"{path} row {lineno}: {len(parts)} fields, header has {len(header)}")
            n += 1
            if parts[col["status"]] != "Approved":
                continue
            hid, sym = parts[col["hgnc_id"]], parts[col["symbol"]]
            if not HGNC_ID.match(hid) or not sym:
                raise ValueError(f"{path} row {lineno}: malformed HGNC id {hid!r} or empty symbol")
            if hid in ids or sym in approved:
                raise ValueError(f"{path} row {lineno}: HGNC id or approved symbol repeated ({hid}, {sym})")
            ids.add(hid)
            approved[sym] = hid
            for target, column in ((prev, "prev_symbol"), (alias, "alias_symbol")):
                for s in parts[col[column]].strip('"').split("|"):
                    if s.strip():
                        target.setdefault(s.strip(), set()).add(hid)
        lines.detach()
        return HgncIndex(approved, {k: frozenset(v) for k, v in prev.items()},
                         {k: frozenset(v) for k, v in alias.items()}, "", n)
    parsed = parse_verified(Path(path), parse, expected_sha256=expected_sha256)
    idx = parsed.value
    return HgncIndex(idx.approved, idx.previous, idx.alias, parsed.sha256, idx.n_rows)


@dataclass(frozen=True)
class ProteinGeneRow:
    protein_id: str
    preferred_name: str
    hgnc_id: str | None
    status: MappingStatus
    mapping_source: str         # "dataset:hgnc" -- the identity table the resolution used


@dataclass(frozen=True)
class ProteinGeneMapping:
    rows: tuple[ProteinGeneRow, ...]
    info_sha256: str
    hgnc_sha256: str
    mapping_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        ids = [r.protein_id for r in self.rows]
        if ids != sorted(set(ids)):
            raise ValueError("mapping rows must be sorted by protein id, one per protein")
        mapped = [r.hgnc_id for r in self.rows if r.status in MAPPED]
        if len(mapped) != len(set(mapped)):
            raise ValueError("a canonical mapping maps no two proteins to one HGNC id")
        if any((r.hgnc_id is None) == (r.status in MAPPED) for r in self.rows):
            raise ValueError("a row has an HGNC id exactly when its status is a mapped status")
        payload = {"schema": "gvc.protein-gene-mapping/1", "info_sha256": self.info_sha256,
                   "hgnc_sha256": self.hgnc_sha256,
                   "rows": [[r.protein_id, r.preferred_name, r.hgnc_id, r.status.value, r.mapping_source]
                            for r in self.rows]}
        object.__setattr__(self, "mapping_sha256", sha256_bytes(canonical_bytes(payload)))

    def gene_of(self) -> dict[str, str]:
        return {r.protein_id: r.hgnc_id for r in self.rows if r.status in MAPPED}  # type: ignore[misc]

    def status_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(r.status.value for r in self.rows).items()))


def map_string_proteins(info: Mapping[str, str], info_sha256: str, hgnc: HgncIndex) -> tuple[ProteinGeneMapping,
                                                                                             dict[str, Any]]:
    """Resolve every STRING protein's preferred name to an HGNC id (approved symbol, else a unique previous symbol), then
    resolve canonical collisions by approved-symbol precedence: when several proteins reach one HGNC id and exactly one is
    named by the approved symbol, it keeps the id and the others become COLLISION_SUPERSEDED; with no approved-symbol
    claimant, every claimant becomes COLLISION_AMBIGUOUS. Returns the mapping and the measured collision report."""
    first: dict[str, tuple[str | None, MappingStatus]] = {p: hgnc.resolve(n) for p, n in info.items()}
    claimants: dict[str, list[str]] = {}
    for p, (hid, st) in first.items():
        if st in MAPPED:
            claimants.setdefault(hid, []).append(p)  # type: ignore[arg-type]
    final = dict(first)
    report_rows = []
    for hid, ps in sorted(claimants.items()):
        if len(ps) < 2:
            continue
        winners = [p for p in ps if first[p][1] is MappingStatus.APPROVED_SYMBOL]
        for p in ps:
            if len(winners) == 1 and p == winners[0]:
                continue
            final[p] = (None, MappingStatus.COLLISION_SUPERSEDED if len(winners) == 1 else MappingStatus.COLLISION_AMBIGUOUS)
        report_rows.append({"hgnc_id": hid, "claimants": [[p, info[p], first[p][1].value] for p in sorted(ps)],
                            "kept": winners[0] if len(winners) == 1 else None})
    alias_resolvable = sum(1 for p, (hid, st) in first.items() if st is MappingStatus.UNRESOLVED
                           and len(hgnc.alias.get(info[p], ())) == 1)
    rows = tuple(ProteinGeneRow(p, info[p], final[p][0], final[p][1], "dataset:hgnc") for p in sorted(info))
    mapping = ProteinGeneMapping(rows, info_sha256, hgnc.source_sha256)
    report = {"proteins": len(rows), "status_counts": mapping.status_counts(), "collisions": report_rows,
              "unresolved_but_unique_alias": alias_resolvable,
              "note": "aliases are not used for canonical identity; the count shows what an alias rule would add"}
    return mapping, report


# =================================================================================================== graph identity
class Regime(str, Enum):
    G0 = "G0"     # independently reviewed experimental observations -- not obtainable from STRING channels
    G1 = "G1"     # STRING channel-restricted, curation-reduced research graph, nested in G2
    G2 = "G2"     # STRING published combined score


@dataclass(frozen=True)
class GraphPolicy:
    """What graph is REQUESTED. Every field is part of ``policy_sha256`` (the relation policy identity)."""

    source: str                       # census dataset, e.g. "dataset:string"
    release: str
    source_sha256: str                # pinned links file
    info_sha256: str                  # pinned protein-info file
    mapping_source: str               # "dataset:hgnc"
    mapping_source_sha256: str
    regime: Regime
    selection_channels: tuple[str, ...]
    attribute_channels: tuple[str, ...]
    threshold_milli: int
    quantization: QuantizationModel | None = None
    threshold_robust_only: bool = False
    edge_reduction: str = "exact_undirected_pair"
    canonicalization: str = "hgnc_id:approved_or_unique_previous_symbol:approved_symbol_precedence"
    node_universe: str = "global_source"

    def __post_init__(self) -> None:
        for d in (self.source_sha256, self.info_sha256, self.mapping_source_sha256):
            if not isinstance(d, str) or not HEX64.match(d):
                raise ValueError("source pins are measured lower-case SHA-256 digests")
        if not self.source.startswith("dataset:") or not self.mapping_source.startswith("dataset:"):
            raise ValueError("sources are census datasets ('dataset:<id>')")
        if not self.release:
            raise ValueError("the release is part of the identity")
        check_threshold_milli(self.threshold_milli)
        _channel_columns(self.selection_channels)
        if self.attribute_channels:
            _channel_columns(self.attribute_channels)
        if self.edge_reduction != "exact_undirected_pair" or self.node_universe != "global_source" or \
                self.canonicalization != "hgnc_id:approved_or_unique_previous_symbol:approved_symbol_precedence":
            raise ValueError("unsupported edge reduction, node universe or canonicalisation")
        if self.regime is Regime.G0:
            raise ValueError("G0 needs an independently reviewed experimental source; STRING's channels are calibrated "
                             "against curated references (KEGG pathways, known complexes), so no STRING policy is G0")
        if self.regime is Regime.G2:
            if tuple(self.selection_channels) != STRING_CHANNELS or self.quantization is not None or \
                    self.threshold_robust_only:
                raise ValueError("G2 selects on the published combined score: all seven channels, no reconstruction")
        if self.regime is Regime.G1:
            if set(self.selection_channels) & STRING_CURATED_CHANNELS:
                raise ValueError("G1 is curation-reduced: it may not select on the database or text-mining channel")
            if self.quantization is None:
                raise ValueError("G1 reconstructs a restricted score and must declare its quantization model")

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "release": self.release, "source_sha256": self.source_sha256,
                "info_sha256": self.info_sha256, "mapping_source": self.mapping_source,
                "mapping_source_sha256": self.mapping_source_sha256, "regime": self.regime.value,
                "selection_channels": list(self.selection_channels), "attribute_channels": list(self.attribute_channels),
                "threshold_milli": self.threshold_milli,
                "quantization": None if self.quantization is None else self.quantization.value,
                "threshold_robust_only": self.threshold_robust_only, "edge_reduction": self.edge_reduction,
                "canonicalization": self.canonicalization, "node_universe": self.node_universe,
                "edge_semantics": "predicted_functional_association_not_physical_interaction",
                "schema": GRAPH_SCHEMA_VERSION}

    @property
    def policy_sha256(self) -> str:
        return sha256_bytes(canonical_bytes(self.as_dict()))


@dataclass(frozen=True)
class GraphMaterialization:
    """Ruling 7f4b69d0 section 9: the materialised graph's identity. Built only by ``materialization_of`` from content."""

    source_sha256: str
    mapping_sha256: str
    node_universe_sha256: str
    relation_policy_sha256: str
    edge_content_sha256: str
    graph_schema_version: str

    def as_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class GeneGraph:
    """A gene-level graph: ``nodes`` sorted HGNC ids (the global source-defined universe), ``edges`` sorted canonical
    (u, v) with u < v, ``attributes`` the declared channel integers per edge (thousandths)."""

    policy: GraphPolicy
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    attribute_channels: tuple[str, ...]
    attributes: np.ndarray
    source_sha256: str
    mapping_sha256: str
    materialization: GraphMaterialization
    report: Mapping[str, Any]


def node_universe_digest(nodes: Sequence[str]) -> str:
    if list(nodes) != sorted(set(nodes)):
        raise ValueError("the node universe is a sorted set")
    return sha256_bytes(canonical_bytes({"schema": "gvc.node-universe/1", "nodes": list(nodes)}))


def edge_content_digest(edges: Sequence[tuple[str, str]], attribute_channels: Sequence[str], attributes: np.ndarray) -> str:
    if list(edges) != sorted(set(edges)) or any(u >= v for u, v in edges):
        raise ValueError("edges are sorted canonical pairs (u < v) without repeats")
    if attributes.shape != (len(edges), len(attribute_channels)):
        raise ValueError("one attribute row per edge, one column per declared channel")
    head = canonical_bytes({"schema": "gvc.edge-content/1", "attribute_channels": list(attribute_channels),
                            "edges_sha256": sha256_bytes("\n".join(f"{u}\t{v}" for u, v in edges).encode("ascii"))})
    return sha256_bytes(head + np.ascontiguousarray(attributes, "<u2").tobytes())


def materialization_of(policy: GraphPolicy, nodes: Sequence[str], edges: Sequence[tuple[str, str]],
                       attributes: np.ndarray, source_sha256: str, mapping_sha256: str) -> GraphMaterialization:
    return GraphMaterialization(source_sha256, mapping_sha256, node_universe_digest(nodes), policy.policy_sha256,
                                edge_content_digest(edges, policy.attribute_channels, attributes), GRAPH_SCHEMA_VERSION)


def materialize_string_graph(table: StringPairTable, mapping: ProteinGeneMapping, policy: GraphPolicy) -> GeneGraph:
    """Build the gene-level graph the policy requests from a verified pair table and a canonical mapping, and measure what
    the reduction did: pairs selected, pairs lost to an unmapped endpoint (by status), pairs that became self-loops."""
    if table.source_sha256 != policy.source_sha256:
        raise ValueError("the pair table was not parsed from the policy's pinned source")
    if mapping.info_sha256 != policy.info_sha256 or mapping.hgnc_sha256 != policy.mapping_source_sha256:
        raise ValueError("the mapping was not built from the policy's pinned protein-info and HGNC sources")
    if policy.regime is Regime.G2:
        selected = table.combined.astype(np.int64) >= policy.threshold_milli
        crosstab = None
    else:
        cls = classify_threshold(table, policy.selection_channels, policy.threshold_milli,
                                 policy.quantization)  # type: ignore[arg-type]
        selected = (cls.status == _STATUS_CODE[ThresholdStatus.SELECTED_ROBUST]) if policy.threshold_robust_only \
            else cls.selected
        crosstab = cls.crosstab()
    gene_of = mapping.gene_of()
    status_of = {r.protein_id: r.status.value for r in mapping.rows}
    missing_info = sorted(set(table.proteins) - set(status_of))
    if missing_info:
        raise ValueError(f"{len(missing_info)} proteins of the links file are absent from the protein-info file")
    prot_gene = np.array([gene_of.get(p, "") for p in table.proteins], dtype=object)
    ga, gb = prot_gene[table.a[selected]], prot_gene[table.b[selected]]
    mapped = (ga != "") & (gb != "")
    lost = Counter()
    prot_status = np.array([status_of[p] for p in table.proteins], dtype=object)
    for i in np.flatnonzero(~mapped):
        for side in (table.a[selected][i], table.b[selected][i]):
            if prot_gene[side] == "":
                lost[prot_status[side]] += 1
    self_loop = mapped & (ga == gb)
    keep = mapped & ~self_loop
    u = np.where(ga[keep] < gb[keep], ga[keep], gb[keep])
    v = np.where(ga[keep] < gb[keep], gb[keep], ga[keep])
    attr_cols = _channel_columns(policy.attribute_channels) if policy.attribute_channels else []
    attrs = table.channels[selected][keep][:, attr_cols] if attr_cols else np.zeros((int(keep.sum()), 0), np.uint16)
    order = sorted(range(len(u)), key=lambda i: (u[i], v[i]))
    edges = tuple((str(u[i]), str(v[i])) for i in order)
    if len(set(edges)) != len(edges):
        raise AssertionError("an injective mapping produced a repeated gene pair")
    attrs = np.ascontiguousarray(attrs[order], np.uint16)
    nodes = tuple(sorted(set(gene_of.values())))
    mat = materialization_of(policy, nodes, edges, attrs, table.source_sha256, mapping.mapping_sha256)
    report = {"pairs": int(len(table)), "selected_pairs": int(selected.sum()), "edges": len(edges),
              "selected_pairs_with_unmapped_endpoint": int((~mapped).sum()),
              "unmapped_endpoint_status_counts": dict(sorted(lost.items())),
              "selected_pairs_becoming_self_loops": int(self_loop.sum()), "nodes": len(nodes),
              "threshold_crosstab": crosstab}
    return GeneGraph(policy, nodes, edges, tuple(policy.attribute_channels), attrs, table.source_sha256,
                     mapping.mapping_sha256, mat, report)


def accept_graph(policy: GraphPolicy, graph: GeneGraph) -> GraphMaterialization:
    """Accept a graph for ``policy`` only if its identity, RECOMPUTED from its own content, matches the policy and the
    manifest it carries. A matching filename, a declared label or a syntactically valid digest proves nothing."""
    if graph.policy.policy_sha256 != policy.policy_sha256:
        raise ValueError("the graph was built for a different graph policy")
    recomputed = materialization_of(policy, graph.nodes, graph.edges, graph.attributes, graph.source_sha256,
                                    graph.mapping_sha256)
    if recomputed != graph.materialization:
        diff = sorted(k for k, v in recomputed.as_dict().items() if graph.materialization.as_dict()[k] != v)
        raise ValueError(f"the graph's manifest disagrees with its content in {diff}")
    if recomputed.source_sha256 != policy.source_sha256:
        raise ValueError("the graph's source is not the policy's pinned source")
    node_set = set(graph.nodes)
    if any(u not in node_set or v not in node_set for u, v in graph.edges):
        raise ValueError("an edge names a node outside the graph's universe")
    return recomputed


# ----------------------------------------------------------------------------------------------- legacy cache identity
LEGACY_GNN = "src/genomic_variant_classifier/models/gnn.py"


def legacy_string_cache_name(repo: Path) -> str:
    """The legacy graph-cache filename template, read from StringDBGraph.build (the code's own f-string)."""
    tree = ast.parse((repo / LEGACY_GNN).read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StringDBGraph")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "build")
    found = [n for n in ast.walk(fn) if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "cache_path"]
    if len(found) != 1 or not isinstance(found[0].value, ast.BinOp) or not isinstance(found[0].value.right, ast.JoinedStr):
        raise ValueError("StringDBGraph.build no longer assigns cache_path = <dir> / f'...' exactly once")
    parts = []
    for v in found[0].value.right.values:
        if isinstance(v, ast.Constant):
            parts.append(str(v.value))
        elif isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Attribute) and v.value.attr == "threshold":
            parts.append("{threshold}")
        else:
            raise ValueError("the legacy cache filename depends on something other than the threshold")
    return "".join(parts)


def legacy_cache_collisions(repo: Path, policies: Sequence[GraphPolicy]) -> dict[str, Any]:
    """Distinct graph policies and the legacy cache files they would share (the legacy key holds only the threshold)."""
    template = legacy_string_cache_name(repo)
    by_name: dict[str, set[str]] = {}
    for p in policies:
        by_name.setdefault(template.format(threshold=p.threshold_milli), set()).add(p.policy_sha256)
    return {"template": template, "policies": len({p.policy_sha256 for p in policies}),
            "filenames": len(by_name), "policies_per_filename": {k: len(v) for k, v in sorted(by_name.items())}}


# ----------------------------------------------------------------------------------------------- census admission
def census_admission(policy: GraphPolicy, ledger: Mapping[str, Any]) -> dict[str, Any]:
    """What the census says about the information this policy uses (measured from the ledger, not asserted): the
    disposition of each STRING information source the policy touches. A strict regime would need every one admit_strict."""
    disp = {s["name"]: s["disposition"] for s in ledger["information_sources"]}
    used = {"string_relation:combined_score_threshold": disp["string_relation:combined_score_threshold"]}
    for c in policy.attribute_channels:
        name = f"string_edge_attribute:{c}"
        used[name] = disp.get(name, "not_in_census")
    strict = all(v == "admit_strict" for v in used.values())
    return {"policy_sha256": policy.policy_sha256, "regime": policy.regime.value, "information_sources": used,
            "strict_admissible": strict,
            "why": "every STRING regime selects through the published combined score (G1 is nested in G2), which the "
                   "census excludes from the strict regime; G1 and G2 are research graphs"}


# =============================================================================================== pathway relations
@dataclass(frozen=True)
class GeneSetSource:
    """One gene-set release: (source id, pathway id) -> canonical genes (HGNC ids), with what did not resolve."""

    source_id: str                                      # e.g. "reactome"
    relation: str                                       # e.g. "shares_pathway"
    sets: Mapping[str, frozenset[str]]                  # pathway id -> HGNC ids
    names: Mapping[str, str]                            # pathway id -> name
    source_sha256: str
    hgnc_sha256: str
    unresolved_members: int
    member_status_counts: Mapping[str, int]
    duplicate_names: int

    @property
    def relation_key(self) -> tuple[str, str]:
        return (self.source_id, self.relation)


def read_gene_sets(path: Path, hgnc: HgncIndex, *, source_id: str, relation: str,
                   expected_sha256: str | None = None) -> GeneSetSource:
    """A GMT file (name <TAB> id <TAB> members...) in one verified pass; sets keyed by their stable id (the second column),
    members resolved to HGNC ids by the same rule as the STRING mapping. A repeated id is refused; repeated names (which
    the legacy parser merges) are counted."""
    def parse(stream: io.BufferedReader) -> tuple[dict[str, frozenset[str]], dict[str, str], int, Counter]:
        lines = io.TextIOWrapper(stream, encoding="utf-8", newline="\n")
        sets: dict[str, frozenset[str]] = {}
        names: dict[str, str] = {}
        status: Counter = Counter()
        unresolved = 0
        for lineno, line in enumerate(lines, start=1):
            _refuse_carriage_return(line, path, lineno)
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0] or not parts[1]:
                raise ValueError(f"{path} line {lineno}: a set needs a name, an id and members")
            sid = parts[1]
            if sid in sets:
                raise ValueError(f"{path} line {lineno}: set id {sid} repeated")
            genes = set()
            for m in parts[2:]:
                if not m:
                    continue
                hid, st = hgnc.resolve(m)
                status[st.value] += 1
                if hid is None:
                    unresolved += 1
                else:
                    genes.add(hid)
            sets[sid] = frozenset(genes)
            names[sid] = parts[0]
        lines.detach()
        return sets, names, unresolved, status
    parsed = parse_verified(Path(path), parse, expected_sha256=expected_sha256)
    sets, names, unresolved, status = parsed.value
    dup = sum(c - 1 for c in Counter(names.values()).values() if c > 1)
    return GeneSetSource(source_id, relation, sets, names, parsed.sha256, hgnc.source_sha256, unresolved,
                         dict(sorted(status.items())), dup)


@dataclass(frozen=True)
class PathwayProjection:
    """Gene-gene edges of ONE relation key: weight = sum over the pathways containing both of 1/(|S_s| - 1), |S_s| the
    GLOBAL canonical size, summed exactly-rounded (math.fsum), so the weight does not depend on summation order."""

    relation_key: tuple[str, str]
    edges: Mapping[tuple[str, str], float]
    max_set_size: int | None
    sets_used: int
    sets_over_cap: int


def project_pathways(src: GeneSetSource, node_universe: Iterable[str], *, max_set_size: int | None) -> PathwayProjection:
    """Project pathway membership onto gene pairs inside ``node_universe``, with the size cap (if any) applied to the
    GLOBAL set size BEFORE any restriction, so an edge's existence and weight depend only on the source."""
    if max_set_size is not None and max_set_size < 2:
        raise ValueError("a size cap below 2 removes every set")
    universe = frozenset(node_universe)
    contrib: dict[tuple[str, str], list[float]] = {}
    used = over = 0
    for sid in sorted(src.sets):
        members = src.sets[sid]
        m = len(members)
        if m < 2:
            continue
        if max_set_size is not None and m > max_set_size:
            over += 1
            continue
        used += 1
        w = 1.0 / (m - 1)
        for pair in combinations(sorted(members & universe), 2):
            contrib.setdefault(pair, []).append(w)
    edges = {k: math.fsum(v) for k, v in sorted(contrib.items())}
    return PathwayProjection(src.relation_key, edges, max_set_size, used, over)


def induced_subgraph_consistency(small: Mapping[tuple[str, str], float], small_universe: Iterable[str],
                                 large: Mapping[tuple[str, str], float]) -> dict[str, int]:
    """Ruling 7f4b69d0 section 10: growing the node universe must not remove or reweight an edge among the nodes already
    present. Compares the small-universe projection with the large one restricted to the small universe."""
    u = frozenset(small_universe)
    induced = {k: w for k, w in large.items() if k[0] in u and k[1] in u}
    removed = sum(1 for k in small if k not in induced)
    added = sum(1 for k in induced if k not in small)
    reweighted = sum(1 for k, w in small.items() if k in induced and induced[k] != w)
    return {"small_edges": len(small), "induced_edges": len(induced), "removed_by_growth": removed,
            "added_by_growth": added, "reweighted_by_growth": reweighted,
            "consistent": int(removed == 0 and added == 0 and reweighted == 0)}


def hypergraph_matvec(sets: Mapping[Any, Iterable[str]], values: Mapping[str, float]) -> dict[str, float]:
    """(B D B^T - diag(B D B^T)) x without materialising cliques (ruling 7f4b69d0 section 10.1, the owner's reference)."""
    out = {g: 0.0 for g in values}
    for key in sorted(sets, key=repr):
        full = set(sets[key])
        if len(full) < 2:
            continue
        active = sorted(full & values.keys())
        if len(active) < 2:
            continue
        w = 1.0 / (len(full) - 1)
        total = math.fsum(values[g] for g in active)
        for g in active:
            out[g] += (total - values[g]) * w
    return out


__all__ = ["GRAPH_SCHEMA_VERSION", "STRING_CHANNELS", "STRING_CURATED_CHANNELS", "STRING_HEADER", "STRING_PRIOR",
           "GeneGraph", "GeneSetSource", "GraphMaterialization", "GraphPolicy", "HgncIndex", "MappingStatus",
           "PathwayProjection", "ProteinGeneMapping", "ProteinGeneRow", "QuantizationModel", "Regime",
           "StringPairTable", "ThresholdClassification", "ThresholdStatus", "accept_graph", "census_admission",
           "channel_bounds", "classify_threshold", "combine_channels", "combine_exact", "edge_content_digest",
           "hypergraph_matvec", "induced_subgraph_consistency", "legacy_cache_collisions", "legacy_string_cache_name",
           "map_string_proteins", "materialization_of", "materialize_string_graph", "model_consistency",
           "node_universe_digest", "pair_table_digest", "project_pathways", "read_gene_sets", "read_hgnc",
           "read_string_info", "read_string_pairs"]
