"""Stage 1D-1c.1C: the GRCh38 reference-identity authority -- one validation rule, two implementations, one identity
contract (owner ruling aa999b96, sections 14 and 15, and revised section 10).

* THE RULE is the reference-allele check of the 101 bp window build: ``data.delta_window_builder.build_window`` with a
  fetch from the reference sequence, and the verdict the window builder stored (``scripts/build_seq_windows.py``: ``ok``
  and the reason up to its first parenthesis). Its identity (``rule_identity``) binds the semantic-closure digest of
  ``build_window`` and of ``verdict_of``, the window, the coordinate convention and the builder version, measured from the
  code now: a code change is a different rule and invalidates every cached result.
* THE REFERENCE is pinned by content (``GRCH38_PRIMARY_ASSEMBLY``): the FASTA's SHA-256 and size, its index's SHA-256 and
  the window manifest's signature, measured on the owner's file (2026-11-16). The path is never part of an identity.
* KNOWN COHORT ROWS reuse the cached result (``data/processed/clinvar_grch38_clean_seq.parquet``, logical dataset
  ``dataset:grch38_reference_check``): ``materialize_reference_status`` aligns it to a population (re-hashing the
  evidence bytes it is given: integrity.require_verified_bytes) and ``write_reference_status`` stores it
  CONTENT-ADDRESSED -- its identifier is the digest of the source digest, the rule, the reference pin, the population
  identity, the row alignment and the result digest; its payload is stored under that identifier and its own SHA-256
  (integrity.store, set "av" revision c). ``load_reference_status`` reuses it only after verifying all of them; a changed
  source, rule, reference or population is refused by name.
* A ``ReferenceStatus`` IS VALID BY CONSTRUCTION (revision c, REFERENCE-VERDICT-ZIP-TRUNCATION-1): one boolean verdict
  and one reason per population row, consistent with each other, read-only, with an alignment to that population and a
  result digest that reproduces. ``result_digest`` refuses verdicts and reasons of different lengths (ordinary ``zip``
  stopped at the shorter one, so a digest could cover only the first rows); for well-formed verdicts it is unchanged.
* A NOVEL VARIANT is validated against the pinned sequence by the same rule (``PinnedReference`` and
  ``validate_variant``), never by membership in the ClinVar-derived table: absent from that table is not invalid.

What the rule does NOT establish, stated rather than implied: allele NORMALIZATION (left alignment and parsimony). The
window build anchors a deletion one base before ``pos`` when that slice matches (the cohort's deletion convention), so a
VCF normalization check is a different rule; it belongs to the stage 1D-1d identity authority (the VRS identifier).
Every verdict says ``normalization: not_established``.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from genomic_variant_classifier.information_sources import integrity as I
from genomic_variant_classifier.information_sources.census import HEX64, canonical_bytes, sha256_bytes, symbol_closure_digest
from genomic_variant_classifier.information_sources.columns import Population, readonly_array
from genomic_variant_classifier.information_sources.graph_identity import frozen_value
from genomic_variant_classifier.information_sources.materialization import Parsed, check_registered

logger = logging.getLogger(__name__)

__all__ = ["EVIDENCE_DATASET", "GRCH38_PRIMARY_ASSEMBLY", "NORMALIZATION", "PinnedReference", "RECEIPT_SCHEMA",
           "ReferencePin", "ReferenceStatus", "ReferenceVerdict", "SCHEMA", "load_reference_status",
           "materialize_reference_status", "result_digest", "rule_identity", "validate_variant", "verdict_of",
           "write_reference_status"]

SCHEMA = "gvc.reference-status.1d1c1c/1"
RECEIPT_SCHEMA = "gvc.reference-status-receipt.1d1c1c/2"
EVIDENCE_DATASET = "dataset:grch38_reference_check"
_THIS_FILE = "src/genomic_variant_classifier/information_sources/reference_status.py"
RULE_SYMBOL = "src/genomic_variant_classifier/data/delta_window_builder.py::build_window"
NORMALIZATION = ("not_established: the window rule checks coordinates and REF only; normalization is the stage 1D-1d "
                 "identity authority's")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------------------------------------------- the reference
@dataclass(frozen=True)
class ReferencePin:
    """A reference sequence pinned by content. ``window_manifest_signature`` is seq_window_manifest.reference_signature
    (SHA-256 of the index bytes then the decimal size), recorded so the pin is tied to the window build."""
    assembly: str
    description: str
    fasta_sha256: str
    fasta_bytes: int
    fai_sha256: str
    window_manifest_signature: str

    def __post_init__(self) -> None:
        for name in ("fasta_sha256", "fai_sha256", "window_manifest_signature"):
            if not HEX64.match(getattr(self, name)):
                raise ValueError(f"{name} must be a lower-case SHA-256")
        if type(self.fasta_bytes) is not int or self.fasta_bytes <= 0:
            raise ValueError("fasta_bytes is a positive int")

    def as_dict(self) -> dict[str, Any]:
        return {"assembly": self.assembly, "description": self.description, "fasta_sha256": self.fasta_sha256,
                "fasta_bytes": self.fasta_bytes, "fai_sha256": self.fai_sha256,
                "window_manifest_signature": self.window_manifest_signature}

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_bytes({"schema": SCHEMA, "reference": self.as_dict()}))


#: Measured 2026-11-16 on the owner's data/external/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa (Ensembl file
#: naming; the release is not recorded in the file). Its signature equals data/processed/seq_windows/seq_windows.manifest.json's
#: reference_signature, so it is the sequence the cached evidence was built from.
GRCH38_PRIMARY_ASSEMBLY = ReferencePin(
    assembly="GRCh38",
    description="Homo_sapiens.GRCh38.dna.primary_assembly.fa (194 sequences; release not recorded in the file)",
    fasta_sha256="1e74081a49ceb9739cc14c812fbb8b3db978eb80ba8e5350beb80d8ad8dfef3b",
    fasta_bytes=3_151_425_851,
    fai_sha256="1411003e3a78242551943185c3a4158920270c7564fae336200384e9e411e813",
    window_manifest_signature="234a0a170c3089b931aa6b462daa8fdc76774095b951c68ef1b7f911e74b91d6",
)


# ------------------------------------------------------------------------------------------------------------ the rule
def verdict_of(result: Any) -> tuple[bool, str]:
    """The verdict the window build stored for one WindowResult: ``ok``, and the reason up to its first parenthesis
    (scripts/build_seq_windows.py stores ``r.reason.split("(")[0] if not r.ok else ""``)."""
    ok = result.ok
    if type(ok) is not bool:
        raise TypeError("a window result's ok is a bool")
    return ok, "" if ok else str(result.reason).split("(")[0]


def rule_identity(repo: Path | None = None) -> dict[str, Any]:
    """The validation rule, measured from the code now. A change to build_window or verdict_of changes it."""
    from genomic_variant_classifier.data.seq_window_manifest import BUILDER_VERSION, CONVENTION, WINDOW

    root = _repo_root() if repo is None else repo
    return {"rule_id": "grch38_window_reference_match/1",
            "implementation": RULE_SYMBOL,
            "implementation_closure_sha256": symbol_closure_digest(root, RULE_SYMBOL),
            "verdict": f"{_THIS_FILE}::verdict_of",
            "verdict_closure_sha256": symbol_closure_digest(root, f"{_THIS_FILE}::verdict_of"),
            "window": WINDOW, "convention": CONVENTION, "builder_version": BUILDER_VERSION,
            "normalization": NORMALIZATION}


def result_digest(ok: Iterable[bool], reason: Iterable[str]) -> str:
    """SHA-256 of canonical [[ok, reason], ...] in row order: the digest the AF ledger records as the reference
    evidence's rows_sha256 and select_strict binds as reference_evidence_sha256. Every verdict is a bool and every reason
    a str, one of each per row: verdicts and reasons of different lengths are refused (REFERENCE-VERDICT-ZIP-TRUNCATION-1;
    an ordinary zip stopped at the shorter, so the digest covered only the first rows)."""
    ok, reason = list(ok), list(reason)
    if len(ok) != len(reason):
        raise ValueError(f"{len(ok)} reference verdicts but {len(reason)} reasons: the reason count differs from the "
                         "verdict count, and a result digest covers every row or is refused "
                         "(REFERENCE-VERDICT-ZIP-TRUNCATION-1)")
    if any(type(o) not in (bool, np.bool_) for o in ok) or any(type(r) is not str for r in reason):
        raise ValueError("a reference verdict is a bool and its reason a str")
    return sha256_bytes(canonical_bytes([[bool(o), r] for o, r in zip(ok, reason, strict=True)]))


@dataclass(frozen=True)
class ReferenceVerdict:
    ok: bool
    reason: str
    rule_sha256: str
    reference_sha256: str
    normalization: str = NORMALIZATION


def _rule_digest(rule: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes({"schema": SCHEMA, "rule": dict(rule)}))


class PinnedReference:
    """Random access to a pinned FASTA through pyfaidx (the library the window build used, pinned in the training lock).
    Opening verifies the index's SHA-256 and the file size against the pin, and -- unless ``verify_content`` is False,
    which only a caller holding a separately verified copy may choose -- the SHA-256 of every byte of the FASTA."""

    def __init__(self, fasta: Path, pin: ReferencePin = GRCH38_PRIMARY_ASSEMBLY, *, verify_content: bool = True):
        import pyfaidx

        fasta = Path(fasta)
        fai = Path(str(fasta) + ".fai")
        if not isinstance(pin, ReferencePin):
            raise TypeError("pin must be a ReferencePin")
        if type(verify_content) is not bool:
            raise TypeError("verify_content is a bool")
        if not fasta.is_file() or not fai.is_file():
            raise FileNotFoundError(f"{fasta} and its .fai index are both required")
        if hashlib.sha256(fai.read_bytes()).hexdigest() != pin.fai_sha256:
            raise ValueError(f"{fai}: the index is not the pinned reference's")
        if fasta.stat().st_size != pin.fasta_bytes:
            raise ValueError(f"{fasta}: {fasta.stat().st_size} bytes, the pinned reference has {pin.fasta_bytes}")
        if verify_content:
            h = hashlib.sha256()
            with open(fasta, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 24), b""):
                    h.update(block)
            if h.hexdigest() != pin.fasta_sha256:
                raise ValueError(f"{fasta}: content SHA-256 {h.hexdigest()} is not the pinned {pin.fasta_sha256}")
        self.pin = pin
        self.content_verified = verify_content
        self._fa = pyfaidx.Fasta(str(fasta), rebuild=False)

    def fetch(self, contig: Any, start0: int, length: int) -> str | None:
        """``length`` bases from the 0-based ``start0``, exactly the window build's fetch: a Python slice of the sequence
        (clipped at its end), None for an unknown sequence name or a negative start."""
        try:
            if start0 < 0:
                return None
            return str(self._fa[str(contig)][start0:start0 + length])
        except Exception:
            return None

    def close(self) -> None:
        self._fa.close()

    def __enter__(self) -> "PinnedReference":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def validate_variant(reference: PinnedReference, chrom: Any, pos: Any, ref: Any, alt: Any) -> ReferenceVerdict:
    """A novel (or any) variant validated against the pinned sequence by the training-time rule."""
    from genomic_variant_classifier.data.delta_window_builder import build_window
    from genomic_variant_classifier.data.seq_window_manifest import WINDOW

    if type(reference) is not PinnedReference:
        raise TypeError("reference must be a PinnedReference")
    if not reference.content_verified:
        raise PermissionError("a verdict needs a reference whose content was verified against its pin")
    ok, reason = verdict_of(build_window(reference.fetch, chrom, pos, ref, alt, WINDOW))
    return ReferenceVerdict(ok, reason, _rule_digest(rule_identity()), reference.pin.digest)


# ------------------------------------------------------------------------------------------------- the cached result
@dataclass(frozen=True, eq=False)
class ReferenceStatus:
    """The cached rule result aligned to one population, with every identity its reuse is checked against. VALID BY
    CONSTRUCTION: refused unless there is exactly one boolean verdict and one str reason per population row, each pair
    'ok with an empty reason' or 'not ok with a reason', the alignment is to this population, the reference is a
    well-formed pin and ``result_sha256`` reproduces. The verdicts are stored bytes-backed (read-only for good) and the
    three mappings read-only."""
    population: Population
    ok: np.ndarray                     # bool, read-only, population order
    reason: tuple[str, ...]            # population order
    source_sha256: str                 # the evidence file's content digest
    rule: Mapping[str, Any]
    reference: Mapping[str, Any]
    alignment: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.population, Population):
            raise TypeError("population must be a columns.Population")
        n = len(self.population)
        if not isinstance(self.ok, np.ndarray) or self.ok.ndim != 1 or self.ok.dtype != np.bool_:
            raise ValueError("reference verdicts are a one-dimensional boolean array")
        if type(self.reason) is not tuple:
            raise TypeError("reference reasons are a tuple of str")
        if len(self.ok) != n or len(self.reason) != n:
            raise ValueError(f"{len(self.ok)} reference verdicts and {len(self.reason)} reasons for a population of {n} "
                             "rows; each row has exactly one of each (REFERENCE-VERDICT-ZIP-TRUNCATION-1)")
        object.__setattr__(self, "ok", readonly_array(self.ok))
        _check_verdicts(self.ok, self.reason)
        for name in ("source_sha256", "result_sha256"):
            if type(getattr(self, name)) is not str or not HEX64.match(getattr(self, name)):
                raise ValueError(f"{name} is a lower-case SHA-256")
        for name in ("rule", "reference", "alignment"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} is a mapping")
            object.__setattr__(self, name, frozen_value(dict(getattr(self, name))))
        ReferencePin(**dict(self.reference))                    # refused unless a well-formed pin
        if self.alignment.get("population_sha256") != self.population.sha256:
            raise ValueError("the row alignment is for a different population")
        if result_digest(self.ok.tolist(), self.reason) != self.result_sha256:
            raise ValueError("the reference verdicts do not reproduce their result digest")

    def identity_payload(self) -> dict[str, Any]:
        verdicts: dict[str, int] = {}
        for o, r in zip(self.ok.tolist(), self.reason, strict=True):
            key = f"{bool(o)}:{r}"
            verdicts[key] = verdicts.get(key, 0) + 1
        return {"schema": SCHEMA,
                "reference_source": {"logical_source": EVIDENCE_DATASET, "content_sha256": self.source_sha256},
                "validation_rule": dict(self.rule), "reference_assembly": dict(self.reference),
                "prediction_units": {"population_id": self.population.population_id,
                                     "population_sha256": self.population.sha256, "rows": len(self.population)},
                "row_alignment": dict(self.alignment),
                "validation_result": {"rows_sha256": self.result_sha256, "verdicts": dict(sorted(verdicts.items()))}}

    @property
    def identity(self) -> str:
        return sha256_bytes(canonical_bytes(self.identity_payload()))


def _check_verdicts(ok: np.ndarray, reason: tuple[str, ...]) -> None:
    if ok.dtype != bool:
        raise ValueError("reference verdicts are booleans")
    if len(ok) != len(reason):
        raise ValueError(f"{len(ok)} reference verdicts but {len(reason)} reasons (REFERENCE-VERDICT-ZIP-TRUNCATION-1)")
    bad = [i for i, (o, r) in enumerate(zip(ok.tolist(), reason, strict=True)) if type(r) is not str or (o != (r == ""))]
    if bad:
        raise ValueError(f"{len(bad)} verdicts are not 'ok with an empty reason' or 'not ok with a reason', e.g. row {bad[0]}")


def _alignment(positions: np.ndarray, population: Population) -> dict[str, Any]:
    pos = np.ascontiguousarray(positions, dtype="<i8")
    return {"key": "variant_id", "match": "exact, each population row to one evidence row",
            "population_sha256": population.sha256,
            "evidence_row_positions_sha256": hashlib.sha256(pos.tobytes()).hexdigest()}


def materialize_reference_status(population: Population, evidence: Parsed, *, registered_datasets: Iterable[str],
                                 reference: ReferencePin = GRCH38_PRIMARY_ASSEMBLY) -> ReferenceStatus:
    """The cached evidence aligned to ``population``: every population row must have exactly one evidence row (refused
    otherwise); the evidence table must name no variant twice anywhere (refused otherwise). The table stays in Arrow
    form; only the population's rows become Python objects (the AF ledger measured a 1.9 GB peak for the whole table)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    check_registered(EVIDENCE_DATASET, tuple(registered_datasets))
    if not isinstance(population, Population):
        raise TypeError("population must be a columns.Population")
    if not isinstance(reference, ReferencePin):
        raise TypeError("reference must be a ReferencePin")
    if not isinstance(evidence, Parsed) or not isinstance(evidence.value, bytes) or not HEX64.match(evidence.sha256):
        raise ValueError("evidence must be the Parsed result of read_verified_bytes")
    I.require_verified_bytes(evidence, what=EVIDENCE_DATASET)      # the digest the status records, measured here
    table = pq.read_table(pa.BufferReader(evidence.value), columns=["variant_id", "ok", "reason"])
    if pc.count_distinct(table["variant_id"], mode="all").as_py() != table.num_rows:
        raise ValueError("the reference evidence names a variant twice")
    wanted = pa.array(population.row_ids, type=table.schema.field("variant_id").type)
    positions = pc.index_in(wanted, value_set=table["variant_id"])
    if positions.null_count:
        raise ValueError(f"{positions.null_count} population rows have no reference evidence")
    pos = positions.to_numpy(zero_copy_only=False).astype(np.int64)
    rows = table.take(pa.array(pos))
    del table
    ok_col = rows["ok"]
    if ok_col.null_count or not pa.types.is_boolean(ok_col.type):
        raise ValueError("the evidence's ok column must be non-null booleans")
    ok = readonly_array(ok_col.to_numpy(zero_copy_only=False).astype(bool))
    reason = tuple("" if r is None else str(r) for r in rows["reason"].to_pylist())
    _check_verdicts(ok, reason)
    return ReferenceStatus(population, ok, reason, evidence.sha256, rule_identity(), reference.as_dict(),
                           _alignment(pos, population), result_digest(ok.tolist(), reason))


def _receipt_bytes(status: ReferenceStatus, table: bytes) -> bytes:
    receipt = {"schema": RECEIPT_SCHEMA, "artifact_id": status.identity, "identity": status.identity_payload(),
               "payload": I.payload_entry(table), "columns": ["variant_id", "ok", "reason"]}
    return (json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def _table_bytes(status: ReferenceStatus) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"variant_id": pa.array(status.population.row_ids, type=pa.string()),
                      "ok": pa.array(status.ok.tolist(), type=pa.bool_()),
                      "reason": pa.array(list(status.reason), type=pa.string())})
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    return buf.getvalue()


def write_reference_status(status: ReferenceStatus, directory: Path) -> tuple[str, str]:
    """Store ``status`` under its content identity and its payload's SHA-256 (``<id>/<payload>.parquet`` and
    ``<id>/<payload>.receipt.json``, integrity.store) and return (identity, payload SHA-256). The same bytes again are a
    no-op only when the stored payload is present and intact; a different record under an existing name is refused, never
    overwritten; a record without its payload is refused (an incomplete store is reported, never repaired)."""
    if type(status) is not ReferenceStatus:
        raise TypeError("status must be a ReferenceStatus")
    ident = status.identity
    table = _table_bytes(status)
    payload = I.store(Path(directory), ident, table, _receipt_bytes(status, table), what="reference status")
    return ident, payload


def load_reference_status(directory: Path, artifact_id: str, payload_sha256: str, *, population: Population,
                          source_sha256: str, reference: ReferencePin = GRCH38_PRIMARY_ASSEMBLY) -> ReferenceStatus:
    """Reuse a stored result only after verifying it: the payload bytes against its record and its name, the arrays
    against the result and alignment digests, and the record against the declared source, the rule measured from the
    code NOW, the pinned reference and ``population``. Each mismatch is refused by name; the stored id must be the
    recomputed one."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if type(artifact_id) is not str or not HEX64.match(artifact_id):
        raise ValueError("artifact_id is a lower-case SHA-256")
    if type(source_sha256) is not str or not HEX64.match(source_sha256):
        raise ValueError("source_sha256 is the declared evidence digest (a lower-case SHA-256)")
    record, raw = I.read_stored(Path(directory), artifact_id, payload_sha256, what="reference status")
    receipt = json.loads(record.decode("ascii"))
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("artifact_id") != artifact_id:
        raise ValueError("the receipt is not a reference-status receipt for this identifier")
    if receipt.get("columns") != ["variant_id", "ok", "reason"]:
        raise ValueError("the receipt does not name the reference-status columns")
    ident = receipt["identity"]
    expected = {"reference_source": {"logical_source": EVIDENCE_DATASET, "content_sha256": source_sha256},
                "validation_rule": rule_identity(), "reference_assembly": reference.as_dict()}
    for key, value in expected.items():
        if ident.get(key) != value:
            raise ValueError(f"reuse refused: the stored {key} differs from the declared one (stored {ident.get(key)}, "
                             f"declared {value}); the cached result is invalid for this preparation")
    pop = ident.get("prediction_units", {})
    if pop.get("population_sha256") != population.sha256 or pop.get("rows") != len(population):
        raise ValueError("reuse refused: the stored result is aligned to a different population "
                         f"({pop.get('population_sha256')}, not {population.sha256})")
    table = pq.read_table(pa.BufferReader(raw))
    if table.column_names != ["variant_id", "ok", "reason"]:
        raise ValueError("the stored table's columns are not the reference-status columns")
    ids = tuple(table["variant_id"].to_pylist())
    if ids != population.row_ids:
        raise ValueError("reuse refused: the stored rows are not the population's rows in population order")
    if table["ok"].null_count or table["reason"].null_count:
        raise ValueError("the stored verdicts and reasons are non-null")
    ok = table["ok"].to_numpy(zero_copy_only=False).astype(bool)
    reason = tuple(table["reason"].to_pylist())
    # The constructor refuses verdicts that are not one per row, inconsistent, unaligned or not reproducing this digest.
    status = ReferenceStatus(population, ok, reason, source_sha256, ident["validation_rule"],
                             ident["reference_assembly"], ident["row_alignment"], ident["validation_result"]["rows_sha256"])
    if status.identity != artifact_id:
        raise ValueError("the stored result does not reproduce its content identifier")
    return status
