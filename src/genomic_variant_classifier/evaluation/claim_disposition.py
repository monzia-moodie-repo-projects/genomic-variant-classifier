"""Scientific claim dispositions: what a historical result is still permitted to support.

Author: Monzia Moodie

Package 1D, stage 1D-0. Owner ruling of 2026-11-08 (decision file SHA-256
56ca0eff9078e933faabe958dddb62f1017ee67c3a6036b8d5573a634f7476a0, sections 2 and 21), and the audit that extended it with
measurements (claude/AUDIT_OF_RULING_PACKAGE1D_PACKAGE2_2026-11-08b.md, refinement R1).

A REPORTED NUMBER AND THE CLAIM IT SUPPORTED ARE DIFFERENT OBJECTS
==================================================================
Run 15 reported an area under the receiver operating characteristic curve (AUROC) of 0.9988 on its unseen-gene holdout
and recorded the C3 falsifier as PASS. The number is a true record of what the code printed. The claim it was meant to
support -- generalization to genes the model never saw -- is not supported, because the experiment violated the
assumptions that claim needs (measured: the holdout rows' gene-count feature equals their own pathogenic labels' count,
the graph neural network was trained on the holdout genes' labels, and string-level gene grouping left 8.28% of holdout
rows sharing a gene with training). The same measurements apply to Run 15's validation and test evaluations.

The ruling's vocabulary separates the two:

    PASS / FAIL / UNKNOWN       what a VALID experiment found -- kept as the HISTORICAL verdict, never rewritten
    VALID                        the claim may be cited as evidence
    INVALIDATED_EVALUATION       the experiment violated assumptions its intended conclusion needs: the hypothesis is
                                 neither confirmed nor falsified, and the size of any optimism is NOT estimated
    SUPERSEDED                   a later evaluation, named by its claim identifier, replaces this one

Downstream reporting consumes the CURRENT disposition, never an old PASS string. `require_admissible` is the gate for
code; `find_unqualified_citations` is the gate for the current-claim documents (`GOVERNED_DOCUMENTS`): every paragraph or
table row there that cites a figure of an inadmissible claim must carry the disposition and the claim identifier beside
it. Historical logs (the changelog, session records, the archive) are records of what was believed, and are not edited.

THE REGISTRY IS APPEND-ONLY AND HASH-CHAINED
============================================
`docs/claims/CLAIM-NNNN_<claim_id>.json`, NNNN = 0001, 0002, ... with no gap. Each record carries `record_sha256`, the
canonical digest (prediction_contract.canonical_digest: sorted keys, ASCII, no NaN) of the record without that field, and
`previous_record_sha256`, the previous record's digest (null for the first). Editing a past record breaks its own digest;
replacing it consistently breaks every later link. A claim's current disposition is its latest record; a changed
disposition is a NEW record. Nothing here estimates inflation: an invalidated record must carry `inflation_estimate: null`
(ruling section 21: "Do not estimate its inflation without a valid replacement experiment").

KNOWN-INVALID DESIGNS DO NOT RENDER VERDICTS
============================================
The live driver (scripts/run_phase2_eval.py) still runs the unseen-gene holdout with the design that invalidated C3 (the
graph neural network is injected before the holdout is carved, the gene count of the holdout rows comes from their own
labels, the groups are gene-symbol strings). Until Package 1D replaces that design, `design_verdict` returns WITHHELD for
it, with the defects named, so a new run cannot print a new PASS from the same experiment.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from genomic_variant_classifier.prediction_contract import ContractError, canonical_digest

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEMA",
    "DIGEST_DOMAIN",
    "REGISTRY_RELATIVE_DIR",
    "GOVERNED_DOCUMENTS",
    "CORRECTED_HISTORICAL_DOCUMENTS",
    "KNOWN_INVALID_DESIGNS",
    "UGH_DESIGN_ID",
    "ClaimDisposition",
    "ClaimRegistryError",
    "InadmissibleClaimError",
    "ClaimRecord",
    "ClaimRegistry",
    "UnqualifiedCitation",
    "record_digest",
    "validate_record",
    "load_registry",
    "load_registry_dir",
    "find_unqualified_citations",
    "design_verdict",
]

SCHEMA = "gvc.scientific-claim-disposition.v1"
DIGEST_DOMAIN = "gvc.scientific-claim-disposition-record.v1"
#: Relative to the repository root.
REGISTRY_RELATIVE_DIR = Path("docs") / "claims"

#: Documents that state PRESENT claims. A figure of an inadmissible claim may appear in them only beside its disposition.
GOVERNED_DOCUMENTS: tuple[str, ...] = (
    "README.md",
    "docs/METRICS.md",
    "docs/ROADMAP.md",
    "docs/runs/RUN17_SCOPE.md",
)

#: Historical records whose stated facts were measured to be false. They are not rewritten; each must open (within its
#: first 40 lines) with a dated correction naming these claim identifiers.
CORRECTED_HISTORICAL_DOCUMENTS: Mapping[str, tuple[str, ...]] = {
    "docs/ABLATION_n_pathogenic_in_gene.md": (
        "run15.test.ensemble_evaluation",
        "run15.validation.ensemble_evaluation",
        "run15.c3.unseen_gene_generalization",
    ),
}
CORRECTION_BANNER = "CORRECTION 2026-11-08"
CORRECTION_BANNER_WITHIN_LINES = 40

#: The holdout design of scripts/run_phase2_eval.py as of this change. Its defects are the C3 record's.
UGH_DESIGN_ID = "run_phase2_eval.unseen_gene_holdout.v1"
KNOWN_INVALID_DESIGNS: Mapping[str, dict] = {
    UGH_DESIGN_ID: {
        "claim_it_cannot_support": "run15.c3.unseen_gene_generalization",
        "defects": (
            "GNN_HOLDOUT_LABEL_LEAKAGE",
            "GENE_TARGET_COUNT_OWN_LABEL_LEAKAGE",
            "GENE_IDENTITY_SPLIT_LEAKAGE",
        ),
        "repair": "Package 1D (stages 1D-1 to 1D-3): a leakage-free gene-disjoint evaluation",
    },
}


class ClaimDisposition(str, Enum):
    VALID = "VALID"
    INVALIDATED_EVALUATION = "INVALIDATED_EVALUATION"
    SUPERSEDED = "SUPERSEDED"


class ClaimRegistryError(ValueError):
    """The registry or one of its records is malformed, broken or ambiguous. Nothing is read from it."""


class InadmissibleClaimError(ValueError):
    """A claim was required as evidence while its current disposition does not admit it."""


_TOP_KEYS = frozenset({
    "schema", "record_sequence", "previous_record_sha256", "recorded_utc", "author", "ruling", "claim",
    "historical_observation", "disposition", "scientific_claim_eligible", "invalidating_defects", "inflation_estimate",
    "partial_replacements", "hypothesis_status", "required_replacement", "replacement_claim_id", "cited_figures",
    "record_sha256",
})
_RULING_KEYS = frozenset({"date", "decision_file_sha256", "sections"})
_CLAIM_KEYS = frozenset({"claim_id", "claim_type", "statement", "evaluation_population"})
_OBSERVATION_KEYS = frozenset({"run", "metrics", "verdict", "criterion", "code_provenance", "source_artifacts"})
_DEFECT_KEYS = frozenset({"defect_id", "statement", "evidence"})
_REPLACEMENT_KEYS = frozenset({"description", "metrics", "addresses_defects", "does_not_address_defects", "source"})
_ARTIFACT_KEYS = frozenset({"path", "sha256", "bytes"})
_HISTORICAL_VERDICTS = frozenset({"PASS", "FAIL", "UNKNOWN", "NOT_RECORDED"})
_HYPOTHESIS_STATUSES = frozenset({"NOT_RESOLVED_BY_THIS_EXPERIMENT", "SUPPORTED", "REFUTED"})

_SHA256 = re.compile(r"[0-9a-f]{64}")
_CLAIM_ID = re.compile(r"[a-z0-9]+(?:\.[a-z0-9_]+)+")
_DEFECT_ID = re.compile(r"[A-Z][A-Z0-9_]*")
_FIGURE = re.compile(r"\d+\.\d+")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_FILENAME = re.compile(r"CLAIM-(\d{4})_([a-z0-9._]+)\.json")


def _fail(where: str, why: str) -> ClaimRegistryError:
    return ClaimRegistryError(f"{where}: {why}")


def _text(value: Any, where: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise _fail(where, "a nonempty string without surrounding whitespace is required")
    return value


def _sha(value: Any, where: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _fail(where, "a full lowercase SHA-256 is required")
    return value


def _keys(obj: Any, expected: frozenset, where: str) -> dict:
    if type(obj) is not dict:
        raise _fail(where, f"an object is required, not {type(obj).__name__}")
    if set(obj) != expected:
        missing, extra = sorted(expected - set(obj)), sorted(set(obj) - expected)
        raise _fail(where, f"keys differ from the schema (missing {missing}, unexpected {extra})")
    return obj


def _text_list(value: Any, where: str, *, pattern: Optional[re.Pattern] = None, nonempty: bool = True) -> list:
    if type(value) is not list or (nonempty and not value):
        raise _fail(where, "a nonempty list is required" if nonempty else "a list is required")
    for i, item in enumerate(value):
        _text(item, f"{where}[{i}]")
        if pattern is not None and pattern.fullmatch(item) is None:
            raise _fail(f"{where}[{i}]", f"{item!r} does not match {pattern.pattern}")
    if len(set(value)) != len(value):
        raise _fail(where, "duplicate entries")
    return value


def _metrics(value: Any, where: str) -> dict:
    if type(value) is not dict or not value:
        raise _fail(where, "a nonempty object of metric -> number is required")
    for name, number in value.items():
        _text(name, f"{where} key")
        if type(number) not in (int, float):
            raise _fail(f"{where}.{name}", "a JSON number is required (bool and strings are refused)")
    return value


def record_digest(record: Mapping[str, Any]) -> str:
    """The canonical digest of a record WITHOUT its `record_sha256` field."""
    body = {k: v for k, v in record.items() if k != "record_sha256"}
    try:
        return canonical_digest(DIGEST_DOMAIN, body)
    except ContractError as exc:
        raise ClaimRegistryError(f"record is not canonical JSON: {exc}") from exc


@dataclass(frozen=True)
class ClaimRecord:
    sequence: int
    claim_id: str
    disposition: ClaimDisposition
    eligible: bool
    defect_ids: tuple[str, ...]
    cited_figures: tuple[str, ...]
    replacement_claim_id: Optional[str]
    record_sha256: str
    previous_record_sha256: Optional[str]
    path: Path
    raw: Mapping[str, Any]

    @property
    def admissible_as_evidence(self) -> bool:
        return self.disposition is ClaimDisposition.VALID


def validate_record(record: Any, where: str = "record") -> ClaimRecord:
    """Every field checked against the schema; the record's own digest re-derived. Returns the typed view."""
    _keys(record, _TOP_KEYS, where)
    if record["schema"] != SCHEMA:
        raise _fail(where, f"schema {record['schema']!r}, expected {SCHEMA!r}")
    seq = record["record_sequence"]
    if type(seq) is not int or seq < 1:
        raise _fail(f"{where}.record_sequence", "a positive integer is required")
    prev = record["previous_record_sha256"]
    if seq == 1:
        if prev is not None:
            raise _fail(f"{where}.previous_record_sha256", "the first record has no predecessor (null)")
    else:
        _sha(prev, f"{where}.previous_record_sha256")
    if type(record["recorded_utc"]) is not str or _UTC.fullmatch(record["recorded_utc"]) is None:
        raise _fail(f"{where}.recorded_utc", "YYYY-MM-DDTHH:MM:SSZ is required")
    _text(record["author"], f"{where}.author")

    ruling = _keys(record["ruling"], _RULING_KEYS, f"{where}.ruling")
    if type(ruling["date"]) is not str or _DATE.fullmatch(ruling["date"]) is None:
        raise _fail(f"{where}.ruling.date", "YYYY-MM-DD is required")
    _sha(ruling["decision_file_sha256"], f"{where}.ruling.decision_file_sha256")
    _text_list(ruling["sections"], f"{where}.ruling.sections")

    claim = _keys(record["claim"], _CLAIM_KEYS, f"{where}.claim")
    claim_id = _text(claim["claim_id"], f"{where}.claim.claim_id")
    if _CLAIM_ID.fullmatch(claim_id) is None:
        raise _fail(f"{where}.claim.claim_id", f"{claim_id!r} does not match {_CLAIM_ID.pattern}")
    for key in ("claim_type", "statement", "evaluation_population"):
        _text(claim[key], f"{where}.claim.{key}")

    obs = _keys(record["historical_observation"], _OBSERVATION_KEYS, f"{where}.historical_observation")
    _text(obs["run"], f"{where}.historical_observation.run")
    _metrics(obs["metrics"], f"{where}.historical_observation.metrics")
    if obs["verdict"] not in _HISTORICAL_VERDICTS:
        raise _fail(f"{where}.historical_observation.verdict", f"one of {sorted(_HISTORICAL_VERDICTS)} is required")
    if obs["criterion"] is not None:
        _text(obs["criterion"], f"{where}.historical_observation.criterion")
    if type(obs["code_provenance"]) is not dict or not obs["code_provenance"]:
        raise _fail(f"{where}.historical_observation.code_provenance", "a nonempty object is required")
    artifacts = obs["source_artifacts"]
    if type(artifacts) is not list or not artifacts:
        raise _fail(f"{where}.historical_observation.source_artifacts", "a nonempty list is required")
    for i, art in enumerate(artifacts):
        w = f"{where}.historical_observation.source_artifacts[{i}]"
        _keys(art, _ARTIFACT_KEYS, w)
        _text(art["path"], f"{w}.path")
        _sha(art["sha256"], f"{w}.sha256")
        if type(art["bytes"]) is not int or art["bytes"] < 0:
            raise _fail(f"{w}.bytes", "a nonnegative integer is required")

    try:
        disposition = ClaimDisposition(record["disposition"])
    except ValueError as exc:
        raise _fail(f"{where}.disposition", f"one of {[d.value for d in ClaimDisposition]} is required") from exc
    eligible = record["scientific_claim_eligible"]
    if type(eligible) is not bool or eligible != (disposition is ClaimDisposition.VALID):
        raise _fail(f"{where}.scientific_claim_eligible", "must be the boolean (disposition == VALID)")

    defects = record["invalidating_defects"]
    if type(defects) is not list:
        raise _fail(f"{where}.invalidating_defects", "a list is required")
    for i, d in enumerate(defects):
        w = f"{where}.invalidating_defects[{i}]"
        _keys(d, _DEFECT_KEYS, w)
        if type(d["defect_id"]) is not str or _DEFECT_ID.fullmatch(d["defect_id"]) is None:
            raise _fail(f"{w}.defect_id", "an upper-case identifier is required")
        _text(d["statement"], f"{w}.statement")
        if type(d["evidence"]) is not dict or not d["evidence"]:
            raise _fail(f"{w}.evidence", "a nonempty object of measured evidence is required")
    defect_ids = tuple(d["defect_id"] for d in defects)
    if len(set(defect_ids)) != len(defect_ids):
        raise _fail(f"{where}.invalidating_defects", "duplicate defect identifiers")
    if disposition is ClaimDisposition.INVALIDATED_EVALUATION:
        if not defects:
            raise _fail(f"{where}.invalidating_defects", "an invalidated evaluation names at least one defect")
        if record["inflation_estimate"] is not None:
            raise _fail(f"{where}.inflation_estimate", "an invalidated evaluation carries no inflation estimate (null)")
        if record["hypothesis_status"] != "NOT_RESOLVED_BY_THIS_EXPERIMENT":
            raise _fail(f"{where}.hypothesis_status", "an invalidated evaluation resolves no hypothesis")
        _text(record["required_replacement"], f"{where}.required_replacement")
    elif defects:
        raise _fail(f"{where}.invalidating_defects", f"a {disposition.value} claim lists no invalidating defect")
    if record["hypothesis_status"] not in _HYPOTHESIS_STATUSES:
        raise _fail(f"{where}.hypothesis_status", f"one of {sorted(_HYPOTHESIS_STATUSES)} is required")

    replacement = record["replacement_claim_id"]
    if disposition is ClaimDisposition.SUPERSEDED:
        if type(replacement) is not str or _CLAIM_ID.fullmatch(replacement) is None or replacement == claim_id:
            raise _fail(f"{where}.replacement_claim_id", "a superseded claim names its (different) replacement")
    elif replacement is not None:
        raise _fail(f"{where}.replacement_claim_id", "only a SUPERSEDED claim names a replacement")

    replacements = record["partial_replacements"]
    if type(replacements) is not list:
        raise _fail(f"{where}.partial_replacements", "a list is required")
    for i, rep in enumerate(replacements):
        w = f"{where}.partial_replacements[{i}]"
        _keys(rep, _REPLACEMENT_KEYS, w)
        _text(rep["description"], f"{w}.description")
        _metrics(rep["metrics"], f"{w}.metrics")
        addressed = _text_list(rep["addresses_defects"], f"{w}.addresses_defects", pattern=_DEFECT_ID)
        open_ = _text_list(rep["does_not_address_defects"], f"{w}.does_not_address_defects", pattern=_DEFECT_ID,
                           nonempty=False)
        _text(rep["source"], f"{w}.source")
        named = set(addressed) | set(open_)
        if set(addressed) & set(open_) or named != set(defect_ids):
            raise _fail(w, "addresses_defects and does_not_address_defects must partition the record's defects")

    figures = tuple(_text_list(record["cited_figures"], f"{where}.cited_figures", pattern=_FIGURE))
    digest = _sha(record["record_sha256"], f"{where}.record_sha256")
    derived = record_digest(record)
    if derived != digest:
        raise _fail(f"{where}.record_sha256", f"states {digest}, the record's canonical digest is {derived}")
    return ClaimRecord(sequence=seq, claim_id=claim_id, disposition=disposition, eligible=eligible,
                       defect_ids=defect_ids, cited_figures=figures, replacement_claim_id=replacement,
                       record_sha256=digest, previous_record_sha256=prev, path=Path(where), raw=record)


def _strict_json(text: str, where: str) -> Any:
    def no_duplicates(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise _fail(where, f"duplicate key {key!r}")
            out[key] = value
        return out

    def no_constant(name):
        raise _fail(where, f"non-finite literal {name} is not JSON")

    try:
        return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=no_constant)
    except json.JSONDecodeError as exc:
        raise _fail(where, f"not JSON: {exc}") from exc


class ClaimRegistry:
    """The verified, ordered chain of claim records. Constructed only by `load_registry`."""

    def __init__(self, records: tuple[ClaimRecord, ...]) -> None:
        self._records = records
        current: dict[str, ClaimRecord] = {}
        for rec in records:
            current[rec.claim_id] = rec
        self._current = current

    @property
    def records(self) -> tuple[ClaimRecord, ...]:
        return self._records

    @property
    def claim_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._current))

    def current(self, claim_id: str) -> ClaimRecord:
        try:
            return self._current[claim_id]
        except KeyError:
            raise ClaimRegistryError(f"no claim record for {claim_id!r}; registered: {list(self.claim_ids)}") from None

    def require_admissible(self, claim_id: str) -> ClaimRecord:
        rec = self.current(claim_id)
        if not rec.admissible_as_evidence:
            raise InadmissibleClaimError(
                f"{claim_id} is {rec.disposition.value} (record {rec.sequence:04d}, {rec.record_sha256[:12]}): it may "
                f"not be cited as scientific evidence. Defects: {', '.join(rec.defect_ids) or 'none'}"
                + (f"; replaced by {rec.replacement_claim_id}" if rec.replacement_claim_id else ""))
        return rec

    def inadmissible_figures(self) -> dict[str, tuple[ClaimRecord, ...]]:
        """figure string -> the CURRENT inadmissible claims citing it."""
        out: dict[str, list[ClaimRecord]] = {}
        for rec in self._current.values():
            if rec.admissible_as_evidence:
                continue
            for fig in rec.cited_figures:
                out.setdefault(fig, []).append(rec)
        return {k: tuple(sorted(v, key=lambda r: r.claim_id)) for k, v in out.items()}


def load_registry(root: Optional[Path] = None) -> ClaimRegistry:
    """Read and verify every record under `<repository root>/docs/claims` (the root resolved and verified by
    paths.runtime_paths). Any defect raises; nothing is returned partially."""
    from genomic_variant_classifier.paths.runtime_paths import resolve_project_root

    return load_registry_dir(resolve_project_root(explicit=root) / REGISTRY_RELATIVE_DIR)


def load_registry_dir(directory: Path) -> ClaimRegistry:
    """Read and verify every record in `directory`. Any defect raises; nothing is returned partially."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ClaimRegistryError(f"the claim registry {directory} does not exist")
    entries = sorted(p.name for p in directory.iterdir())
    unknown = [n for n in entries if n != "README.md" and _FILENAME.fullmatch(n) is None]
    if unknown:
        raise ClaimRegistryError(f"{directory}: files that are not claim records: {unknown}")
    names = [n for n in entries if _FILENAME.fullmatch(n)]
    if not names:
        raise ClaimRegistryError(f"{directory}: no claim record")
    records: list[ClaimRecord] = []
    previous: Optional[str] = None
    for expected_seq, name in enumerate(names, start=1):
        seq_text, claim_id = _FILENAME.fullmatch(name).groups()
        path = directory / name
        raw = _strict_json(path.read_text(encoding="utf-8"), name)
        rec = validate_record(raw, where=name)
        if rec.sequence != expected_seq or int(seq_text) != expected_seq:
            raise ClaimRegistryError(f"{name}: sequence {rec.sequence} where {expected_seq} is next (no gaps, no reuse)")
        if rec.claim_id != claim_id:
            raise ClaimRegistryError(f"{name}: the file name names {claim_id!r}, the record {rec.claim_id!r}")
        if rec.previous_record_sha256 != previous:
            raise ClaimRegistryError(f"{name}: previous_record_sha256 {rec.previous_record_sha256} breaks the chain "
                                     f"(the preceding record's digest is {previous})")
        previous = rec.record_sha256
        records.append(ClaimRecord(**{**rec.__dict__, "path": path}))
    logger.debug("Claim registry %s: %d record(s), head %s", directory, len(records), previous)
    return ClaimRegistry(tuple(records))


@dataclass(frozen=True)
class UnqualifiedCitation:
    line: int
    figure: str
    claims: tuple[str, ...]
    text: str


def _units(text: str) -> list[tuple[int, str]]:
    """(first line number, text) per citation unit: a table row, a line inside a fenced block, or a prose paragraph."""
    units: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    fenced = False

    def flush():
        nonlocal buf
        if buf:
            units.append((start, "\n".join(buf)))
        buf = []

    for no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            fenced = not fenced
            units.append((no, line))
            continue
        if fenced or stripped.startswith("|"):
            flush()
            units.append((no, line))
            continue
        if not stripped:
            flush()
            continue
        if not buf:
            start = no
        buf.append(line)
    flush()
    return units


def find_unqualified_citations(text: str, registry: ClaimRegistry) -> list[UnqualifiedCitation]:
    """Every unit citing a figure of an inadmissible claim without that claim's disposition AND identifier beside it."""
    figures = registry.inadmissible_figures()
    patterns = {fig: re.compile(r"(?<![\d.])" + re.escape(fig) + r"(?!\d)") for fig in figures}
    found: list[UnqualifiedCitation] = []
    for line, unit in _units(text):
        for fig, pattern in patterns.items():
            if not pattern.search(unit):
                continue
            claims = figures[fig]
            qualified = any(rec.disposition.value in unit and rec.claim_id in unit for rec in claims)
            if not qualified:
                found.append(UnqualifiedCitation(line=line, figure=fig, claims=tuple(r.claim_id for r in claims),
                                                 text=unit[:160]))
    return found


def design_verdict(design_id: str, observed: float, threshold: float) -> dict:
    """The verdict a falsifier may report. A known-invalid design is WITHHELD with its defects; otherwise PASS/FAIL."""
    if type(observed) is not float or type(threshold) is not float:
        raise ContractError("design_verdict: observed and threshold must be Python floats")
    known = KNOWN_INVALID_DESIGNS.get(design_id)
    if known is not None:
        return {"design_id": design_id, "design_status": "INVALID_DESIGN", "verdict": "WITHHELD",
                "observed": observed, "criterion": f">= {threshold}",
                "defects": list(known["defects"]), "claim_it_cannot_support": known["claim_it_cannot_support"],
                "repair": known["repair"]}
    return {"design_id": design_id, "design_status": "NOT_KNOWN_INVALID", "verdict":
            "PASS" if observed >= threshold else "FAIL", "observed": observed, "criterion": f">= {threshold}",
            "defects": [], "claim_it_cannot_support": None, "repair": None}
