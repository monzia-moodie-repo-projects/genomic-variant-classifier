"""Stage 1D-1a scientific information-source census: the reviewed ledger and its structural gate.

Adopted from the owner's integrated reference (decision file c34eb717, 2026-11-09; gvc_1d1a_integrated_reference.py,
sha256 4ca06c71...), with these refinements, each motivated in docs/census/README.md:

R1  Observed consumption is MEASURED, not optional. The reference's verify() accepted
    `required_external_surfaces=()` by default and its command line never passed any, so a census with no consumer
    inventory passed. Here every surface in consumers.SURFACES is measured from the code, each entry's `consumers` must
    EQUAL the set of surfaces where it is observed, and every discovered consumer site must be classified.
R2  Identity is bound to MEANING, not to unrelated bytes. The reference bound the census to the whole 3,825-line
    variant_ensemble.py, so editing model code invalidated it, while the producers that live in other files
    (enrich_gene_counts, the connectors) were not bound at all. Here the census binds (a) a digest of the ordered
    contract itself and (b) a comment-insensitive "semantic closure" digest of every cited producer: the function's
    code plus every module-level definition and import it transitively references.
R3  Evidence references are CHECKED. `code:<path>#<text>` must occur exactly once in that file; `record:<path>` must
    exist; `doi:` must be well formed. (A DOI's content is not verified offline, and the report says so.)
R4  Admission is MONOTONE over dependencies: an admit_strict entry may depend only on admit_strict entries, and an
    entry whose dependencies are measured (the GNN scores' node attributes) must declare every measured dependency.
R5  Non-tabular information sources (graph relations, edge attributes, supervision labels) are census entries with the
    same disposition vocabulary, so a relation cannot re-introduce what the tabular census excluded.
R7  (Stage 1D-1b.2, ADMISSION-CYCLIC-LINEAGE-1.) The dependency graph is acyclic -- every cyclic component is named --
    and every leaf names a dataset in the ledger's own registry (lineage.py).

The gate verifies structure, coverage, measured consumption, anchors and digests. It does NOT establish the scientific
truth of the reviewer's assertions, and its report says so in those words.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import logging
import math
import re
import textwrap
import tokenize
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

SCHEMA = "gvc.feature-census.1d1a/2"
CONTRACT_FILE = "src/genomic_variant_classifier/models/variant_ensemble.py"
LEDGER_PATH = "docs/census/FEATURE_CENSUS_1D1a.json"
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
DOI = re.compile(r"10\.\d{4,9}/\S+\Z")
SCIENTIFIC_TRUTH_LINE = "Scientific truth of all source claims: NOT ESTABLISHED BY STRUCTURAL VALIDATION ALONE"


class Disposition(str, Enum):
    ADMIT_STRICT = "admit_strict"
    EXCLUDE_STRICT = "exclude_strict"
    RETIRE_PREDICTIVE = "retire_predictive"
    CONDITIONAL_ONLY = "conditional_only"
    WITHHOLD_PENDING_EVIDENCE = "withhold_pending_evidence"
    PENDING_REVIEW = "pending_review"


class SupervisionStatus(str, Enum):
    NONE = "none"
    PROXY_NONCLINICAL = "proxy_nonclinical"
    CLINICAL = "clinical"
    MIXED = "mixed"
    UNKNOWN = "unknown"          # reviewed and still unknown; not unreviewed
    UNREVIEWED = "unreviewed"


class UpstreamOverlap(str, Enum):
    VERIFIED_DISJOINT = "verified_disjoint"
    VERIFIED_OVERLAP = "verified_overlap"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    UNREVIEWED = "unreviewed"


class TimeStatus(str, Enum):
    VERIFIED_AS_OF = "verified_as_of"
    CURRENT_ONLY = "current_only"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    UNREVIEWED = "unreviewed"


class SourceState(str, Enum):
    VERIFIED = "verified"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class RelationOrigin(str, Enum):
    INDEPENDENT_BIOLOGY = "independent_biology"
    CLINICAL_DISEASE = "clinical_disease"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class LabelCriterionUse(str, Enum):
    """Whether the evidence behind a feature is evidence that curators use to ASSIGN the label (ACMG/AMP criteria;
    Richards et al. 2015). Such a feature can reproduce the classification process rather than independent biology
    (annotation circularity), whatever its supervision. Refinement R6."""
    POPULATION_DATA = "acmg_population_data"                # BA1, BS1, BS2, PM2
    COMPUTATIONAL_EVIDENCE = "acmg_computational_evidence"  # PP3, BP4, BP7
    VARIANT_TYPE = "acmg_variant_type"                      # PVS1, PM4, BP7 (consequence class)
    GENE_DISEASE_CURATION = "acmg_gene_disease_curation"    # gene-disease validity underlying every criterion
    CLINICAL_CLASSIFICATION = "clinical_classification"      # another resource's or the label's own classification
    NONE_KNOWN = "none_known"
    UNKNOWN = "unknown"


class SourceKind(str, Enum):
    GRAPH_RELATION = "graph_relation"
    EDGE_ATTRIBUTE = "edge_attribute"
    SUPERVISION_LABEL = "supervision_label"


#: The owner's ruling overlay (decision files 25691d66 and c34eb717). It is NOT a designation of the other features.
PINNED: Mapping[str, Disposition] = {
    "n_pathogenic_in_gene": Disposition.RETIRE_PREDICTIVE,
    "gene_has_known_disease": Disposition.RETIRE_PREDICTIVE,
    "lovd_variant_class": Disposition.RETIRE_PREDICTIVE,
    "cosmic_sig_tier": Disposition.RETIRE_PREDICTIVE,
    "cosmic_recurrence": Disposition.EXCLUDE_STRICT,
    "kegg_pathway_count": Disposition.EXCLUDE_STRICT,
    "kegg_disease_pathway_flag": Disposition.EXCLUDE_STRICT,
    "rnaseq_log2fc": Disposition.EXCLUDE_STRICT,
    "rnaseq_de_neglog10p": Disposition.EXCLUDE_STRICT,
    "n_known_pathogenic_protein_variants": Disposition.EXCLUDE_STRICT,
    "has_uniprot_annotation": Disposition.EXCLUDE_STRICT,
    "omim_n_diseases": Disposition.EXCLUDE_STRICT,
    "omim_n_diseases_molecular": Disposition.EXCLUDE_STRICT,
    "omim_is_autosomal_dominant": Disposition.EXCLUDE_STRICT,
    "clingen_validity_score": Disposition.EXCLUDE_STRICT,
    "eve_score": Disposition.CONDITIONAL_ONLY,
    "reactome_pathway_count": Disposition.WITHHOLD_PENDING_EVIDENCE,
    "gnn_score": Disposition.CONDITIONAL_ONLY,
    "hetero_gnn_score": Disposition.CONDITIONAL_ONLY,
}

ENTRY_FIELDS = frozenset({
    "name", "disposition", "producer", "meaning", "source_ids", "source_state", "consumers", "evidence_refs",
    "reason", "upstream_supervision", "threshold_supervision", "upstream_overlap", "source_time_status", "depends_on",
    "label_criterion_use",
})
SOURCE_FIELDS = frozenset({
    "name", "kind", "origin", "disposition", "producer", "meaning", "source_ids", "consumers", "evidence_refs",
    "reason", "upstream_supervision", "upstream_overlap",
})
TOP_FIELDS = frozenset({
    "schema", "authority", "expected_feature_count", "entries", "information_sources", "datasets",
    "producer_digests", "surfaces", "consumer_sites", "measured_dependencies", "registered_defects",
    "graph_inventory",
})


# --------------------------------------------------------------------------------------------------------------------
# Canonical bytes and strict loading (the reference's rules, unchanged)
# --------------------------------------------------------------------------------------------------------------------
def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def strict_loads(text: str) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=reject)


def strict_load(path: Path) -> Any:
    return strict_loads(path.read_text(encoding="utf-8"))


def ledger_bytes(ledger: Mapping[str, Any]) -> bytes:
    """The committed form: sorted keys, two-space indent, ASCII, LF, one trailing newline."""
    return (json.dumps(ledger, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


# --------------------------------------------------------------------------------------------------------------------
# The declared contract
# --------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Contract:
    names: tuple[str, ...]
    contract_digest: str      # sha256 of the canonical ordered names and count -- the census binds this
    file_sha256: str          # whole-file bytes at capture -- recorded, informational only (refinement R2)


def source_contract(repo: Path) -> Contract:
    """Read the literal ordered contract without importing model code (the reference's rules, unchanged)."""
    path = repo / CONTRACT_FILE
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    found: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in {"TABULAR_FEATURES", "EXPECTED_TABULAR_FEATURE_COUNT"}:
                if target.id in found:
                    raise ValueError(f"duplicate authoritative assignment: {target.id}")
                found[target.id] = ast.literal_eval(value)
    if set(found) != {"TABULAR_FEATURES", "EXPECTED_TABULAR_FEATURE_COUNT"}:
        raise ValueError("source feature declaration or count is missing")
    names, expected = found["TABULAR_FEATURES"], found["EXPECTED_TABULAR_FEATURE_COUNT"]
    if not isinstance(names, list) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("TABULAR_FEATURES must be a nonempty literal list of names")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected != len(names):
        raise ValueError("declared feature count does not reconcile")
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names")
    digest = sha256_bytes(canonical_bytes({"names": list(names), "count": expected}))
    return Contract(tuple(names), digest, sha256_bytes(raw))


# --------------------------------------------------------------------------------------------------------------------
# Producer semantic-closure digests (refinement R2)
# --------------------------------------------------------------------------------------------------------------------
def _strip_comments(segment: str) -> str:
    """Remove comment tokens, trailing blanks and blank lines; dedent. Stable across Python 3.11 and 3.12 (f-string
    tokenisation changed in 3.12, but COMMENT tokens did not)."""
    text = textwrap.dedent(segment)
    lines = text.splitlines(keepends=True)
    cuts: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            cuts[row] = min(cuts.get(row, col), col)
    kept = []
    for i, line in enumerate(lines, start=1):
        line = line[: cuts[i]] if i in cuts else line
        line = line.rstrip()
        if line.strip():
            kept.append(line)
    return "\n".join(kept)


def _module_table(tree: ast.Module) -> dict[str, ast.AST]:
    """Top-level bindings, including those made inside top-level try/if blocks."""
    table: dict[str, ast.AST] = {}

    def visit(body: Sequence[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                table.setdefault(node.name, node)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            table.setdefault(n.id, node)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                table.setdefault(node.target.id, node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    table.setdefault((alias.asname or alias.name).split(".")[0], node)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for h in node.handlers:
                    visit(h.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.If):
                visit(node.body)
                visit(node.orelse)
    visit(tree.body)
    return table


def _resolve(tree: ast.Module, qualname: str) -> ast.AST:
    scope: Sequence[ast.stmt] = tree.body
    node: ast.AST | None = None
    for part in qualname.split("."):
        node = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and n.name == part), None)
        if node is None:
            raise ValueError(f"symbol {qualname!r}: {part!r} is not defined at that scope")
        scope = node.body
    assert node is not None
    return node


def parse_symbol(symbol: str) -> tuple[str, str]:
    if symbol.count("::") != 1:
        raise ValueError(f"code symbol must be '<path>::<qualname>': {symbol!r}")
    rel, qual = symbol.split("::")
    if not rel.endswith(".py") or not qual or rel.startswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"malformed code symbol: {symbol!r}")
    return rel, qual


def symbol_closure_digest(repo: Path, symbol: str) -> str:
    """Digest of a function or class plus every module-level binding it transitively references (comment-insensitive)."""
    rel, qual = parse_symbol(symbol)
    path = repo / rel
    if not path.is_file():
        raise ValueError(f"symbol {symbol!r}: file is missing")
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=rel)
    table = _module_table(tree)
    root = _resolve(tree, qual)
    segments: list[list[str]] = []
    seen: set[int] = set()

    def segment_of(node: ast.AST) -> str:
        parts = [ast.get_source_segment(text, d) or "" for d in getattr(node, "decorator_list", [])]
        parts.append(ast.get_source_segment(text, node) or "")
        return "\n".join(_strip_comments(p) for p in parts)

    def include(label: str, node: ast.AST) -> None:
        if id(node) in seen:
            return
        seen.add(id(node))
        segments.append([label, segment_of(node)])
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return
        names = sorted({n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)})
        for d in getattr(node, "decorator_list", []):
            names.extend(n.id for n in ast.walk(d) if isinstance(n, ast.Name))
        for name in sorted(set(names)):
            if name in table:
                include(name, table[name])

    include(qual, root)
    return sha256_bytes(canonical_bytes({"symbol": symbol, "segments": segments}))


# --------------------------------------------------------------------------------------------------------------------
# Evidence references (refinement R3)
# --------------------------------------------------------------------------------------------------------------------
def check_evidence_ref(repo: Path, ref: str) -> str:
    """Return the kind of a checked reference; refuse anything unverifiable in form."""
    if ref.startswith("code:"):
        body = ref[len("code:"):]
        if "#" not in body:
            raise ValueError(f"code evidence needs '<path>#<exact text>': {ref!r}")
        rel, anchor = body.split("#", 1)
        if not anchor.strip():
            raise ValueError(f"empty anchor: {ref!r}")
        path = repo / rel
        if not path.is_file():
            raise ValueError(f"evidence file is missing: {ref!r}")
        count = path.read_text(encoding="utf-8").count(anchor)
        if count != 1:
            raise ValueError(f"evidence anchor occurs {count} times (must be exactly once): {ref!r}")
        return "code"
    if ref.startswith("record:"):
        rel = ref[len("record:"):]
        if not (repo / rel).is_file():
            raise ValueError(f"evidence record is missing: {ref!r}")
        return "record"
    if ref.startswith("doi:"):
        if not DOI.match(ref[len("doi:"):]):
            raise ValueError(f"malformed DOI: {ref!r}")
        return "doi"
    if ref.startswith("url:https://"):
        return "url"
    raise ValueError(f"unknown evidence reference kind: {ref!r}")


# --------------------------------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------------------------------
def _nonempty_str(record: Mapping[str, Any], field: str, who: str) -> None:
    if not isinstance(record.get(field), str) or not record[field].strip():
        raise ValueError(f"{who}: missing {field}")


def _nonempty_unique_list(record: Mapping[str, Any], field: str, who: str, *, allow_empty: bool = False) -> list[str]:
    value = record.get(field)
    if not isinstance(value, list) or (not value and not allow_empty) or any(
            not isinstance(s, str) or not s.strip() for s in value):
        raise ValueError(f"{who}: missing {field}")
    if len(set(value)) != len(value):
        raise ValueError(f"{who}: duplicate {field}")
    return value


def _enum_value(record: Mapping[str, Any], field: str, enum: type[Enum], who: str, *, unreviewed: str | None,
                what: str) -> str:
    value = record.get(field)
    if value not in {e.value for e in enum}:
        raise ValueError(f"{who}: invalid {what}")
    if unreviewed is not None and value == unreviewed:
        raise ValueError(f"{who}: {what} not assessed")
    return value


@dataclass(frozen=True)
class CensusReport:
    strictly_admitted: tuple[str, ...]
    disposition_counts: dict[str, int]
    information_source_counts: dict[str, int]
    surfaces: dict[str, int]
    evidence_kinds: dict[str, int]
    semantic_digest: str

    def lines(self) -> list[str]:
        return [
            "Census structure: VERIFIED",
            "Feature coverage: VERIFIED (every declared feature exactly once, in contract order)",
            "Observed consumption coverage: VERIFIED (every surface measured from code; every consumer site classified)",
            "Producer identity: VERIFIED (semantic-closure digests of every cited producer)",
            f"Source references present: VERIFIED ({self.evidence_kinds.get('code', 0)} code anchors found exactly "
            f"once, {self.evidence_kinds.get('record', 0)} records present; {self.evidence_kinds.get('doi', 0)} DOIs "
            f"and {self.evidence_kinds.get('url', 0)} URLs well formed but NOT fetched)",
            SCIENTIFIC_TRUTH_LINE,
            f"Strictly admitted features: {len(self.strictly_admitted)} (a measured output, not a target)",
            f"Dispositions: {dict(sorted(self.disposition_counts.items()))}",
            f"Information-source dispositions: {dict(sorted(self.information_source_counts.items()))}",
            f"Semantic contract digest: {self.semantic_digest}",
        ]


def verify_census(repo: Path, ledger: Mapping[str, Any], *, engineered: Sequence[str] | None = None,
                  discovered: Mapping[str, Sequence[str]] | None = None) -> CensusReport:
    """Structural census gate. Refuses on the first defect; returns what was verified.

    `engineered` and `discovered` let a caller that has already measured the builder's output and the consumer sites
    of THIS repository reuse them (the test suite measures once per module); by default both are measured here.
    """
    from genomic_variant_classifier.information_sources import consumers as _consumers

    if not isinstance(ledger, Mapping) or set(ledger) != TOP_FIELDS or ledger.get("schema") != SCHEMA:
        raise ValueError("unexpected ledger schema/fields")
    contract = source_contract(repo)
    authority = ledger["authority"]
    if not isinstance(authority, Mapping) or set(authority) != {"path", "contract_digest", "file_sha256_at_capture"}:
        raise ValueError("authority must hold path, contract_digest and file_sha256_at_capture")
    if authority["path"] != CONTRACT_FILE or authority["contract_digest"] != contract.contract_digest:
        raise ValueError("feature authority changed after census capture (names, order or count)")
    if not isinstance(authority["file_sha256_at_capture"], str) or not HEX64.match(authority["file_sha256_at_capture"]):
        raise ValueError("file_sha256_at_capture must be a SHA-256")
    if ledger["expected_feature_count"] != len(contract.names):
        raise ValueError("ledger count disagrees with actual source")

    entries = ledger["entries"]
    if not isinstance(entries, list) or len(entries) != len(contract.names):
        raise ValueError("feature census is incomplete")
    if tuple(e.get("name") if isinstance(e, Mapping) else None for e in entries) != contract.names:
        raise ValueError("census names or order do not match authoritative contract")

    sources = ledger["information_sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("information_sources must be a non-empty list")
    source_names = [s.get("name") if isinstance(s, Mapping) else None for s in sources]
    if len(set(source_names)) != len(source_names) or set(source_names) & set(contract.names):
        raise ValueError("information-source names must be unique and distinct from feature names")
    universe = set(contract.names) | set(source_names)
    graph = _consumers.observe_graph_sources(repo)
    if ledger["graph_inventory"] != graph:
        raise ValueError("recorded graph inventory differs from the measured one; re-measure and re-review")
    required = _consumers.required_information_sources(graph)
    if sorted(required) != sorted(source_names):
        raise ValueError(f"information sources must match the measured graph inventory exactly: missing "
                         f"{sorted(set(required) - set(source_names))}, extra {sorted(set(source_names) - set(required))}")

    datasets = ledger["datasets"]
    if not isinstance(datasets, Mapping) or not datasets:
        raise ValueError("datasets must be a non-empty object")
    evidence_kinds: dict[str, int] = {}

    def check_refs(refs: Sequence[str], who: str) -> None:
        for ref in refs:
            kind = check_evidence_ref(repo, ref)
            evidence_kinds[kind] = evidence_kinds.get(kind, 0) + 1
        if not any(r.startswith(("code:", "record:")) for r in refs):
            raise ValueError(f"{who}: needs at least one repository-checkable evidence reference (code: or record:)")

    for ds_id, ds in datasets.items():
        if not isinstance(ds, Mapping) or set(ds) != {"description", "access", "evidence_refs"}:
            raise ValueError(f"dataset {ds_id}: needs description, access and evidence_refs")
        _nonempty_str(ds, "description", f"dataset {ds_id}")
        _nonempty_str(ds, "access", f"dataset {ds_id}")
        check_refs(_nonempty_unique_list(ds, "evidence_refs", f"dataset {ds_id}"), f"dataset {ds_id}")

    # Measured consumption (R1).
    surfaces = _consumers.observe_surfaces(repo, contract.names, engineered=engineered)
    if ledger["surfaces"] != surfaces:
        raise ValueError("recorded surface inventory differs from the measured one; re-measure and re-review")
    sites = _consumers.discover_sites(repo) if discovered is None else discovered
    _consumers.check_site_classification(sites, ledger["consumer_sites"])
    observed_in: dict[str, list[str]] = {n: [] for n in contract.names}
    for surface, record in surfaces.items():
        unreviewed = sorted(set(record["observed"]) - universe)
        if unreviewed:
            raise ValueError(f"surface {surface} consumes unreviewed information: {unreviewed}")
        for name in record["observed"]:
            observed_in[name].append(surface)

    # Producer digests (R2).
    digests = ledger["producer_digests"]
    if not isinstance(digests, Mapping):
        raise ValueError("producer_digests must be an object")
    cited: set[str] = set()

    def code_symbols(record: Mapping[str, Any]) -> list[str]:
        syms = [record["producer"]] + [s[len("code:"):] for s in record["source_ids"] if s.startswith("code:")]
        for s in syms:
            parse_symbol(s)
        return syms

    def check_sources(record: Mapping[str, Any], who: str) -> None:
        for s in record["source_ids"]:
            if s.startswith("dataset:"):
                if s[len("dataset:"):] not in datasets:
                    raise ValueError(f"{who}: cites unknown dataset {s}")
            elif not s.startswith("code:"):
                raise ValueError(f"{who}: source ids are 'dataset:<id>' or 'code:<path>::<qualname>': {s!r}")

    dispositions: dict[str, Disposition] = {}
    approved: list[str] = []
    counts: dict[str, int] = {}
    for record in entries:
        name = record.get("name")
        if set(record) != ENTRY_FIELDS:
            raise ValueError(f"{name}: unexpected entry shape")
        disp = Disposition(record["disposition"])
        if name in PINNED and disp is not PINNED[name]:
            raise ValueError(f"{name}: conflicts with pinned ruling {PINNED[name].value}")
        if disp is Disposition.PENDING_REVIEW:
            raise ValueError(f"{name}: unreviewed disposition")
        for field in ("producer", "meaning", "reason"):
            _nonempty_str(record, field, name)
        _nonempty_unique_list(record, "source_ids", name)
        check_sources(record, name)
        consumers_ = _nonempty_unique_list(record, "consumers", name, allow_empty=True)
        if sorted(consumers_) != sorted(observed_in[name]):
            raise ValueError(f"{name}: recorded consumers {sorted(consumers_)} differ from measured "
                             f"{sorted(observed_in[name])}")
        check_refs(_nonempty_unique_list(record, "evidence_refs", name), name)
        _enum_value(record, "source_state", SourceState, name, unreviewed=None, what="source state")
        _enum_value(record, "upstream_supervision", SupervisionStatus, name,
                    unreviewed=SupervisionStatus.UNREVIEWED.value, what="upstream supervision")
        _enum_value(record, "threshold_supervision", SupervisionStatus, name,
                    unreviewed=SupervisionStatus.UNREVIEWED.value, what="threshold supervision")
        _enum_value(record, "upstream_overlap", UpstreamOverlap, name,
                    unreviewed=UpstreamOverlap.UNREVIEWED.value, what="upstream overlap")
        _enum_value(record, "source_time_status", TimeStatus, name,
                    unreviewed=TimeStatus.UNREVIEWED.value, what="source timing")
        _enum_value(record, "label_criterion_use", LabelCriterionUse, name, unreviewed=None,
                    what="label-criterion use")
        if disp is Disposition.ADMIT_STRICT and record["source_state"] != SourceState.VERIFIED.value:
            raise ValueError(f"{name}: admitted without a verified source")
        cited.update(code_symbols(record))
        dispositions[name] = disp
        counts[disp.value] = counts.get(disp.value, 0) + 1
        if disp is Disposition.ADMIT_STRICT:
            approved.append(name)

    source_counts: dict[str, int] = {}
    for record in sources:
        name = record.get("name")
        if set(record) != SOURCE_FIELDS:
            raise ValueError(f"{name}: unexpected information-source shape")
        SourceKind(record["kind"])
        origin = RelationOrigin(record["origin"])
        disp = Disposition(record["disposition"])
        if disp is Disposition.PENDING_REVIEW:
            raise ValueError(f"{name}: unreviewed disposition")
        if disp is Disposition.ADMIT_STRICT and origin is not RelationOrigin.INDEPENDENT_BIOLOGY:
            raise ValueError(f"{name}: an information source not of independent biological origin cannot be admitted")
        for field in ("producer", "meaning", "reason"):
            _nonempty_str(record, field, name)
        _nonempty_unique_list(record, "source_ids", name)
        check_sources(record, name)
        consumers_ = _nonempty_unique_list(record, "consumers", name)
        bad = [c for c in consumers_ if c not in _consumers.GRAPH_CONSUMERS]
        if bad:
            raise ValueError(f"{name}: unknown consumer surfaces {bad}")
        check_refs(_nonempty_unique_list(record, "evidence_refs", name), name)
        _enum_value(record, "upstream_supervision", SupervisionStatus, name,
                    unreviewed=SupervisionStatus.UNREVIEWED.value, what="upstream supervision")
        _enum_value(record, "upstream_overlap", UpstreamOverlap, name,
                    unreviewed=UpstreamOverlap.UNREVIEWED.value, what="upstream overlap")
        cited.update(code_symbols(record))
        dispositions[name] = disp
        source_counts[disp.value] = source_counts.get(disp.value, 0) + 1

    if set(digests) != cited:
        raise ValueError(f"producer_digests must cover exactly the cited code symbols: missing "
                         f"{sorted(cited - set(digests))}, extra {sorted(set(digests) - cited)}")
    for symbol in sorted(cited):
        actual = symbol_closure_digest(repo, symbol)
        if digests[symbol] != actual:
            users = sorted(r["name"] for r in [*entries, *sources] if symbol in code_symbols(r))
            raise ValueError(f"producer changed since census capture: {symbol} (re-review {users})")

    # Monotone admission over dependencies (R4).
    measured = ledger["measured_dependencies"]
    if not isinstance(measured, Mapping):
        raise ValueError("measured_dependencies must be an object")
    for record in entries:
        deps = _nonempty_unique_list(record, "depends_on", record["name"], allow_empty=True)
        unknown = sorted(set(deps) - universe)
        if unknown or record["name"] in deps:
            raise ValueError(f"{record['name']}: depends_on names unknown or itself: {unknown}")
        if dispositions[record["name"]] is Disposition.ADMIT_STRICT:
            weaker = sorted(d for d in deps if dispositions[d] is not Disposition.ADMIT_STRICT)
            if weaker:
                raise ValueError(f"{record['name']}: admitted while depending on non-admitted {weaker}")
    # Lineage integrity (stage 1D-1b.2, ADMISSION-CYCLIC-LINEAGE-1): no cycle, and every leaf grounded in a registered
    # dataset, so no admission can rest on a self-supporting dependency or on an unregistered source string.
    from genomic_variant_classifier.information_sources import lineage as _lineage

    lineage_deps = {r["name"]: tuple(r["depends_on"]) for r in entries}
    lineage_deps.update({r["name"]: () for r in sources})
    _lineage.require_acyclic(lineage_deps)
    ungrounded = _lineage.ungrounded_leaves({r["name"]: r["source_ids"] for r in [*entries, *sources]}, lineage_deps,
                                            datasets)
    if ungrounded:
        raise ValueError(f"lineage leaves name no registered dataset: {ungrounded}")
    for name, rule in measured.items():
        if name not in contract.names or not isinstance(rule, list) or not rule:
            raise ValueError(f"measured dependency rule for {name!r} is malformed")
        record = next(r for r in entries if r["name"] == name)
        need: set[str] = set()
        for item in rule:
            if item.startswith("surface:"):
                need |= set(surfaces[item[len("surface:"):]]["observed"])
            elif item in universe:
                need.add(item)
            else:
                raise ValueError(f"measured dependency of {name} cites unknown {item!r}")
        need.discard(name)
        missing = sorted(need - set(record["depends_on"]))
        if missing:
            raise ValueError(f"{name}: depends_on omits measured dependencies {missing}")

    defects = ledger["registered_defects"]
    if not isinstance(defects, list):
        raise ValueError("registered_defects must be a list")
    ids = [d.get("id") for d in defects if isinstance(d, Mapping)]
    if len(ids) != len(defects) or len(set(ids)) != len(ids):
        raise ValueError("registered defects need unique ids")
    for d in defects:
        if set(d) != {"id", "statement", "affects", "evidence_refs"}:
            raise ValueError(f"defect {d.get('id')}: needs id, statement, affects and evidence_refs")
        _nonempty_str(d, "statement", d["id"])
        affects = _nonempty_unique_list(d, "affects", d["id"])
        if sorted(set(affects) - universe):
            raise ValueError(f"defect {d['id']}: affects unknown names {sorted(set(affects) - universe)}")
        check_refs(_nonempty_unique_list(d, "evidence_refs", d["id"]), d["id"])

    return CensusReport(tuple(approved), counts, source_counts, {k: len(v["observed"]) for k, v in surfaces.items()},
                        evidence_kinds, semantic_contract_digest(ledger))


def semantic_contract_digest(ledger: Mapping[str, Any]) -> str:
    """Digest of the reviewed semantics: entries and information sources, without measured or byte-level fields."""
    keep = ("name", "disposition", "meaning", "reason", "producer", "source_ids", "upstream_supervision",
            "threshold_supervision", "upstream_overlap", "source_time_status", "depends_on", "kind", "origin",
            "label_criterion_use")
    projection = {
        "version": ledger["schema"],
        "ordered_entries": [{k: r[k] for k in keep if k in r} for r in ledger["entries"]],
        "information_sources": [{k: r[k] for k in keep if k in r} for r in ledger["information_sources"]],
    }
    return sha256_bytes(canonical_bytes(projection))


def materialization_digest(semantic_digest: str, *, source_digests: Mapping[str, str], row_population_digest: str,
                           parser_digests: Mapping[str, str]) -> str:
    """Future fit/inference identity (the reference's rule, unchanged). The caller must measure every digest from the
    bytes actually consumed; 1D-1a fabricates none of them."""
    for digest in (semantic_digest, row_population_digest, *source_digests.values(), *parser_digests.values()):
        if not isinstance(digest, str) or not HEX64.match(digest):
            raise ValueError("invalid SHA-256 digest")
    return sha256_bytes(canonical_bytes({
        "semantic_digest": semantic_digest, "source_digests": dict(sorted(source_digests.items())),
        "parser_digests": dict(sorted(parser_digests.items())), "row_population_digest": row_population_digest}))


# --------------------------------------------------------------------------------------------------------------------
# EVE coverage by source-join membership (the reference's design; EVE-COVERAGE-IDENTITY-1)
# --------------------------------------------------------------------------------------------------------------------
class Coverage(str, Enum):
    MATCHED_SCORE = "matched_score"
    GENE_MODELED_VARIANT_UNSCORED = "gene_modeled_variant_unscored"
    GENE_NOT_MODELED = "gene_not_modeled"


@dataclass(frozen=True)
class ProteinVariantKey:
    accession_with_isoform: str
    reference_aa: str
    position: int
    alternate_aa: str

    def __post_init__(self) -> None:
        if not self.accession_with_isoform or self.position <= 0:
            raise ValueError("invalid protein identity")
        if len(self.reference_aa) != 1 or len(self.alternate_aa) != 1:
            raise ValueError("invalid amino-acid substitution")
        if self.reference_aa == self.alternate_aa:
            raise ValueError("reference and alternate amino acids are equal")


@dataclass(frozen=True)
class EveLookup:
    score_by_variant: Mapping[ProteinVariantKey, float | None]
    modeled_proteins: frozenset[str]
    source_sha256: str


@dataclass(frozen=True)
class EveObservation:
    status: Coverage
    score: float | None
    protein_key: ProteinVariantKey

    @property
    def has_observed_score(self) -> bool:
        return self.status is Coverage.MATCHED_SCORE


def annotate_eve(keys: Sequence[ProteinVariantKey], lookup: EveLookup) -> tuple[EveObservation, ...]:
    """Coverage from SOURCE-JOIN MEMBERSHIP, never from `score != 0.5`; a matched 0.5 is an observed score.

    A matched tuple still requires reference-residue and isoform verification in the production adapter (stage 1D-1d).
    """
    if not isinstance(lookup.source_sha256, str) or not HEX64.match(lookup.source_sha256):
        raise ValueError("unverified EVE lookup digest")
    for key in lookup.score_by_variant:
        if key.accession_with_isoform not in lookup.modeled_proteins:
            raise ValueError("EVE score table contains an undeclared protein")
    out = []
    for key in keys:
        if key in lookup.score_by_variant:
            value = lookup.score_by_variant[key]
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("matched EVE record has invalid numeric score")
            status = Coverage.MATCHED_SCORE if value is not None else Coverage.GENE_MODELED_VARIANT_UNSCORED
            out.append(EveObservation(status, None if value is None else float(value), key))
        elif key.accession_with_isoform in lookup.modeled_proteins:
            out.append(EveObservation(Coverage.GENE_MODELED_VARIANT_UNSCORED, None, key))
        else:
            out.append(EveObservation(Coverage.GENE_NOT_MODELED, None, key))
    return tuple(out)


@dataclass(frozen=True)
class GraphRelationEvidence:
    name: str
    origin: RelationOrigin
    source_digest: str | None


def admit_strict_graph_relation(relation: GraphRelationEvidence) -> None:
    """Default-deny screening (the reference's rule): only independent biology with a measured source digest."""
    if relation.origin is not RelationOrigin.INDEPENDENT_BIOLOGY:
        raise ValueError(f"{relation.name}: graph relation not independently biological")
    if not isinstance(relation.source_digest, str) or not HEX64.match(relation.source_digest):
        raise ValueError(f"{relation.name}: source digest unavailable")


# --------------------------------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the stage 1D-1a information-source census.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=None, help=f"default: <repo>/{LEDGER_PATH}")
    args = parser.parse_args(argv)
    path = args.ledger or args.repo / LEDGER_PATH
    raw = path.read_bytes()
    ledger = strict_loads(raw.decode("ascii"))
    if ledger_bytes(ledger) != raw:
        raise ValueError("ledger is not in its canonical committed form")
    for line in verify_census(args.repo.resolve(), ledger).lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
