"""Stage 1D-1b.4: population identity, split isolation and fit authorization (SHADOW MODE: nothing in production imports
this module).

Owner ruling 68ca322f (2026-11-10: LABEL-TYPE-IDENTITY-1, VARIANT-MULTIGENE-ISOLATION-1; split groups are built from
components, which loci bind a group is the owner's stage 1D-1d decision) and the owner's 1D-1b.4 reference packages
cee6713d ("TaskBound"), 7ba10b3b ("TaskScoped") and 7b277d7c ("EvidenceScope"), audited and synthesized on 2026-11-13
(AUDIT_1D1b4_REFERENCES_2026-11-13.md, probes T1-T8 and E1-E7). Today the split groups rows by ONE gene symbol
(data/real_data_prep.py) and the 1D-1b.3 label authority by ONE canonical gene per row; a variant listed under several
genes, or a gene field that does not fully resolve, is invisible to both.

* ``PopulationRow``: one row of a population, strictly typed (native ``str`` identifiers, enum partitions). Variant
  identity is qualified by an enum, never a caller boolean: ``VRS_VERIFIED`` (the Variation Representation Specification)
  is set only by the stage 1D-1d identity authority, ``UNNORMALIZED`` is an exact source string. A row carries its gene
  SET and the tokens that did not resolve; one unresolved token makes the gene set unknown, not smaller.
* ``resolve_gene_field``: a cohort gene field to a gene set, by the canonical rule of ``graph_identity.HgncIndex``
  (approved symbol, else a unique previous symbol). A token holding whitespace is descriptive text, not a symbol
  (ClinVar writes "covers 26 genes, ..." for large copy-number variants; GENE-FIELD-PROSE-1).
* ``split_components``: rows joined through a shared variant identity or ANY listed HGNC locus (grouping (a)), with
  content-derived component identifiers.
* ``audit_split``: the partitions a declared ``SplitProtocol`` requires, audited pairwise (genes, variants, components):
  VIOLATED, UNKNOWN (gene-incomplete rows, or unnormalized variants under a VRS protocol), NOT_EVALUATED (an empty required
  partition), else SUPPORTED_FOR_SUPPLIED_IDENTITIES -- never "supported" outright, because the identities are supplied.
* ``authorize_fit``: two named authorities. STRICT refuses unless the split is supported. CONDITIONAL_COMPONENT
  (exploratory only) authorizes a gene-complete training row only if its component -- built over the rows that can carry
  labels into the fit (gene-complete training rows) plus every held-out row -- holds no held-out row. Measured on Run 15:
  a gene-direct rule admits 14,967 training rows joined to a held-out row through another row (probe T1).
* ``tempered_component_weights``: weights proportional to n_c^-alpha with mean 1, and the effective sample size.
* ``component_checkpoint_split``: the inner early-stopping split of authorized training rows, component-disjoint and
  class-stratified (GNN-EARLY-STOP-ROW-SPLIT-1).

Every verdict is conditional on the supplied identities; nothing here computes HGNC mappings or VRS identifiers.
In-process Python objects are not a security boundary. numpy is required; scikit-learn for the checkpoint split.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from genomic_variant_classifier.information_sources.census import HEX64, canonical_bytes, sha256_bytes
from genomic_variant_classifier.information_sources.graph_identity import HGNC_ID, HgncIndex, readonly_array

logger = logging.getLogger(__name__)

SCHEMA = "gvc.population.1d1b4/1"
#: The separators ClinVar uses between the genes of one gene field (measured on Run 15: ';' in 729 multi-gene fields).
GENE_FIELD_SEPARATORS = re.compile(r"[;,|/]")


def digest(domain: str, value: Any) -> str:
    """A project digest: SHA-256 of canonical JSON, separated by schema and domain. It is NOT a GA4GH VRS identifier."""
    return sha256_bytes(canonical_bytes({"schema": SCHEMA, "domain": domain, "value": value}))


def _nonempty_str(value: object, what: str) -> None:
    if type(value) is not str or not value.strip():
        raise TypeError(f"{what} must be a nonempty str, got {type(value).__name__}")


# =========================================================================================== identity and gene fields
class Partition(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    CALIBRATION = "calibration"
    TEST = "test"


class VariantIdentity(str, Enum):
    VRS_VERIFIED = "vrs_verified"        # set only by the stage 1D-1d identity authority
    UNNORMALIZED = "unnormalized"        # an exact source string: spellings of one allele stay distinct


@dataclass(frozen=True, slots=True)
class GeneField:
    """A gene field read as a SET: the HGNC ids that resolved, the tokens that did not, and whether a token is prose."""

    gene_ids: frozenset[str]
    unresolved_tokens: tuple[str, ...]
    prose: bool


def resolve_gene_field(value: str, hgnc: HgncIndex) -> GeneField:
    """Split on ``GENE_FIELD_SEPARATORS``, strip each token, resolve it by ``HgncIndex.resolve`` (approved symbol, else a
    unique previous symbol; aliases are not identity). An empty field resolves to the empty set (gene-incomplete)."""
    if type(value) is not str:
        raise TypeError(f"a gene field is a str, got {type(value).__name__}")
    if not isinstance(hgnc, HgncIndex):
        raise TypeError("hgnc must be a graph_identity.HgncIndex")
    ids: set[str] = set()
    unresolved: set[str] = set()
    for token in (t.strip() for t in GENE_FIELD_SEPARATORS.split(value)):
        if not token:
            continue
        hid, _ = hgnc.resolve(token)
        if hid is None:
            unresolved.add(token)
        else:
            ids.add(hid)
    prose = any(any(ch.isspace() for ch in t) for t in unresolved)
    return GeneField(frozenset(ids), tuple(sorted(unresolved)), prose)


@dataclass(frozen=True, slots=True)
class PopulationRow:
    row_id: str
    variant_id: str
    variant_identity: VariantIdentity
    gene_ids: frozenset[str]
    unresolved_gene_tokens: tuple[str, ...]
    partition: Partition

    def __post_init__(self) -> None:
        _nonempty_str(self.row_id, "row_id")
        _nonempty_str(self.variant_id, "variant_id")
        if not isinstance(self.variant_identity, VariantIdentity):
            raise TypeError("variant_identity must be a VariantIdentity")
        if type(self.gene_ids) is not frozenset or any(type(g) is not str or not HGNC_ID.match(g) for g in self.gene_ids):
            raise ValueError("gene_ids must be a frozenset of canonical HGNC identifiers")
        if type(self.unresolved_gene_tokens) is not tuple or any(
                type(t) is not str or not t.strip() for t in self.unresolved_gene_tokens):
            raise TypeError("unresolved_gene_tokens must be a tuple of nonempty str")
        if not isinstance(self.partition, Partition):
            raise TypeError("partition must be a Partition")

    @property
    def gene_complete(self) -> bool:
        """Every listed locus resolved: one unresolved token makes the gene set unknown, not smaller."""
        return bool(self.gene_ids) and not self.unresolved_gene_tokens


def _unique(rows: Iterable[PopulationRow]) -> tuple[PopulationRow, ...]:
    out = tuple(rows)
    if any(not isinstance(r, PopulationRow) for r in out):
        raise TypeError("rows must be PopulationRow")
    if len({r.row_id for r in out}) != len(out):
        raise ValueError("duplicate row_id")
    return out


# ================================================================================================ components and audit
def split_components(rows: Iterable[PopulationRow]) -> Mapping[str, str]:
    """{row_id: component id}. Indivisible split units: rows joined through a shared variant identity or ANY shared HGNC
    locus (grouping (a)). A component id is the digest of its sorted member identities, so it does not depend on order."""
    rows = _unique(rows)
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x: tuple[str, str]) -> tuple[str, str]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in rows:
        v = find(("variant", r.variant_id))
        for g in r.gene_ids:
            root = find(("gene", g))
            if root != v:
                parent[root] = v
    members: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for node in list(parent):
        members[find(node)].append(node)
    ident = {root: digest("component", sorted(m)) for root, m in members.items()}
    return MappingProxyType({r.row_id: ident[find(("variant", r.variant_id))] for r in rows})


class SplitVerdict(str, Enum):
    SUPPORTED_FOR_SUPPLIED_IDENTITIES = "supported_for_supplied_identities"
    VIOLATED = "violated"
    UNKNOWN = "unknown"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True)
class SplitProtocol:
    """The partitions a split must have, and whether variant identity must be VRS-verified."""

    required: tuple[Partition, ...] = (Partition.TRAIN, Partition.VALIDATION, Partition.TEST)
    require_vrs: bool = True

    def __post_init__(self) -> None:
        if type(self.required) is not tuple or any(not isinstance(p, Partition) for p in self.required):
            raise TypeError("required is a tuple of Partition")
        if Partition.TRAIN not in self.required or len(set(self.required)) != len(self.required):
            raise ValueError("a protocol requires TRAIN and lists each partition once")
        if type(self.require_vrs) is not bool:
            raise TypeError("require_vrs is a bool")

    def as_dict(self) -> dict[str, Any]:
        return {"required": [p.value for p in self.required], "require_vrs": self.require_vrs}


def population_digest(row_ids: Iterable[str]) -> str:
    return digest("population", sorted(row_ids))


def audit_split(rows: Iterable[PopulationRow], protocol: SplitProtocol = SplitProtocol()) -> dict[str, Any]:
    """The pairwise isolation audit of every present partition, judged against ``protocol``."""
    rows = _unique(rows)
    if not isinstance(protocol, SplitProtocol):
        raise TypeError("protocol must be a SplitProtocol")
    by = {p: [r for r in rows if r.partition is p] for p in Partition}
    sizes = {p.value: len(by[p]) for p in Partition}
    base = {"protocol": protocol.as_dict(), "sizes": sizes, "population_digest": population_digest(r.row_id for r in rows),
            "partition_digest": digest("partition", sorted([r.row_id, r.partition.value] for r in rows))}
    if any(not by[p] for p in protocol.required):
        return {"verdict": SplitVerdict.NOT_EVALUATED.value, **base, "reason": "a required partition is empty"}
    comp = split_components(rows)
    present = [p for p in Partition if by[p]]
    pairs = {}
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            ga = set().union(*(r.gene_ids for r in by[a]))
            gb = set().union(*(r.gene_ids for r in by[b]))
            pairs[f"{a.value}|{b.value}"] = {
                "shared_genes": len(ga & gb),
                "shared_variants": len({r.variant_id for r in by[a]} & {r.variant_id for r in by[b]}),
                "shared_components": len({comp[r.row_id] for r in by[a]} & {comp[r.row_id] for r in by[b]})}
    gene_incomplete = {p.value: sum(not r.gene_complete for r in by[p]) for p in present}
    unnormalized = {p.value: sum(r.variant_identity is VariantIdentity.UNNORMALIZED for r in by[p]) for p in present}
    partitions_of: dict[str, set[Partition]] = defaultdict(set)
    for r in rows:
        partitions_of[comp[r.row_id]].add(r.partition)
    if any(any(v.values()) for v in pairs.values()):
        verdict = SplitVerdict.VIOLATED
    elif any(gene_incomplete.values()) or (protocol.require_vrs and any(unnormalized.values())):
        verdict = SplitVerdict.UNKNOWN
    else:
        verdict = SplitVerdict.SUPPORTED_FOR_SUPPLIED_IDENTITIES
    return {"verdict": verdict.value, **base, "pairs": pairs, "gene_incomplete_rows": gene_incomplete,
            "unnormalized_variant_rows": unnormalized, "components": len(partitions_of),
            "components_spanning_partitions": sum(len(s) > 1 for s in partitions_of.values())}


# =================================================================================================== fit authorization
class FitAuthority(str, Enum):
    STRICT = "strict"                                  # the whole split must be supported
    CONDITIONAL_COMPONENT = "conditional_component"    # exploratory: clean components only


@dataclass(frozen=True)
class FitAuthorization:
    """Which training rows ONE fit may take labels from. Its digest is computed here from every field it binds, so an
    authorization cannot carry a digest of other content."""

    authority: FitAuthority
    protocol: SplitProtocol
    partition_digest: str
    authorized_rows: frozenset[str]
    registered_rows: frozenset[str]
    report: Mapping[str, Any]
    population_digest: str = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.authority, FitAuthority) or not isinstance(self.protocol, SplitProtocol):
            raise TypeError("authority and protocol must be a FitAuthority and a SplitProtocol")
        if type(self.partition_digest) is not str or not HEX64.match(self.partition_digest):
            raise ValueError("partition_digest is a lower-case SHA-256")
        for name in ("authorized_rows", "registered_rows"):
            v = getattr(self, name)
            if type(v) is not frozenset or any(type(x) is not str or not x for x in v):
                raise TypeError(f"{name} is a frozenset of nonempty str")
        if not self.authorized_rows:
            raise PermissionError("empty authorization")
        if not self.authorized_rows <= self.registered_rows:
            raise ValueError("an authorized row is not a registered row")
        object.__setattr__(self, "report", MappingProxyType(dict(self.report)))
        object.__setattr__(self, "population_digest", population_digest(self.registered_rows))
        object.__setattr__(self, "digest", digest("fit-authorization", {
            "authority": self.authority.value, "protocol": self.protocol.as_dict(),
            "population": self.population_digest, "partition": self.partition_digest,
            "authorized": sorted(self.authorized_rows)}))


def authorize_fit(rows: Iterable[PopulationRow], authority: FitAuthority,
                  protocol: SplitProtocol = SplitProtocol()) -> FitAuthorization:
    """Derive -- never accept -- the rows a fit may take labels from."""
    rows = _unique(rows)
    if not isinstance(authority, FitAuthority):
        raise TypeError("authority must be a FitAuthority")
    audit = audit_split(rows, protocol)
    train = [r for r in rows if r.partition is Partition.TRAIN]
    if authority is FitAuthority.STRICT:
        if audit["verdict"] != SplitVerdict.SUPPORTED_FOR_SUPPLIED_IDENTITIES.value:
            raise PermissionError(f"strict fit refused: split verdict is {audit['verdict']}")
        allowed = frozenset(r.row_id for r in train)
        report: dict[str, Any] = {"split_verdict": audit["verdict"], "claims_permitted": "strict",
                                  "training_rows": len(train), "authorized_rows": len(allowed)}
    else:
        if audit["verdict"] == SplitVerdict.NOT_EVALUATED.value:
            raise PermissionError("conditional fit refused: the split protocol is not evaluable")
        # Leakage travels through LABELLED training rows. A gene-incomplete training row is never authorized, so it carries
        # no label across components; components are therefore built over the rows that can carry labels into the fit
        # (gene-complete training rows) and every held-out row (whose links all count, known or not).
        usable = [r for r in rows if r.partition is not Partition.TRAIN or r.gene_complete]
        comp = split_components(usable)
        dirty = {comp[r.row_id] for r in usable if r.partition is not Partition.TRAIN}
        allowed = frozenset(r.row_id for r in train if r.gene_complete and comp[r.row_id] not in dirty)
        report = {"split_verdict": audit["verdict"], "claims_permitted": "exploratory only",
                  "training_rows": len(train), "authorized_rows": len(allowed),
                  "excluded_gene_incomplete": sum(not r.gene_complete for r in train),
                  "excluded_component_shares_heldout": sum(r.gene_complete and comp[r.row_id] in dirty for r in train),
                  "heldout_gene_incomplete_rows": sum(not r.gene_complete for r in rows if r.partition is not Partition.TRAIN),
                  "unnormalized_authorized_rows": sum(r.variant_identity is VariantIdentity.UNNORMALIZED
                                                      for r in train if r.row_id in allowed),
                  "limitation": "conditional on supplied identities; unnormalized variant strings may hide duplicates"}
    if not allowed:
        raise PermissionError("empty authorization")
    out = FitAuthorization(authority, protocol, audit["partition_digest"], allowed, frozenset(r.row_id for r in rows), report)
    if out.population_digest != audit["population_digest"]:
        raise AssertionError("the authorization's population is not the audited population")
    return out


# ============================================================================================ weights and checkpoints
def tempered_component_weights(component_ids: Sequence[str], alpha: float) -> dict[str, Any]:
    """w_i proportional to n_c(i)^-alpha, scaled to mean 1: alpha = 0 is row-balanced, alpha = 1 component-balanced.
    Reports the effective sample size (sum w)^2 / sum w^2. Choose alpha inside training folds, never from held-out results."""
    comps = tuple(component_ids)
    if not comps or any(type(c) is not str or not c for c in comps):
        raise ValueError("component ids must be nonempty str")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite int or float in [0, 1] (bool and NumPy scalars are refused)")
    counts = Counter(comps)
    raw = np.array([counts[c] ** -float(alpha) for c in comps])
    w = raw * (len(comps) / raw.sum())
    ess = float(w.sum() ** 2 / (w * w).sum())
    mass: dict[str, float] = defaultdict(float)
    for c, x in zip(comps, w):
        mass[c] += float(x)
    return {"weights": readonly_array(w), "effective_sample_size": ess, "ess_fraction": ess / len(comps),
            "max_to_min_weight": float(w.max() / w.min()), "component_mass_range": (min(mass.values()), max(mass.values())),
            "rows": len(comps), "components": len(counts), "alpha": float(alpha)}


def component_checkpoint_split(y: np.ndarray, components: np.ndarray, *, fraction: float = 0.15,
                               seed: int = 42) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The inner early-stopping split of AUTHORIZED TRAINING rows: component-disjoint and class-stratified. Of the
    K = round(1 / fraction) StratifiedGroupKFold folds, the one whose positive rate is closest to the overall rate is taken
    (ties: the size closest to ``fraction``). Both classes are required on both sides."""
    from sklearn.model_selection import StratifiedGroupKFold

    y = np.asarray(y)
    g = np.asarray(components)
    if y.ndim != 1 or g.shape != y.shape or len(y) < 4:
        raise ValueError("misaligned or too few training rows")
    if y.dtype.kind not in "iu" or not np.isin(y, (0, 1)).all():
        raise TypeError("labels must be an integer array of 0 and 1 (never bool or float)")
    if type(fraction) is not float or not 0 < fraction < 0.5:
        raise ValueError("fraction must be a float in (0, 0.5)")
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    k = round(1 / fraction)
    n_components = len(np.unique(g))
    if n_components < k:
        raise ValueError(f"too few components ({n_components}) for {k} folds")
    rate = float(y.mean())
    best = None
    for tr, va in StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups=g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        key = (abs(float(y[va].mean()) - rate), abs(len(va) / len(y) - fraction))
        if best is None or key < best[0]:
            best = (key, tr, va)
    if best is None:
        raise ValueError("no component-disjoint fold has both classes on both sides")
    _, tr, va = best
    if set(g[tr].tolist()) & set(g[va].tolist()):
        raise AssertionError("component overlap between inner training and inner validation")
    return readonly_array(tr), readonly_array(va), {
        "folds": k, "seed": seed, "achieved_fraction": len(va) / len(y), "positive_rate_all": rate,
        "positive_rate_inner_train": float(y[tr].mean()), "positive_rate_inner_validation": float(y[va].mean()),
        "inner_train_components": int(len(np.unique(g[tr]))), "inner_validation_components": int(len(np.unique(g[va]))),
        "component_overlap": 0, "scope": "authorized training rows only"}


__all__ = ["FitAuthority", "FitAuthorization", "GENE_FIELD_SEPARATORS", "GeneField", "Partition", "PopulationRow",
           "SCHEMA", "SplitProtocol", "SplitVerdict", "VariantIdentity", "audit_split", "authorize_fit",
           "component_checkpoint_split", "digest", "population_digest", "resolve_gene_field", "split_components",
           "tempered_component_weights"]
