"""The evaluation population contract.

WHY THIS MODULE EXISTS
======================
Ruled 2026-07-27:

    No numerical kernel may select, filter, normalise or redefine its evaluation
    population. Population construction is an explicit upstream operation, and
    every result must describe exactly that population.

Commit 2a enforced half of that: predicted scores and probabilities are validated
and fail closed rather than being silently filtered. It deliberately left the
other half standing. Reference labels are legitimately missing -- withheld labels
are first-class in this project and are carried as NaN by `CanonicalVariantTable`
-- and selecting on them is a POPULATION decision, so it could not simply be
deleted. It was parked behind a named transitional selector,
`metrics.select_finite_reference_labels`, precisely so that this commit would
have one deletion target rather than an anonymous clause to hunt for.

This module is that target's replacement.

THE DEFECT SHAPE THIS PREVENTS
------------------------------
Three defects recorded in `registry.py` share one shape: destroy a distinction,
measure the destroyed distinction, declare success.

    n01 + n11 == 203      held only after applicability had been erased
    85 and 107            printed as a partition after their overlap was forgotten
    np.minimum/np.maximum sorted quantile bounds BEFORE the crossing rate was
                          measured, so that rate was structurally zero

A narrowed population is the same shape. Once rows are gone, nothing downstream
can tell they were ever there, which is why `n_observations = 1000` sat beside a
number computed over 980 rows and no assertion could catch it.

An `EvaluationPopulation` cannot lose that distinction. Narrowing is the only
operation offered; every narrowing is strict, states its reason, and keeps a
reference to what it narrowed FROM. The lineage is the evidence.

WHAT THIS IS NOT
----------------
It is not a filter, a mask utility, or a convenience wrapper over indexing. It is
a claim about which rows a number describes, carried alongside the number -- an
enforceable estimand contract. It offers no way to widen, reorder, duplicate,
relabel or silently repair itself, because each would break the claim.

ADDRESSING MODEL
----------------
`indices` are absolute positions into the ORIGINAL source frame, never into a
parent. `take` is therefore a single fancy-index against the source arrays with
no chain to walk, and a population five narrowings deep still states plainly
which original rows it covers. The parent link is provenance, not address
translation.

Absolute indices are meaningful only relative to a NAMED source. Two populations
over different frames can carry identical indices, identical `n_source` and
identical `scope` and still describe entirely different rows, so `source_id` is
mandatory, is inherited unchanged by every child, and participates in the
membership fingerprint. Without it the lineage is locally coherent but not
globally auditable.
"""
from __future__ import annotations

import hashlib
from enum import Enum
import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["EvaluationPopulation", "PopulationComparison", "PopulationError",
           "PopulationTypeError"]


class PopulationError(ValueError):
    """A population invariant was violated.

    Subclasses `ValueError` so ordinary callers are not surprised, while
    remaining a single greppable class for code that wants to distinguish "this
    population claim is incoherent" from any other bad argument. Nothing here
    warns or repairs; every violation raises.
    """


class PopulationTypeError(PopulationError, TypeError):
    """A population invariant was violated by the TYPE of an argument.

    Inherits from both `PopulationError` and `TypeError`, so `except TypeError`,
    `except ValueError` and `except PopulationError` all catch it. A wrong dtype
    is genuinely a type error AND a contract violation; callers should not have
    to know which spelling this module happened to choose.
    """


def _membership_fingerprint(source_id: str, n_source: int,
                            indices: np.ndarray) -> str:
    """Deterministic identity of a population's membership.

    Absolute indices plus source identity completely determine membership, but a
    report should not have to serialise or compare potentially enormous index
    vectors to check that two numbers describe the same rows. This makes that an
    O(1) string comparison.

    It catches the defect cardinality cannot: two equal-sized but DIFFERENT
    subsets. `n = 980` beside `n = 980` says nothing about whether the same 980
    rows were used.

    Because indices are strictly increasing, membership has exactly one byte
    representation, so the digest is canonical rather than order-dependent. The
    little-endian cast is explicit so the value cannot vary with host byte
    order; a fingerprint that differed between machines would be worse than
    none, because it would fail comparisons that ought to succeed.
    """
    digest = hashlib.sha256()
    digest.update(source_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(n_source, dtype="<i8").tobytes())
    digest.update(np.asarray(indices, dtype="<i8").tobytes())
    return f"sha256:{digest.hexdigest()}"


class PopulationComparison(str, Enum):
    """The result of asking whether two populations describe the same rows.

    THREE-VALUED, DELIBERATELY. A boolean cannot express the difference between
    "proven different" and "not knowable from the provenance available", and
    collapsing them is how an unattributed population comes to be treated as
    comparable. `False` would read as "different rows", which is a claim; the
    honest answer when nothing identifies the frame is that the question cannot
    be answered.
    """

    SAME = "same"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


@dataclass(frozen=True, eq=False)
class EvaluationPopulation:
    """An immutable, auditable claim about which rows a metric describes.

    Generated equality is DISABLED (`eq=False`). NumPy array comparison does not
    yield a scalar boolean, so a dataclass-generated `__eq__` over an array field
    either raises or evaluates ambiguously. Semantic comparison is explicit:
    `same_membership_as`, or `membership_fingerprint`.

    Attributes
    ----------
    indices
        Strictly increasing `int64` positions into the SOURCE frame. Strictly
        increasing rather than merely unique, because a population is a SET of
        rows; permitting an order would invite downstream code to depend on it.
        Stored as an owned, read-only copy.
    scope
        What this population is, in words. Appears in every result computed over
        it, so it must be meaningful to a reader of the artifact.
    n_source
        Size of the original frame.
    source_id
        Identity of the frame the indices address; inherited unchanged by every
        child. Prefer something naming the actual artifact, for example
        `canonical_variant_table:sha256:<digest>` or
        `evaluation_frame:<artifact-id>:<schema-version>`, over a bare label.
    restriction_reason
        Why this population differs from its parent. Named `restriction` rather
        than `exclusion` because not every narrowing is an exclusion: a subgroup
        selection is an analytical restriction, and calling it an exclusion is
        semantically strained. A report may still describe a restriction as an
        exclusion where that is the right word. `None` if and only if root.
    parent
        The population this was narrowed from. `None` if and only if root.
        Excluded from `repr` to keep failure messages readable on deep lineages.
    """

    indices: np.ndarray
    scope: str
    n_source: int
    source_id: Optional[str]
    restriction_reason: Optional[str] = None
    parent: Optional["EvaluationPopulation"] = field(default=None, repr=False,
                                                     compare=False)

    # ------------------------------------------------------------------ #
    # Invariants. Every one raises; none warns; none repairs.
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        if isinstance(self.n_source, bool) or not isinstance(
                self.n_source, (int, np.integer)):
            raise PopulationTypeError(
                f"n_source must be an integer, got {type(self.n_source).__name__}")
        if self.n_source < 0:
            raise PopulationError(f"n_source must be non-negative, got {self.n_source}")
        object.__setattr__(self, "n_source", int(self.n_source))

        if not isinstance(self.scope, str) or not self.scope.strip():
            raise PopulationError(
                "scope must be a non-empty string naming what this population "
                "is; an unnamed population cannot be reported meaningfully")
        # ATTRIBUTION IS OPTIONAL, BUT NEVER FAKED (2026-07-28).
        #
        # `source_id=None` means the caller could not identify the frame these
        # indices address. That is a real and common state -- `evaluate()`
        # receives arrays, not a canonical table -- and it must be representable
        # WITHOUT inventing an identity.
        #
        # A sentinel string was considered and rejected. Combined with the normal
        # fingerprint algorithm it produces a value that looks cryptographically
        # authoritative while identifying only `sentinel + n_source + positions`,
        # so two equal-length calls over entirely different rows would certify an
        # equivalence nobody established. A consumer might notice the sentinel;
        # every generic comparison of `membership_fingerprint` would not.
        #
        # An UNATTRIBUTED population therefore has NO fingerprint at all, and
        # comparison against it returns UNKNOWN rather than True or False.
        if self.source_id is not None:
            if not isinstance(self.source_id, str) or not self.source_id.strip():
                raise PopulationError(
                    "source_id must be a non-empty string identifying the frame "
                    "these indices address, or None to state explicitly that the "
                    "frame is unattributed. A blank string is neither.")

        raw = np.asarray(self.indices)
        # REJECT before casting. `np.array([1.7], dtype=np.int64)` silently
        # yields [1]; a truncating coercion inside a class whose whole purpose is
        # to prevent silent membership changes would be self-defeating. Booleans
        # are rejected separately because `np.issubdtype(np.bool_, np.integer)`
        # is False but a bool array would otherwise read as positions 0 and 1.
        if raw.dtype == np.bool_ or not np.issubdtype(raw.dtype, np.integer):
            raise PopulationTypeError(
                f"indices must be an integer array, got dtype {raw.dtype}. "
                "Casting would silently truncate, and a boolean array would be "
                "read as positions rather than as a mask -- use `restrict` for "
                "masks.")
        # OWNED COPY, then read-only. Setting the write flag on a VIEW leaves the
        # writable base reachable by the caller, who could then mutate a
        # supposedly immutable population from outside it.
        indices = np.array(raw, dtype=np.int64, copy=True)

        if indices.ndim != 1:
            raise PopulationError(
                f"indices must be one-dimensional, got shape {indices.shape}")
        if indices.size:
            if indices[0] < 0 or indices[-1] >= self.n_source:
                raise PopulationError(
                    f"indices must lie in [0, {self.n_source}); observed range "
                    f"[{int(indices[0])}, {int(indices[-1])}]")
            if not np.all(indices[1:] > indices[:-1]):
                if np.unique(indices).size != indices.size:
                    raise PopulationError(
                        "indices contain duplicates; a population is a set of "
                        "rows, and a duplicated row would be counted twice in "
                        "every metric computed over it")
                raise PopulationError(
                    "indices must be strictly increasing; a population is a set "
                    "of rows and must not carry an ordering that downstream code "
                    "could come to depend on")

        if self.parent is None:
            if self.restriction_reason is not None:
                raise PopulationError(
                    "a root population cannot carry a restriction reason: it has "
                    "restricted nothing")
            if not np.array_equal(indices, np.arange(self.n_source, dtype=np.int64)):
                raise PopulationError(
                    f"a root population must contain every one of the "
                    f"{self.n_source} source rows; a partial population must be "
                    "derived through `restrict` so its reason and parent are "
                    "recorded")
        else:
            if not isinstance(self.restriction_reason, str) or \
                    not self.restriction_reason.strip():
                raise PopulationError(
                    "a derived population requires a restriction reason; a "
                    "narrowing that cannot say why it narrowed is the defect "
                    "this class exists to prevent")
            if self.n_source != self.parent.n_source:
                raise PopulationError(
                    f"child n_source ({self.n_source}) must equal the parent's "
                    f"({self.parent.n_source}); a narrowing cannot change the "
                    "frame it is measured against")
            if self.source_id != self.parent.source_id:
                raise PopulationError(
                    f"child source_id {self.source_id!r} must equal the parent's "
                    f"{self.parent.source_id!r}; a narrowing cannot change which "
                    "frame its indices address")
            # STRICT narrowing. An unchanged population must not acquire
            # artificial lineage: `label_eligible(n=1000, reason=...)` beneath
            # `attempted_cohort(n=1000)` asserts a restriction that never
            # happened. Relabelling identical membership is a DIFFERENT
            # operation and is deliberately not offered here.
            if indices.size >= self.parent.indices.size:
                raise PopulationError(
                    f"a restriction must strictly narrow: child has "
                    f"{indices.size} rows, parent has {self.parent.indices.size}. "
                    "An unchanged population must not acquire artificial lineage "
                    "claiming a restriction that did not occur.")
            # THE LOAD-BEARING INVARIANT. Smaller, ordered, unique and in range is
            # NOT enough: parent [0, 2, 4, 6] with child [1, 3] satisfies every
            # one of those and still re-admits rows the parent removed.
            #
            # searchsorted is O(m log n) against the parent's already-sorted
            # indices, rather than the O((m+n) log(m+n)) of a set intersection.
            parent_idx = self.parent.indices
            if parent_idx.size == 0:
                raise PopulationError(
                    "child indices must be a subset of the parent's, but the "
                    "parent population is empty")
            positions = np.clip(np.searchsorted(parent_idx, indices),
                                0, parent_idx.size - 1)
            if not np.array_equal(parent_idx[positions], indices):
                missing = indices[parent_idx[positions] != indices]
                raise PopulationError(
                    f"{missing.size} row(s) are not present in the parent "
                    f"population (first: {int(missing[0])}). A population may "
                    "only be narrowed, never widened: widening would silently "
                    "re-admit rows a named restriction had already removed, and "
                    "no downstream assertion could detect it.")

        indices.setflags(write=False)
        object.__setattr__(self, "indices", indices)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def full(cls, n_source: int, *, scope: str,
             source_id: Optional[str]) -> "EvaluationPopulation":
        """The whole attempted cohort: every row, nothing restricted."""
        if isinstance(n_source, bool) or not isinstance(n_source, (int, np.integer)):
            raise PopulationTypeError(
                f"n_source must be an integer, got {type(n_source).__name__}")
        if n_source < 0:
            raise PopulationError(f"n_source must be non-negative, got {n_source}")
        return cls(indices=np.arange(int(n_source), dtype=np.int64), scope=scope,
                   n_source=int(n_source), source_id=source_id)

    def restrict(self, mask: Sequence, *, scope: str,
                 reason: str) -> "EvaluationPopulation":
        """Narrow this population, recording why.

        `mask` is boolean and indexed RELATIVE TO THIS POPULATION -- its length
        must equal `self.n`, not `self.n_source`. That is the ergonomic choice,
        since the natural call computes the predicate on already-projected
        arrays: `pop.restrict(np.isfinite(pop.take(y)), ...)`. Accepting a
        source-length mask as well would make the two silently interchangeable
        whenever a population happens to be complete -- the case that occurs in
        every test fixture and almost never in production.

        The mask must remove at least one row. A caller who finds that
        inconvenient wants either no restriction at all -- keep the population as
        it is -- or a relabelling, which is a different operation.
        """
        mask_array = np.asarray(mask)
        if mask_array.dtype.kind != "b":
            raise PopulationTypeError(
                f"restriction mask must be boolean, got dtype {mask_array.dtype}; "
                "an integer mask -- even one containing only 0 and 1 -- would be "
                "interpreted as POSITIONS and would silently select the wrong rows")
        if mask_array.ndim != 1:
            raise PopulationError(
                f"restriction mask must be one-dimensional, got shape "
                f"{mask_array.shape}")
        if mask_array.size != self.n:
            raise PopulationError(
                f"restriction mask has length {mask_array.size} but this "
                f"population has {self.n} rows. The mask is relative to THIS "
                f"population, not to the source frame of {self.n_source} rows.")
        if bool(mask_array.all()):
            raise PopulationError(
                "a restriction must remove at least one row; an unchanged "
                "population must not acquire artificial lineage. Guard the call "
                "with `if not mask.all():` and keep the existing population when "
                "nothing is removed.")
        return EvaluationPopulation(
            indices=self.indices[mask_array], scope=scope, n_source=self.n_source,
            source_id=self.source_id, restriction_reason=reason, parent=self)

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    @property
    def n(self) -> int:
        """Rows in this population."""
        return int(self.indices.size)

    def __len__(self) -> int:
        return self.n

    @property
    def n_excluded_from_parent(self) -> int:
        """Rows this restriction removed relative to its parent. Zero at a root."""
        return 0 if self.parent is None else self.parent.n - self.n

    @property
    def n_excluded_from_source(self) -> int:
        """Rows absent relative to the ORIGINAL frame, however many narrowings."""
        return self.n_source - self.n

    @property
    def is_complete(self) -> bool:
        """True when this population still covers the whole source frame."""
        return self.n == self.n_source

    @property
    def membership_fingerprint(self) -> str:
        """Deterministic identity of this population's membership.

        Memoised on first access rather than computed at construction. Hashing is
        O(n) in the index vector, most populations are never fingerprinted, and
        populations are constructed more often than compared; paying at
        construction would tax the common path for the rare one. Memoising keeps
        repeated access O(1), which matters because the report invariant compares
        it once per metric.
        """
        if self.source_id is None:
            # NOT a fingerprint of "nothing" -- the ABSENCE of a fingerprint.
            # Returning a digest here would let `a.fingerprint == b.fingerprint`
            # answer True for two populations whose equivalence is unknown.
            return None
        cached = getattr(self, "_fingerprint_cache", None)
        if cached is None:
            cached = _membership_fingerprint(self.source_id, self.n_source,
                                             self.indices)
            object.__setattr__(self, "_fingerprint_cache", cached)
        return cached

    @property
    def is_attributed(self) -> bool:
        """Whether anything identifies the frame these indices address."""
        return self.source_id is not None

    def compare_membership(self, other: "EvaluationPopulation") -> PopulationComparison:
        """THE AUTHORITATIVE COMPARISON. Three-valued, because two are not enough.

        `None == None` is True in Python, so a caller comparing two absent
        fingerprints directly would conclude that two unattributed populations
        describe the same rows. They might; nothing establishes it. This is the
        only comparison that distinguishes proven-same from proven-different from
        not-knowable, and it is the one callers should use.
        """
        if not isinstance(other, EvaluationPopulation):
            raise TypeError(
                f"can only compare against another EvaluationPopulation, got "
                f"{type(other).__name__}")
        if not self.is_attributed or not other.is_attributed:
            return PopulationComparison.UNKNOWN
        return (PopulationComparison.SAME
                if self.membership_fingerprint == other.membership_fingerprint
                else PopulationComparison.DIFFERENT)

    def same_membership_as(self, other: "EvaluationPopulation") -> bool:
        """Do these describe exactly the same rows of the same frame?

        Explicit because generated equality is disabled: array comparison does
        not produce a scalar boolean. The frame is compared first, so two
        populations over different sources are never equal however their indices
        happen to line up.
        """
        if not isinstance(other, EvaluationPopulation):
            return NotImplemented
        # An unattributed population is never PROVEN to share membership, so this
        # returns False there. Callers who need to distinguish "different" from
        # "unknown" must use `compare_membership`; this boolean cannot express it
        # and is retained only because it predates the distinction.
        if not self.is_attributed or not other.is_attributed:
            return False
        return (self.source_id == other.source_id
                and self.n_source == other.n_source
                and np.array_equal(self.indices, other.indices))

    def take(self, values: Sequence) -> np.ndarray:
        """Project a SOURCE-aligned array onto this population.

        The length check is the point. Passing an already-projected array is the
        obvious mistake, and silently accepting it would produce a shorter array
        that still looks entirely plausible.
        """
        array = np.asarray(values)
        if array.ndim == 0:
            raise PopulationError("cannot project a scalar")
        if array.shape[0] != self.n_source:
            raise PopulationError(
                f"take() requires an array aligned to the original source frame: "
                f"expected {self.n_source} rows, received {array.shape[0]}. An "
                "array that has already been projected must not be projected "
                "again.")
        return array[self.indices]

    def lineage(self) -> Tuple[dict, ...]:
        """The full chain from the root, oldest first.

        The audit trail: which rows this number describes and what removed the
        rest, without re-deriving anything.
        """
        chain = []
        node: Optional[EvaluationPopulation] = self
        while node is not None:
            chain.append({"scope": node.scope, "n": node.n,
                          "removed_here": node.n_excluded_from_parent,
                          "reason": node.restriction_reason})
            node = node.parent
        return tuple(reversed(chain))

    def describe(self) -> str:
        """One-line human summary, for logs and failure messages."""
        return " -> ".join(
            f"{step['scope']}(n={step['n']}"
            + (f", -{step['removed_here']} {step['reason']})" if step["reason"] else ")")
            for step in self.lineage())
