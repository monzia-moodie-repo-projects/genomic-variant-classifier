"""What a feature matrix IS, independently of any matrix.

DRIFT-1 Phase 1B. Created 2026-08-27.

WHY THIS EXISTS
---------------
`DriftDetector` refuses a comparison whose columns it cannot align, and
`_aligned_lsif_matrices` refuses same-width-but-reordered frames, because such
a comparison is meaningless. Both refusals are correct and both happen at the
moment of comparison, on data already in memory.

A representation identity states the same contract BEFORE either side exists,
so a reference and a candidate can be proven to inhabit one representation
rather than discovered to differ.

MEASURED 2026-08-27, and this type INVENTS NO FINGERPRINT:

    policy_fingerprint()   models/model_preprocessing.py:171 -- a PURE function
        of a policy mapping. No estimator, no fitted state, no data. So a
        representation can carry a preprocessing digest without instantiating
        anything.

    DECLARED_MISSINGNESS   models/model_preprocessing.py -- the missingness
        policy, already declared and already fingerprinted.

    TABULAR_FEATURES       models/variant_ensemble.py -- the ordered feature
        contract, with EXPECTED_TABULAR_FEATURE_COUNT as its fail-loud guard.

    EvaluationPopulation.membership_fingerprint
        evaluation/population.py:408 -- population identity, which this type
        deliberately does NOT touch. A representation says what the columns
        are; a population says which rows. Two facts, two owners.

WHY THE CONTRACT DIGEST IS DERIVED, NOT STORED
----------------------------------------------
An earlier design carried BOTH `feature_names` and `feature_contract_sha256`.
Two fields, one fact -- and a caller who edits the names and forgets the digest
produces an identity that is internally false, with nothing to notice.

The attestation schema faced the same choice on 2026-08-26 and answered it
differently for a reason that does not apply here: `pre_head` cannot be derived
from `pre_head_oid`, because git chooses the abbreviation length. So version 3
records both and BINDS them -- "recording both is evidence only if they agree."

A contract digest IS derivable, exactly, from the ordered names. So the
superior form is not two bound fields; it is ONE field and a derivation.

`preprocessing_policy_sha256` is different again, and is stored: it digests a
mapping owned by another module, which this type neither holds nor should. That
is a reference to a separate authority, not a duplicate of local data.

WHAT IS DELIBERATELY REQUIRED
-----------------------------
`source_manifest_sha256` has NO owner in this repository -- measured, the one
candidate was `moe_identity.py:192`'s `anchor_manifest_sha256`, an unrelated
concept about mechanistic anchor sets. It is required here anyway.

Making it optional would be the nullable-union defect this programme has now
repaired three times: a field that is sometimes absent becomes a field nobody
supplies. Requiring it means the release manifest must exist before any
representation can be identified, which is the intended order.

Acronyms: SHA-256 = Secure Hash Algorithm 256-bit.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

#: Separator for the contract digest. A newline cannot occur in a feature name
#: -- enforced below -- so the joined form is unambiguous and two different
#: name tuples cannot produce one digest.
_JOIN = "\n"


class RepresentationPlane(str, Enum):
    """WHICH space a comparison is made in.

    These answer different scientific questions and must never share one
    result object:

    SEMANTIC_TABULAR
        The engineered features as they mean something -- gnomAD allele
        frequency, SpliceAI score, annotation missingness. Interpretable.

    MODEL_INPUT
        What the estimator actually consumes after imputation and scaling.
        Operationally what the deployed model experiences.

    LEARNED_REPRESENTATION
        A learned latent space. Can reveal joint shifts invisible
        feature-by-feature, and is the least interpretable of the three.
    """

    SEMANTIC_TABULAR = "semantic_tabular"
    MODEL_INPUT = "model_input"
    LEARNED_REPRESENTATION = "learned_representation"


@dataclass(frozen=True)
class RepresentationIdentity:
    """The contract two feature frames must share to be comparable at all.

    Equality is on the whole object. That is the point: a same-width check
    (`shape[1] == 95`) admits a frame whose columns were reordered or
    substituted, and `_aligned_lsif_matrices` already refuses such comparisons
    because the resulting density ratio is uninterpretable.
    """

    plane: RepresentationPlane
    feature_names: Tuple[str, ...]
    preprocessing_policy_sha256: str
    source_manifest_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.plane, RepresentationPlane):
            raise ValueError(
                "plane is {!r}; a representation must name its plane from the "
                "declared vocabulary, because semantic and model-input drift "
                "are different questions".format(self.plane))

        names = self.feature_names
        if not isinstance(names, tuple):
            raise ValueError(
                "feature_names is {}; it must be a TUPLE, so the order is part "
                "of the identity and cannot be mutated after construction"
                .format(type(names).__name__))
        if not names:
            raise ValueError("a representation must enumerate its features")
        for name in names:
            if not isinstance(name, str) or not name:
                raise ValueError("feature name {!r} is not a non-empty string"
                                 .format(name))
            if _JOIN in name:
                raise ValueError(
                    "feature name {!r} contains a newline, which is the digest "
                    "separator. Two different name tuples could then produce "
                    "one digest.".format(name))
        if len(set(names)) != len(names):
            duplicated = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                "feature names must be unique; duplicated: {}".format(duplicated))

        for field_name in ("preprocessing_policy_sha256",
                           "source_manifest_sha256"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not _SHA256.match(value):
                raise ValueError(
                    "{} is {!r}; expected 64 lowercase hexadecimal characters. "
                    "A prefix is not a digest.".format(field_name, value))

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def feature_contract_digest(self) -> str:
        """A digest of the ORDERED names. DERIVED, never stored.

        Because it is derived, `a.feature_names == b.feature_names` and
        `a.feature_contract_digest == b.feature_contract_digest` are the same
        statement. Storing it as a field would make them two statements that
        can disagree.
        """
        payload = _JOIN.join(self.feature_names).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def describe(self) -> str:
        return (
            "{plane} | {n} features | contract {contract} | policy {policy} | "
            "manifest {manifest}".format(
                plane=self.plane.value, n=self.n_features,
                contract=self.feature_contract_digest[:12],
                policy=self.preprocessing_policy_sha256[:12],
                manifest=self.source_manifest_sha256[:12]))


class RepresentationMismatch(ValueError):
    """Two frames do not inhabit one representation."""


def assert_same_representation(reference: RepresentationIdentity,
                               candidate: RepresentationIdentity) -> None:
    """Refuse a comparison across two representations, naming WHICH field.

    A single equality check would answer "these differ" and leave a reader to
    find out how. Every failure here names the field and shows both values,
    because the interesting cases -- one reordered column, one substituted
    feature, a changed missingness policy -- look identical in a boolean.
    """
    if reference.plane is not candidate.plane:
        raise RepresentationMismatch(
            "representation plane differs: reference {!r}, candidate {!r}. "
            "Semantic-feature drift and model-input drift are different "
            "questions and must not be compared."
            .format(reference.plane.value, candidate.plane.value))

    if reference.feature_names != candidate.feature_names:
        if set(reference.feature_names) == set(candidate.feature_names):
            raise RepresentationMismatch(
                "the two representations name the same {} features in a "
                "DIFFERENT ORDER. Column position carries meaning here; a "
                "same-width comparison would pair up the wrong features."
                .format(len(reference.feature_names)))
        missing = sorted(set(reference.feature_names)
                         - set(candidate.feature_names))
        extra = sorted(set(candidate.feature_names)
                       - set(reference.feature_names))
        raise RepresentationMismatch(
            "feature sets differ: candidate is missing {} and adds {}"
            .format(missing[:8] or "nothing", extra[:8] or "nothing"))

    if (reference.preprocessing_policy_sha256
            != candidate.preprocessing_policy_sha256):
        raise RepresentationMismatch(
            "preprocessing policy differs: reference {}, candidate {}. The "
            "same values imputed or scaled differently are not the same "
            "observation.".format(reference.preprocessing_policy_sha256[:16],
                                  candidate.preprocessing_policy_sha256[:16]))

    if reference.source_manifest_sha256 != candidate.source_manifest_sha256:
        raise RepresentationMismatch(
            "source manifest differs: reference {}, candidate {}. A change in "
            "the annotation releases is MEASUREMENT-PROCESS drift, and "
            "reporting it as population drift would be a scientific error."
            .format(reference.source_manifest_sha256[:16],
                    candidate.source_manifest_sha256[:16]))
