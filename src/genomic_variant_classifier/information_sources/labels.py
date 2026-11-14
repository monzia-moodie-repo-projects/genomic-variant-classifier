"""Stage 1D-1b.3, repaired in stage 1D-1b.4: authorized training labels (SHADOW MODE: nothing in production imports this
module).

Owner rulings f2ec92cd (Unit 1D-1b.3: "for an ordinary biological feature producer, evaluation labels must be inaccessible;
for a supervised model fitter, only explicitly authorized training labels are permitted"), 7a2a6ade (section 13: the
fold-scoped label store and the required non-interference test) and 68ca322f (LABEL-TYPE-IDENTITY-1 and
LABEL-DENIED-READ-AUDIT-1). Today the boundary is wherever the CALLER puts it: the run driver writes
``gnn_df["acmg_label"] = y_train.values`` and the graph builders read every labeled row they are given
(models/gnn.py build_pyg_dataset, models/hetero_gnn_scorer.py assemble_node_features_and_focal).

* ``TrainingLabels`` (stage 1D-1b.4; replaces the 1D-1b.3 LabelRecord, FoldScopedLabelStore and LabelAuthority, which
  chose rows by ONE canonical gene per row, accepted any value equal to 0 or 1, and left refused reads unrecorded): the only
  label interface a fitting worker receives. It is built from a ``population.FitAuthorization`` -- derived from the split's
  components, never asserted by the caller -- and keeps the labels of the authorized rows only, as native ``int`` 0 or 1;
  no held-out value is ever stored. Every read is recorded with its outcome (authorized, denied for a registered row that
  is not authorized, unknown for a row the population does not hold); a refused read raises ``LabelAccessError``.
  ``masked`` gives a LEGACY builder its label vector (authorized labels, NaN elsewhere) and records the bulk read. The
  object cannot be copied or serialised, so its receipt accounts for every read made through it.
* ``label_noninterference``: the required test. Alter the held-out labels (flip them), rebuild with everything else fixed,
  and compare the outputs byte for byte. A builder given the raw labels fails it; given ``TrainingLabels.masked`` it passes.
* ``discover_label_sites``: every module of src/ and scripts/ that names the graph builders' label column (``acmg_label``)
  or takes a ``label_col`` parameter, so a new graph-label reader cannot appear unreviewed (the census's discovery-closure
  rule, applied to labels). Its scope is stated, not implied: the tabular fitters receive labels as ``y`` arrays and the
  cohort column ``label``, whose fold boundary is the leakage-free baseline's (stage 1D-2); this discovery does not cover
  them (LABEL-ACCESS-DISCOVERY-SCOPE-1).

As the owner states, a Python wrapper is not a security sandbox: these objects make the authorized path the only
convenient one and make violations measurable; a production boundary must also keep the raw label source unreachable.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from genomic_variant_classifier.information_sources.graph_identity import readonly_array
from genomic_variant_classifier.information_sources.population import FitAuthorization, digest

logger = logging.getLogger(__name__)

LABEL_COLUMN = "acmg_label"
LABEL_PARAMETERS = frozenset({"label_col", "label_column"})
FIT_SCOPE_KEYS = frozenset({"model", "population", "split"})


class LabelAccessError(PermissionError):
    """A label read the fit's authorization does not cover (the row is held out, refused, or unknown)."""


class TrainingLabels:
    """Authorized training labels for ONE fit, and the record of every read made through this object."""

    __slots__ = ("_labels", "_registered", "_events", "_digest", "_authorization_digest")

    def __init__(self, authorization: FitAuthorization, label_source: Mapping[str, int], *,
                 fit_scope: Mapping[str, str]):
        if not isinstance(authorization, FitAuthorization):
            raise TypeError("authorization must be a population.FitAuthorization")
        missing = sorted(r for r in authorization.authorized_rows if r not in label_source)
        if missing:
            raise ValueError(f"the label source lacks {len(missing)} authorized row(s), e.g. {missing[0]}")
        kept = {r: label_source[r] for r in sorted(authorization.authorized_rows)}
        bad = next((r for r, v in kept.items() if type(v) is not int or v not in (0, 1)), None)
        if bad is not None:
            raise TypeError(f"{bad}: a reference label is the native int 0 or 1, got {type(kept[bad]).__name__} "
                            f"{kept[bad]!r} (convert explicitly at the adapter)")
        scope = dict(fit_scope)
        if set(scope) != FIT_SCOPE_KEYS or any(type(v) is not str or not v.strip() for v in scope.values()):
            raise ValueError(f"fit_scope names exactly {sorted(FIT_SCOPE_KEYS)}, each a nonempty str")
        self._labels = MappingProxyType(kept)
        self._registered = authorization.registered_rows
        self._events: list[tuple[Any, ...]] = []
        self._authorization_digest = authorization.digest
        self._digest = digest("training-labels", {
            "authorization": authorization.digest, "authority": authorization.authority.value, "fit_scope": scope,
            "labels": digest("label-payload", [[k, v] for k, v in kept.items()])})

    def _state(self, row_id: object) -> str:
        if type(row_id) is str and row_id in self._labels:
            return "authorized"
        return "denied" if type(row_id) is str and row_id in self._registered else "unknown"

    def read(self, row_id: str) -> int:
        state = self._state(row_id)
        self._events.append(("read", row_id if type(row_id) is str else repr(row_id), state))
        if state != "authorized":
            raise LabelAccessError(f"{state} label read: row {row_id!r}")
        return self._labels[row_id]

    def masked(self, row_ids: Sequence[str]) -> np.ndarray:
        """Float labels aligned with ``row_ids`` for a legacy builder: authorized labels, NaN everywhere else (held-out,
        refused and unknown rows). Recorded as one bulk read with its counts and the digest of the requested rows."""
        ids = list(row_ids)
        states = [self._state(r) for r in ids]
        out = np.full(len(ids), np.nan)
        for i, (r, s) in enumerate(zip(ids, states)):
            if s == "authorized":
                out[i] = float(self._labels[r])
        c = Counter(states)
        self._events.append(("masked", digest("masked-rows", [r if type(r) is str else repr(r) for r in ids]),
                             c["authorized"], c["denied"], c["unknown"]))
        return readonly_array(out)

    @property
    def authorized_rows(self) -> tuple[str, ...]:
        return tuple(self._labels)

    @property
    def digest(self) -> str:
        return self._digest

    def receipt(self) -> dict[str, Any]:
        """Every read through this object: single reads by outcome, bulk reads by their counts, and the digest of the
        ordered event log."""
        events = tuple(self._events)
        single = Counter(e[2] for e in events if e[0] == "read")
        bulk = [e for e in events if e[0] == "masked"]
        return {"training_labels_digest": self._digest, "authorization_digest": self._authorization_digest,
                "authorized": single["authorized"], "denied": single["denied"], "unknown": single["unknown"],
                "masked_calls": len(bulk), "masked_authorized": sum(e[2] for e in bulk),
                "masked_denied": sum(e[3] for e in bulk), "masked_unknown": sum(e[4] for e in bulk),
                "events_digest": digest("label-events", [list(e) for e in events])}

    def __reduce_ex__(self, protocol: object):     # no copy may leave this object's receipt
        raise TypeError("training labels cannot be serialised or copied")

    def __copy__(self):
        raise TypeError("training labels cannot be serialised or copied")

    def __deepcopy__(self, memo: object):
        raise TypeError("training labels cannot be serialised or copied")


def flip_heldout(labels: np.ndarray, heldout: np.ndarray) -> np.ndarray:
    """D -> D': every held-out label flipped (0 <-> 1); NaN stays NaN; nothing else changes."""
    out = np.asarray(labels, dtype=np.float64).copy()
    m = np.asarray(heldout, dtype=bool) & ~np.isnan(out)
    out[m] = 1.0 - out[m]
    return out


def output_digest(value: Any) -> str:
    """A canonical digest of a builder's output: numpy arrays by dtype, shape and bytes; mappings by sorted key; sequences
    in order; scalars by repr. Two outputs are identical exactly when their digests are."""
    h = hashlib.sha256()

    def feed(v: Any) -> None:
        if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):   # a torch tensor
            v = v.detach().cpu().numpy()
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            h.update(f"nd:{a.dtype.str}:{a.shape}:".encode())
            h.update(a.tobytes())
        elif isinstance(v, Mapping):
            h.update(b"map:")
            for k in sorted(v, key=repr):
                h.update(repr(k).encode())
                feed(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(f"seq:{len(v)}:".encode())
            for x in v:
                feed(x)
        elif isinstance(v, float) and math.isnan(v):
            h.update(b"nan")
        else:
            h.update(f"{type(v).__name__}:{v!r}".encode())
    feed(value)
    return h.hexdigest()


@dataclass(frozen=True)
class NonInterference:
    holds: bool
    digest_original: str
    digest_altered: str
    heldout_labeled_rows: int


def label_noninterference(build: Callable[[np.ndarray], Any], labels: np.ndarray, heldout: np.ndarray) -> NonInterference:
    """Ruling 7a2a6ade section 13: ``build`` maps a label vector to the producer's outputs with every other input fixed.
    The property holds when flipping every held-out label leaves the outputs byte-identical."""
    labels = np.asarray(labels, dtype=np.float64)
    heldout = np.asarray(heldout, dtype=bool)
    if labels.shape != heldout.shape:
        raise ValueError("labels and the held-out mask must align")
    n = int((heldout & ~np.isnan(labels)).sum())
    if n == 0:
        raise ValueError("the test is vacuous: no held-out row carries a label to alter")
    d0 = output_digest(build(labels.copy()))
    d1 = output_digest(build(flip_heldout(labels, heldout)))
    return NonInterference(d0 == d1, d0, d1, n)


# ------------------------------------------------------------------------------------------------- static discovery
def discover_label_sites(repo: Path, roots: Sequence[str] = ("src", "scripts")) -> dict[str, list[str]]:
    """{relative path: sorted markers} for every module that names the label column (a string constant equal to
    LABEL_COLUMN) or defines a function with a ``label_col``/``label_column`` parameter. Markers are
    'constant:<enclosing function or <module>>' and 'parameter:<function>'."""
    sites: dict[str, set[str]] = {}
    for root in roots:
        base = repo / root
        if not base.is_dir():
            raise ValueError(f"discovery root is missing: {root}")
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(repo).as_posix()
            try:
                tree = ast.parse(path.read_bytes(), filename=rel)
            except SyntaxError as exc:
                raise ValueError(f"cannot parse {rel} for label discovery: {exc}") from exc
            found: set[str] = set()

            def visit(node: ast.AST, scope: str) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        inner = child.name if scope == "<module>" else f"{scope}.{child.name}"
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = child.args
                            names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
                            if names & LABEL_PARAMETERS:
                                found.add(f"parameter:{inner}")
                        visit(child, inner)
                    else:
                        if isinstance(child, ast.Constant) and child.value == LABEL_COLUMN:
                            found.add(f"constant:{scope}")
                        visit(child, scope)
            visit(tree, "<module>")
            if found:
                sites[rel] = found
    return {k: sorted(v) for k, v in sorted(sites.items())}


#: label_definition: assigns the reference label; label_writer: hands labels to a builder (sets the boundary); fitter: reads
#: whatever labels it is given; evaluation: scores predictions against labels; diagnostic / experiment: not a model input.
SITE_ROLES = frozenset({"label_definition", "label_writer", "fitter", "evaluation", "diagnostic", "experiment"})


def check_label_sites(discovered: Mapping[str, Sequence[str]], recorded: Mapping[str, Mapping[str, Any]]) -> None:
    """The reviewed table must equal discovery exactly (no unreviewed reader, no stale entry), with a role and reason."""
    missing = sorted(set(discovered) - set(recorded))
    stale = sorted(set(recorded) - set(discovered))
    if missing or stale:
        raise ValueError(f"label sites unaccounted for: unreviewed {missing}; reviewed but absent {stale}")
    for rel, rec in recorded.items():
        if list(rec.get("markers", [])) != list(discovered[rel]):
            raise ValueError(f"{rel}: reviewed markers {rec.get('markers')} differ from discovered {discovered[rel]}")
        roles = rec.get("roles")
        if not isinstance(roles, list) or not roles or len(set(roles)) != len(roles) or set(roles) - SITE_ROLES:
            raise ValueError(f"{rel}: roles must be a non-empty list drawn from {sorted(SITE_ROLES)}")
        if not isinstance(rec.get("reason"), str) or not rec["reason"].strip():
            raise ValueError(f"{rel}: a reviewed site needs a reason")


__all__ = ["FIT_SCOPE_KEYS", "LABEL_COLUMN", "LABEL_PARAMETERS", "LabelAccessError", "NonInterference", "SITE_ROLES",
           "TrainingLabels", "check_label_sites", "discover_label_sites", "flip_heldout", "label_noninterference",
           "output_digest"]
