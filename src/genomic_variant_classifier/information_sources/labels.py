"""Stage 1D-1b.3: authorized labels (SHADOW MODE: nothing in production imports this module).

Owner rulings f2ec92cd (Unit 1D-1b.3: "for an ordinary biological feature producer, evaluation labels must be inaccessible;
for a supervised model fitter, only explicitly authorized training labels are permitted") and 7a2a6ade (section 13: the
fold-scoped label store and the required non-interference test). Today the boundary is wherever the CALLER puts it: the run
driver writes ``gnn_df["acmg_label"] = y_train.values`` and the graph builders read every labeled row they are given
(models/gnn.py build_pyg_dataset, models/hetero_gnn_scorer.py assemble_node_features_and_focal). Nothing in the builders
knows which rows are authorized.

* ``LabelAuthority`` holds the labels and the partition, keyed by row and CANONICAL gene (an HGNC id, not a symbol: two
  symbols of one gene in different partitions are one gene, GENE-SYMBOL-SPLIT-IDENTITY-1). It authorizes, for a fit, the
  training rows whose gene has no held-out row; a training row that shares its gene with a held-out row is refused and
  counted. Held-out labels never leave the authority.
* ``FoldScopedLabelStore`` (the owner's reference, kept): a fitter reads labels one row at a time, every read is recorded,
  and a read outside the authorized genes raises ``PermissionError``. It offers no accessor for the underlying table.
* ``masked_labels``: the label vector a LEGACY builder may be given -- authorized labels kept, every other label NaN.
* ``label_noninterference``: the required test. Alter the held-out labels (flip them), rebuild with everything else fixed,
  and compare the outputs byte for byte. A builder given the raw labels fails it; given ``masked_labels`` it passes.
* ``discover_label_sites``: every module of src/ and scripts/ that names the graph builders' label column (``acmg_label``)
  or takes a ``label_col`` parameter, so a new graph-label reader cannot appear unreviewed (the census's discovery-closure
  rule, applied to labels). Its scope is stated, not implied: the tabular fitters receive labels as ``y`` arrays and the
  cohort column ``label``, whose fold boundary is the leakage-free baseline's (stage 1D-2); this discovery does not cover
  them.

As the owner states, a Python wrapper is not a security sandbox: these objects make the authorized path the only
convenient one and make violations measurable; a production boundary must also keep the raw label source unreachable.
"""
from __future__ import annotations

import ast
import hashlib
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

LABEL_COLUMN = "acmg_label"
LABEL_PARAMETERS = frozenset({"label_col", "label_column"})
PARTITIONS = frozenset({"train", "heldout"})


@dataclass(frozen=True)
class LabelRecord:
    row_id: str
    gene_id: str
    label: int
    partition: str

    def __post_init__(self) -> None:
        if not self.row_id or not self.gene_id:
            raise ValueError("label records need a row id and a canonical gene id")
        if isinstance(self.label, bool) or self.label not in (0, 1):
            raise ValueError(f"{self.row_id}: a reference label is the integer 0 or 1")
        if self.partition not in PARTITIONS:
            raise ValueError(f"{self.row_id}: partition must be one of {sorted(PARTITIONS)}")


class FoldScopedLabelStore:
    """Authorized training labels for ONE fit. Reads are recorded; a read of an unauthorized gene's row is refused."""

    __slots__ = ("_data", "_allowed", "_accesses")

    def __init__(self, data: Mapping[str, tuple[str, int]], allowed_genes: frozenset[str]):
        if not allowed_genes:
            raise ValueError("empty authorization scope")
        self._data = MappingProxyType(dict(data))
        self._allowed = frozenset(allowed_genes)
        self._accesses: list[tuple[str, str]] = []

    def read(self, row_id: str) -> int:
        gene, label = self._data[row_id]
        if gene not in self._allowed:
            raise PermissionError(f"unauthorized reference-label read: row {row_id}, gene {gene}")
        self._accesses.append((row_id, gene))
        return label

    def authorized_rows(self) -> tuple[str, ...]:
        return tuple(r for r, (g, _) in self._data.items() if g in self._allowed)

    @property
    def accesses(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._accesses)

    def __reduce__(self):            # no pickled copy of the labels can leave the process through this object
        raise TypeError("a fold-scoped label store cannot be serialised")


@dataclass(frozen=True)
class Authorization:
    authorized_rows: int
    authorized_genes: int
    heldout_rows: int
    heldout_genes: int
    refused_training_rows: int          # training rows whose canonical gene also has a held-out row
    refused_genes: tuple[str, ...]


class LabelAuthority:
    """The only holder of the reference labels. ``authorize()`` derives -- never accepts -- the authorized scope."""

    __slots__ = ("_records", "_auth", "_allowed")

    def __init__(self, records: Iterable[LabelRecord]):
        recs = list(records)
        ids = [r.row_id for r in recs]
        if len(set(ids)) != len(ids):
            raise ValueError("a row id appears twice")
        if not recs:
            raise ValueError("no label records")
        heldout_genes = {r.gene_id for r in recs if r.partition == "heldout"}
        train_genes = {r.gene_id for r in recs if r.partition == "train"}
        shared = train_genes & heldout_genes
        allowed = frozenset(train_genes - heldout_genes)
        refused = sum(1 for r in recs if r.partition == "train" and r.gene_id in shared)
        self._records = {r.row_id: r for r in recs}
        self._allowed = allowed
        self._auth = Authorization(sum(1 for r in recs if r.partition == "train" and r.gene_id in allowed), len(allowed),
                                   sum(1 for r in recs if r.partition == "heldout"), len(heldout_genes), refused,
                                   tuple(sorted(shared)))

    def authorize(self) -> Authorization:
        return self._auth

    def store_for_fit(self) -> FoldScopedLabelStore:
        data = {rid: (r.gene_id, r.label) for rid, r in self._records.items()
                if r.partition == "train" and r.gene_id in self._allowed}
        return FoldScopedLabelStore(data, self._allowed)

    def masked_labels(self, row_ids: Sequence[str]) -> np.ndarray:
        """Float labels aligned with ``row_ids``: authorized training labels, NaN everywhere else (held-out rows, refused
        training rows, rows the authority does not know)."""
        out = np.full(len(row_ids), np.nan)
        for i, rid in enumerate(row_ids):
            r = self._records.get(str(rid))
            if r is not None and r.partition == "train" and r.gene_id in self._allowed:
                out[i] = float(r.label)
        return out

    def __reduce__(self):
        raise TypeError("the label authority cannot be serialised")


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


__all__ = ["Authorization", "FoldScopedLabelStore", "LABEL_COLUMN", "LabelAuthority", "LabelRecord", "NonInterference",
           "SITE_ROLES", "check_label_sites", "discover_label_sites", "flip_heldout", "label_noninterference",
           "output_digest"]
