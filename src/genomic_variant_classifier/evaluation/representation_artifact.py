"""The extraction boundary: persist the GNN's focal representation, with an
immutable record of what it is and the order its rows came in.

WHY THIS MODULE EXISTS
======================
`models/gnn.py` now exposes the focal-node representation through
`GNNOutput.focal_embeddings` (commit 59d017f). Exposing it is not the same as
having it: Panel R's R3 through R7 need a representation that OUTLIVES the forward
pass, carries the partition it came from, and cannot silently reorder between
extraction and analysis. This module is that persistence layer, and nothing more.

Per the capability ladder (capability_lifecycle.py), this moves R3, R4 and R5
from NOT_IMPLEMENTED to IMPLEMENTED_NO_OUTPUT -- exactly one rung. The ladder's
own words for the step it satisfies: "a typed model output exposes the
representation, but nothing persists it." This module persists it. It runs NO
probe, fits NO whitening transform, computes NO geometry metric; those are the
OUTPUT_AVAILABLE and VALIDATED rungs above, and they are later commits. Doing
them here would be the NOT_IMPLEMENTED-straight-to-VALIDATED jump the ladder
exists to forbid.

THE FOUR GUARANTEES
-------------------
1. IDENTITY. The persisted matrix is exactly GNNOutput.focal_embeddings, detached
   from autograd and moved to CPU as float64 numpy. forward() deliberately leaves
   the embedding attached to the graph; the extraction boundary is the ONE place
   a detach is correct, because here the tensor stops being part of training and
   becomes a scientific object.

2. ROW ORDER IS HASHED. A representation matrix is meaningless without knowing
   which variant each row is. If the row order changes between extraction and a
   per-row probe, every per-row metric is silently corrupted. row_order_sha256 is
   a SHA-256 over the ordered, newline-joined row keys; a reorder changes the
   hash, so a downstream consumer can assert the order it received is the order
   that was hashed.

3. PROVENANCE IS IMMUTABLE. git_sha, partition_role, representation_name, the
   producing model class, n_rows, dim, dtype and an ISO-8601 UTC timestamp are
   frozen onto the artifact. A frozen dataclass cannot be edited after the fact
   to look like it came from a different run or partition.

4. PARTITION ROLE TRAVELS WITH THE MATRIX. R3 and R4 probes must fit on TRAIN
   only -- a whitening transform fit on STRUCTURE or TEST is leakage even with no
   labels (capability_lifecycle.py, R4 step). If the role were not bound to the
   matrix, a probe could not enforce train-only fitting. Binding it here is the
   leakage guard at the source.

Mirrors prediction_artifacts.py: atomic .tmp-then-rename writes, a JSON manifest
carrying the git SHA, parquet for the matrix (never pickle -- not portable, not
auditable), and no logging configuration at module scope (library-module rule).

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "RepresentationArtifact",
    "extract_focal_embeddings",
    "RowOrderMismatch",
    "hash_row_order",
]

# The partition roles a representation may carry. TRAIN is the only role a probe
# or whitening transform may be FIT on; the others are apply-only. Kept as a
# frozenset so a typo ("TARIN") fails loudly rather than silently mislabelling.
VALID_PARTITION_ROLES = frozenset(
    {"TRAIN", "TUNE", "STRUCTURE", "TEST", "STRUCTURE_TEST", "UNPARTITIONED"}
)


class RowOrderMismatch(ValueError):
    """Raised when a consumer's row keys do not hash to the artifact's recorded
    row_order_sha256 -- i.e. the representation has been reordered since
    extraction, which invalidates every per-row metric computed against it."""


def hash_row_order(row_keys: Sequence[str]) -> str:
    """SHA-256 over the ordered row keys, newline-joined and UTF-8 encoded.

    The hash is ORDER-SENSITIVE by construction: it is the whole point. Two
    matrices with the same rows in a different order produce different hashes.
    A newline join (rather than concatenation) means keys "ab","c" and "a","bc"
    do not collide.
    """
    if not all(isinstance(k, str) for k in row_keys):
        raise TypeError("row keys must all be strings")
    joined = "\n".join(row_keys).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _git_sha() -> str:
    """Best-effort current commit SHA. Returns 'unknown' rather than raising, so
    an artifact produced outside a git checkout (a fresh cloud box mid-transfer)
    still persists with an honest 'unknown' marker instead of failing the run."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class RepresentationArtifact:
    """An extracted focal representation, plus the immutable record of what it is.

    Frozen: once extracted, neither the matrix reference nor its provenance can be
    reassigned. The embeddings array is additionally marked read-only (see
    extract_focal_embeddings) so the values themselves cannot be edited in place.
    """

    embeddings: np.ndarray            # (n_rows, dim) float64, read-only
    row_keys: tuple[str, ...]         # length n_rows, the per-row identity
    row_order_sha256: str             # hash of row_keys, order-sensitive
    representation_name: str          # e.g. "gnn.focal.pre_classifier"
    partition_role: str               # one of VALID_PARTITION_ROLES
    model_class: str                  # e.g. "VariantGAT"
    git_sha: str
    created_utc: str                  # ISO-8601, UTC
    n_rows: int
    dim: int
    dtype: str

    def __post_init__(self) -> None:
        # Validate invariants at construction. A frozen dataclass still runs
        # __post_init__; object.__setattr__ is the sanctioned way to normalise.
        if self.partition_role not in VALID_PARTITION_ROLES:
            raise ValueError(
                f"partition_role {self.partition_role!r} not in "
                f"{sorted(VALID_PARTITION_ROLES)}")
        if self.embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {self.embeddings.ndim}-D")
        if self.embeddings.shape[0] != len(self.row_keys):
            raise ValueError(
                f"row count {self.embeddings.shape[0]} != "
                f"{len(self.row_keys)} row keys")
        if self.n_rows != self.embeddings.shape[0]:
            raise ValueError(f"n_rows {self.n_rows} != matrix rows "
                             f"{self.embeddings.shape[0]}")
        if self.dim != self.embeddings.shape[1]:
            raise ValueError(f"dim {self.dim} != matrix width "
                             f"{self.embeddings.shape[1]}")
        recomputed = hash_row_order(self.row_keys)
        if recomputed != self.row_order_sha256:
            raise ValueError(
                "row_order_sha256 does not match row_keys; the artifact's own "
                "hash is inconsistent with its own keys")

    def verify_row_order(self, row_keys: Sequence[str]) -> None:
        """Assert that `row_keys` is the exact order this artifact was hashed
        with. A downstream probe calls this before trusting per-row alignment;
        a mismatch means a reorder happened and every per-row metric is void."""
        got = hash_row_order(row_keys)
        if got != self.row_order_sha256:
            raise RowOrderMismatch(
                "row order does not match the extracted artifact: expected hash "
                f"{self.row_order_sha256[:12]}..., got {got[:12]}.... The "
                "representation has been reordered since extraction; per-row "
                "metrics against it would be silently wrong.")

    def to_manifest(self) -> dict:
        """The provenance record, without the matrix. JSON-serialisable."""
        return {
            "representation_name": self.representation_name,
            "partition_role": self.partition_role,
            "model_class": self.model_class,
            "git_sha": self.git_sha,
            "created_utc": self.created_utc,
            "n_rows": self.n_rows,
            "dim": self.dim,
            "dtype": self.dtype,
            "row_order_sha256": self.row_order_sha256,
        }

    def save(self, output_dir: Path | str) -> dict[str, Path]:
        """Persist matrix + row keys (parquet) and provenance (JSON), atomically.

        Returns the two written paths. Mirrors prediction_artifacts.py: write to
        a .tmp sibling, fsync, then os.replace -- so a crash leaves either the
        old files or the complete new ones, never a half-written matrix.
        """
        import pandas as pd

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{self.representation_name}.{self.partition_role}".replace("/", "_")

        # matrix + keys as one parquet: row_key column then e0000..e{dim-1}
        cols = {"row_key": list(self.row_keys)}
        for j in range(self.dim):
            cols[f"e{j:04d}"] = self.embeddings[:, j]
        df = pd.DataFrame(cols)

        def _write_parquet(path: Path) -> None:
            df.to_parquet(path, index=False)

        def _write_manifest(path: Path) -> None:
            path.write_text(json.dumps(self.to_manifest(), indent=2))

        mat_path = _atomic_write(out, f"{stem}.parquet", _write_parquet)
        man_path = _atomic_write(out, f"{stem}.manifest.json", _write_manifest)
        logger.info("Representation artifact written: %s", mat_path)
        return {"matrix": mat_path, "manifest": man_path}


def _atomic_write(output_dir: Path, filename: str, write_fn) -> Path:
    """Write via .tmp sibling + fsync + os.replace. Copied idiom from
    prediction_artifacts.RunArtifactWriter._atomic_write, kept consistent so the
    whole project fails and recovers the same way."""
    dst = output_dir / filename
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{filename}.", suffix=".tmp",
                                        dir=str(output_dir))
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        write_fn(tmp)
        with open(tmp, "r+b") as fh:
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dst


def extract_focal_embeddings(
    output,                              # GNNOutput (duck-typed to avoid a hard
                                         # torch import in a torch-free consumer)
    row_keys: Sequence[str],
    *,
    representation_name: str,
    partition_role: str,
    model_class: str,
    git_sha: Optional[str] = None,
) -> RepresentationArtifact:
    """Turn a GNNOutput carrying focal_embeddings into a frozen, hashed artifact.

    This is THE extraction boundary. It is the one sanctioned place the embedding
    is detached from autograd -- forward() keeps it on the graph on purpose, and
    the detach here marks the transition from "part of training" to "scientific
    object". It fits nothing and measures nothing; it persists identity, order,
    provenance and partition, and stops.

    Raises if the output carries no embeddings (return_embeddings was False), or
    if row_keys length does not match the matrix -- both are caller errors that
    must fail loudly, not produce a mislabelled artifact.
    """
    if not getattr(output, "has_embeddings", False):
        raise ValueError(
            "GNNOutput carries no focal_embeddings; call the model with "
            "return_embeddings=True before extracting")

    emb = output.focal_embeddings
    # Detach + CPU + float64 numpy. Duck-typed: if it is a torch tensor it has
    # .detach()/.cpu()/.numpy(); if a consumer already passed numpy, np.asarray
    # is a no-op. float64 so downstream linear algebra is not silently float32.
    if hasattr(emb, "detach"):
        arr = emb.detach().cpu().numpy()
    else:
        arr = np.asarray(emb)
    arr = np.ascontiguousarray(arr, dtype=np.float64)

    keys = tuple(str(k) for k in row_keys)
    if arr.shape[0] != len(keys):
        raise ValueError(
            f"matrix has {arr.shape[0]} rows but {len(keys)} row keys were given")

    # Freeze the values: a read-only view means a downstream bug cannot mutate the
    # persisted representation in place and leave the hash describing stale data.
    arr.setflags(write=False)

    return RepresentationArtifact(
        embeddings=arr,
        row_keys=keys,
        row_order_sha256=hash_row_order(keys),
        representation_name=representation_name,
        partition_role=partition_role,
        model_class=model_class,
        git_sha=git_sha if git_sha is not None else _git_sha(),
        created_utc=datetime.now(timezone.utc).isoformat(),
        n_rows=arr.shape[0],
        dim=arr.shape[1],
        dtype=str(arr.dtype),
    )
