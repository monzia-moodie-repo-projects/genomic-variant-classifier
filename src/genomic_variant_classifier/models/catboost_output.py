"""Bound CatBoost training diagnostics. This is not filesystem sandboxing.

No model serialization policy is implemented here. No snapshots/resume are
supported. Callers create the private run-owned output root before fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Mapping


CONTROLLED_OUTPUT_OPTIONS = frozenset({
    "allow_writing_files", "train_dir", "save_snapshot", "snapshot_file",
    "snapshot_interval", "output_borders", "json_log", "learn_error_log",
    "test_error_log", "time_left_log", "roc_file",
})


@dataclass(frozen=True)
class BackendOutput:
    enabled: bool
    fit_dir: str | None

    def catboost_parameters(self) -> dict:
        params = {"allow_writing_files": self.enabled, "save_snapshot": False}
        if self.fit_dir is not None:
            params["train_dir"] = self.fit_dir
        return params


def prepare_catboost_output(
    *, allow_writing_files: bool, output_root: str | None,
    extra: Mapping[str, object],
) -> BackendOutput:
    """Validate first, then allocate a fresh directory for this fit only.

    The root must already exist and belong exclusively to this run/attempt.
    Never construct it from a sample identifier or untrusted path fragment.
    A unique leaf prevents collisions between clones, refits and processes.
    Its parent remains caller-owned: ACLs/containers must enforce confinement.
    """
    if type(allow_writing_files) is not bool:
        raise TypeError("allow_writing_files must be a bool")
    unexpected = CONTROLLED_OUTPUT_OPTIONS.intersection(extra)
    if unexpected:
        raise ValueError(f"Output options must use the policy: {sorted(unexpected)}")
    if not allow_writing_files:
        if output_root is not None:
            raise ValueError("Disabled diagnostics require output_root=None")
        return BackendOutput(False, None)
    if not isinstance(output_root, str) or not output_root:
        raise ValueError("Diagnostics require an explicit absolute output root")
    root = Path(output_root)
    if not root.is_absolute():
        raise ValueError("Output root must be absolute; CWD is not an authority")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Output root must be an existing directory")
    fit_dir = mkdtemp(prefix="catboost-fit-", dir=root)
    return BackendOutput(True, fit_dir)
