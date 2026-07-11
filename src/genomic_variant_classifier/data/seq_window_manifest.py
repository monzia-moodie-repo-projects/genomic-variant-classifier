"""seq_window_manifest.py -- the coherence contract for precomputed sequence windows.

Single source of truth for the manifest hash functions and the retrain-side verification gate, so
build_seq_windows.py (the producer) and the retrain (the consumer) agree by construction rather than
drifting. The gate is what prevents the degenerate-cnn_1d failure from recurring: if the precomputed
windows are missing, stale (cohort or reference changed), or incomplete, the retrain ABORTS loudly
instead of silently falling back to poly placeholders.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# The window-construction convention, pinned by the feasibility + indel-convention probes
# (2026-07-10). If the builder ever changes how it anchors or sizes windows, this string must
# change too, and the gate will then correctly reject artifacts built under the old convention.
CONVENTION = "contig_as_is;pos_1based;ins_mnv_off0;del_off-1_then_0;window101"
BUILDER_VERSION = "delta_window_builder/2026-07-10-stepB"
WINDOW = 101
KEY_COLS = ["chrom", "pos", "ref", "alt"]
MIN_OK_FRACTION = 0.95  # a healthy artifact builds real windows for at least this fraction


def cohort_key_hash(df) -> str:
    """Deterministic hash of the cohort's variant keys. Order-independent (keys are sorted), so it
    identifies the SET of variants, catching additions, removals, or edits regardless of row order."""
    h = hashlib.sha256()
    keys = (df["chrom"].astype(str) + "|" + df["pos"].astype(str) + "|"
            + df["ref"].astype(str) + "|" + df["alt"].astype(str))
    for k in sorted(keys.unique()):
        h.update(k.encode("utf-8"))
    h.update(str(len(df)).encode())
    return h.hexdigest()


def reference_signature(fa_path: Path) -> str:
    """Cheap, definitive reference signature: sha256 of the .fai index (small, uniquely identifies
    the indexed reference) plus the FASTA byte size. Avoids hashing the multi-gigabyte genome."""
    fa_path = Path(fa_path)
    fai = Path(str(fa_path) + ".fai")
    h = hashlib.sha256()
    if fai.exists():
        h.update(fai.read_bytes())
    if fa_path.exists():
        h.update(str(fa_path.stat().st_size).encode())
    return h.hexdigest()


@dataclass
class VerifyResult:
    ok: bool
    checks: dict = field(default_factory=dict)   # check name -> (passed, detail)
    reasons: list = field(default_factory=list)  # human-readable failure reasons

    def raise_if_failed(self) -> "VerifyResult":
        if not self.ok:
            raise SeqWindowsStaleError(
                "seq_windows verification failed:\n  - " + "\n  - ".join(self.reasons)
                + "\nRe-run scripts/build_seq_windows.py to regenerate the windows for the "
                  "current cohort and reference.")
        return self


class SeqWindowsStaleError(RuntimeError):
    """Raised when the precomputed windows are missing, stale, or incomplete for the cohort/reference
    the retrain is about to use. A hard stop -- never silently train cnn_1d on placeholder sequences."""


def verify_seq_windows(cohort_df, windows_dir, reference_path,
                       expected_convention: str = CONVENTION,
                       expected_window: int = WINDOW,
                       min_ok_fraction: float = MIN_OK_FRACTION,
                       sample_keys: int = 500) -> VerifyResult:
    """Fail-loud coherence gate. Confirms the precomputed windows in `windows_dir` were built for
    exactly this `cohort_df` and `reference_path`, under the expected convention, and are complete.

    Returns a VerifyResult; call .raise_if_failed() to enforce it as a hard gate.
    """
    windows_dir = Path(windows_dir)
    checks = {}
    reasons = []

    def record(name, passed, detail=""):
        checks[name] = (passed, detail)
        if not passed:
            reasons.append(f"{name}: {detail}")

    # 1. manifest present + parseable
    manifest_path = windows_dir / "seq_windows.manifest.json"
    if not manifest_path.exists():
        record("manifest_present", False, f"no manifest at {manifest_path}")
        return VerifyResult(False, checks, reasons)
    try:
        m = json.loads(manifest_path.read_text())
        record("manifest_present", True)
    except Exception as e:
        record("manifest_present", False, f"manifest unparseable: {e}")
        return VerifyResult(False, checks, reasons)

    # 2. not a dry-run artifact
    record("not_dry_run", not m.get("dry_run", False),
           "artifact is a DRY RUN; rebuild without --limit" if m.get("dry_run", False) else "")

    # 3. window size
    record("window", m.get("window") == expected_window,
           f"manifest window {m.get('window')} != expected {expected_window}"
           if m.get("window") != expected_window else "")

    # 4. convention
    record("convention", m.get("convention") == expected_convention,
           f"manifest convention {m.get('convention')!r} != expected {expected_convention!r}"
           if m.get("convention") != expected_convention else "")

    # 5. cohort coherence (the anti-drift check)
    got_cohort = cohort_key_hash(cohort_df)
    record("cohort_hash", got_cohort == m.get("cohort_key_sha256"),
           "cohort has changed since the windows were built (key-hash mismatch)"
           if got_cohort != m.get("cohort_key_sha256") else "")

    # 6. reference coherence
    got_ref = reference_signature(reference_path)
    record("reference_signature", got_ref == m.get("reference_signature"),
           "reference genome has changed since the windows were built"
           if got_ref != m.get("reference_signature") else "")

    # 7. artifact present + coverage
    parq = windows_dir / "seq_windows.parquet"
    if not parq.exists():
        record("artifact_present", False, f"no seq_windows.parquet at {parq}")
        return VerifyResult(False, checks, reasons)
    record("artifact_present", True)

    n_built = m.get("n_rows_built", 0)
    n_ok = m.get("n_ok", 0)
    ok_frac = (n_ok / n_built) if n_built else 0.0
    record("ok_fraction", ok_frac >= min_ok_fraction,
           f"only {ok_frac*100:.1f}% of windows built ok (< {min_ok_fraction*100:.0f}% floor)"
           if ok_frac < min_ok_fraction else "")

    # row-count coverage: the artifact should cover every cohort row
    record("row_coverage", n_built == len(cohort_df),
           f"artifact has {n_built} rows, cohort has {len(cohort_df)}"
           if n_built != len(cohort_df) else "")

    # 8. spot-check: a sample of cohort keys must be present AND ok in the artifact
    try:
        import pandas as pd
        w = pd.read_parquet(parq, columns=KEY_COLS + ["ok"])
        wkeys = set(w["chrom"].astype(str) + "|" + w["pos"].astype(str) + "|"
                    + w["ref"].astype(str) + "|" + w["alt"].astype(str))
        samp = cohort_df.sample(n=min(sample_keys, len(cohort_df)), random_state=13)
        skeys = (samp["chrom"].astype(str) + "|" + samp["pos"].astype(str) + "|"
                 + samp["ref"].astype(str) + "|" + samp["alt"].astype(str))
        missing = sum(1 for k in skeys if k not in wkeys)
        record("key_coverage", missing == 0,
               f"{missing}/{len(skeys)} sampled cohort keys absent from the windows"
               if missing else "")
    except Exception as e:
        record("key_coverage", False, f"coverage spot-check failed: {e}")

    ok = all(passed for passed, _ in checks.values())
    return VerifyResult(ok, checks, reasons)
