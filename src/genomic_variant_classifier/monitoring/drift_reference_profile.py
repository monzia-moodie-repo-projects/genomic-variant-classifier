"""An AGGREGATE-ONLY reference profile -- so drift monitoring can run without the raw cohort.

Created 2026-07-13 (roadmap 6.20).

WHY THIS EXISTS
---------------
The scheduled drift monitor (`.github/workflows/drift_monitor.yml`) needs a reference
distribution to compare each new ClinVar release against. Until today it tried to obtain one
by downloading `outputs/run15_rerun_report/full/splits/X_train.parquet` (23.8 MB) from Google
Drive -- and that download was never implemented, so the workflow reported "no drift" every
month having never looked at anything (see roadmap 6.20).

Wiring that download turned out to be a bad idea, though NOT for the reason first claimed.

WHAT WAS FIRST CLAIMED, AND WHY IT WAS WRONG
--------------------------------------------
An earlier draft of this file asserted that `X_train.parquet` could not be published because
it carries dbNSFP-derived columns, and that `configs/data_manifest.yaml` marks dbNSFP
`tier: controlled` -- "LICENSED (paid). Do NOT place on personal cloud."

That was false, and it was false because a line number was trusted instead of read. The
"LICENSED (paid)" note at `data_manifest.yaml:286` belongs to **hgmd**, not dbnsfp. Read the
file:

    dbnsfp   (line  86):  tier: academic     class: public_redownloadable
                          notes: "Academic license. Confirm terms before any cloud copy."
    hgmd     (line 277):  tier: controlled   notes: "LICENSED (paid). Do NOT place on
                                                     personal cloud."

The `controlled` tier -- the one the rule "controlled/licensed => NEVER personal cloud"
actually governs -- is: **omim, hgmd, cosmic, tcga, topmed.** dbNSFP is not in it.

WHAT IS ACTUALLY TRUE (measured against the real matrix, 2026-07-13)
--------------------------------------------------------------------
* The four controlled-tier columns in X_train -- `omim_n_diseases`,
  `omim_is_autosomal_dominant`, `hgmd_is_disease_mutation`, `hgmd_n_reports` -- are
  **CONSTANT ZERO** (nunique == 1). They were never populated. The matrix therefore carries
  NO controlled-tier information whatsoever.
* What it does carry is dbNSFP-derived values at `tier: academic`, and they are **z-scored**,
  not raw scores (cadd_phred: mean -0.0000, range [-2.34, 10.76]).

BUT FROM RUN 17 ONWARD, THE LICENSING QUESTION BECOMES REAL -- FOR OMIM
------------------------------------------------------------------------
The Run-15 OMIM columns are zero because the connector then read `mim2gene.txt`, which has no
inheritance field and no phenotype counts. That was fixed: `omim.py::_count_phenotypes` now
parses `genemap2.txt` (real phenotype counts, the `(3)` molecular mapping key, and autosomal-
dominant inheritance), and `scripts/launch_run17_*.sh` HARD-ABORTS (exit 8) if genemap2 is
absent -- "omim_n_diseases/omim_n_diseases_molecular/omim_is_autosomal_dominant would
silent-zero".

So from Run 17, `X_train.parquet` WILL carry real OMIM-derived values. OMIM is `tier:
controlled` in data_manifest.yaml, held under an institutional license whose terms do not
permit redistribution. Publishing that matrix to a PUBLIC repository would then be a genuine
problem -- not because of dbNSFP, as this file once wrongly claimed, but because of OMIM.

A ten-bin histogram of a z-scored gene-level disease count is not an OMIM data product; it is
a summary statistic. The profile therefore stays clean where the raw matrix would not.

SO WHY NOT PUBLISH THE MATRIX?
------------------------------
Because the profile is simply the better artifact, and it keeps the licensing question moot
rather than answered-under-pressure:

  * 1.4 MB of histograms instead of 23.8 MB of variant rows.
  * Committable, versioned beside the code, no credentials, no cloud, no expiring token --
    which is what lets the monitor run at all.
  * It redistributes no per-variant annotation from any source, academic or otherwise, so
    "confirm terms before any cloud copy" never has to be litigated in the first place.

Publishing a million rows of derived annotations to a PUBLIC repository is a decision that
should be made deliberately, with the dbNSFP academic terms in hand -- not slipped in as a
side effect of wanting a drift monitor to run. The profile means nobody has to make it.

THE INSIGHT
-----------
The drift monitor does not need the variant rows. It needs the reference DISTRIBUTION.

Read `DriftDetector._psi`:

    lo, hi  = np.percentile(ref, 1), np.percentile(ref, 99)
    edges   = np.linspace(lo, hi, n_bins + 1)
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    new_pct = np.histogram(new, bins=edges)[0] / max(len(new), 1)
    ref_pct = np.clip(ref_pct, 1e-4, None)
    new_pct = np.clip(new_pct, 1e-4, None)
    return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

Everything the reference contributes is THREE AGGREGATES:

    1. the 1st and 99th percentiles  (which fix the bin edges)
    2. the ten reference bin COUNTS
    3. the count of finite rows       (the denominator)

Given those, the Population Stability Index is reproduced **EXACTLY** -- not approximately.
Bit-for-bit. And `DriftDetector._check_feature` decides each feature's action from PSI ALONE:

    if   psi > PSI_RETRAIN: action = "retrain"
    elif psi > PSI_MONITOR: action = "monitor"
    else:                   action = "none"

So an aggregate-only profile yields a FULLY FUNCTIONAL per-feature drift monitor, with no
variant-level data and no licensed annotation values -- only histogram buckets.

WHAT THIS PROFILE CANNOT DO, AND SAYS SO
----------------------------------------
* Maximum Mean Discrepancy and the Szekely-Rizzo energy test are MULTIVARIATE permutation
  tests. They need actual reference SAMPLES and cannot be reconstructed from marginal
  aggregates. From a profile they are NOT RUN, and the report records `joint_tests_run=False`
  with a reason. They are never reported as passing.

  THIS MATTERS. `DriftDetector.check` escalates on `mmd_pvalue < 0.001`. If a profile-based
  run silently substituted `mmd_pvalue = 1.0`, that escalation would be permanently disarmed
  while appearing to work -- which is EXACTLY the defect this whole file exists to fix.

* Kolmogorov-Smirnov and Wasserstein are computed against a quantile-reconstructed reference
  (see `quantiles` below). They are APPROXIMATE, to the resolution of the stored grid, and are
  labelled as such. They are informational -- no action depends on them.

WHAT IS IN THE FILE, AND WHAT IS NOT
------------------------------------
IN:  per feature -- 1st/99th percentile, 10 bin counts, a quantile grid, mean, std, finite-n.
NOT: any variant. Any identifier. Any per-variant annotation value.

It is a histogram. It is a few hundred kilobytes. It is committable, versioned beside the
code, needs no credentials, no cloud, and no expiring OAuth token -- and it lets the drift
monitor ACTUALLY RUN, every month, on a hosted runner, for the first time.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Bumped whenever the on-disk shape changes. A profile whose version this code does not
#: understand is a HARD ERROR -- never a silent fallback to "no drift".
PROFILE_FORMAT_VERSION = 1

#: Number of quantiles stored per feature. 1001 points (0.000 ... 1.000 in 0.001 steps)
#: reconstructs the reference empirical cumulative distribution function to ~0.1% resolution,
#: which is ample for the INFORMATIONAL Kolmogorov-Smirnov and Wasserstein figures. It costs
#: ~8 KB per feature.
N_QUANTILES = 1001


def _dumps_compact_arrays(payload: dict) -> str:
    """Serialise the profile with the NUMERIC ARRAYS ON ONE LINE EACH.

    WHY THIS EXISTS
    ---------------
    `json.dumps(payload, indent=1)` -- the obvious call, and what this file used first -- puts
    every element of every array on its own line. With 78 features x 1,001 quantiles that is
    **79,805 lines** of JSON for 1.4 MB of data.

    It is not wrong. It is bad repository hygiene, in three specific ways:

      * No human will ever read 78,078 lines of floats, so the "readability" the indent was
        bought for does not exist.
      * Every regeneration produces a whole-file diff, so `git log` on this artifact tells you
        nothing about WHAT changed -- only that it did. A reference distribution is exactly the
        thing you want to be able to diff meaningfully.
      * It bloats every clone of a PUBLIC repository for no benefit.

    Header keys stay pretty-printed, one per line, so the provenance -- source, build time,
    row count, feature list -- remains readable at a glance. The bin counts and quantile grids
    go on one line each: still greppable, still diffable per-feature, ~6x fewer lines.

    THE CONTENTS ARE UNCHANGED. This is a formatting decision and nothing else. `json.dumps`
    emits floats via `repr`, which is round-trip exact, so every stored value -- and therefore
    every Population Stability Index derived from it -- is bit-for-bit identical to what the
    indented form produced. `tests/unit/test_drift_reference_profile.py::
    test_psi_identical_after_save_and_reload` asserts that with `==`, and the build script
    re-verifies PSI against the raw matrix on every regeneration.

    HAND-ROLLED JSON IS HOW FILES GET QUIETLY CORRUPTED, so this does not trust itself: it
    parses its own output back and asserts it equals the payload before returning. A
    serialiser that has not been round-tripped is a serialiser you are hoping about.
    """
    def _kv(key: str, value: Any, indent: str) -> str:
        return f"{indent}{json.dumps(key)}: {json.dumps(value)}"

    lines: list[str] = ["{"]

    # -- header: one key per line, human-readable provenance ------------------------------
    for key in ("format_version", "source", "built_at_utc", "n_bins", "n_ref_samples"):
        lines.append(_kv(key, payload[key], " ") + ",")
    lines.append(_kv("feature_names", payload["feature_names"], " ") + ",")

    # -- features: one line per scalar, ONE LINE per array ---------------------------------
    lines.append(' "features": {')
    items = list(payload["features"].items())
    for i, (name, spec) in enumerate(items):
        lines.append(f"  {json.dumps(name)}: {{")
        keys = ("p01", "p99", "hist", "n_finite", "mean", "std", "quantiles")
        for j, k in enumerate(keys):
            comma = "," if j < len(keys) - 1 else ""
            lines.append(_kv(k, spec[k], "   ") + comma)
        lines.append("  }" + ("," if i < len(items) - 1 else ""))
    lines.append(" }")
    lines.append("}")

    text = "\n".join(lines) + "\n"

    # ROUND-TRIP OR DIE. Never write a file this function has not proven it can read back.
    reparsed = json.loads(text)
    if reparsed != payload:
        raise RuntimeError(
            "_dumps_compact_arrays produced JSON that does not round-trip to the payload it "
            "was given. The serialiser is broken and REFUSES to write the file. Do not 'fix' "
            "this by falling back to json.dumps and moving on -- find out what diverged, "
            "because a drift reference that silently differs from what was measured is worse "
            "than no drift reference at all."
        )
    return text


@dataclass
class FeatureProfile:
    """The aggregate reference for ONE feature. No variant rows. No licensed values."""

    name: str
    #: 1st and 99th percentile of the reference column. These FIX the PSI bin edges, so they
    #: must be stored, not recomputed -- recomputing them from new data would compare the new
    #: release against ITSELF.
    p01: float
    p99: float
    #: Reference histogram COUNTS over the 10 PSI bins spanned by [p01, p99].
    hist: list[int]
    #: Number of FINITE reference rows for this feature -- the PSI denominator. Note this is
    #: per-feature: DriftDetector._check_feature filters non-finite values per column.
    n_finite: int
    #: Reference mean / standard deviation (reported; also used for mean_shift_sigmas).
    mean: float
    std: float
    #: Quantile grid of the reference column -- a compressed empirical cumulative distribution
    #: function. Used ONLY to reconstruct an approximate reference sample for the
    #: INFORMATIONAL Kolmogorov-Smirnov and Wasserstein figures.
    quantiles: list[float]

    def psi_edges(self, n_bins: int = 10) -> np.ndarray:
        """The exact bin edges DriftDetector._psi would have computed. Degenerate -> empty."""
        if self.p01 == self.p99:
            return np.array([])
        return np.linspace(self.p01, self.p99, n_bins + 1)

    def reference_sample(self, n: int = 10_000) -> np.ndarray:
        """Reconstruct an approximate reference sample from the stored quantile grid.

        This is a *distributional* reconstruction: it has the reference's empirical cumulative
        distribution function to grid resolution, and NOTHING ELSE. It is not, and cannot be,
        the original rows -- the grid is a monotone summary and the row ordering, identities
        and joint structure are all gone.

        Used only for the INFORMATIONAL Kolmogorov-Smirnov statistic and Wasserstein distance.
        No action depends on either.
        """
        q = np.asarray(self.quantiles, dtype=float)
        probs = np.linspace(0.0, 1.0, len(q))
        want = np.linspace(0.0, 1.0, n)
        return np.interp(want, probs, q)


@dataclass
class DriftReferenceProfile:
    """The whole aggregate reference. Committable. Contains no variant-level data."""

    format_version: int
    source: str
    built_at_utc: str
    n_bins: int
    n_ref_samples: int
    feature_names: list[str]
    features: dict[str, FeatureProfile] = field(default_factory=dict)

    # ---------------------------------------------------------------- build --
    @classmethod
    def from_reference(
        cls,
        X_ref: pd.DataFrame,
        source: str,
        n_bins: int = 10,
        n_quantiles: int = N_QUANTILES,
    ) -> "DriftReferenceProfile":
        """Build the profile from the raw reference matrix. RUN THIS WHERE THE DATA LIVES."""
        from datetime import datetime, timezone

        feature_names = list(X_ref.columns)
        features: dict[str, FeatureProfile] = {}

        for name in feature_names:
            col = X_ref[name].to_numpy(dtype=np.float64)
            finite = col[np.isfinite(col)]

            if finite.size == 0:
                logger.warning(
                    "feature %r has NO finite reference values; PSI will be 0.0 for it and "
                    "it can never signal drift. That is a data problem, not a profile problem.",
                    name,
                )
                features[name] = FeatureProfile(
                    name=name, p01=0.0, p99=0.0, hist=[0] * n_bins, n_finite=0,
                    mean=0.0, std=0.0, quantiles=[0.0] * n_quantiles,
                )
                continue

            # EXACTLY what DriftDetector._psi does -- same percentiles, same edges, same
            # histogram, same denominator. Any deviation here silently changes every PSI.
            p01 = float(np.percentile(finite, 1))
            p99 = float(np.percentile(finite, 99))
            if p01 == p99:
                hist = [0] * n_bins
            else:
                edges = np.linspace(p01, p99, n_bins + 1)
                hist = [int(c) for c in np.histogram(finite, bins=edges)[0]]

            features[name] = FeatureProfile(
                name=name,
                p01=p01,
                p99=p99,
                hist=hist,
                n_finite=int(finite.size),
                mean=float(np.mean(finite)),
                std=float(np.std(finite)),
                quantiles=[
                    float(v)
                    for v in np.quantile(finite, np.linspace(0.0, 1.0, n_quantiles))
                ],
            )

        return cls(
            format_version=PROFILE_FORMAT_VERSION,
            source=source,
            built_at_utc=datetime.now(timezone.utc).isoformat(),
            n_bins=n_bins,
            n_ref_samples=int(len(X_ref)),
            feature_names=feature_names,
            features=features,
        )

    # ------------------------------------------------------------- persist --
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": self.format_version,
            "source": self.source,
            "built_at_utc": self.built_at_utc,
            "n_bins": self.n_bins,
            "n_ref_samples": self.n_ref_samples,
            "feature_names": self.feature_names,
            "features": {
                name: {
                    "p01": f.p01,
                    "p99": f.p99,
                    "hist": f.hist,
                    "n_finite": f.n_finite,
                    "mean": f.mean,
                    "std": f.std,
                    "quantiles": f.quantiles,
                }
                for name, f in self.features.items()
            },
        }
        path.write_text(_dumps_compact_arrays(payload), encoding="utf-8")
        size_kb = path.stat().st_size / 1024
        logger.info(
            "Drift reference profile -> %s (%.0f KB, %d features, %d reference rows). "
            "Contains histogram counts and quantile grids ONLY -- no variant-level data.",
            path, size_kb, len(self.feature_names), self.n_ref_samples,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "DriftReferenceProfile":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Drift reference profile not found: {path}\n"
                f"\n"
                f"This is NOT 'no drift'. Without a reference there is NOTHING TO COMPARE\n"
                f"AGAINST, and the monitor cannot run. Build it where the raw cohort lives:\n"
                f"\n"
                f"    python scripts/build_drift_reference_profile.py \\\n"
                f"        --reference outputs/run15_rerun_report/full/splits/X_train.parquet \\\n"
                f"        --out data/reference/drift/run15_reference_profile.json\n"
                f"\n"
                f"See roadmap 6.20."
            )

        payload = json.loads(path.read_text(encoding="utf-8"))

        version = payload.get("format_version")
        if version != PROFILE_FORMAT_VERSION:
            # A HARD ERROR, deliberately. A profile this code does not understand must never
            # degrade into a silent "no drift" -- that is the exact failure this file exists
            # to end.
            raise ValueError(
                f"{path} has format_version={version!r}, but this code understands "
                f"{PROFILE_FORMAT_VERSION}. Refusing to guess. Rebuild the profile with "
                f"scripts/build_drift_reference_profile.py."
            )

        features = {
            name: FeatureProfile(name=name, **spec)
            for name, spec in payload["features"].items()
        }
        return cls(
            format_version=version,
            source=payload["source"],
            built_at_utc=payload["built_at_utc"],
            n_bins=payload["n_bins"],
            n_ref_samples=payload["n_ref_samples"],
            feature_names=payload["feature_names"],
            features=features,
        )

    # ----------------------------------------------------------------- PSI --
    def psi(self, feature: str, new_col: np.ndarray) -> float:
        """Population Stability Index for one feature. EXACT -- not an approximation.

        Reproduces `DriftDetector._psi` bit-for-bit, using the stored aggregates in place of
        the reference column. There is a test that asserts this equality against the raw
        matrix (tests/unit/test_drift_reference_profile.py); if it ever fails, the profile and
        the detector have drifted apart and every PSI in every report is suspect.
        """
        prof = self.features[feature]
        new_col = np.asarray(new_col, dtype=np.float64)
        new_col = new_col[np.isfinite(new_col)]

        if prof.p01 == prof.p99:
            return 0.0

        edges = np.linspace(prof.p01, prof.p99, self.n_bins + 1)
        ref_pct = np.asarray(prof.hist, dtype=np.float64) / prof.n_finite
        new_pct = np.histogram(new_col, bins=edges)[0] / max(len(new_col), 1)

        ref_pct = np.clip(ref_pct, 1e-4, None)
        new_pct = np.clip(new_pct, 1e-4, None)
        return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
