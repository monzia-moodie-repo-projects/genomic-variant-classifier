"""Pin the P6 adjudication probe's answers on a synthetic cohort.

WHY THIS EXISTS
===============
`scripts/probe_clean_cohort_p6_2026-07-25.py` generates scientific evidence: the
counters in docs/measurements/CLEAN_COHORT_P6_AUDIT_2026-07-25.txt, which gate
cohort-version-2 certification. That probe is about to be restructured for the R2
provenance correction, and a restructuring of evidence-generating code may only be
trusted if it reproduces the prior answers exactly.

Synthetic fixtures cannot prove equivalence on the real 4,420,180-row cohort --
only a golden capture from that input can. What they CAN do is pin the probe's
behaviour on a cohort whose expected answers are derivable by hand from the
policy definitions, so a logic change is caught immediately rather than at the
next real run. That is what this file does.

THE FIXTURE
-----------
Eight variants, each constructed against a specific code path rather than chosen
arbitrarily:

  V1  singleton, pathogenic                     -- nothing changes anywhere
  V2  two rows, same tier, both pathogenic      -- concordant; representative may
                                                   move, label does not
  V3  two rows, same tier, pathogenic + benign  -- opposed binary at the best tier.
                                                   BOTH legacy and P6 quarantine it,
                                                   so neither has a label: no group
                                                   label change, no representative
  V4  best tier holds an explicit conflict and
      a binary                                  -- Rule 4 AMBIGUOUS_AT_BEST_TIER:
                                                   label withheld, representative
                                                   still selected
  V5  best tier holds only uncertain            -- Rule 6 NO_BINARY_AT_BEST_TIER
  V6  two rows at different tiers               -- tier decides; stable
  V7  legacy takes a lower-tier row by file
      order, P6 takes the better-tier row       -- representative-row LABEL changes
  V8  pathogenic at "criteria provided, single
      submitter" (legacy 3 / unified 3) plus
      benign at "criteria provided, conflicting
      classifications" (legacy 4 / unified 3)   -- THE DIVERGENT CASE. The legacy
                                                   best-tier set is the pathogenic
                                                   row alone, so legacy resolves and
                                                   KEEPS it; the unified best-tier set
                                                   is BOTH rows, so P6 Rule 3 fires and
                                                   QUARANTINES. Legacy label 1, P6
                                                   label None: the group label changes
                                                   and there is NO P6 representative
                                                   row to compare against.

V8 is the population that falsified the roadmap's original shared-universe
assumption (`n01 + n11 == 203`). It exists here so that assumption can never be
silently reintroduced.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# pyarrow is imported DIRECTLY, not via pytest.importorskip. A module-level
# importorskip collapses every test in this file into a SINGLE skip entry when the
# package is absent -- measured here on 2026-07-26: eleven tests became "1 skipped"
# on an interpreter without it, and the suite-size ratchet cannot see the
# difference because collection happens before skipping. pyarrow is pinned at
# requirements.txt:89 (pyarrow==23.0.1) and is what the probe reads the cohort
# with; its absence is a broken environment and must fail loudly. This mirrors the
# reasoning already recorded in test_ci_failure_alert_workflow.py for PyYAML.

_REPO = Path(__file__).resolve().parents[2]
_PROBE = _REPO / "scripts" / "probe_clean_cohort_p6_2026-07-25.py"

# variant_id, review_status, clinical_sig, source_id, ref, alt, consequence, gene_symbol
_ROWS = [
    ("V1", "criteria provided, single submitter", "pathogenic", "s1", "A", "T", "missense_variant", "GENEA"),

    ("V2", "criteria provided, single submitter", "pathogenic", "s2", "A", "G", "missense_variant", "GENEA"),
    ("V2", "criteria provided, single submitter", "likely pathogenic", "s1", "A", "G", "missense_variant", "GENEA"),

    ("V3", "criteria provided, single submitter", "pathogenic", "s2", "C", "T", "missense_variant", "GENEB"),
    ("V3", "criteria provided, single submitter", "benign", "s1", "C", "T", "missense_variant", "GENEB"),

    ("V4", "criteria provided, single submitter", "pathogenic", "s2", "G", "A", "missense_variant", "GENEC"),
    ("V4", "criteria provided, single submitter", "conflicting classifications of pathogenicity",
     "s1", "G", "A", "missense_variant", "GENEC"),

    ("V5", "criteria provided, single submitter", "uncertain significance", "s1", "T", "C", "missense_variant", "GENED"),
    ("V5", "criteria provided, single submitter", "uncertain significance", "s2", "T", "C", "missense_variant", "GENED"),

    ("V6", "no assertion criteria provided", "pathogenic", "s2", "A", "C", "missense_variant", "GENEE"),
    ("V6", "criteria provided, multiple submitters, no conflicts", "benign", "s1", "A", "C", "missense_variant", "GENEE"),

    ("V7", "no assertion criteria provided", "pathogenic", "s2", "G", "T", "missense_variant", "GENEF"),
    ("V7", "criteria provided, multiple submitters, no conflicts", "benign", "s1", "G", "T", "missense_variant", "GENEF"),

    ("V8", "criteria provided, single submitter", "pathogenic", "s2", "T", "A", "missense_variant", "GENEG"),
    ("V8", "criteria provided, conflicting classifications", "benign", "s1", "T", "A", "missense_variant", "GENEG"),
]


def _write_cohort(path: Path) -> None:
    cols = {
        "variant_id":   [r[0] for r in _ROWS],
        "metadata":     [{"review_status": r[1]} for r in _ROWS],
        "clinical_sig": [r[2] for r in _ROWS],
        "source_id":    [r[3] for r in _ROWS],
        "ref":          [r[4] for r in _ROWS],
        "alt":          [r[5] for r in _ROWS],
        "consequence":  [r[6] for r in _ROWS],
        "gene_symbol":  [r[7] for r in _ROWS],
    }
    pq.write_table(pa.table(cols), path)


def _run(tmp_path: Path, tag: str, emit_json: Path | None):
    """Load a FRESH module instance, point it at the fixture, run it."""
    spec = importlib.util.spec_from_file_location(f"_p6probe_{tag}", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_p6probe_{tag}"] = mod
    spec.loader.exec_module(mod)

    cohort = tmp_path / "cohort.parquet"
    if not cohort.exists():
        _write_cohort(cohort)
    mod.RAW = cohort
    mod.CLEAN = tmp_path / "absent.parquet"
    mod.CC = _REPO / "scripts" / "clean_cohort.py"
    mod.OUT = tmp_path / "artifact.txt"

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = mod.main(emit_json)
    return rc, buf.getvalue(), mod.OUT.read_bytes()


@pytest.fixture(scope="module")
def both_runs(tmp_path_factory):
    """Both runs in ONE directory, so the artifact path -- and therefore the
    'WROTE <path>' line -- is identical between them. Two directories would make
    the outputs differ for a reason that has nothing to do with the probe, which
    is exactly the kind of false signal these tests exist to exclude."""
    d = tmp_path_factory.mktemp("p6")
    off = _run(d, "off", None)
    j = d / "capture.json"
    rc, out, art = _run(d, "on", j)
    on = (rc, out, art, json.loads(j.read_text(encoding="utf-8")))
    return off, on


@pytest.fixture(scope="module")
def probe_run(both_runs):
    return both_runs[0]


@pytest.fixture(scope="module")
def probe_run_with_capture(both_runs):
    return both_runs[1]


# --------------------------------------------------------------------------- #
# The capture must not change the evidence
# --------------------------------------------------------------------------- #
def test_the_probe_runs_on_the_fixture(probe_run):
    rc, out, art = probe_run
    assert rc == 0
    assert b"P0-P6 POLICY TABLE" in art


def test_the_capture_does_not_change_the_artifact(probe_run, probe_run_with_capture):
    """The whole point: an addition to evidence-generating code must leave the
    evidence byte-identical."""
    _, _, art_off = probe_run
    _, _, art_on, _ = probe_run_with_capture
    assert hashlib.sha256(art_on).hexdigest() == hashlib.sha256(art_off).hexdigest()


def test_the_capture_adds_exactly_one_line_of_output(probe_run, probe_run_with_capture):
    _, out_off, _ = probe_run
    _, out_on, _, _ = probe_run_with_capture
    added = [l for l in out_on.splitlines() if l not in out_off.splitlines()]
    assert len(added) == 1 and added[0].startswith("WROTE ")


def test_the_capture_is_strict_json(probe_run_with_capture):
    """allow_nan=False: NaN and Infinity are not JSON number literals, and a
    non-finite counter would be a computation that failed silently."""
    *_, cap = probe_run_with_capture
    json.dumps(cap, allow_nan=False)
    assert set(cap) == {"schema", "generated", "golden", "supplementary"}


# --------------------------------------------------------------------------- #
# The probe's answers on a cohort whose expectations are derivable by hand
# --------------------------------------------------------------------------- #
def test_the_policy_table_is_pinned(probe_run_with_capture):
    *_, cap = probe_run_with_capture
    p6 = cap["golden"]["policy_table"]["P6"]
    assert p6["repr"] == 3, "representative-row changes vs legacy"
    assert p6["label"] == 1, "representative-row LABEL changes (the 63-analogue)"
    assert p6["quar"] == 1, "quarantine changes vs legacy -- V8 alone"
    assert cap["golden"]["p6_group_adjudicated_label_changes"] == 2, "the 203-analogue"


def test_the_two_counts_measure_different_populations(probe_run_with_capture):
    """The defect the R2 correction exists to fix, reproduced in miniature.

    One variant (V7) changes its representative-row label; a DIFFERENT variant
    (V4) changes its group-adjudicated label; and a third (V8) changes its group
    label while having no representative row at all. If these were one estimand
    the two counters would agree. They do not, and they must not.
    """
    *_, cap = probe_run_with_capture
    repr_label_changes = cap["golden"]["policy_table"]["P6"]["label"]
    group_label_changes = cap["golden"]["p6_group_adjudicated_label_changes"]
    assert repr_label_changes != group_label_changes


def test_the_not_applicable_population_exists(probe_run_with_capture):
    """V8: quarantined by P6, kept by legacy. It has a group-label change and NO
    representative row, so `representative_row_label_changed` is undefined for it.
    This is why that field must be nullable rather than False."""
    sup = probe_run_with_capture[3]["supplementary"]
    assert sup["p6_variants_total"] == 8
    assert sup["p6_variants_with_representative"] == 6, "V3 and V8 have none"
    assert sup["p6_quarantined_variants"] == 2, "V3 (both quarantine) and V8 (P6 only)"
    assert sup["p6_variants_total"] - sup["p6_variants_with_representative"] == 2


def test_explicit_conflicts_preserved_counts_states_not_explicit_conflicts(probe_run_with_capture):
    """A second overloaded label found on 2026-07-26.

    The acceptance line reads "P6 explicit conflicts preserved (not discarded)",
    but the quantity is `sum(state in {IRREDUCIBLE_CONFLICT, AMBIGUOUS_AT_BEST_TIER})`
    -- withheld-label STATES. IRREDUCIBLE_CONFLICT means an opposed binary at the
    best tier, which need not involve any explicit "conflicting classifications"
    value: V3 has none and still counts. This pins the arithmetic so the R2
    artifact can rename it honestly.
    """
    cap = probe_run_with_capture[3]
    states = cap["supplementary"]["p6_state_counts"]
    published = cap["golden"]["p6_explicit_conflicts_preserved"]
    assert published == states.get("IRREDUCIBLE_CONFLICT", 0) + states.get("AMBIGUOUS_AT_BEST_TIER", 0)
    assert states["IRREDUCIBLE_CONFLICT"] == 2
    assert states["AMBIGUOUS_AT_BEST_TIER"] == 1
    assert published == 3


def test_order_invariance_holds_on_the_fixture(probe_run_with_capture):
    *_, cap = probe_run_with_capture
    assert cap["golden"]["order_invariant"] is True
    per = cap["golden"]["order_invariance_per_permutation"]
    assert len(per) == 4, "reverse + three random permutations"
    assert all(v["ok"] for v in per.values())


def test_the_vocabulary_check_is_clean_on_the_fixture(probe_run_with_capture):
    """Falsifiable anchor: every clinical_sig value in the fixture is recognised.
    If the classifier's vocabulary changes, this fires."""
    *_, cap = probe_run_with_capture
    assert cap["golden"]["unrecognised_clinical_sig_distinct"] == 0
    assert cap["golden"]["would_raise"] == 0


def test_the_source_map_has_not_drifted(probe_run_with_capture):
    """The probe transcribes clean_cohort.REVIEW_STATUS_TIER. If clean_cohort
    changes and the probe is not re-transcribed, every tier in this evidence is
    wrong -- so the probe checks, and this pins the check."""
    *_, cap = probe_run_with_capture
    assert cap["golden"]["source_map_check"].startswith("OK:")
