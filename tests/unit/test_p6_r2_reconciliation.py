"""Battery for the P6 R2 reconciliation.

The reconciliation is pure: it takes a collection of immutable PolicyDelta records
and returns tables. So most of this file never touches a filesystem -- it builds a
handful of variants whose expected 2x2 cells are obvious by inspection, and asserts
them. That is the point of the layering.

The end-to-end tests use the same eight-variant synthetic cohort as
test_p6_probe_contract.py, so the two files agree on what the fixture means.

THE CENTRAL CLAIM UNDER TEST
============================
`representative_row_label_changed` is NULLABLE. For a variant P6 quarantines there
is no P6 representative row, so the comparison is undefined rather than False.
Encoding it as False would preserve the original repair plan's equation
`n01 + n11 == 203` syntactically, by changing what the predicate means -- and would
put "a label exists and is unchanged" into the same cell as "no label exists".
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# pyarrow imported DIRECTLY, never via importorskip: a module-level importorskip
# collapses this whole file into a single skip entry when the package is absent,
# and pyarrow is pinned at requirements.txt:89. See test_p6_probe_contract.py.

_REPO = Path(__file__).resolve().parents[2]
_R2 = _REPO / "scripts" / "probe_p6_r2_reconciliation.py"


def _load():
    spec = importlib.util.spec_from_file_location("_p6r2_undertest", _R2)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p6r2_undertest"] = mod
    spec.loader.exec_module(mod)
    return mod


r2 = _load()


def delta(*, vid="V", legacy_repr=1, p6_repr=1, legacy_label=1, p6_label=1,
          legacy_out=1, p6_out=1, state="PATHOGENIC",
          legacy_quarantined=False, p6_quarantined=False,
          repr_label_changed=None, group_changed=None):
    """Build a PolicyDelta whose derived fields are CONSISTENT by construction.

    The previous helper took `quar` and set `p6_quarantined = quar or legacy_quar`,
    which made a LEGACY-ONLY quarantine -- legacy True, P6 False -- impossible to
    express. That is exactly the 107-to-85 transition the real cohort exhibits, so
    the helper could not construct the case whose mishandling produced the defect
    it was meant to test. The two states are now independent inputs and
    `quarantine_changed` is derived from them.

    `representative_row_changed` and the nullability of the label comparison are
    likewise derived from the row identities, so a test cannot accidentally build
    a record that PolicyDelta.__post_init__ would reject for the wrong reason.
    """
    applies = legacy_repr is not None and p6_repr is not None
    if applies and repr_label_changed is None:
        repr_label_changed = legacy_label != p6_label
    if not applies:
        repr_label_changed = None
    if group_changed is None:
        group_changed = legacy_out != p6_out
    return r2.PolicyDelta(
        variant_id=vid,
        representative_row_changed=(legacy_repr != p6_repr),
        representative_row_label_changed=repr_label_changed,
        final_adjudicated_label_changed=group_changed,
        trainability_changed=(legacy_out is None) != (p6_out is None),
        quarantine_changed=(legacy_quarantined != p6_quarantined),
        legacy_representative_row=legacy_repr,
        p6_representative_row=p6_repr,
        legacy_representative_label=legacy_label,
        p6_representative_label=p6_label,
        legacy_output_label=legacy_out,
        p6_output_label=p6_out,
        p6_state=state,
        legacy_quarantined=legacy_quarantined,
        p6_quarantined=p6_quarantined,
    )


def na(*, vid="Q", miss_legacy=False, miss_p6=False, group_changed=False,
       legacy_quarantined=None, p6_quarantined=None):
    """A not-applicable variant: at least one representative row is absent."""
    assert miss_legacy or miss_p6, "a not-applicable record must miss a row"
    return delta(
        vid=vid,
        legacy_repr=None if miss_legacy else 1,
        p6_repr=None if miss_p6 else 2,
        legacy_label=None if miss_legacy else 1,
        p6_label=None if miss_p6 else 1,
        legacy_out=None if miss_legacy else 1,
        p6_out=(None if miss_p6 else 1) if not group_changed else (0 if not miss_p6 else None),
        group_changed=group_changed,
        legacy_quarantined=miss_legacy if legacy_quarantined is None else legacy_quarantined,
        p6_quarantined=miss_p6 if p6_quarantined is None else p6_quarantined,
    )


# --------------------------------------------------------------------------- #
# Group 1 -- the 2x2, one variant per cell
# --------------------------------------------------------------------------- #
def test_each_cell_of_table_a_gets_exactly_one_variant():
    deltas = [
        delta(vid="A", legacy_label=1, p6_label=1, legacy_out=1, p6_out=1),  # n00
        delta(vid="B", legacy_label=1, p6_label=1, legacy_out=1, p6_out=0),  # n01
        delta(vid="C", legacy_label=1, p6_label=0, legacy_out=1, p6_out=1),  # n10
        delta(vid="D", legacy_label=1, p6_label=0, legacy_out=1, p6_out=0),  # n11
    ]
    a = r2.summarize(deltas).table_a
    assert (a.n00, a.n01, a.n10, a.n11) == (1, 1, 1, 1)
    assert a.total == 4


def test_a_null_comparison_goes_to_table_b_not_into_n00():
    s_ = r2.summarize([delta(vid="A"), na(vid="Q", miss_p6=True, group_changed=True)])
    assert s_.table_a.total == 1 and s_.table_a.n00 == 1
    assert (s_.table_b.n_na0, s_.table_b.n_na1) == (0, 1)


def test_the_domain_partitions_exactly():
    ds = [delta(vid=f"V{i}", p6_label=(0 if i % 2 else 1)) for i in range(6)]
    ds += [na(vid=f"Q{i}", miss_p6=True, group_changed=bool(i % 2)) for i in range(4)]
    s_ = r2.summarize(ds)
    assert s_.table_a.total + s_.table_b.total == s_.n_total == 10
    assert not r2.check_invariants(s_)


# --------------------------------------------------------------------------- #
# Group 2 -- the reconciliations, and their falsifiability
# --------------------------------------------------------------------------- #
def test_both_reconciliations_hold_on_a_mixed_collection():
    ds = [
        delta(vid="a"),
        delta(vid="b", legacy_out=1, p6_out=0),
        delta(vid="c", p6_label=0),
        delta(vid="d", p6_label=0, legacy_out=1, p6_out=0),
        na(vid="e", miss_p6=True, group_changed=True),
        na(vid="f", miss_p6=True, group_changed=False),
    ]
    s_ = r2.summarize(ds)
    assert s_.table_a.n10 + s_.table_a.n11 == s_.representative_label_changed == 2
    assert s_.table_a.n01 + s_.table_a.n11 + s_.table_b.n_na1 == s_.group_label_changed == 3
    assert not r2.check_invariants(s_)


def test_the_original_plans_equation_is_false_on_that_same_collection():
    ds = [delta(vid="b", legacy_out=1, p6_out=0),
          na(vid="e", miss_p6=True, group_changed=True)]
    s_ = r2.summarize(ds)
    assert s_.table_a.n01 + s_.table_a.n11 == 1
    assert s_.group_label_changed == 2


def test_check_invariants_reports_ALL_failures_not_just_the_first():
    broken = r2.Reconciliation(
        table_a=r2.TableA(0, 0, 5, 0),
        table_b=r2.TableB(0, 0, 0, 0, 0, 0),
        n_total=99, representative_selection_changed=0,
        representative_label_changed=1, group_label_changed=7,
        quarantine_changed=3, trainability_changed=0,
        selection_change_breakdown={}, label_transitions={}, state_counts={},
        newly_quarantined_with_label_loss=0)
    f = r2.check_invariants(broken)
    assert len(f) >= 3, f
    assert any("domain partition" in x for x in f)
    assert any("row margin" in x for x in f)
    assert any("full reconciliation" in x for x in f)


# --------------------------------------------------------------------------- #
# Group 3 -- selection changes, quarantine states, and the JOINT table
# --------------------------------------------------------------------------- #
def test_selection_changes_are_decomposed_by_kind():
    ds = [
        delta(vid="replaced", legacy_repr=1, p6_repr=2),
        na(vid="removed", miss_p6=True, group_changed=True),
        na(vid="p6only", miss_legacy=True, group_changed=True),
    ]
    s_ = r2.summarize(ds)
    assert s_.selection_change_breakdown == {
        "legacy_representative_removed": 1,
        "p6_only_representative": 1,
        "replaced_by_a_different_row": 1,
    }
    assert s_.representative_selection_changed == 3


@pytest.mark.parametrize("lq,pq,changed", [
    (False, False, False),
    (True,  False, True),
    (False, True,  True),
    (True,  True,  False),
])
def test_all_four_quarantine_combinations_are_representable(lq, pq, changed):
    """The previous helper forced p6_quarantined = quar OR legacy_quar, so a
    LEGACY-ONLY quarantine was inexpressible -- and that is exactly the
    107-to-85 transition the real cohort shows."""
    d = delta(vid="V", legacy_quarantined=lq, p6_quarantined=pq)
    assert d.legacy_quarantined is lq and d.p6_quarantined is pq
    assert d.quarantine_changed is changed


def test_the_joint_table_pairs_each_label_change_with_ITS_population():
    """The defect independent marginals cannot detect: two cohorts with the same
    row totals and the same column totals but different joint structure."""
    x = r2.summarize([na(vid=f"n{i}", miss_p6=True, miss_legacy=True) for i in range(3)]
                     + [na(vid="c", miss_legacy=True, group_changed=True)])
    y = r2.summarize([na(vid="n0", miss_p6=True, miss_legacy=True, group_changed=True)]
                     + [na(vid=f"n{i}", miss_p6=True, miss_legacy=True) for i in (1, 2)]
                     + [na(vid="c", miss_legacy=True)])
    assert x.table_b.neither_side == y.table_b.neither_side == 3
    assert x.table_b.legacy_missing_only == y.table_b.legacy_missing_only == 1
    assert x.table_b.n_na1 == y.table_b.n_na1 == 1, "identical MARGINS"
    assert x.table_b.legacy_missing_only_changed == 1
    assert y.table_b.neither_changed == 1
    assert x.table_b != y.table_b, "but DIFFERENT joint structure -- now distinguishable"


def test_every_margin_is_derived_from_the_six_cells():
    b = r2.TableB(neither_unchanged=85, neither_changed=0,
                  legacy_missing_only_unchanged=5, legacy_missing_only_changed=17,
                  p6_missing_only_unchanged=0, p6_missing_only_changed=0)
    assert (b.neither_side, b.legacy_missing_only, b.p6_missing_only) == (85, 22, 0)
    assert (b.n_na0, b.n_na1, b.total) == (90, 17, 107)
    assert sum(b.cells) == b.total
    assert b.legacy_without_representative == 107
    assert b.p6_without_representative == 85


def test_the_direction_sentence_is_DERIVED_not_hard_coded():
    un = r2.TableB(85, 0, 5, 17, 0, 0)
    assert "UN-QUARANTINES" in un.quarantine_direction
    newly = r2.TableB(85, 0, 0, 0, 5, 17)
    assert "NEWLY QUARANTINES" in newly.quarantine_direction
    same = r2.TableB(10, 0, 3, 0, 3, 0)
    assert "UNCHANGED" in same.quarantine_direction


def test_negative_or_boolean_cells_are_refused():
    bad = r2.Reconciliation(
        table_a=r2.TableA(0, 0, 0, 0), table_b=r2.TableB(-1, 0, 0, 0, 0, 0),
        n_total=-1, representative_selection_changed=0,
        representative_label_changed=0, group_label_changed=0,
        quarantine_changed=0, trainability_changed=0,
        selection_change_breakdown={}, label_transitions={}, state_counts={},
        newly_quarantined_with_label_loss=0)
    assert any("non-negative integers" in x for x in r2.check_invariants(bad))


def test_the_availability_transitions_must_equal_the_quarantine_changes():
    s_ = r2.summarize([na(vid="a", miss_legacy=True), na(vid="b", miss_p6=True)])
    assert not r2.check_invariants(s_)
    wrong = r2.Reconciliation(
        table_a=r2.TableA(0, 0, 0, 0), table_b=r2.TableB(0, 0, 1, 0, 1, 0),
        n_total=2, representative_selection_changed=0,
        representative_label_changed=0, group_label_changed=0,
        quarantine_changed=99, trainability_changed=0,
        selection_change_breakdown={}, label_transitions={}, state_counts={},
        newly_quarantined_with_label_loss=0)
    assert any("symmetric difference" in x for x in r2.check_invariants(wrong))


# --------------------------------------------------------------------------- #
# Group 3b -- PolicyDelta refuses inconsistent records
# --------------------------------------------------------------------------- #
def test_an_applicable_comparison_cannot_be_None():
    with pytest.raises(ValueError, match="cannot be None"):
        delta(vid="V", repr_label_changed=None, legacy_repr=1, p6_repr=2,
              legacy_label=1, p6_label=1).__class__(
            variant_id="V", representative_row_changed=True,
            representative_row_label_changed=None,
            final_adjudicated_label_changed=False, trainability_changed=False,
            quarantine_changed=False, legacy_representative_row=1,
            p6_representative_row=2, legacy_representative_label=1,
            p6_representative_label=1, legacy_output_label=1, p6_output_label=1,
            p6_state="PATHOGENIC", legacy_quarantined=False, p6_quarantined=False)


def test_a_non_applicable_comparison_must_be_None():
    with pytest.raises(ValueError, match="must be None"):
        r2.PolicyDelta(
            variant_id="V", representative_row_changed=True,
            representative_row_label_changed=False,
            final_adjudicated_label_changed=False, trainability_changed=False,
            quarantine_changed=False, legacy_representative_row=None,
            p6_representative_row=2, legacy_representative_label=None,
            p6_representative_label=1, legacy_output_label=None, p6_output_label=1,
            p6_state="PATHOGENIC", legacy_quarantined=True, p6_quarantined=False)


def test_representative_row_changed_must_agree_with_the_row_identities():
    with pytest.raises(ValueError, match="disagrees with the row identities"):
        r2.PolicyDelta(
            variant_id="V", representative_row_changed=False,
            representative_row_label_changed=False,
            final_adjudicated_label_changed=False, trainability_changed=False,
            quarantine_changed=False, legacy_representative_row=1,
            p6_representative_row=2, legacy_representative_label=1,
            p6_representative_label=1, legacy_output_label=1, p6_output_label=1,
            p6_state="PATHOGENIC", legacy_quarantined=False, p6_quarantined=False)


def test_quarantine_changed_must_agree_with_the_quarantine_states():
    with pytest.raises(ValueError, match="disagrees with the states"):
        r2.PolicyDelta(
            variant_id="V", representative_row_changed=False,
            representative_row_label_changed=False,
            final_adjudicated_label_changed=False, trainability_changed=False,
            quarantine_changed=False, legacy_representative_row=1,
            p6_representative_row=1, legacy_representative_label=1,
            p6_representative_label=1, legacy_output_label=1, p6_output_label=1,
            p6_state="PATHOGENIC", legacy_quarantined=True, p6_quarantined=False)


# --------------------------------------------------------------------------- #
# Group 4 -- end to end on the synthetic cohort
# --------------------------------------------------------------------------- #
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

_FIXTURE_GOLDEN = {"golden": {
    "policy_table": {"P6": {"repr": 3, "label": 1, "quar": 1}},
    "p6_group_adjudicated_label_changes": 2}}


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    d = tmp_path_factory.mktemp("p6r2")
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
    pq.write_table(pa.table(cols), d / "cohort.parquet")
    return d


def _cfg(d: Path, *, golden: dict | None, tag: str):
    gp = d / f"golden_{tag}.json"
    if golden is not None:
        gp.write_text(json.dumps(golden), encoding="utf-8", newline="\n")
    sp = d / f"supersede_{tag}.txt"
    sp.write_text("ORIGINAL ARTIFACT\n", encoding="utf-8", newline="\n")
    return r2.ProbeConfig(repo_root=_REPO, raw_path=d / "cohort.parquet",
                          golden_path=gp, output_path=d / f"R2_{tag}.txt",
                          sidecar_path=d / f"R2_{tag}.json",
                          supersede_path=sp)


def _run(cfg):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = r2.run_probe(cfg)
    return rc, buf.getvalue()


def test_end_to_end_reproduces_the_golden_and_supersedes_once(workspace):
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="ok")
    rc, out = _run(cfg)
    assert rc == r2.EXIT_OK
    assert "reproduces the frozen reference" in out
    assert cfg.supersede_path.read_text().count("SUPERSEDED BY") == 1
    assert cfg.supersede_path.read_text().splitlines()[0] == "ORIGINAL ARTIFACT"


def test_superseding_is_idempotent(workspace):
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="twice")
    _run(cfg); _run(cfg)
    text = cfg.supersede_path.read_text()
    assert text.count("SUPERSEDED BY") == 1
    assert text.count("=" * 78) == 1, "the block must not be repeated"


def test_a_wrong_golden_fails_and_refuses_to_supersede(workspace):
    bad = {"golden": {"policy_table": {"P6": {"repr": 3, "label": 999, "quar": 1}},
                      "p6_group_adjudicated_label_changes": 2}}
    cfg = _cfg(workspace, golden=bad, tag="bad")
    rc, out = _run(cfg)
    assert rc == r2.EXIT_RECONCILIATION_FAILED
    assert "GOLDEN reproduction failed" in out
    assert "SUPERSEDED BY" not in cfg.supersede_path.read_text()


def test_a_missing_golden_never_claims_reproduction(workspace):
    """An earlier version printed 'reproduces the frozen reference EXACTLY' when
    the golden file was absent, because the golden checks were skipped and the
    failure list was therefore empty. A claim that passed because it never ran."""
    cfg = _cfg(workspace, golden=None, tag="nogolden")
    rc, out = _run(cfg)
    assert rc == r2.EXIT_ENVIRONMENT
    assert "NOT VERIFIED" in out
    assert "reproduces the frozen reference" not in out
    assert "SUPERSEDED BY" not in cfg.supersede_path.read_text()


def test_the_report_states_all_three_estimands_distinctly(workspace):
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="names")
    _, out = _run(cfg)
    assert "representative-row SELECTION changed" in out
    assert "representative-row LABEL changed" in out
    assert "group-adjudicated LABEL changed" in out
    assert "canonical" not in out.split("WHY R2 EXISTS")[0], \
        "the overloaded word must not head any count"


def test_the_fixture_exercises_the_not_applicable_population(workspace):
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="na")
    _, out = _run(cfg)
    assert "TABLE B" in out
    assert "EITHER side lacks a representative row" in out
    assert "neither side has a representative" in out
    assert "legacy missing, P6 present" in out
    assert "P6 missing, legacy present" in out
    assert "MUTUALLY EXCLUSIVE representative-availability" in out
    assert "every margin above is derived from the six cells" in out


def test_the_report_states_the_measured_quarantine_DIRECTION(workspace):
    """An earlier draft asserted that P6 newly quarantines and therefore always
    loses a binary label. The real cohort falsified it: legacy quarantines 107,
    P6 quarantines 85, and P6 never newly quarantines. The report must state the
    measurement, not the assumption."""
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="direction")
    _, out = _run(cfg)
    assert "DIRECTION OF THE QUARANTINE CHANGE, derived from the table" in out
    assert "newly quarantined by P6 AND lost a binary label" in out
    # On THIS fixture V8 is newly quarantined, so the derived sentence must say so
    # -- the opposite of the real cohort. That is the proof it is not hard-coded.
    assert "NEWLY QUARANTINES" in out


def test_the_strict_figure_is_not_described_as_both_having_a_LABEL(workspace):
    """29 of the 53 have a legacy row whose own label is None. Calling the strict
    count 'both sides had a label' would be false, and would be the same species
    of overloaded wording the artifact exists to remove."""
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="wording")
    _, out = _run(cfg)
    assert "both had a REPRESENTATIVE ROW" in out
    assert "both sides had a label" not in out


def test_the_overloading_note_follows_the_full_reconciliation(workspace):
    """It must not be wedged between the two reconciliation lines, orphaning the
    203. Ordering is part of whether evidence reads correctly."""
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="order")
    _, out = _run(cfg)
    assert out.index("n01 + n11 + n_na1") < out.index("THIRD OVERLOADED QUANTITY")


# --------------------------------------------------------------------------- #
# Group 5 -- the published figure is REPLAYED, not derived
# --------------------------------------------------------------------------- #
def test_the_replay_reproduces_the_published_definition_exactly():
    """The published count iterates the P6 map and uses base_vlabels.GET(v), so a
    variant with no legacy representative compares against None and is counted.
    Replaying that is the only way to reproduce it; deriving it from a stricter
    predicate gave 53 against a frozen 63 on the real cohort, and the gate caught it.
    """
    # rows 0..3; variants A (both kept), B (P6 only -- legacy has no representative)
    row_variant = ["A", "A", "B", "B"]
    row_label = [1, 0, 1, None]
    as_pub, bridge, ex = r2.replay_published_representative_label_changes(
        p6_kept={1, 3}, base_kept={0}, row_variant=row_variant, row_label=row_label)
    # A: legacy label 1 vs P6 label 0 -> counted, both present
    # B: absent from base_vlabels -> get() is None, P6 label None -> None == None, NOT counted
    assert as_pub == 1
    assert bridge == {"counted_both_labels_binary_and_differ": 1}


def test_the_bridge_names_a_comparison_against_a_MISSING_legacy_row():
    row_variant = ["A", "B"]
    row_label = [1, 1]
    as_pub, bridge, ex = r2.replay_published_representative_label_changes(
        p6_kept={0, 1}, base_kept={0}, row_variant=row_variant, row_label=row_label)
    assert as_pub == 1
    assert bridge == {"counted_but_legacy_had_NO_representative": 1}
    assert ex["counted_but_legacy_had_NO_representative"] == ["B"]


def test_the_bridge_distinguishes_a_present_row_carrying_no_label():
    row_variant = ["A"]
    row_label = [None]
    as_pub, bridge, _ = r2.replay_published_representative_label_changes(
        p6_kept={0}, base_kept=set(), row_variant=row_variant, row_label=row_label)
    assert as_pub == 0, "None vs absent-None compares equal; not a change"
    assert bridge == {}


def test_the_report_states_both_figures_and_the_bridge(workspace):
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="bridge")
    _, out = _run(cfg)
    assert "AS PUBLISHED on 2026-07-25" in out
    assert "the STRICT comparison" in out
    assert "THIRD OVERLOADED QUANTITY" in out
    assert "strict + definition bridge" in out


# --------------------------------------------------------------------------- #
# Group 6 -- the machine-readable sidecar
# --------------------------------------------------------------------------- #
def test_the_sidecar_comes_from_the_SAME_object_as_the_text(workspace):
    """The text is for a reader; the sidecar is for regression checks and audit
    tooling. Both are produced from one Reconciliation, so the displayed table and
    the persisted numbers cannot diverge. Reconstructing the sidecar independently
    would reintroduce the exact failure mode this artifact removes."""
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="sidecar")
    rc, out = _run(cfg)
    assert rc == r2.EXIT_OK
    d = json.loads(cfg.sidecar_path.read_text(encoding="utf-8"))

    assert d["schema_version"] == r2.SIDECAR_SCHEMA_VERSION
    assert d["golden_reproduced"] is True
    # the six cells and every derived margin agree with the rendered table
    tb = d["table_b"]
    assert sum(tb.values()) == d["derived"]["table_b_total"]
    assert d["derived"]["table_b_row_totals"] == [
        tb["neither_unchanged"] + tb["neither_changed"],
        tb["legacy_missing_only_unchanged"] + tb["legacy_missing_only_changed"],
        tb["p6_missing_only_unchanged"] + tb["p6_missing_only_changed"]]
    assert d["derived"]["table_b_column_totals"] == [
        tb["neither_unchanged"] + tb["legacy_missing_only_unchanged"]
        + tb["p6_missing_only_unchanged"],
        tb["neither_changed"] + tb["legacy_missing_only_changed"]
        + tb["p6_missing_only_changed"]]
    ta = d["table_a"]
    assert ta["n01"] + ta["n11"] + d["derived"]["table_b_column_totals"][1] \
        == d["derived"]["group_label_changed"]
    assert ta["total" if "total" in ta else "n00"] is not None  # shape sanity


def test_the_sidecar_is_strict_json_with_a_stable_key_order(workspace):
    cfg = _cfg(workspace, golden=_FIXTURE_GOLDEN, tag="strict")
    _run(cfg)
    raw = cfg.sidecar_path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw, parse_constant=lambda c: pytest.fail(f"bare {c} literal"))
    d = json.loads(raw)
    assert list(d) == sorted(d), "sort_keys gives a stable diff"


def test_the_direction_token_is_stable_and_separate_from_the_prose():
    """Prose may be reworded; the token may not."""
    assert r2.TableB(85, 0, 5, 17, 0, 0).quarantine_direction_token == "P6_UNQUARANTINES"
    assert r2.TableB(85, 0, 0, 0, 5, 17).quarantine_direction_token == "P6_NEWLY_QUARANTINES"
    assert r2.TableB(10, 0, 3, 0, 3, 0).quarantine_direction_token \
        == "QUARANTINE_CARDINALITY_UNCHANGED"


def test_a_failed_run_records_golden_reproduced_false(workspace):
    bad = {"golden": {"policy_table": {"P6": {"repr": 3, "label": 999, "quar": 1}},
                      "p6_group_adjudicated_label_changes": 2}}
    cfg = _cfg(workspace, golden=bad, tag="sidecarbad")
    rc, _ = _run(cfg)
    assert rc == r2.EXIT_RECONCILIATION_FAILED
    d = json.loads(cfg.sidecar_path.read_text(encoding="utf-8"))
    assert d["golden_reproduced"] is False, \
        "a sidecar from a failed run must not claim reproduction"
