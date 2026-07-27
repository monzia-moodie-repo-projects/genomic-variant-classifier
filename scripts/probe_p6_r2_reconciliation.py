"""P6 R2 reconciliation: one per-variant PolicyDelta pass, two tables, six invariants.

WHAT THIS CORRECTS
==================
`docs/measurements/CLEAN_COHORT_P6_AUDIT_2026-07-25.txt` publishes two counts that
its own prose calls by the same name:

    line 49  kept-row label changes vs P0 ... 63
    line 65  Comparing legacy P0 kept-row label vs P6 CANONICAL label
    line 87  P6 CANONICAL-label changes vs legacy : 63

Line 87 names 63 "canonical"; line 65 uses "canonical" for the basis of the 203.
Both underlying measurements are correct. Only the naming is overloaded, and an
incomplete local edit that merged them into a single row of 203 was worse -- it
deleted a real distinction -- so it was reverted rather than committed.

READING THE SOURCE FALSIFIED THE ORIGINAL REPAIR PLAN
-----------------------------------------------------
The plan of record required a 2x2 overlap satisfying

    n10 + n11 == 63     and     n01 + n11 == 203

The second equation cannot hold. Probe lines 490-491 sum over variants that HAVE a
P6 representative row; line 493 sums over EVERY variant; and `run_p6` selects no
representative at all for a quarantined variant. The two counts are computed over
DIFFERENT UNIVERSES, so a single 2x2 cannot reconcile them.

The golden capture measured that population exactly: 4,415,977 variants, of which
4,415,892 have a P6 representative row. The 85 that do not are precisely the
IRREDUCIBLE_CONFLICT set, and precisely the quarantined set.

WHAT REPLACES IT
----------------
`representative_row_label_changed` is NULLABLE. For a quarantined variant the
comparison is not False, it is undefined: no P6 representative label exists to
compare against. Encoding that as False would preserve the equation syntactically
by changing the meaning of the predicate, and would put two incompatible states --
"a label exists and is unchanged" and "no label exists" -- into one cell.

    Table A, over variants where the comparison APPLIES:

                                     final adjudicated label changed
                                          no        yes
        repr-row label unchanged         n00        n01
        repr-row label changed           n10        n11

    Table B, over variants where it does NOT apply:

        final adjudicated label unchanged   n_na0
        final adjudicated label changed     n_na1

    n10 + n11         == 63       (conditional reconciliation)
    n01 + n11 + n_na1 == 203      (full reconciliation)

THREE ESTIMANDS, NOT TWO
------------------------
The file already contains three related, non-substitutable measurements. The R2
artifact reports all three under names that cannot be confused:

    representative-row SELECTION changed   232   row availability/choice
    representative-row LABEL changed        63   only where a P6 representative exists
    group-adjudicated LABEL changed        203   over the full variant universe

ARCHITECTURE
------------
Pure logic depends on neither paths nor the command line:

    load_inputs(config)      -> LoadedCohort
    compute_policy_deltas()  -> tuple[PolicyDelta, ...]     ONE per-variant pass
    summarize()              -> Reconciliation
    render_report()          -> str

so the reconciliation can be tested on hand-computed fixtures that never touch a
filesystem. The adjudication functions are IMPORTED from
`probe_clean_cohort_p6_2026-07-25.py` rather than reimplemented: that probe is the
golden reference, and a second copy of the policy would be a second thing to drift.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

REPO_DEFAULT = Path(r"C:\Projects\genomic-variant-classifier")
PROBE_NAME = "probe_clean_cohort_p6_2026-07-25.py"

EXIT_OK = 0
EXIT_RECONCILIATION_FAILED = 1
EXIT_ENVIRONMENT = 2


# --------------------------------------------------------------------------- #
# Layer 2 -- typed configuration. Exactly one place where defaults live.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbeConfig:
    """Every filesystem location this program touches, resolved once.

    The reconciliation never reads a module constant, an environment variable or
    a command-line argument. It receives this object. That is what makes the
    dependency explicit and the logic testable against a temporary workspace.
    """

    repo_root: Path
    raw_path: Path
    golden_path: Path
    output_path: Path
    sidecar_path: Path
    supersede_path: Path

    @staticmethod
    def default(repo_root: Path = REPO_DEFAULT) -> "ProbeConfig":
        m = repo_root / "docs" / "measurements"
        return ProbeConfig(
            repo_root=repo_root,
            raw_path=repo_root / "data" / "processed" / "clinvar_grch38.parquet",
            golden_path=m / "CLEAN_COHORT_P6_GOLDEN_2026-07-26.json",
            output_path=m / "CLEAN_COHORT_P6_AUDIT_2026-07-25_R2.txt",
            sidecar_path=m / "CLEAN_COHORT_P6_AUDIT_2026-07-25_R2.json",
            supersede_path=m / "CLEAN_COHORT_P6_AUDIT_2026-07-25.txt",
        )


# --------------------------------------------------------------------------- #
# Layer 1 -- the immutable per-variant record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PolicyDelta:
    """What changed for ONE variant between the legacy policy and P6.

    Four total booleans and ONE NULLABLE comparison. The nullable one is not a
    convenience: `representative_row_label_changed` is undefined when P6 selects
    no representative row, and saying False there would assert something untrue.

    The evidence that produced each delta travels with it, so the 2x2, the
    transition matrices and the acceptance counts are all derived from the same
    immutable objects and cannot accidentally count different universes.
    """

    variant_id: str

    representative_row_changed: bool
    representative_row_label_changed: Optional[bool]
    final_adjudicated_label_changed: bool
    trainability_changed: bool
    quarantine_changed: bool

    legacy_representative_row: Optional[int]
    p6_representative_row: Optional[int]
    legacy_representative_label: Optional[int]
    p6_representative_label: Optional[int]
    legacy_output_label: Optional[int]
    p6_output_label: Optional[int]
    p6_state: Optional[str]
    legacy_quarantined: bool
    p6_quarantined: bool

    def __post_init__(self) -> None:
        """Make an internally inconsistent per-variant record UNCONSTRUCTABLE.

        Every table is derived from this collection, so a malformed record here
        propagates into every published number. Rejecting it at construction is
        the only place the check cannot be forgotten.
        """
        applies = (self.legacy_representative_row is not None
                   and self.p6_representative_row is not None)
        if applies and self.representative_row_label_changed is None:
            raise ValueError(
                f"{self.variant_id}: both representative rows exist, so the "
                "representative-label comparison APPLIES and cannot be None.")
        if not applies and self.representative_row_label_changed is not None:
            raise ValueError(
                f"{self.variant_id}: a representative row is missing, so the "
                "comparison is undefined and must be None, not "
                f"{self.representative_row_label_changed!r}.")

        expected_row_changed = (self.legacy_representative_row
                                != self.p6_representative_row)
        if self.representative_row_changed != expected_row_changed:
            raise ValueError(
                f"{self.variant_id}: representative_row_changed="
                f"{self.representative_row_changed} disagrees with the row "
                f"identities {self.legacy_representative_row!r} vs "
                f"{self.p6_representative_row!r}.")

        expected_quar_changed = self.legacy_quarantined != self.p6_quarantined
        if self.quarantine_changed != expected_quar_changed:
            raise ValueError(
                f"{self.variant_id}: quarantine_changed={self.quarantine_changed} "
                f"disagrees with the states legacy={self.legacy_quarantined} "
                f"p6={self.p6_quarantined}.")

    @property
    def representative_label_comparison_applies(self) -> bool:
        """Derived, never stored twice: the comparison applies exactly when both
        sides have a representative row to take a label from."""
        return (self.legacy_representative_row is not None
                and self.p6_representative_row is not None)


@dataclass(frozen=True)
class TableA:
    """The 2x2 overlap, restricted to variants where the comparison applies."""

    n00: int
    n01: int
    n10: int
    n11: int

    @property
    def total(self) -> int:
        return self.n00 + self.n01 + self.n10 + self.n11


@dataclass(frozen=True)
class TableB:
    """Joint 3x2 table for the non-applicable representative-label comparisons.

    SIX CELLS ARE STORED; every published number is DERIVED from them. Storing
    the row and column marginals independently -- as an earlier version did --
    lets correct totals be paired with the wrong population: a cohort whose 17
    label changes all fell among legacy-missing variants and one whose 17 all
    fell among neither-side variants serialise identically. Correct margins,
    different science, and no invariant over those margins can tell them apart.

    THE THREE ROWS ARE MUTUALLY EXCLUSIVE representative-availability transitions:

      neither_side          neither the legacy policy nor P6 selected a
                            representative row;
      legacy_missing_only   the legacy policy selected none, while P6 selected one;
      p6_missing_only       the legacy policy selected one, while P6 selected none.

    They sum exactly to the Table B universe. The two columns are whether the
    group-adjudicated label changed.
    """

    neither_unchanged: int
    neither_changed: int
    legacy_missing_only_unchanged: int
    legacy_missing_only_changed: int
    p6_missing_only_unchanged: int
    p6_missing_only_changed: int

    @property
    def cells(self) -> tuple:
        return (self.neither_unchanged, self.neither_changed,
                self.legacy_missing_only_unchanged, self.legacy_missing_only_changed,
                self.p6_missing_only_unchanged, self.p6_missing_only_changed)

    # --- row totals: representative-availability transitions -----------------
    @property
    def neither_side(self) -> int:
        return self.neither_unchanged + self.neither_changed

    @property
    def legacy_missing_only(self) -> int:
        return self.legacy_missing_only_unchanged + self.legacy_missing_only_changed

    @property
    def p6_missing_only(self) -> int:
        return self.p6_missing_only_unchanged + self.p6_missing_only_changed

    # --- column totals: group-adjudicated label change -----------------------
    @property
    def n_na0(self) -> int:
        return (self.neither_unchanged + self.legacy_missing_only_unchanged
                + self.p6_missing_only_unchanged)

    @property
    def n_na1(self) -> int:
        return (self.neither_changed + self.legacy_missing_only_changed
                + self.p6_missing_only_changed)

    @property
    def total(self) -> int:
        return self.neither_side + self.legacy_missing_only + self.p6_missing_only

    # --- derived quarantine cardinalities and direction ----------------------
    @property
    def legacy_without_representative(self) -> int:
        return self.neither_side + self.legacy_missing_only

    @property
    def p6_without_representative(self) -> int:
        return self.neither_side + self.p6_missing_only

    @property
    def quarantine_direction_token(self) -> str:
        """A stable identifier for machine consumption, beside the human sentence.
        Prose may be reworded; this may not."""
        lg, p6 = self.legacy_without_representative, self.p6_without_representative
        if lg > p6:
            return "P6_UNQUARANTINES"
        if p6 > lg:
            return "P6_NEWLY_QUARANTINES"
        return "QUARANTINE_CARDINALITY_UNCHANGED"

    @property
    def quarantine_direction(self) -> str:
        """DERIVED, never hard-coded. The renderer also runs on synthetic cohorts
        where the direction differs, so the sentence must come from the table."""
        lg, p6 = self.legacy_without_representative, self.p6_without_representative
        if lg > p6:
            return (f"P6 UN-QUARANTINES: the legacy policy withholds a representative "
                    f"from {lg:,} variants and P6 from {p6:,}")
        if p6 > lg:
            return (f"P6 NEWLY QUARANTINES: the legacy policy withholds a representative "
                    f"from {lg:,} variants and P6 from {p6:,}")
        return (f"quarantine cardinality UNCHANGED at {lg:,}, though the membership "
                f"may still differ")


@dataclass(frozen=True)
class Reconciliation:
    table_a: TableA
    table_b: TableB
    n_total: int
    representative_selection_changed: int
    representative_label_changed: int
    group_label_changed: int
    quarantine_changed: int
    trainability_changed: int
    selection_change_breakdown: dict
    label_transitions: dict
    state_counts: dict
    newly_quarantined_with_label_loss: int
    representative_label_changed_as_published: int = 0
    definition_bridge: dict = field(default_factory=dict)
    failures: tuple = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Layer 1 -- pure computation
# --------------------------------------------------------------------------- #
def compute_policy_deltas(
    *,
    groups: dict,
    base_kept: set,
    base_quar: set,
    p6_kept: set,
    p6_quar: set,
    p6_labels: dict,
    p6_states: dict,
    row_variant: Sequence[str],
    row_label: Sequence[Optional[int]],
) -> tuple:
    """ONE pass over the variants, producing one immutable record each.

    Every downstream table is derived from the returned collection, so no two
    tables can silently describe different populations.
    """
    base_repr = {}
    for i in base_kept:
        base_repr[row_variant[i]] = i
    p6_repr = {}
    for i in p6_kept:
        p6_repr[row_variant[i]] = i

    deltas = []
    for v in groups:
        lr = base_repr.get(v)
        pr = p6_repr.get(v)
        ll = row_label[lr] if lr is not None else None
        pl = row_label[pr] if pr is not None else None

        legacy_out = ll                       # legacy label IS the kept row's label
        p6_out = p6_labels.get(v)             # P6 adjudicates at the group level

        applies = lr is not None and pr is not None
        deltas.append(PolicyDelta(
            variant_id=v,
            representative_row_changed=(lr != pr),
            representative_row_label_changed=((ll != pl) if applies else None),
            final_adjudicated_label_changed=(legacy_out != p6_out),
            trainability_changed=((legacy_out is None) != (p6_out is None)),
            quarantine_changed=((v in base_quar) != (v in p6_quar)),
            legacy_representative_row=lr,
            p6_representative_row=pr,
            legacy_representative_label=ll,
            p6_representative_label=pl,
            legacy_output_label=legacy_out,
            p6_output_label=p6_out,
            p6_state=p6_states.get(v),
            legacy_quarantined=(v in base_quar),
            p6_quarantined=(v in p6_quar),
        ))
    return tuple(deltas)


def replay_published_representative_label_changes(
    *, p6_kept, base_kept, row_variant, row_label) -> tuple:
    """Recompute the PUBLISHED 63 by replaying the original computation exactly.

    probe_clean_cohort_p6_2026-07-25.py lines 464 and 514-517 build

        base_vlabels     = {variant: label of its KEPT LEGACY row}
        p6_reprrow_label = {variant: label of its KEPT P6 row}

    and then count, over the SECOND map's keys,

        base_vlabels.get(v) != p6_reprrow_label.get(v)

    The `.get()` is load-bearing: a variant absent from `base_vlabels` compares as
    None. So the published figure counts a P6 representative label differing from
    the ABSENCE of a legacy representative as a "label change" -- which is a
    comparison against a missing row, not against another label.

    This function replays that computation rather than deriving it from
    PolicyDelta, so the frozen counter is reproduced BY CONSTRUCTION. The stricter
    quantity -- both sides present -- is computed separately from the deltas, and
    `bridge` classifies every variant where the two definitions disagree, so the
    difference is measured rather than argued about.
    """
    base_vlabels = {}
    for i in base_kept:
        base_vlabels[row_variant[i]] = row_label[i]
    p6_reprrow_label = {}
    for i in p6_kept:
        p6_reprrow_label[row_variant[i]] = row_label[i]

    as_published = 0
    bridge = Counter()
    examples = {}
    for v, pl in p6_reprrow_label.items():
        present = v in base_vlabels
        ll = base_vlabels.get(v)
        if ll != pl:
            as_published += 1
            if not present:
                bridge["counted_but_legacy_had_NO_representative"] += 1
                examples.setdefault("counted_but_legacy_had_NO_representative", []).append(v)
            elif ll is None:
                bridge["counted_legacy_row_present_but_label_None"] += 1
                examples.setdefault("counted_legacy_row_present_but_label_None", []).append(v)
            elif pl is None:
                bridge["counted_p6_label_None"] += 1
                examples.setdefault("counted_p6_label_None", []).append(v)
            else:
                bridge["counted_both_labels_binary_and_differ"] += 1
    for k in list(examples):
        examples[k] = sorted(examples[k])[:5]
    return as_published, dict(sorted(bridge.items())), examples


def summarize(deltas: Sequence[PolicyDelta], *,
              as_published: int = 0, bridge: Optional[dict] = None) -> Reconciliation:
    """Derive every table from the single collection, then check it against itself."""
    n00 = n01 = n10 = n11 = 0
    jb = Counter()
    sel_break = Counter()
    transitions = Counter()
    states = Counter()
    newly_quar_label_loss = 0

    for d in deltas:
        states[d.p6_state] += 1
        transitions[(d.legacy_output_label, d.p6_output_label)] += 1

        if d.representative_row_changed:
            if d.legacy_representative_row is None:
                sel_break["p6_only_representative"] += 1
            elif d.p6_representative_row is None:
                sel_break["legacy_representative_removed"] += 1
            else:
                sel_break["replaced_by_a_different_row"] += 1

        if d.representative_row_label_changed is None:
            # SIX JOINT CELLS. Row and column marginals are DERIVED from these,
            # never stored beside them: independent marginals let correct totals
            # be paired with the wrong population, which no invariant over those
            # marginals can detect.
            miss_p6 = d.p6_representative_row is None
            miss_lg = d.legacy_representative_row is None
            chg = d.final_adjudicated_label_changed
            if miss_p6 and miss_lg:
                jb["neither_changed" if chg else "neither_unchanged"] += 1
            elif miss_lg:
                jb["legacy_missing_only_changed" if chg
                   else "legacy_missing_only_unchanged"] += 1
            else:
                jb["p6_missing_only_changed" if chg
                   else "p6_missing_only_unchanged"] += 1
        elif d.representative_row_label_changed:
            if d.final_adjudicated_label_changed:
                n11 += 1
            else:
                n10 += 1
        else:
            if d.final_adjudicated_label_changed:
                n01 += 1
            else:
                n00 += 1

        if (d.p6_quarantined and not d.legacy_quarantined
                and d.legacy_output_label is not None):
            newly_quar_label_loss += 1

    a = TableA(n00, n01, n10, n11)
    b = TableB(
        neither_unchanged=jb["neither_unchanged"],
        neither_changed=jb["neither_changed"],
        legacy_missing_only_unchanged=jb["legacy_missing_only_unchanged"],
        legacy_missing_only_changed=jb["legacy_missing_only_changed"],
        p6_missing_only_unchanged=jb["p6_missing_only_unchanged"],
        p6_missing_only_changed=jb["p6_missing_only_changed"],
    )
    return Reconciliation(
        table_a=a, table_b=b, n_total=len(deltas),
        representative_selection_changed=sum(1 for d in deltas if d.representative_row_changed),
        representative_label_changed=sum(1 for d in deltas if d.representative_row_label_changed is True),
        group_label_changed=sum(1 for d in deltas if d.final_adjudicated_label_changed),
        quarantine_changed=sum(1 for d in deltas if d.quarantine_changed),
        trainability_changed=sum(1 for d in deltas if d.trainability_changed),
        selection_change_breakdown=dict(sorted(sel_break.items())),
        label_transitions={f"{k[0]}->{k[1]}": v for k, v in sorted(transitions.items(), key=lambda kv: str(kv[0]))},
        state_counts=dict(sorted(states.items(), key=lambda kv: str(kv[0]))),
        newly_quarantined_with_label_loss=newly_quar_label_loss,
        representative_label_changed_as_published=as_published,
        definition_bridge=dict(bridge or {}),
    )


def check_invariants(r: Reconciliation, golden: Optional[dict] = None) -> list:
    """Every acceptance assertion, returned rather than raised so ALL failures show.

    A single `assert` reports the first violation and hides the rest, which is the
    wrong shape for evidence: if three counts disagree, a reader needs all three.
    """
    f = []
    a, b = r.table_a, r.table_b

    if a.total + b.total != r.n_total:
        f.append(f"domain partition: {a.total} applicable + {b.total} not-applicable "
                 f"!= {r.n_total} total")
    if a.n10 + a.n11 != r.representative_label_changed:
        f.append(f"table A row margin: n10+n11={a.n10 + a.n11} != "
                 f"representative-label changes {r.representative_label_changed}")
    if any((not isinstance(x, int)) or isinstance(x, bool) or x < 0 for x in b.cells):
        f.append(f"table B cells must be non-negative integers; got {b.cells}")
    if sum(b.cells) != b.total:
        f.append(f"table B cells {b.cells} sum to {sum(b.cells)}, not {b.total}")
    if b.n_na0 + b.n_na1 != b.total:
        f.append(f"table B columns {b.n_na0}+{b.n_na1} != universe {b.total}")
    if b.total != r.n_total - a.total:
        f.append(f"table B universe {b.total} != total {r.n_total} minus "
                 f"table A {a.total}")
    if b.legacy_missing_only + b.p6_missing_only != r.quarantine_changed:
        f.append(f"availability transitions {b.legacy_missing_only}+"
                 f"{b.p6_missing_only} != quarantine changes "
                 f"{r.quarantine_changed} (symmetric difference)")
    if a.n01 + a.n11 + b.n_na1 != r.group_label_changed:
        f.append(f"full reconciliation: n01+n11+n_na1={a.n01 + a.n11 + b.n_na1} != "
                 f"group-label changes {r.group_label_changed}")

    if golden:
        g = golden.get("golden", {})
        p6 = g.get("policy_table", {}).get("P6", {})
        for name, got, want in (
            ("representative-row SELECTION changed", r.representative_selection_changed, p6.get("repr")),
            ("representative-row LABEL changed (as published)",
             r.representative_label_changed_as_published, p6.get("label")),
            ("group-adjudicated LABEL changed", r.group_label_changed,
             g.get("p6_group_adjudicated_label_changes")),
            ("quarantine changed", r.quarantine_changed, p6.get("quar")),
        ):
            if want is not None and got != want:
                f.append(f"GOLDEN reproduction failed -- {name}: recomputed {got}, "
                         f"frozen reference {want}")
    return f


# --------------------------------------------------------------------------- #
# Layer 1 -- rendering
# --------------------------------------------------------------------------- #
def render_report(r: Reconciliation, *, cohort: str, rows: int,
                  golden_available: bool, failures: Sequence[str]) -> str:
    L = []
    def e(s=""):
        L.append(s)

    e("=" * 78)
    e("STEP 1b CLEAN_COHORT P6 EVIDENCE-ADJUDICATION AUDIT -- R2 (read-only)")
    e("SUPERSEDES CLEAN_COHORT_P6_AUDIT_2026-07-25.txt")
    e("=" * 78)
    e()
    e(f"  cohort: {cohort}   rows: {rows:,}   variants: {r.n_total:,}")
    e()
    e("  WHY R2 EXISTS. The 2026-07-25 artifact used the word 'canonical' for two")
    e("  different estimands: line 87 applied it to 63, while line 65 applied it to")
    e("  the basis of 203. Both measurements were correct; the naming was not. R2")
    e("  renames them so they cannot be confused and adds the overlap that shows how")
    e("  they relate. No policy, threshold or rule has changed.")
    e()
    e("=" * 78)
    e("THREE ESTIMANDS -- RELATED, NOT SUBSTITUTABLE")
    e("-" * 78)
    e(f"  representative-row SELECTION changed : {r.representative_selection_changed:>8,}"
      f"   over all {r.n_total:,} variants")
    for k, v in r.selection_change_breakdown.items():
        e(f"      {k:<34} {v:>8,}")
    e(f"  representative-row LABEL changed     : {r.representative_label_changed_as_published:>8,}"
      f"   AS PUBLISHED on 2026-07-25")
    e(f"      of which both had a REPRESENTATIVE ROW : {r.representative_label_changed:>8,}"
      f"  the STRICT comparison")
    e("      (not 'both had a label' -- most of these compare a binary label against")
    e("       a representative row whose own label is None)")
    for k, v in r.definition_bridge.items():
        e(f"      {k:<48} {v:>8,}")
    e(f"  group-adjudicated LABEL changed      : {r.group_label_changed:>8,}"
      f"   over all {r.n_total:,} variants")
    e()
    e("  These have DIFFERENT DENOMINATORS. A representative-row label comparison is")
    e("  undefined for a variant P6 quarantines, because no P6 representative row")
    e("  exists to compare against. That is why the tables below are split.")
    e()
    e("=" * 78)
    e("TABLE A -- OVERLAP, where the representative-label comparison APPLIES")
    e("-" * 78)
    e(f"  universe: {r.table_a.total:,} variants with BOTH a legacy and a P6 representative row")
    e()
    e("                                      group-adjudicated label changed")
    e("                                            no            yes")
    e(f"  representative-row label unchanged  {r.table_a.n00:>12,}   {r.table_a.n01:>12,}")
    e(f"  representative-row label changed    {r.table_a.n10:>12,}   {r.table_a.n11:>12,}")
    e()
    e(f"  n10 + n11 = {r.table_a.n10 + r.table_a.n11:,}  -- the STRICT representative-row label changes")
    e(f"  n11       = {r.table_a.n11:,}  -- BOTH changed; the overlap the prior artifact never reported")
    e()
    e("=" * 78)
    e("TABLE B -- the representative-label comparison does NOT apply")
    e("-" * 78)
    e(f"  universe: {r.table_b.total:,} variants where EITHER side lacks a representative row")
    e()
    e("                                            group-adjudicated label changed")
    e("                                                 no        yes      total")
    b = r.table_b
    e(f"  neither side has a representative      {b.neither_unchanged:>10,} {b.neither_changed:>10,} {b.neither_side:>10,}")
    e(f"  legacy missing, P6 present             {b.legacy_missing_only_unchanged:>10,} {b.legacy_missing_only_changed:>10,} {b.legacy_missing_only:>10,}")
    e(f"  P6 missing, legacy present             {b.p6_missing_only_unchanged:>10,} {b.p6_missing_only_changed:>10,} {b.p6_missing_only:>10,}")
    e(f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}")
    e(f"  total                                  {b.n_na0:>10,} {b.n_na1:>10,} {b.total:>10,}")
    e()
    e("  The three rows are MUTUALLY EXCLUSIVE representative-availability")
    e("  transitions and every margin above is derived from the six cells, so no")
    e("  count can be paired with the wrong population.")
    e()
    e("  DIRECTION OF THE QUARANTINE CHANGE, derived from the table:")
    e(f"    {b.quarantine_direction}.")
    e(f"    variants whose group label changed while P6 withheld a representative: "
      f"{b.neither_changed + b.p6_missing_only_changed:,}")
    e(f"    newly quarantined by P6 AND lost a binary label : "
      f"{r.newly_quarantined_with_label_loss:,}")
    e()
    e("  The mechanism is select_repr_row's conflict branch, which keeps a row only")
    e("  when the legacy best tier holds exactly one class AND that class is binary.")
    e("  A best tier holding only non-binary rows gives classes == {None}: one class,")
    e("  but not a subset of {0, 1}, so the legacy policy withholds a representative.")
    e("  The unified best tier is always a superset (the tier map merges, e.g. legacy")
    e("  4 -> unified 3), so it can include a binary row and give P6 a label where")
    e("  legacy had none.")
    e()
    e("=" * 78)
    e("RECONCILIATION")
    e("-" * 78)
    e(f"  n00 + n01 + n10 + n11            = {r.table_a.total:,}   (applicable)")
    e(f"  n_na0 + n_na1                    = {r.table_b.total:,}   (not applicable)")
    e(f"  applicable + not applicable      = {r.n_total:,}   (all variants)")
    e()
    e(f"  n10 + n11                        = {r.table_a.n10 + r.table_a.n11:,}"
      f"   == STRICT representative-row label changes")
    e(f"  strict + definition bridge       = {r.representative_label_changed_as_published:,}"
      f"   == the figure published as 63 on 2026-07-25")
    e(f"  n01 + n11 + n_na1                = {r.table_a.n01 + r.table_a.n11 + r.table_b.n_na1:,}"
      f"   == group-adjudicated label changes")
    e()
    e("  THIRD OVERLOADED QUANTITY. The published figure counts a P6 representative")
    e("  label differing from the ABSENCE of a legacy representative as a 'label")
    e("  change'. That is a comparison against a missing row, not against a label.")
    e("  Both numbers are reported above; neither is discarded.")
    e()
    e("  NOTE. The repair plan of record required n01 + n11 == 203. That equation")
    e("  assumed the two counts shared a universe. They do not, and it is replaced by")
    e("  the reconciliation above. See the session document dated 2026-07-26.")
    e()
    e("=" * 78)
    e("LABEL TRANSITIONS (legacy output label -> P6 output label)")
    e("-" * 78)
    for k, v in sorted(r.label_transitions.items(), key=lambda kv: -kv[1]):
        e(f"  {k:<20} {v:>12,}")
    e()
    e("P6 CANONICAL STATE COUNTS")
    e("-" * 78)
    for k, v in sorted(r.state_counts.items(), key=lambda kv: -kv[1]):
        e(f"  {str(k):<28} {v:>12,}")
    e()
    e("=" * 78)
    e("GOLDEN REPRODUCTION")
    e("-" * 78)
    if not golden_available:
        # NEVER claim reproduction when nothing was compared. An earlier version
        # printed "reproduces the frozen reference EXACTLY" when the golden file
        # was absent, because the golden checks were skipped and the failure list
        # was therefore empty -- a claim that passed because it never ran.
        e("  *** NOT VERIFIED *** the golden reference was not found, so no counter")
        e("  was compared against it. This report is NOT evidence of reproduction.")
    elif failures:
        e("  *** FAILED ***")
        for x in failures:
            e(f"    {x}")
    else:
        e("  Every counter recomputed here reproduces the frozen reference in")
        e("  CLEAN_COHORT_P6_GOLDEN_2026-07-26.json EXACTLY.")
    e()
    return "\n".join(L) + "\n"


SIDECAR_SCHEMA_VERSION = 1


def serialize_json(r: Reconciliation, *, n_total: int, rows: int, cohort: str,
                   golden_reproduced: bool) -> str:
    """Machine-readable evidence, from the SAME Reconciliation the text renders.

    The text report is for a human reader; this is for exact regression checks,
    downstream audit tooling and future comparison. Both are produced from one
    object, so the displayed table and the persisted numbers cannot diverge --
    reconstructing these values independently would reintroduce the very failure
    mode this artifact exists to remove.

    Strict serialization: sort_keys for a stable diff, allow_nan=False because a
    non-finite counter is a computation that failed silently.
    """
    b, a = r.table_b, r.table_a
    payload = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "cohort": cohort,
        "rows": rows,
        "n_total": n_total,
        "table_a": {"n00": a.n00, "n01": a.n01, "n10": a.n10, "n11": a.n11},
        "table_b": {
            "neither_unchanged": b.neither_unchanged,
            "neither_changed": b.neither_changed,
            "legacy_missing_only_unchanged": b.legacy_missing_only_unchanged,
            "legacy_missing_only_changed": b.legacy_missing_only_changed,
            "p6_missing_only_unchanged": b.p6_missing_only_unchanged,
            "p6_missing_only_changed": b.p6_missing_only_changed,
        },
        "derived": {
            "table_a_total": a.total,
            "table_b_row_totals": [b.neither_side, b.legacy_missing_only, b.p6_missing_only],
            "table_b_column_totals": [b.n_na0, b.n_na1],
            "table_b_total": b.total,
            "legacy_without_representative": b.legacy_without_representative,
            "p6_without_representative": b.p6_without_representative,
            "quarantine_direction": b.quarantine_direction_token,
            "representative_selection_changed": r.representative_selection_changed,
            "representative_label_changed_strict": r.representative_label_changed,
            "representative_label_changed_as_published":
                r.representative_label_changed_as_published,
            "group_label_changed": r.group_label_changed,
            "quarantine_changed": r.quarantine_changed,
        },
        "definition_bridge": dict(r.definition_bridge),
        "label_transitions": dict(r.label_transitions),
        "state_counts": {str(k): v for k, v in r.state_counts.items()},
        "golden_reproduced": bool(golden_reproduced),
    }
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- #
# Layer 2 -- loading
# --------------------------------------------------------------------------- #
def _load_probe(config: ProbeConfig):
    src = config.repo_root / "scripts" / PROBE_NAME
    if not src.is_file():
        raise SystemExit(f"ABORT: probe not found at {src}")
    spec = importlib.util.spec_from_file_location("_p6probe_r2", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_p6probe_r2"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_probe(config: ProbeConfig) -> int:
    import pyarrow.parquet as pq

    p6 = _load_probe(config)
    if not config.raw_path.is_file():
        raise SystemExit(f"ABORT: cohort not found at {config.raw_path}")

    tbl = pq.read_table(config.raw_path)
    have = set(tbl.column_names)
    vids = tbl.column("variant_id").to_pylist()
    sig_col = "clinical_sig" if "clinical_sig" in have else "clinvar_clinical_significance"
    sigs = tbl.column(sig_col).to_pylist()

    if "metadata" in have:
        meta = tbl.column("metadata").to_pylist()
        rev = [(m or {}).get("review_status") for m in meta]
    else:
        rev = tbl.column("review_status").to_pylist()

    kf = tuple(f for f in ("source_id", "clinical_sig", "ref", "alt", "consequence",
                           "gene_symbol") if f in have)
    rowdicts = [{f: tbl.column(f)[i].as_py() for f in kf} for i in range(tbl.num_rows)]

    labels = [p6.norm_label(x) for x in sigs]
    leg_tiers = [p6.legacy_tier(r) for r in rev]
    uni_pairs = [p6.unified_tier(r) for r in rev]
    uni_tiers = [p6.SENTINEL_UNMATCHED if t is None else t for t, _ in uni_pairs]

    groups = {}
    for i, v in enumerate(vids):
        groups.setdefault(v, []).append(i)

    order = list(range(len(vids)))
    # run_single_row_policy returns (kept, quar) -- two values, verified against
    # the call site at probe line 452. Legacy P0 uses the LEGACY tier map.
    base_kept, base_quar = p6.run_single_row_policy(
        vids, leg_tiers, labels, rev, rowdicts, kf, "P0", order, groups)
    p6_kept, p6_quar, p6_labels, p6_states = p6.run_p6(
        vids, uni_tiers, sigs, rev, rowdicts, kf, order, groups)

    deltas = compute_policy_deltas(
        groups=groups, base_kept=base_kept, base_quar=base_quar,
        p6_kept=p6_kept, p6_quar=p6_quar, p6_labels=p6_labels, p6_states=p6_states,
        row_variant=vids, row_label=labels)

    as_pub, bridge, bridge_examples = replay_published_representative_label_changes(
        p6_kept=p6_kept, base_kept=base_kept, row_variant=vids, row_label=labels)
    recon = summarize(deltas, as_published=as_pub, bridge=bridge)
    if bridge_examples:
        print("\n  DEFINITION BRIDGE -- example variants per category:")
        for k, vs in sorted(bridge_examples.items()):
            print(f"    {k}: {vs}")
    golden = None
    if config.golden_path.is_file():
        golden = json.loads(config.golden_path.read_text(encoding="utf-8"))
    failures = check_invariants(recon, golden)

    golden_available = golden is not None
    text = render_report(recon, cohort=config.raw_path.name, rows=tbl.num_rows,
                         golden_available=golden_available, failures=failures)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(text, encoding="utf-8", newline="\n")
    sidecar = serialize_json(recon, n_total=len(deltas), rows=tbl.num_rows,
                             cohort=config.raw_path.name,
                             golden_reproduced=(golden_available and not failures))
    config.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    config.sidecar_path.write_text(sidecar, encoding="utf-8", newline="\n")
    print(text)
    print(f"WROTE {config.output_path}")
    print(f"WROTE {config.sidecar_path}")

    if failures:
        print("\nRECONCILIATION FAILED:", file=sys.stderr)
        for x in failures:
            print(f"  {x}", file=sys.stderr)
        return EXIT_RECONCILIATION_FAILED

    if not golden_available:
        # An unverified run must not supersede verified evidence.
        print(f"\nABORT: golden reference not found at {config.golden_path}. The R2 "
              "artifact was written but the original was NOT superseded, because "
              "nothing was verified against the frozen counters.", file=sys.stderr)
        return EXIT_ENVIRONMENT

    _append_supersession_pointer(config)
    return EXIT_OK


def _append_supersession_pointer(config: ProbeConfig) -> None:
    """Append a pointer to the superseded artifact. Its numbers are NOT edited:
    provenance is preserved by pointing forward, never by rewriting history."""
    if not config.supersede_path.is_file():
        return
    marker = "SUPERSEDED BY"
    cur = config.supersede_path.read_text(encoding="utf-8")
    if marker in cur:
        return
    # Built as a LIST OF LINES, never as a mixed expression. Writing
    #     "\n" + "=" * 78 + "\n" f"..." "-" * 78 + ...
    # concatenates the ADJACENT LITERALS first and then applies * 78 to the joined
    # string, which appended this block seventy-eight times. Measured 2026-07-26.
    block = [
        "",
        "=" * 78,
        f"{marker} {config.output_path.name} (2026-07-26).",
        "-" * 78,
        "  The counts in this file are CORRECT and are left exactly as measured.",
        "  What was wrong was the NAMING: the word 'canonical' was used above for two",
        "  different estimands -- the representative-row label change (63) and the",
        "  group-adjudicated label change (203). The successor renames them, adds the",
        "  overlap between them, and reports the third estimand (representative-row",
        "  SELECTION changed, 232) under a name that cannot be confused with either.",
        "  Nothing here is edited; provenance is preserved by pointing forward.",
    ]
    cur = cur.rstrip("\n") + "\n" + "\n".join(block) + "\n"
    config.supersede_path.write_text(cur, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- #
# Layer 3 -- command line. Only this layer knows argparse exists.
# --------------------------------------------------------------------------- #
def config_from_args(args: argparse.Namespace) -> ProbeConfig:
    base = ProbeConfig.default(Path(args.repo))
    return ProbeConfig(
        repo_root=base.repo_root,
        raw_path=Path(args.raw) if args.raw else base.raw_path,
        golden_path=Path(args.golden) if args.golden else base.golden_path,
        output_path=Path(args.out) if args.out else base.output_path,
        sidecar_path=Path(args.sidecar) if args.sidecar else base.sidecar_path,
        supersede_path=Path(args.supersede) if args.supersede else base.supersede_path,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="P6 R2 reconciliation: one PolicyDelta pass, two tables, "
                    "six invariants. Read-only except for the R2 artifact.")
    ap.add_argument("--repo", default=str(REPO_DEFAULT))
    ap.add_argument("--raw", default=None)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--supersede", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    return run_probe(config_from_args(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
