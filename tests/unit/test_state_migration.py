"""Tests for the literature-scout state migration -- STATE-FILE-DUPLICATES-1.

Author: Monzia Moodie
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")
from migrate_literature_scout_state import (  # noqa: E402
    NESTED_COPY, ORDERING_KEY, ROOT_COPY, MigrationRefused,
    StateMigrationRecord, choose_source, compare,
)


def _pair(root_extra=None, nested_extra=None, root_ts="2026-06-13T19:14:10Z",
          nested_ts="2026-06-20T02:30:41Z"):
    root = {ORDERING_KEY: root_ts, "literature_scout.deps_outdated_count": 99}
    nested = {ORDERING_KEY: nested_ts, "literature_scout.deps_outdated_count": 110}
    root.update(root_extra or {})
    nested.update(nested_extra or {})
    return root, nested


# ---- which copy supersedes, and why -------------------------------------
def test_the_later_copy_supersedes():
    """MEASURED: the nested copy's last_run is 2026-06-20T02:30:41Z against the
    root copy's 2026-06-13T19:14:10Z, and all five differing values are later
    observations."""
    root, nested = _pair()
    src, sup, sv, _ = choose_source(root, nested, compare(root, nested))
    assert src == NESTED_COPY
    assert sup == ROOT_COPY
    assert sv[ORDERING_KEY].startswith("2026-06-20")


def test_the_direction_is_decided_by_the_VALUE_not_the_file():
    """Not "the nested one wins" -- whichever records the later run wins. If the
    root copy were newer this must reverse, or the rule is a hard-coded
    preference wearing a comparison."""
    root, nested = _pair(root_ts="2026-07-01T00:00:00Z",
                         nested_ts="2026-06-20T02:30:41Z")
    src, sup, _, _ = choose_source(root, nested, compare(root, nested))
    assert src == ROOT_COPY
    assert sup == NESTED_COPY


def test_a_DIFFERING_key_set_REFUSES():
    """A newer copy that has LOST a key is not a superset, and merging it
    blindly would discard a change-detection baseline."""
    root, nested = _pair(root_extra={"literature_scout.only_here": "x"})
    with pytest.raises(MigrationRefused) as exc:
        choose_source(root, nested, compare(root, nested))
    assert "Neither copy is a superset" in str(exc.value)


def test_a_key_only_in_the_NEWER_copy_also_REFUSES():
    """Symmetric: the rule is about supersets, not about which side gained."""
    root, nested = _pair(nested_extra={"literature_scout.new_key": "y"})
    with pytest.raises(MigrationRefused):
        choose_source(root, nested, compare(root, nested))


def test_contemporaneous_copies_REFUSE():
    """Equal ordering values cannot rank five differing observations."""
    root, nested = _pair(root_ts="2026-06-20T02:30:41Z",
                         nested_ts="2026-06-20T02:30:41Z")
    with pytest.raises(MigrationRefused) as exc:
        choose_source(root, nested, compare(root, nested))
    assert "contemporaneous" in str(exc.value)


def test_deleting_the_ordering_key_trips_the_SUPERSET_guard_first():
    """Order of guards, stated rather than assumed.

    An earlier version of this test deleted ORDERING_KEY from one copy and
    asserted the ORDERING message -- but removing a key changes the KEY SETS,
    so the superset guard fires first. The test was wrong about which guard
    would catch it, and the message told me so.
    """
    root, nested = _pair()
    del nested[ORDERING_KEY]
    with pytest.raises(MigrationRefused) as exc:
        choose_source(root, nested, compare(root, nested))
    assert "Neither copy is a superset" in str(exc.value)


def test_a_NULL_ordering_value_REFUSES_on_ordering():
    """The ordering guard, reached properly: the key is present in BOTH copies
    so the superset check passes, but one value is not a string."""
    root, nested = _pair()
    nested[ORDERING_KEY] = None
    with pytest.raises(MigrationRefused) as exc:
        choose_source(root, nested, compare(root, nested))
    assert "logical ordering cannot be established" in str(exc.value)


def test_a_non_string_ordering_value_REFUSES():
    root, nested = _pair()
    nested[ORDERING_KEY] = 1750000000
    with pytest.raises(MigrationRefused):
        choose_source(root, nested, compare(root, nested))


# ---- the comparison itself ----------------------------------------------
def test_compare_reports_both_directions_and_the_differences():
    root, nested = _pair(root_extra={"a": 1}, nested_extra={"b": 2})
    c = compare(root, nested)
    assert c["only_root"] == ("a",)
    assert c["only_nested"] == ("b",)
    assert ORDERING_KEY in c["differing"]
    assert "literature_scout.deps_outdated_count" in c["differing"]


def test_compare_finds_no_difference_between_identical_copies():
    root, _ = _pair()
    c = compare(root, dict(root))
    assert c == {"only_root": (), "only_nested": (), "differing": ()}


# ---- the record is the point --------------------------------------------
def test_the_record_carries_every_digest_and_the_reasoning():
    """A copy leaves no answer to "why does this store's history jump from
    2026-06-13 to 2026-06-20?" The record answers it."""
    r = StateMigrationRecord(
        migration_id="STATE-FILE-DUPLICATES-1",
        performed_at="2026-08-15T00:00:00+00:00",
        source_path=NESTED_COPY, source_sha256="22fe38e9", source_key_count=25,
        superseded_path=ROOT_COPY, superseded_sha256="e28c673b",
        superseded_key_count=25,
        destination_path=".gvc-state/literature_scout/state.json",
        destination_before_sha256="", destination_after_sha256="abc123",
        keys_only_in_source=(), keys_only_in_superseded=(),
        differing_keys=(ORDERING_KEY,), ordering_key=ORDERING_KEY,
        source_ordering_value="2026-06-20T02:30:41Z",
        superseded_ordering_value="2026-06-13T19:14:10Z",
        decision="the nested copy supersedes", schema="gvc.literature-scout-state",
        schema_version=1, generation=1,
        legacy_files_retained=(ROOT_COPY, NESTED_COPY))
    d = r.as_dict()
    for key in ("source_sha256", "superseded_sha256", "destination_before_sha256",
                "destination_after_sha256", "differing_keys", "decision",
                "legacy_files_retained"):
        assert key in d, key
    assert json.dumps(d)  # serialisable
    assert isinstance(d["legacy_files_retained"], list)


def test_the_record_names_the_retained_legacy_files():
    """They are the only surviving record of what the cwd-relative path
    produced. Deleting them on an assumption of redundancy is a mistake this
    session made three times and caught three times."""
    r = StateMigrationRecord(
        migration_id="x", performed_at="x", source_path=NESTED_COPY,
        source_sha256="x", source_key_count=25, superseded_path=ROOT_COPY,
        superseded_sha256="x", superseded_key_count=25, destination_path="x",
        destination_before_sha256="", destination_after_sha256="x",
        keys_only_in_source=(), keys_only_in_superseded=(), differing_keys=(),
        ordering_key=ORDERING_KEY, source_ordering_value="x",
        superseded_ordering_value="x", decision="x", schema="x",
        schema_version=1, generation=1,
        legacy_files_retained=(ROOT_COPY, NESTED_COPY))
    assert ROOT_COPY in r.legacy_files_retained
    assert NESTED_COPY in r.legacy_files_retained


def test_every_record_field_is_REQUIRED():
    """No field may carry a default.

    Sabotage W8 gave `legacy_files_retained` a default of () and went
    undetected, because every test passes it explicitly. A record whose fields
    can be OMITTED can be constructed claiming no legacy files were retained
    when two were -- and the omission would look like a fact.

    A migration record is evidence. Every field is a claim someone must make
    deliberately.
    """
    import dataclasses
    fields = dataclasses.fields(StateMigrationRecord)
    defaulted = [f.name for f in fields
                 if f.default is not dataclasses.MISSING
                 or f.default_factory is not dataclasses.MISSING]
    assert not defaulted, (
        "these field(s) carry defaults and can be silently omitted: {}"
        .format(defaulted))
    assert len(fields) >= 20, len(fields)


def test_omitting_any_field_RAISES():
    """The behavioural companion: the type refuses a partial record."""
    with pytest.raises(TypeError):
        StateMigrationRecord(migration_id="x", performed_at="x")


def test_the_record_is_immutable():
    import dataclasses
    r = StateMigrationRecord(
        migration_id="x", performed_at="x", source_path="x", source_sha256="x",
        source_key_count=1, superseded_path="x", superseded_sha256="x",
        superseded_key_count=1, destination_path="x",
        destination_before_sha256="", destination_after_sha256="x",
        keys_only_in_source=(), keys_only_in_superseded=(), differing_keys=(),
        ordering_key="k", source_ordering_value="x",
        superseded_ordering_value="x", decision="x", schema="x",
        schema_version=1, generation=1, legacy_files_retained=())
    try:
        r.decision = "rewritten"
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("the migration record was mutable")
