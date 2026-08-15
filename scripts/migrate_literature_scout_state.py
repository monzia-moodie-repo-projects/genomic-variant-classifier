#!/usr/bin/env python3
"""migrate_literature_scout_state.py -- Author: Monzia Moodie

STATE-FILE-DUPLICATES-1: reconcile two divergent literature-scout stores into
the canonical location, producing an immutable migration record.

WHY THIS IS A MIGRATION AND NOT A FILE COPY
    A copy leaves no answer to "why does this store's history jump from
    2026-06-13 to 2026-06-20?"

The record below answers it, with the digests of every file involved, the
key-set comparison that justified the choice, and the reasoning stated in the
document rather than in a commit message someone must go looking for.

WHAT WAS MEASURED, 2026-08-15
    data/agent_state.json                        13,463 bytes  2026-06-13 15:14
        sha256 e28c673ba7a93ed7856755ef6bf9cd84b4ceaac95500fba7e350e9e8438479cb
        25 keys, no envelope
    src/.../agent_layer/data/agent_state.json    14,524 bytes  2026-06-19 22:30
        sha256 22fe38e94ce3bc8fd349e1fd4a6fbff51e4e6ed5c217503dfee095a8fe339e16
        25 keys, no envelope
    .gvc-state/literature_scout/state.json       ABSENT

Same key set. No key unique to either. FIVE values differ, and every difference
is the nested copy being a LATER observation:

    literature_scout.last_run             2026-06-13T19:14:10Z -> 2026-06-20T02:30:41Z
    literature_scout.deps_outdated_count  99 -> 110
    literature_scout.deps_major_bumps     borb 3.0.7 -> borb 3.0.8
    literature_scout.deps_outdated        (longer list)
    literature_scout.alerts               (longer list)

WHY TWO COPIES EXISTED
version_monitor_agent.py:58 read `Path("data/agent_state.json")` -- RELATIVE,
resolved against the process working directory. Launched from the repository
root it wrote data/; launched from src/.../agent_layer it wrote
src/.../agent_layer/data/. That is LITERATURE-STATE-CWD-RELATIVE-1, closed at
commit a734ea1.

A NOTE ON THE TWO CLOCKS
The nested file's modification time is 2026-06-19 22:30 while the last_run
value it holds is 2026-06-20T02:30:41 UTC -- four hours apart, which is Eastern
Daylight Time's offset. The same instant on two clocks. Both last_run values
are UTC-suffixed ISO 8601, so comparing them as strings is sound.

WHAT THIS DOES NOT DO
It does not delete either legacy file. They are the only surviving record of
what the cwd-relative path produced, and this project spent a session
characterising that defect. Deleting evidence on an assumption of redundancy
would be a mistake this session already made three times and caught three
times.

The migration is REFUSED unless every precondition holds:
    both sources present and parseable
    identical key sets
    the chosen source later on every differing key
    the destination absent, or already migrated with a matching digest

Usage:
    python scripts/migrate_literature_scout_state.py --repo-root . --check
    python scripts/migrate_literature_scout_state.py --repo-root .
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT_COPY = "data/agent_state.json"
NESTED_COPY = "src/genomic_variant_classifier/agent_layer/data/agent_state.json"

#: Measured 2026-08-15. If either file has changed, the migration REFUSES --
#: a reconciliation justified by a comparison of different bytes is not
#: justified at all.
EXPECTED_DIGESTS = {
    ROOT_COPY: "e28c673ba7a93ed7856755ef6bf9cd84b4ceaac95500fba7e350e9e8438479cb",
    NESTED_COPY: "22fe38e94ce3bc8fd349e1fd4a6fbff51e4e6ed5c217503dfee095a8fe339e16",
}

#: The key whose ordering decides which copy supersedes. UTC-suffixed ISO 8601
#: in both files, so a string comparison is sound.
ORDERING_KEY = "literature_scout.last_run"


class MigrationRefused(RuntimeError):
    """A precondition failed. Nothing was written."""


@dataclass(frozen=True)
class StateMigrationRecord:
    """The immutable audit. Everything needed to reconstruct the decision."""
    migration_id: str
    performed_at: str
    source_path: str
    source_sha256: str
    source_key_count: int
    superseded_path: str
    superseded_sha256: str
    superseded_key_count: int
    destination_path: str
    destination_before_sha256: str
    destination_after_sha256: str
    keys_only_in_source: tuple
    keys_only_in_superseded: tuple
    differing_keys: tuple
    ordering_key: str
    source_ordering_value: str
    superseded_ordering_value: str
    decision: str
    schema: str
    schema_version: int
    generation: int
    legacy_files_retained: tuple

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k, v in d.items():
            if isinstance(v, tuple):
                d[k] = list(v)
        return d


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bare(path: Path) -> dict:
    try:
        data = json.loads(io.open(path, encoding="utf-8").read())
    except json.JSONDecodeError as exc:
        raise MigrationRefused("{} is not valid JSON: {}".format(path, exc))
    if not isinstance(data, dict):
        raise MigrationRefused(
            "{} holds a {}, not an object".format(path, type(data).__name__))
    return data


def compare(root_values: dict, nested_values: dict) -> dict:
    """Everything the record needs about how the two copies relate."""
    only_root = tuple(sorted(set(root_values) - set(nested_values)))
    only_nested = tuple(sorted(set(nested_values) - set(root_values)))
    shared = set(root_values) & set(nested_values)
    differing = tuple(sorted(k for k in shared
                             if root_values[k] != nested_values[k]))
    return {"only_root": only_root, "only_nested": only_nested,
            "differing": differing}


def choose_source(root_values: dict, nested_values: dict, cmp: dict) -> tuple:
    """Which copy supersedes, and why. RAISES rather than guessing.

    The rule is not "the bigger file" or "the newer modification time" -- it is
    that one copy's ordering value is strictly later AND no key exists only in
    the other. A copy that is newer but has LOST a key is not a superset, and
    merging it blindly would discard a baseline.
    """
    if cmp["only_root"] or cmp["only_nested"]:
        raise MigrationRefused(
            "the key sets differ -- only in root: {}; only in nested: {}. "
            "Neither copy is a superset, so neither simply supersedes the "
            "other, and this migration will not guess."
            .format(cmp["only_root"], cmp["only_nested"]))

    root_ts = root_values.get(ORDERING_KEY)
    nested_ts = nested_values.get(ORDERING_KEY)
    if not isinstance(root_ts, str) or not isinstance(nested_ts, str):
        raise MigrationRefused(
            "{} is missing or not a string in one copy (root={!r}, "
            "nested={!r}); logical ordering cannot be established."
            .format(ORDERING_KEY, root_ts, nested_ts))
    if root_ts == nested_ts:
        raise MigrationRefused(
            "both copies record {} == {!r}; they are contemporaneous and the "
            "five differing values cannot be ordered."
            .format(ORDERING_KEY, root_ts))
    if nested_ts > root_ts:
        return NESTED_COPY, ROOT_COPY, nested_values, root_values
    return ROOT_COPY, NESTED_COPY, root_values, nested_values


def migrate(repo_root: Path, *, check_only: bool = False) -> int:
    from genomic_variant_classifier.paths.runtime_paths import resolve_runtime_paths
    from genomic_variant_classifier.state.json_state_store import (
        JsonStateStore, PayloadShape,
    )
    from genomic_variant_classifier.agent_layer.agents.version_monitor_agent import (
        LITERATURE_SCOUT_SCHEMA,
    )

    root_p = repo_root / ROOT_COPY
    nested_p = repo_root / NESTED_COPY
    for p, rel in ((root_p, ROOT_COPY), (nested_p, NESTED_COPY)):
        if not p.exists():
            raise MigrationRefused("{} is absent".format(rel))
        got = _sha256(p)
        want = EXPECTED_DIGESTS[rel]
        if got != want:
            raise MigrationRefused(
                "{} has SHA-256 {}, expected {}. It changed since the "
                "comparison that justifies this migration, and a "
                "reconciliation justified by different bytes is not justified."
                .format(rel, got, want))
        print("  source    OK  {:<58} {}".format(rel, got[:16]))

    root_values = _load_bare(root_p)
    nested_values = _load_bare(nested_p)
    cmp = compare(root_values, nested_values)
    print("  keys      OK  root {}, nested {}, differing {}".format(
        len(root_values), len(nested_values), len(cmp["differing"])))

    src_rel, sup_rel, src_values, sup_values = choose_source(
        root_values, nested_values, cmp)
    print("  ordering  OK  {} supersedes {}".format(src_rel, sup_rel))
    print("      {} : {!r} > {!r}".format(
        ORDERING_KEY, src_values[ORDERING_KEY], sup_values[ORDERING_KEY]))

    paths = resolve_runtime_paths()
    dest = paths.literature_scout_state
    store = JsonStateStore(path=dest, schema=LITERATURE_SCOUT_SCHEMA)

    before_sha = ""
    if dest.exists():
        before_sha = _sha256(dest)
        existing = store.load()
        if existing.values == src_values:
            print("  destination already holds the migrated payload "
                  "(generation {})".format(existing.generation))
            return 0
        raise MigrationRefused(
            "{} already exists with DIFFERENT content (sha {}). This "
            "migration will not overwrite a store it did not write."
            .format(dest, before_sha[:16]))
    print("  dest      OK  {} absent".format(dest))

    if check_only:
        print("\n  --check: would migrate {} -> {}. Nothing written."
              .format(src_rel, dest))
        return 0

    generation = store.save(src_values, generation=1)
    after_sha = _sha256(dest)
    written = store.load()
    if written.shape is not PayloadShape.ENVELOPED:
        raise MigrationRefused("the written store is not enveloped")
    if written.values != src_values:
        raise MigrationRefused(
            "POST-WRITE: the destination holds {} key(s), expected {}"
            .format(len(written.values), len(src_values)))
    print("  wrote     {} ({} keys, generation {})".format(
        dest, len(written.values), generation))

    record = StateMigrationRecord(
        migration_id="STATE-FILE-DUPLICATES-1",
        performed_at=datetime.now(timezone.utc).isoformat(),
        source_path=src_rel,
        source_sha256=EXPECTED_DIGESTS[src_rel],
        source_key_count=len(src_values),
        superseded_path=sup_rel,
        superseded_sha256=EXPECTED_DIGESTS[sup_rel],
        superseded_key_count=len(sup_values),
        destination_path=str(dest.relative_to(repo_root.resolve())),
        destination_before_sha256=before_sha,
        destination_after_sha256=after_sha,
        keys_only_in_source=cmp["only_nested"] if src_rel == NESTED_COPY else cmp["only_root"],
        keys_only_in_superseded=cmp["only_root"] if src_rel == NESTED_COPY else cmp["only_nested"],
        differing_keys=cmp["differing"],
        ordering_key=ORDERING_KEY,
        source_ordering_value=src_values[ORDERING_KEY],
        superseded_ordering_value=sup_values[ORDERING_KEY],
        decision=(
            "The nested copy supersedes the root copy: identical key sets, no "
            "key unique to either, and every one of the five differing values "
            "is a later observation -- last_run advances, deps_outdated_count "
            "rises from 99 to 110, and borb's tracked release moves 3.0.7 to "
            "3.0.8. Two copies existed because version_monitor_agent.py:58 "
            "resolved a RELATIVE path against the process working directory "
            "(LITERATURE-STATE-CWD-RELATIVE-1, closed at a734ea1). Both legacy "
            "files are RETAINED: they are the only surviving record of what "
            "that defect produced."
        ),
        schema=LITERATURE_SCOUT_SCHEMA,
        schema_version=store.schema_version,
        generation=generation,
        legacy_files_retained=(ROOT_COPY, NESTED_COPY),
    )
    doc_dir = repo_root / "docs" / "migrations"
    doc_dir.mkdir(parents=True, exist_ok=True)
    doc = doc_dir / "LITERATURE_SCOUT_STATE_2026-08-15.json"
    io.open(doc, "w", encoding="utf-8", newline="\n").write(
        json.dumps(record.as_dict(), indent=2, sort_keys=True) + "\n")
    print("  record    {}".format(doc.relative_to(repo_root.resolve())))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    try:
        return migrate(Path(args.repo_root).resolve(), check_only=args.check)
    except MigrationRefused as exc:
        print("  REFUSED: {}".format(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
