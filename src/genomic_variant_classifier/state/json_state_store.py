"""A JSON state store that is atomic, identified, and fails closed.

STATE-STORE-1
=============
Two mutable JSON stores exist in this project and they behave differently:

    agent_layer/shared_state.py     SharedState: atomic write via tmp + rename,
                                    load returns a default on corruption but
                                    leaves the corrupt file untouched
    version_monitor_agent.py:58-85  _load_state/_save_state: cwd-relative path,
                                    direct write_text, corruption swallowed
                                    into {} which the next save then PERSISTS

The second is worse in a way that matters. Measured at lines 64-70:

    except (json.JSONDecodeError, OSError):
        return {}                       # a truncated file reads as "no history"
    ...
    _STATE_PATH.write_text(...)         # and the next save writes that over it

A crash mid-write produces truncation; truncation reads as empty; empty is then
persisted as the new truth. This agent's entire purpose is detecting when
upstream sources change, and its baselines -- ClinVar header hashes,
AlphaMissense entity tags -- are exactly what that sequence destroys.

WHY THIS MODULE AND NOT A SECOND IMPLEMENTATION
SharedState's atomic write is CORRECT and was read before this was designed:

    tmp_fd, tmp_path = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp_path, self._path)

Same directory so the rename is atomic, cleanup on failure, and a re-raise so
the caller sees it. Writing a second, subtly different atomic write beside it
would be the parallel-vocabulary defect this project keeps eliminating -- so
that mechanism is reproduced here deliberately and identically, with fsync
ADDED, and the intent is that SharedState later adopts this module rather than
the reverse.

THREE DIFFERENCES FROM SharedState, EACH DELIBERATE
    fsync before rename. SharedState omits it, so os.replace is atomic against
    other PROCESSES while the bytes may still sit in the operating system cache
    at power loss. For change-detection baselines that is worth the cost.

    Corruption RAISES. SharedState logs and returns a default; this raises
    StateCorruptionError. A store that answers "empty" when it means "damaged"
    is the same shape as a parser reporting a 310KB lock as zero packages.

    Schema identity in the payload. Two files named agent_state.json hold
    unrelated schemas -- a flat literature-scout key-value log and the
    orchestrator's structured state. Reasoning from the filename nearly merged
    them. An envelope makes that a loud failure instead.

LEGACY PAYLOADS ARE READABLE, AND SAID SO
MEASURED 2026-08-14: data/agent_state.json is a BARE flat dict of 25 keys with
no envelope. A store that refused unenveloped payloads could not read the data
it exists to migrate. So load() accepts both and reports which it found, and
the caller decides -- migration reads legacy deliberately; ordinary operation
can require an envelope.

Author: Monzia Moodie
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

#: Envelope keys. Named once so a typo cannot silently create a second shape.
SCHEMA_KEY = "schema"
SCHEMA_VERSION_KEY = "schema_version"
GENERATION_KEY = "generation"
UPDATED_AT_KEY = "updated_at"
VALUES_KEY = "values"


class StateStoreError(RuntimeError):
    """Base for every failure this module raises rather than absorbs."""


class StateCorruptionError(StateStoreError):
    """The file exists and could not be parsed.

    Raised INSTEAD of returning an empty mapping. "Empty" and "damaged" are
    different answers, and conflating them is how a truncated file becomes a
    silently reset baseline.
    """


class StateSchemaMismatch(StateStoreError):
    """The payload belongs to a different store.

    Two files named agent_state.json held unrelated schemas. Pointing a reader
    at the wrong one previously SUCCEEDED and returned a dictionary that meant
    something else.
    """


class PayloadShape(str, Enum):
    """What load() found on disk.

    LEGACY_BARE is not an error. The literature-scout store is a bare 25-key
    dict today, and a migration must be able to read it deliberately.
    """
    ABSENT = "absent"
    ENVELOPED = "enveloped"
    LEGACY_BARE = "legacy_bare"


@dataclass(frozen=True)
class LoadedState:
    """What was read, and what shape it was in."""
    values: dict
    shape: PayloadShape
    schema: str = None
    schema_version: int = None
    generation: int = 0
    updated_at: str = None

    def require_enveloped(self) -> "LoadedState":
        """Assert the payload carried identity. For callers that are not
        migrations."""
        if self.shape is PayloadShape.LEGACY_BARE:
            raise StateSchemaMismatch(
                "the payload has no envelope, so its schema is unverifiable. "
                "Only a migration should read a legacy store; ordinary "
                "operation requires identity.")
        return self


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JsonStateStore:
    """One mutable JSON store: atomic, identified, fail-closed.

    `path` is supplied by the caller -- normally from RuntimePaths, never
    computed from repository layout inside this module. That separation is
    OUTPUT-ROOT-CONFLATION-1 stated as a constructor argument.
    """

    path: Path
    schema: str
    schema_version: int = 1

    def load(self, *, allow_legacy: bool = False) -> LoadedState:
        """Read the store. RAISES on corruption; returns empty only if ABSENT.

        `allow_legacy` must be passed explicitly to read an unenveloped
        payload, so reading legacy data is always a deliberate act.
        """
        if not self.path.exists():
            return LoadedState(values={}, shape=PayloadShape.ABSENT,
                               schema=self.schema,
                               schema_version=self.schema_version)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise StateCorruptionError(
                "{} exists but is not valid JSON: {}. Returning an empty store "
                "here would report DAMAGE as ABSENCE, and the next save would "
                "persist that emptiness over the original."
                .format(self.path, exc)) from exc
        except OSError as exc:
            raise StateCorruptionError(
                "{} could not be read: {}".format(self.path, exc)) from exc

        if not isinstance(raw, dict):
            raise StateCorruptionError(
                "{} holds a {}, not an object.".format(
                    self.path, type(raw).__name__))

        if SCHEMA_KEY in raw and VALUES_KEY in raw:
            if raw[SCHEMA_KEY] != self.schema:
                raise StateSchemaMismatch(
                    "{} declares schema {!r}; this store is {!r}. Two files "
                    "named agent_state.json held unrelated schemas, and "
                    "reading the wrong one previously SUCCEEDED."
                    .format(self.path, raw[SCHEMA_KEY], self.schema))
            values = raw.get(VALUES_KEY)
            if not isinstance(values, dict):
                raise StateCorruptionError(
                    "{}: {!r} is not an object.".format(self.path, VALUES_KEY))
            return LoadedState(
                values=dict(values), shape=PayloadShape.ENVELOPED,
                schema=raw[SCHEMA_KEY],
                schema_version=raw.get(SCHEMA_VERSION_KEY),
                generation=int(raw.get(GENERATION_KEY, 0)),
                updated_at=raw.get(UPDATED_AT_KEY))

        if not allow_legacy:
            raise StateSchemaMismatch(
                "{} has no envelope, so its schema cannot be verified. Pass "
                "allow_legacy=True to read it -- only a migration should."
                .format(self.path))
        return LoadedState(values=dict(raw), shape=PayloadShape.LEGACY_BARE,
                           schema=None, schema_version=None, generation=0)

    def save(self, values: dict, *, generation: int = None) -> int:
        """Write atomically. Returns the generation written.

        The temporary file is created in the SAME directory so `os.replace` is
        an atomic rename rather than a cross-device copy. fsync precedes the
        rename, which SharedState omits.
        """
        if not isinstance(values, dict):
            raise StateStoreError(
                "values must be a dict, got {}".format(type(values).__name__))
        if generation is None:
            try:
                generation = self.load(allow_legacy=True).generation + 1
            except StateStoreError:
                # A damaged store must not silently reset the counter; start
                # from 1 and let the corruption surface on the next load.
                generation = 1

        payload = {
            SCHEMA_KEY: self.schema,
            SCHEMA_VERSION_KEY: self.schema_version,
            GENERATION_KEY: generation,
            UPDATED_AT_KEY: _utc_now(),
            VALUES_KEY: values,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True, default=str)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return generation

    def get(self, key: str, default: Any = None) -> Any:
        """One value. Replaces version_monitor_agent's `_get`."""
        return self.load().values.get(key, default)

    def update(self, updates: dict) -> int:
        """Merge and persist. Replaces `_set_many`.

        Read-modify-write, as the original was -- but the read RAISES on
        corruption instead of silently starting from an empty mapping.
        """
        if not isinstance(updates, dict):
            raise StateStoreError(
                "updates must be a dict, got {}".format(type(updates).__name__))
        current = self.load(allow_legacy=True)
        merged = dict(current.values)
        merged.update(updates)
        return self.save(merged, generation=current.generation + 1)
