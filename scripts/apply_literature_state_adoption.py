#!/usr/bin/env python3
"""apply_literature_state_adoption.py -- Author: Monzia Moodie

LITERATURE-STATE-CWD-RELATIVE-1: version_monitor_agent adopts the state store.

THE DEFECT, MEASURED AT LINES 58-85
    _STATE_PATH = Path("data/agent_state.json")     # cwd-relative
    except (json.JSONDecodeError, OSError):
        return {}                                    # corruption -> empty
    _STATE_PATH.write_text(...)                      # non-atomic

Those compound. A crash mid-write truncates the file; truncation reads as an
empty store; the next _set_many persists that emptiness as the new truth. The
agent is LIVE -- registered in the orchestrator at line 166 and scheduled in
the version_monitor and adaptation pipelines -- and its baselines are exactly
what that sequence destroys.

Two divergent copies exist as a result, at data/ and at
src/.../agent_layer/data/, same 25 keys, five values differing, every
difference the nested copy being a later observation. That is
STATE-FILE-DUPLICATES-1, reconciled separately.

FOUR EDITS, EACH ANCHOR BYTE-EXACT AND VERIFIED TO OCCUR ONCE

  1. version_monitor_agent.py lines 58-85: six definitions become a module
     store constructed from RuntimePaths, with _get and _set_many kept as thin
     delegates so the THREE call sites at lines 156, 202 and 496 are untouched.

     _set is DROPPED. It was defined at lines 77-80 and called NOWHERE --
     verified by an abstract-syntax-tree call census across src, scripts and
     tests.

  2. version_monitor_agent.py line 39: two imports added after the BaseAgent
     import.

  3. .gitignore after line 102: /.gvc-state/ added, ANCHORED.

     MEASURED 2026-08-15: git check-ignore returned NOTHING for
     .gvc-state/literature_scout/state.json. The first real agent run would
     have created it and left it untracked in git status, where someone
     eventually commits mutable operational state. That is
     REPORTS-DIR-IGNORED-1 inverted.

     The leading slash matters for the same reason it did there: a nested
     .gvc-state under src/ must stay VISIBLE, and the anchored rule keeps it so
     -- verified by probe in both directions.

  4. tests/unit/test_ignore_boundary.py: one sentinel added to CASES, asserting
     the canonical state path IS ignored. That file already carries the
     instrument and nine sentinels; a tenth there is better than a parallel
     assertion elsewhere.

WHY THE NEW TESTS EXIST
The two pre-existing test files for this agent stub _run_watch_targets AND pass
dry_run=True. Line 495 reads `if not dry_run: _set_many(...)`, so NEITHER ever
reaches the store -- replacing it would have been invisible to the whole suite.
The new file drives it through an INJECTED store, so nothing touches the real
repository file.

VERIFIED
  16 store-adoption tests: 16 fail before the edit, all pass after, 9 of 9
  sabotage mutations detected -- including reverting to the cwd-relative path,
  ignoring the injected store, using the orchestrator's schema, replacing
  instead of merging, SWALLOWING CORRUPTION AGAIN, refusing legacy payloads,
  and reintroducing the dead _set.

  The boundary case fails without the .gitignore rule and passes with it,
  confirmed by reverting the rule in an isolated repository.

Idempotent, ast-verifies before AND after writing, backs up to
.pre_litstate.bak, and rolls back every file if any post-write check fails.

Usage:  python scripts/apply_literature_state_adoption.py --repo-root . --check
        python scripts/apply_literature_state_adoption.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

OLD_BLOCK = '_STATE_PATH = Path("data/agent_state.json")\n\ndef _load_state() -> dict[str, Any]:\n    if _STATE_PATH.exists():\n        try:\n            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))\n        except (json.JSONDecodeError, OSError):\n            return {}\n    return {}\n\ndef _save_state(state: dict[str, Any]) -> None:\n    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)\n    _STATE_PATH.write_text(\n        json.dumps(state, indent=2, default=str), encoding="utf-8"\n    )\n\ndef _get(key: str, default: Any = None) -> Any:\n    return _load_state().get(key, default)\n\ndef _set(key: str, value: Any) -> None:\n    state = _load_state()\n    state[key] = value\n    _save_state(state)\n\ndef _set_many(updates: dict[str, Any]) -> None:\n    state = _load_state()\n    state.update(updates)\n    _save_state(state)\n'

NEW_BLOCK = '#: The literature-scout store. A FLAT key-value change-detection log --\n#: NOT the orchestrator SharedState, which is structured and lives elsewhere.\n#: Two files named agent_state.json held these unrelated schemas, and reading\n#: the wrong one previously SUCCEEDED and returned a dict that meant something\n#: else. The envelope now makes that a loud failure.\nLITERATURE_SCOUT_SCHEMA = "gvc.literature-scout-state"\n\n#: Retained for provenance: the path this store used before STATE-STORE-1.\n#: It was CWD-RELATIVE, so the destination depended on where the process was\n#: launched -- and two divergent copies exist as a result, at data/ and at\n#: src/.../agent_layer/data/. STATE-FILE-DUPLICATES-1 reconciles them.\n_LEGACY_STATE_PATH = Path("data/agent_state.json")\n\n_store_override: JsonStateStore | None = None\n\n\ndef set_state_store(store: JsonStateStore | None) -> None:\n    """Inject a store, for hermetic tests. Pass None to restore the default.\n\n    The two pre-existing tests for this agent stub _run_watch_targets and pass\n    dry_run=True, so NEITHER reaches the store at all. Without injection a new\n    test could only drive it by writing to the real repository.\n    """\n    global _store_override\n    _store_override = store\n\n\ndef _state_store() -> JsonStateStore:\n    """The store to use: injected, else anchored to RuntimePaths.\n\n    Anchored rather than cwd-relative. The previous Path("data/agent_state.json")\n    resolved against the working directory, which is how the same logical store\n    came to exist at two depths with divergent contents.\n    """\n    if _store_override is not None:\n        return _store_override\n    return JsonStateStore(\n        path=resolve_runtime_paths().literature_scout_state,\n        schema=LITERATURE_SCOUT_SCHEMA,\n    )\n\n\ndef _get(key: str, default: Any = None) -> Any:\n    """One value. Corruption RAISES rather than reading as absent.\n\n    The previous _load_state swallowed JSONDecodeError into {}, so a truncated\n    file reported "no history" and the next _set_many persisted that emptiness\n    over the original -- destroying exactly the ClinVar and AlphaMissense\n    baselines this agent exists to keep.\n    """\n    return _state_store().load(allow_legacy=True).values.get(key, default)\n\n\ndef _set_many(updates: dict[str, Any]) -> None:\n    """Merge and persist ATOMICALLY.\n\n    The previous _save_state wrote with write_text directly, so an interrupted\n    write left partial JSON -- which the previous _load_state then read as an\n    empty store. Those two defects compounded.\n    """\n    _state_store().update(updates)\n'

IMPORT_OLD = 'from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent\n'

IMPORT_NEW = 'from genomic_variant_classifier.agent_layer.agents.base_agent import BaseAgent\nfrom genomic_variant_classifier.paths.runtime_paths import resolve_runtime_paths\nfrom genomic_variant_classifier.state.json_state_store import JsonStateStore\n'

GITIGNORE_OLD = '/reports/\nnotebooks/genomic_variant_classifier/reports/\n'

GITIGNORE_NEW = '/reports/\nnotebooks/genomic_variant_classifier/reports/\n/.gvc-state/\n'

LOG_OLD = '        _set_many(all_updates)\n        logger.info("LiteratureScoutAgent: state written to %s", _STATE_PATH)\n'

LOG_NEW = '        _set_many(all_updates)\n        logger.info("LiteratureScoutAgent: state written to %s",\n                    _state_store().path)\n'

BOUNDARY_OLD = '    ("outputs/drift_reports/concept_drift/report.json", True,\n     "drift agents write under outputs/, which is separately ignored"),\n'

BOUNDARY_NEW = '    ("outputs/drift_reports/concept_drift/report.json", True,\n     "drift agents write under outputs/, which is separately ignored"),\n    (".gvc-state/literature_scout/state.json", True,\n     "CANONICAL MUTABLE STATE. RuntimePaths puts the literature-scout "\n     "store here, and version_monitor_agent creates it on first write. "\n     "Measured 2026-08-15: before this rule, git check-ignore returned "\n     "NOTHING for it -- so the first real agent run would have left an "\n     "untracked directory that someone eventually commits. This is the "\n     "state analogue of REPORTS-DIR-IGNORED-1, inverted: canonical state "\n     "MUST be ignored, while state appearing under src/ must be VISIBLE"),\n'

AGENT = "src/genomic_variant_classifier/agent_layer/agents/version_monitor_agent.py"
BOUNDARY = "tests/unit/test_ignore_boundary.py"
GITIGNORE = ".gitignore"

#: (path, old, new, marker) -- marker present means already applied.
EDITS = (
    (AGENT, OLD_BLOCK, NEW_BLOCK, "LITERATURE_SCOUT_SCHEMA"),
    (AGENT, IMPORT_OLD, IMPORT_NEW, "state.json_state_store import JsonStateStore"),
    # The FIFTH edit, and the one that broke the first attempt. run() logs
    # `_STATE_PATH` at what was line 497 pre-adoption and 532 post-adoption --
    # a NAME LOAD of a constant the block edit deletes. I had read that line,
    # quoted it in an earlier exchange, and recorded it as "a log message"
    # rather than as a reference to a name I was about to remove. The suite
    # caught it with NameError and the installer rolled everything back.
    (AGENT, LOG_OLD, LOG_NEW, "_state_store().path"),
    (GITIGNORE, GITIGNORE_OLD, GITIGNORE_NEW, "/.gvc-state/"),
    (BOUNDARY, BOUNDARY_OLD, BOUNDARY_NEW, ".gvc-state/literature_scout/state.json"),
)


def _verify_agent(source: str) -> tuple:
    """Structural checks on the patched agent, by AST rather than substring.

    ROOTFIX-VERIFY-TEXTUAL-1 recorded that `if "root" not in source` is
    satisfied by an unrelated identifier. These walk the tree.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax error after patch: {}".format(exc)

    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in ("_get", "_set_many", "_state_store", "set_state_store"):
        if required not in names:
            return False, "{} is missing".format(required)
    if "_set" in names:
        return False, "the dead _set survived; it had zero callers"
    for gone in ("_load_state", "_save_state"):
        if gone in names:
            return False, "{} survived; the store replaces it".format(gone)

    assigns = {t.id for n in tree.body if isinstance(n, ast.Assign)
               for t in n.targets if isinstance(t, ast.Name)}
    if "_STATE_PATH" in assigns:
        return False, "_STATE_PATH survived"
    if "LITERATURE_SCOUT_SCHEMA" not in assigns:
        return False, "LITERATURE_SCOUT_SCHEMA is missing"

    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                imported.add(n.module + "." + a.name)
    if "genomic_variant_classifier.state.json_state_store.JsonStateStore" not in imported:
        return False, "JsonStateStore is not imported"
    if "genomic_variant_classifier.paths.runtime_paths.resolve_runtime_paths" not in imported:
        return False, "resolve_runtime_paths is not imported"

    # NO DANGLING REFERENCE to a deleted name may remain. This is the check
    # whose absence let `logger.info(..., _STATE_PATH)` through: confirming the
    # DEFINITIONS are gone says nothing about whether anything still LOADS
    # them. An ast.Name in Load context is exactly that question.
    deleted = {"_STATE_PATH", "_load_state", "_save_state", "_set"}
    dangling = sorted({(n.lineno, n.id) for n in ast.walk(tree)
                       if isinstance(n, ast.Name) and n.id in deleted
                       and isinstance(n.ctx, ast.Load)})
    if dangling:
        return False, "dangling reference(s) to deleted name(s): {}".format(
            ", ".join("{} at line {}".format(nm, ln) for ln, nm in dangling))

    # The three call sites must survive unchanged.
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) in ("_get", "_set_many")]
    if len(calls) < 3:
        return False, "expected at least 3 call sites, found {}".format(len(calls))
    return True, "store adopted; _set dropped; no dangling refs; 3 call sites intact"


def _verify_boundary(source: str) -> tuple:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax error after patch: {}".format(exc)
    for n in tree.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CASES":
            cases = ast.literal_eval(n.value)
            hits = [c for c in cases if c[0].startswith(".gvc-state/")]
            if not hits:
                return False, "no .gvc-state sentinel in CASES"
            if hits[0][1] is not True:
                return False, "the .gvc-state sentinel does not require IGNORED"
            return True, "{} sentinel(s), canonical state asserted ignored".format(len(cases))
    return False, "CASES not found"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.repo_root)

    plans = []
    already = 0
    for rel, old, new, marker in EDITS:
        p = root / rel
        if not p.exists():
            print("  ERROR: not found: {}".format(rel))
            return 2
        src = p.read_text(encoding="utf-8")
        if marker in src:
            already += 1
            print("  {:<62} already applied".format(rel + " [" + marker[:22] + "]"))
            continue
        n = src.count(old)
        if n != 1:
            print("  {:<62} ERROR: anchor occurs {} time(s), expected 1; "
                  "aborting with NO changes to ANY file.".format(rel, n))
            return 1
        print("  {:<62} anchor OK".format(rel + " [" + marker[:22] + "]"))
        plans.append((p, src, old, new, rel))

    if args.check:
        print("\n  --check: {} pending, {} already applied. Nothing written."
              .format(len(plans), already))
        return 0
    if not plans:
        print("\n  All {} edit(s) already applied. Nothing to do.".format(len(EDITS)))
        return 0

    # Patch every file in memory and verify BEFORE writing any of them.
    patched = {}
    for p, src, old, new, rel in plans:
        current = patched.get(p, src)
        result = current.replace(old, new, 1)
        if result == current:
            print("  ERROR: {} unchanged after replace; no file written.".format(rel))
            return 1
        patched[p] = result

    for p, text in patched.items():
        rel = str(p).replace("\\", "/")
        if rel.endswith("version_monitor_agent.py"):
            ok, msg = _verify_agent(text)
        elif rel.endswith("test_ignore_boundary.py"):
            ok, msg = _verify_boundary(text)
        else:
            ok, msg = (True, "no structural check for this file type")
        if not ok:
            print("  ERROR: {} failed verification BEFORE writing ({}); "
                  "no file written.".format(rel, msg))
            return 1
        print("  pre-write  {:<48} {}".format(Path(rel).name, msg))

    written = []
    for p, text in patched.items():
        backup = p.with_suffix(p.suffix + ".pre_litstate.bak")
        if not backup.exists():
            backup.write_bytes(p.read_bytes())
        p.write_text(text, encoding="utf-8", newline="\n")
        written.append((p, backup))
        print("  wrote {}".format(p.name))

    for p, backup in written:
        text = p.read_text(encoding="utf-8")
        rel = str(p).replace("\\", "/")
        if rel.endswith("version_monitor_agent.py"):
            ok, msg = _verify_agent(text)
        elif rel.endswith("test_ignore_boundary.py"):
            ok, msg = _verify_boundary(text)
        else:
            ok, msg = (True, "written")
        if not ok:
            for p2, b2 in written:
                p2.write_bytes(b2.read_bytes())
            print("  ERROR: {} failed POST-WRITE verification ({}); "
                  "ROLLED BACK all {} file(s).".format(rel, msg, len(written)))
            return 1

    print("\n  {} edit(s) applied; {} already were.".format(len(plans), already))
    return 0


if __name__ == "__main__":
    sys.exit(main())
