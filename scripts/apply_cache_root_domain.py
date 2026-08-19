#!/usr/bin/env python3
"""apply_cache_root_domain.py -- Author: Monzia Moodie

INSTALLER-TRANSACTION-1, step 2: a fifth path domain for the transaction
journal.

    A transaction may temporarily own a rollback state. The repository never
    does.

WHY A FIFTH DOMAIN AND NOT state_root
MEASURED 2026-08-19: state_root defaults to <project>/.gvc-state -- a
repository subdirectory, git-ignored by the `/.gvc-state/` rule added at
a734ea1. That is CORRECT for what it holds: literature-scout and orchestrator
state belong to THIS checkout.

A transaction journal does not. It must survive an interrupted installer even
if the working tree is reset, and the invariant this unit serves is that a
successful installer leaves NO rollback artefact in the repository. Putting the
journal under state_root would place it back inside the thing it repairs.

    repository identity  -> project_root
    artifact identity    -> artifact_root
    checkout state       -> state_root
    machine-scoped cache -> cache_root      <- new

This is the pattern OUTPUT-ROOT-CONFLATION-1 established, applied again: a path
derives from the authority that owns what the path contains.

WHAT IT REPLACES
MEASURED 2026-08-19: 148 `.bak_<timestamp>` siblings had accumulated inside the
repository across eight days -- 17,640,928 bytes, invisible to `git status`
because .gitignore carries `*.bak_*`. Retired at 5447362 with a manifest.

SEVEN EDITS, EACH ANCHOR VERIFIED TO OCCUR ONCE
    1. ENV_CACHE_ROOT beside the three existing environment names
    2. the cache_root field, AFTER state_root
    3. the transaction_journal property
    4. two describe() keys
    5. _default_cache_root(), a new helper
    6. resolve_runtime_paths() gains a keyword parameter
    7. the resolution call and the construction site

MEASURED BEFORE WRITING ANY OF THEM
    runtime_paths.py         11,646 bytes, 279 lines, LF-only, pure ASCII
    RuntimePaths fields      project_root, artifact_root, state_root
    construction sites       ONE, at line 278, KEYWORD-only -- so a fourth
                             field cannot break a positional call anywhere
    _resolve_secondary       already generic (explicit, environment, default)
    describe() test          asserts MEMBERSHIP, not equality, so two new keys
                             break nothing
    tests constructing RuntimePaths directly : NONE

Because of the last two, this unit edits no existing test.

FIELD ORDER IS ASSERTED EVEN THOUGH IT IS LATENT
cache_root goes AFTER state_root. With one keyword-only construction site,
order cannot matter today -- but a future positional call would silently
mis-assign three paths, and sabotage F9 confirms the test catches a reordering.

A NOTE ON WHAT CANNOT BE TESTED FROM WINDOWS
Passing a fake environment with XDG_STATE_HOME="/home/runner/.local/state"
selects the right BRANCH but produces "C:/home/runner/..." -- MEASURED. Path
flavour is baked into the platform, not the environment. The tests therefore
assert RELATIONSHIPS (outside the repository, beneath the chosen base,
absolute, project-named) which hold on both platforms; the literal POSIX form
is verified when the runner executes them.

VERIFIED
16 tests. Import error before the edits, all passing after. 9 of 9 sabotage
mutations detected: cache_root defaulting inside the repository, the journal
moved under state_root or project_root, a relative fallback, the project name
dropped, the environment variable ignored, describe keys dropped, the
dataclass unfrozen, and the field order reversed.

Idempotent, ast-verifies before AND after writing, backs up to
.pre_cacheroot.bak, and rolls back if any post-write check fails.

Usage:  python scripts/apply_cache_root_domain.py --repo-root . --check
        python scripts/apply_cache_root_domain.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ENV_OLD = 'ENV_STATE_ROOT = "GVC_STATE_ROOT"\n'

ENV_NEW = 'ENV_STATE_ROOT = "GVC_STATE_ROOT"\n\n#: The transaction journal lives OUTSIDE the repository, so an interrupted\n#: installer survives a working-tree reset and a successful one leaves the\n#: repository with zero rollback artefacts.\n#:\n#: This is a FIFTH path domain, not a synonym for state_root. state_root\n#: defaults to <project>/.gvc-state -- correct for agent state, which belongs\n#: to THIS checkout. A transaction journal does not: it must outlive the\n#: checkout it is repairing.\nENV_CACHE_ROOT = "GVC_CACHE_ROOT"\n'

FIELD_OLD = '    state_root: Path\n'

FIELD_NEW = '    state_root: Path\n    cache_root: Path\n'

DESCRIBE_OLD = '            "state_root": str(self.state_root),\n'

DESCRIBE_NEW = '            "state_root": str(self.state_root),\n            "cache_root": str(self.cache_root),\n'

DESC2_OLD = '            "orchestrator_state": str(self.orchestrator_state),\n'

DESC2_NEW = '            "orchestrator_state": str(self.orchestrator_state),\n            "transaction_journal": str(self.transaction_journal),\n'

PROP_OLD = '    def describe(self) -> dict:\n'

PROP_NEW = '    @property\n    def transaction_journal(self) -> Path:\n        """Where an in-flight installer transaction records its preimages.\n\n        INSTALLER-TRANSACTION-1. Under cache_root, never under project_root:\n        a successful installer must leave NO rollback artefact in the\n        repository, and an interrupted one must still be recoverable.\n\n        MEASURED 2026-08-19: 148 `.bak_<timestamp>` siblings had accumulated\n        inside the repository across eight days, 17,640,928 bytes, invisible\n        to `git status` because .gitignore carries `*.bak_*`. What was\n        designed as a rollback implementation detail had become a permanent\n        archival system by omission.\n        """\n        return self.cache_root / "transactions"\n\n    def describe(self) -> dict:\n'

SIG_OLD = 'def resolve_runtime_paths(*, project_root=None, artifact_root=None,\n                          state_root=None, environ=None) -> RuntimePaths:\n    """The single entry point. Resolves all three roots, or raises."""\n'

SIG_NEW = 'def _default_cache_root(env) -> Path:\n    """A user-scoped location OUTSIDE any repository.\n\n    Order, and why each link exists:\n\n        LOCALAPPDATA    on Windows. MEASURED 2026-08-19: set, and\n                        AppData/Local resolves outside the repository.\n        XDG_STATE_HOME  the POSIX convention for state that should persist\n                        but is not configuration.\n        home/.local/state\n                        the fallback that ALWAYS resolves. MEASURED: with\n                        HOME unset on Windows, Path.home() still returned\n                        C:/Users/monzi via USERPROFILE. On POSIX it falls\n                        back to the password database.\n\n    A NOTE ON WHAT CANNOT BE TESTED FROM WINDOWS. Passing a fake environment\n    with XDG_STATE_HOME="/home/runner/.local/state" selects the right BRANCH\n    but produces "C:/home/runner/..." -- because path flavour is baked into\n    the platform, not into the environment. So tests here assert\n    RELATIONSHIPS (outside the repository, beneath the chosen base) rather\n    than literal paths, and the literal POSIX form is verified on the runner.\n    """\n    if os.name == "nt" and env.get("LOCALAPPDATA"):\n        base = Path(env["LOCALAPPDATA"])\n    elif env.get("XDG_STATE_HOME"):\n        base = Path(env["XDG_STATE_HOME"])\n    else:\n        base = Path.home() / ".local" / "state"\n    return (base / "GenomicVariantClassifier").expanduser().resolve()\n\n\ndef resolve_runtime_paths(*, project_root=None, artifact_root=None,\n                          state_root=None, cache_root=None,\n                          environ=None) -> RuntimePaths:\n    """The single entry point. Resolves all FOUR roots, or raises."""\n'

RESOLVE_OLD = '    state = _resolve_secondary(\n        state_root, env.get(ENV_STATE_ROOT), project / ".gvc-state", "state_root")\n    return RuntimePaths(project_root=project, artifact_root=artifacts,\n                        state_root=state)\n'

RESOLVE_NEW = '    state = _resolve_secondary(\n        state_root, env.get(ENV_STATE_ROOT), project / ".gvc-state", "state_root")\n    cache = _resolve_secondary(\n        cache_root, env.get(ENV_CACHE_ROOT), _default_cache_root(env),\n        "cache_root")\n    return RuntimePaths(project_root=project, artifact_root=artifacts,\n                        state_root=state, cache_root=cache)\n'

TARGET = "src/genomic_variant_classifier/paths/runtime_paths.py"
MARKER = "ENV_CACHE_ROOT"

EDITS = (
    (ENV_OLD, ENV_NEW, "ENV_CACHE_ROOT"),
    (FIELD_OLD, FIELD_NEW, "cache_root: Path"),
    (PROP_OLD, PROP_NEW, "def transaction_journal"),
    (DESCRIBE_OLD, DESCRIBE_NEW, '"cache_root": str'),
    (DESC2_OLD, DESC2_NEW, '"transaction_journal": str'),
    (SIG_OLD, SIG_NEW, "def _default_cache_root"),
    (RESOLVE_OLD, RESOLVE_NEW, "cache_root=cache"),
)

#: The field order the tests assert. Latent today -- the sole construction site
#: is keyword-only -- but a future positional call would mis-assign silently.
EXPECTED_FIELDS = ("project_root", "artifact_root", "state_root", "cache_root")


def _verify(source: str) -> tuple:
    """Structural checks by AST, per ROOTFIX-VERIFY-TEXTUAL-1.

    A substring check for "cache_root" would be satisfied by the docstrings
    this edit inserts, which name it in prose repeatedly.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax error after patch: {}".format(exc)

    cls = None
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == "RuntimePaths":
            cls = n
    if cls is None:
        return False, "RuntimePaths is missing"

    fields = tuple(m.target.id for m in cls.body
                   if isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name))
    if fields != EXPECTED_FIELDS:
        return False, ("RuntimePaths fields are {}, expected {}".format(
            fields, EXPECTED_FIELDS))

    frozen = False
    for d in cls.decorator_list:
        if isinstance(d, ast.Call) and getattr(d.func, "id", None) == "dataclass":
            frozen = any(k.arg == "frozen" and getattr(k.value, "value", False)
                         for k in d.keywords)
    if not frozen:
        return False, "RuntimePaths is not frozen"

    props = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
    if "transaction_journal" not in props:
        return False, "the transaction_journal property is missing"

    # The journal must derive from cache_root -- NOT from state_root or
    # project_root, which would put it back inside the repository.
    for m in cls.body:
        if isinstance(m, ast.FunctionDef) and m.name == "transaction_journal":
            reads = {sub.attr for sub in ast.walk(m)
                     if isinstance(sub, ast.Attribute)}
            if "cache_root" not in reads:
                return False, ("transaction_journal does not derive from "
                               "cache_root (reads {})".format(sorted(reads)))
            for forbidden in ("state_root", "project_root"):
                if forbidden in reads:
                    return False, ("transaction_journal derives from {}; the "
                                   "journal must live OUTSIDE the repository"
                                   .format(forbidden))

    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "_default_cache_root" not in names:
        return False, "_default_cache_root is missing"

    assigns = {n.targets[0].id for n in tree.body
               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    if "ENV_CACHE_ROOT" not in assigns:
        return False, "ENV_CACHE_ROOT is missing"

    # The default must not be repository-relative.
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "_default_cache_root":
            src_seg = ast.dump(n)
            if "project" in src_seg and "project_root" in src_seg:
                return False, ("_default_cache_root references the project "
                               "root; the cache is machine-scoped")

    # resolve_runtime_paths must accept and pass the new keyword.
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_runtime_paths":
            kwonly = {a.arg for a in n.args.kwonlyargs}
            if "cache_root" not in kwonly:
                return False, ("resolve_runtime_paths does not accept "
                               "cache_root (has {})".format(sorted(kwonly)))
            calls = [c for c in ast.walk(n) if isinstance(c, ast.Call)
                     and getattr(c.func, "id", None) == "RuntimePaths"]
            if not calls:
                return False, "resolve_runtime_paths does not construct RuntimePaths"
            passed = {k.arg for k in calls[0].keywords}
            if passed != set(EXPECTED_FIELDS):
                return False, ("the construction passes {}, expected {}".format(
                    sorted(passed), sorted(EXPECTED_FIELDS)))

    return True, ("4 fields in order; frozen; journal from cache_root; "
                  "resolver accepts and passes it")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    p = Path(args.repo_root) / TARGET
    if not p.exists():
        print("  ERROR: not found: {}".format(TARGET))
        return 2

    # Path.read_text(newline=...) is 3.13+; the default would translate line
    # endings. This file is LF-only, but reading bytes makes that a MEASURED
    # fact rather than an assumption.
    raw = p.read_bytes()
    src = raw.decode("utf-8")
    print("  target: {} bytes, CRLF {}, non-ASCII {}".format(
        len(raw), raw.count(b"\r\n"), sum(1 for b in raw if b > 0x7F)))

    if MARKER in src:
        ok, msg = _verify(src)
        print("  already applied; current state: {}".format(msg))
        return 0 if ok else 1

    patched = src
    for i, (old, new, label) in enumerate(EDITS, 1):
        n = patched.count(old)
        if n != 1:
            print("  ERROR: anchor {} ({!r}) occurs {} time(s), expected 1; "
                  "NOTHING written.".format(i, label[:30], n))
            return 1
        patched = patched.replace(old, new, 1)
        print("  anchor {} OK  {}".format(i, label))

    ok, msg = _verify(patched)
    if not ok:
        print("  ERROR: verification failed BEFORE writing ({}); "
              "NOTHING written.".format(msg))
        return 1
    print("  pre-write  {}".format(msg))

    if args.check:
        print("\n  --check: {} edit(s) pending. Nothing written.".format(len(EDITS)))
        return 0

    backup = p.with_suffix(p.suffix + ".pre_cacheroot.bak")
    if not backup.exists():
        backup.write_bytes(raw)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        fh.write(patched)
    after = p.read_bytes()
    if after.count(b"\r\n") != raw.count(b"\r\n"):
        p.write_bytes(raw)
        print("  ERROR: line endings changed; ROLLED BACK.")
        return 1
    print("  wrote {}  (CRLF {}, non-ASCII {})".format(
        TARGET, after.count(b"\r\n"), sum(1 for b in after if b > 0x7F)))

    ok, msg = _verify(after.decode("utf-8"))
    if not ok:
        p.write_bytes(raw)
        print("  ERROR: POST-WRITE failed ({}); ROLLED BACK.".format(msg))
        return 1
    print("  post-write {}".format(msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
