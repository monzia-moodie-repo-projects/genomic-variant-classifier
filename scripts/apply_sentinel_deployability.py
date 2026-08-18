#!/usr/bin/env python3
"""apply_sentinel_deployability.py -- Author: Monzia Moodie

RUNTIME-SENTINEL-TEST-ARTEFACT-1: a sentinel must exist in every environment
where the root must be found.

THE DEFECT
    PROJECT_SENTINELS = (
        "pyproject.toml",
        "src/genomic_variant_classifier",
        "tests/EXPECTED_SUITE_SIZE",      <-- a TEST-SUITE ARTEFACT
    )

The third entry identifies a DEPLOYMENT root by a file every correct deployment
excludes. The conjunction could not hold in a container by construction.

MEASURED 2026-08-17 against this repository's own build files:

    Dockerfile:185    COPY . .            (trainer stage, WORKDIR /app)
    .dockerignore     tests/              excluded

    pyproject.toml                  in trainer image : True
    src/genomic_variant_classifier  in trainer image : True
    tests/EXPECTED_SUITE_SIZE       in trainer image : FALSE

So resolve_project_root() would have RAISED on import of any module reaching
agent_layer.config inside the trainer image -- which is where cloud training
runs. The API image is unaffected for a different reason: Dockerfile lines
109-126 copy only api/, models/, utils/, monitoring/model_registry.py and the
package __init__.py, so agent_layer/ is not present there at all.

The defect was latent from 69a9597 (2026-08-14) only because nothing imported
runtime_paths yet. PROJECT-ROOT-HARDCODED-1 is its first consumer, and
config.py is imported at MODULE SCOPE by thirteen modules -- so the failure
would have been total inside that image.

WHY THE REPAIR COSTS NOTHING
Two sentinels plus the declared project name lose NO discrimination. Measured:

    this repository : True
    C:/Users/monzi  : False      C:/Projects : False
    C:/Windows      : False      C:/         : False

HOW I FOUND IT
Not by inspection. config.py is imported at module scope by thirteen modules,
so I asked what happens if resolution FAILS -- and that question required
measuring a second environment. The existing test asserting "every sentinel is
load-bearing" passed throughout, because it only ever ran inside the
repository. It was invisible to that test by construction.

THE RATCHET DECREASES BY ONE
test_every_sentinel_is_load_bearing is parametrized over PROJECT_SENTINELS, so
it drops from three cases to two. A deliberate decrease, recorded as
PHYLOPTEST-DUP-1 was. The test still asserts every REMAINING sentinel is
load-bearing; it simply has one fewer to check, because one was wrong to
require.

Idempotent, ast-verifies before AND after writing, backs up to
.pre_sentinel.bak, and rolls back if any post-write check fails.

Usage:  python scripts/apply_sentinel_deployability.py --repo-root . --check
        python scripts/apply_sentinel_deployability.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import sys
from pathlib import Path

SENTINELS_OLD = '#: A directory is this repository only if it holds ALL of these AND declares\n#: the project name above. Each alone is common; the conjunction is not.\nPROJECT_SENTINELS = (\n    "pyproject.toml",\n    "src/genomic_variant_classifier",\n    "tests/EXPECTED_SUITE_SIZE",\n)\n'

SENTINELS_NEW = '#: A directory is this repository only if it holds ALL of these AND declares\n#: the project name above. Each alone is common; the conjunction is not.\n#:\n#: RUNTIME-SENTINEL-TEST-ARTEFACT-1 (2026-08-17). This tuple once included\n#: "tests/EXPECTED_SUITE_SIZE". That is a TEST-SUITE ARTEFACT being used to\n#: identify a DEPLOYMENT root, and every correct deployment excludes tests --\n#: so the conjunction could not hold in a container by construction.\n#:\n#: MEASURED against this repository\'s own Dockerfile and .dockerignore:\n#:\n#:     pyproject.toml                  copied into the trainer image : True\n#:     src/genomic_variant_classifier  copied into the trainer image : True\n#:     tests/EXPECTED_SUITE_SIZE       copied into the trainer image : FALSE\n#:                                     (excluded by `tests/` at .dockerignore)\n#:\n#: The trainer image runs `COPY . .`, so discovery would have RAISED on import\n#: of any module reaching agent_layer.config -- which is where cloud training\n#: runs. The defect was latent from 69a9597 only because nothing imported this\n#: module yet; PROJECT-ROOT-HARDCODED-1 is its first consumer.\n#:\n#: Two sentinels plus the declared name lose NO discrimination. Measured:\n#:\n#:     this repository : True\n#:     C:/Users/monzi  : False      C:/Projects : False\n#:     C:/Windows      : False      C:/         : False\n#:\n#: A sentinel must be present in EVERY environment where the root must be\n#: found. Test artefacts, documentation and build outputs are not.\nPROJECT_SENTINELS = (\n    "pyproject.toml",\n    "src/genomic_variant_classifier",\n)\n'

TARGET = "src/genomic_variant_classifier/paths/runtime_paths.py"
MARKER = "RUNTIME-SENTINEL-TEST-ARTEFACT-1"

#: Artefact roots that correct deployments exclude. A sentinel under any of
#: these is a category error regardless of what .dockerignore says today.
FORBIDDEN_ROOTS = {"tests", "test", "docs", "doc", "build", "dist",
                   "notebooks", "htmlcov", ".github", "logs", "outputs"}


def _dockerignore_patterns(repo: Path) -> list:
    p = repo / ".dockerignore"
    if not p.is_file():
        return []
    return [l.strip().rstrip("/")
            for l in io.open(p, encoding="utf-8").read().splitlines()
            if l.strip() and not l.strip().startswith("#")]


def _survives_copy(rel: str, patterns: list) -> bool:
    top = rel.split("/")[0]
    return not any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(top, pat)
                   for pat in patterns)


def _verify(source: str, repo: Path) -> tuple:
    """Structural checks by AST, per ROOTFIX-VERIFY-TEXTUAL-1.

    `if "tests/" not in source` would be satisfied by the docstring, which
    quotes the removed sentinel deliberately. This reads the tuple.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "syntax error after patch: {}".format(exc)

    sentinels = None
    for n in tree.body:
        if (isinstance(n, ast.Assign) and n.targets
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "PROJECT_SENTINELS"):
            try:
                sentinels = ast.literal_eval(n.value)
            except ValueError:
                return False, "PROJECT_SENTINELS is not a literal"
    if sentinels is None:
        return False, "PROJECT_SENTINELS is missing"
    if len(sentinels) < 2:
        return False, ("PROJECT_SENTINELS has {} entr(ies); a single sentinel "
                       "is not a conjunction".format(len(sentinels)))

    bad_root = [s for s in sentinels if s.split("/")[0] in FORBIDDEN_ROOTS]
    if bad_root:
        return False, ("sentinel(s) under an artefact root that correct "
                       "deployments exclude: {}".format(bad_root))

    patterns = _dockerignore_patterns(repo)
    if patterns:
        excluded = [s for s in sentinels if not _survives_copy(s, patterns)]
        if excluded:
            return False, ("sentinel(s) excluded by .dockerignore, so they "
                           "cannot be found in the trainer image: {}"
                           .format(excluded))

    # The declared-name check must survive; it carries the discrimination the
    # removed sentinel was credited with.
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for required in ("_has_sentinels", "_declares_project_name",
                     "looks_like_project_root"):
        if required not in names:
            return False, "{} is missing".format(required)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "looks_like_project_root":
            called = {getattr(c.func, "id", None) for c in ast.walk(n)
                      if isinstance(c, ast.Call)}
            if "_declares_project_name" not in called:
                return False, ("looks_like_project_root no longer checks the "
                               "declared project name")
    return True, "{} sentinel(s), all deployable, name check intact".format(
        len(sentinels))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    p = repo / TARGET
    if not p.exists():
        print("  ERROR: not found: {}".format(TARGET))
        return 2
    src = p.read_text(encoding="utf-8")

    if MARKER in src:
        print("  {} already applied".format(MARKER))
        ok, msg = _verify(src, repo)
        print("  current state: {}".format(msg))
        return 0 if ok else 1

    n = src.count(SENTINELS_OLD)
    if n != 1:
        print("  ERROR: the anchor occurs {} time(s), expected 1; "
              "NOTHING written.".format(n))
        return 1
    print("  anchor OK  (1 occurrence)")

    patched = src.replace(SENTINELS_OLD, SENTINELS_NEW, 1)
    ok, msg = _verify(patched, repo)
    if not ok:
        print("  ERROR: verification failed BEFORE writing ({}); "
              "NOTHING written.".format(msg))
        return 1
    print("  pre-write  {}".format(msg))

    if args.check:
        print("\n  --check: 1 edit pending. Nothing written.")
        return 0

    backup = p.with_suffix(p.suffix + ".pre_sentinel.bak")
    if not backup.exists():
        backup.write_bytes(p.read_bytes())
    p.write_text(patched, encoding="utf-8", newline="\n")
    print("  wrote {}".format(TARGET))

    ok, msg = _verify(p.read_text(encoding="utf-8"), repo)
    if not ok:
        p.write_bytes(backup.read_bytes())
        print("  ERROR: POST-WRITE verification failed ({}); ROLLED BACK."
              .format(msg))
        return 1
    print("  post-write {}".format(msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
