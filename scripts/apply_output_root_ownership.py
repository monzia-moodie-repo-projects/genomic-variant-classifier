#!/usr/bin/env python3
"""apply_output_root_ownership.py -- Author: Monzia Moodie

OUTPUT-ROOT-CONFLATION-1: a path derives from the authority that owns what the
path contains.

THE DEFECT
    SHAP_REPORT_DIR       = PROJECT_ROOT / "reports" / "shap"
    LITERATURE_DIGEST_DIR = PROJECT_ROOT / "reports" / "literature"

Both are ARTIFACT DESTINATIONS computed from REPOSITORY identity. Where output
goes is a deployment decision, not a fact about where the source lives.

MEASURED 2026-08-19, at ec8e51b
    config.py            17,852 bytes, 422 lines, CRLF 421, non-ASCII 41
    SHAP_REPORT_DIR       line 194   (was 174 before the resolver comment block)
    LITERATURE_DIGEST_DIR line 327   (was 307)
    readers               interpretability_agent and literature_scout_agent,
                          one each
    artifact_root == project_root on this workstation -- so the defect is
    INVISIBLE under the default configuration.

ONE AUTHORITY, NOT TWO
    _RUNTIME_PATHS = resolve_runtime_paths()
    PROJECT_ROOT   = _RUNTIME_PATHS.project_root
    SHAP_REPORT_DIR       = _RUNTIME_PATHS.reports_root / "shap"
    LITERATURE_DIGEST_DIR = _RUNTIME_PATHS.reports_root / "literature"

NOT resolve_project_root() alongside a second resolve_runtime_paths() call.
That would create TWO authorities for one process -- the parallel-vocabulary
defect this project keeps removing -- and each call performs a full discovery
walk.

The single call is a CONFIGURATION SNAPSHOT, not an optimisation: runtime path
configuration is immutable for the lifetime of a process. Fresh process, fresh
resolution; existing process, stable paths.

_RUNTIME_PATHS is PRIVATE deliberately. The authority belongs in
paths.runtime_paths; a public name would invite `from config import
RUNTIME_PATHS` imports that replace the old global constants with one global
service locator.

TWO TESTS FROM ec8e51b ARE REVISED, AND THAT IS THE POINT
    test_project_root_is_assigned_from_the_resolver
    test_the_resolver_is_imported

Both required a PARTICULAR implementation -- the exact call, the exact import
-- rather than the property the commit existed to guarantee. Appropriate as
migration guards for PROJECT-ROOT-HARDCODED-1; wrong as permanent
architectural constraints. This is the milder form of the /probe_anchor
mistake: encoding HOW as the contract for WHAT.

And test_derived_constants_track_the_resolved_root asserted

    SHAP_REPORT_DIR == PROJECT_ROOT / "reports" / "shap"

which LITERALLY SPECIFIES the defect being closed.

THE RELEASE-BLOCKING TEST NEEDS SEPARATED ROOTS
On this workstation both roots are equal, so no test confined to it can
validate the boundary -- the same lesson the sentinel repair taught, where a
rule valid inside the checkout was impossible inside the trainer image.

    An artifact path contract must be tested in an environment where artifact
    identity DIFFERS from repository identity.

test_the_two_root_domains_can_DIVERGE uses GVC_ARTIFACT_ROOT, the supported
injection mechanism, and asserts BOTH directions: reports follow artifact
identity, checkpoints do NOT. The repair is an OWNERSHIP correction, not a
blanket move of every path under artifact_root.

VERIFIED
    OLD test + OLD config    16 passed
    NEW test + OLD config     3 FAILED   (one authority; reports_root; divergence)
    NEW test + NEW config     20 passed
    6 of 6 sabotage mutations detected, including a second authority added,
    reports reverting to PROJECT_ROOT, and checkpoints or models swept under
    artifact_root.

NOT IN THIS UNIT
CONFIG-DEAD-PATHS-1. The seven stale constants remain asserted present; the 28
unwired roadmap values are not proven dead merely by lacking consumers today.

Idempotent, ast-verifies before AND after writing, backs up each file, and
rolls BOTH back if any post-write check fails.

Usage:  python scripts/apply_output_root_ownership.py --repo-root . --check
        python scripts/apply_output_root_ownership.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

CFG_IMPORT_OLD = 'from genomic_variant_classifier.paths.runtime_paths import resolve_project_root\r\n'

CFG_IMPORT_NEW = 'from genomic_variant_classifier.paths.runtime_paths import resolve_runtime_paths\r\n'

CFG_ROOT_OLD = 'PROJECT_ROOT = resolve_project_root()\r\n'

CFG_ROOT_NEW = '#\r\n# OUTPUT-ROOT-CONFLATION-1 (2026-08-19). ONE authority, resolved ONCE.\r\n#\r\n# PROJECT-ROOT-HARDCODED-1 replaced a Windows literal with\r\n# resolve_project_root(). That was correct for its scope, but artifact\r\n# destinations also lived here and were computed from REPOSITORY identity:\r\n#\r\n#     SHAP_REPORT_DIR       = PROJECT_ROOT / "reports" / "shap"\r\n#     LITERATURE_DIGEST_DIR = PROJECT_ROOT / "reports" / "literature"\r\n#\r\n# Where output goes is a DEPLOYMENT decision, not a fact about where the\r\n# source lives. RuntimePaths separates the two, so both now derive from\r\n# artifact identity while repository-owned paths stay repository-owned.\r\n#\r\n# ONE resolver call, not two. Adding resolve_runtime_paths() alongside\r\n# resolve_project_root() would create TWO authorities for one process --\r\n# exactly the parallel-vocabulary defect this project keeps removing -- and\r\n# each call performs a full discovery walk.\r\n#\r\n# THIS IS A CONFIGURATION SNAPSHOT, not merely an optimisation. Runtime path\r\n# configuration is immutable for the lifetime of a process: environment\r\n# variables are input to resolution at initialisation, and mutating them\r\n# afterwards does NOT reconfigure a running process. A process whose source\r\n# identity is one directory while its artifact identity silently moves\r\n# mid-execution would be far harder to reason about.\r\n#\r\n#     fresh process   -> fresh resolution\r\n#     existing process -> stable paths\r\n#\r\n# Kept PRIVATE deliberately. The authority belongs in paths.runtime_paths,\r\n# not here; a public name would invite `from config import RUNTIME_PATHS`\r\n# imports that replace the old global constants with one global service\r\n# locator. Consumers should eventually receive RuntimePaths by injection.\r\n_RUNTIME_PATHS = resolve_runtime_paths()\r\n\r\nPROJECT_ROOT = _RUNTIME_PATHS.project_root\r\n'

CFG_SHAP_OLD = 'SHAP_REPORT_DIR        = PROJECT_ROOT / "reports" / "shap"\r\n'

CFG_SHAP_NEW = 'SHAP_REPORT_DIR        = _RUNTIME_PATHS.reports_root / "shap"\r\n'

CFG_LIT_OLD = 'LITERATURE_DIGEST_DIR  = PROJECT_ROOT / "reports" / "literature"\r\n'

CFG_LIT_NEW = 'LITERATURE_DIGEST_DIR  = _RUNTIME_PATHS.reports_root / "literature"\r\n'

T1_OLD = 'def test_project_root_is_assigned_from_the_resolver():\n    """By syntax tree, not by substring: the assignment\'s value must be a CALL\n    to resolve_project_root, not a Path(os.getenv(...)) expression."""\n    tree = ast.parse(io.open(_CONFIG, encoding="utf-8").read())\n    for n in tree.body:\n        if (isinstance(n, ast.Assign) and n.targets\n                and isinstance(n.targets[0], ast.Name)\n                and n.targets[0].id == "PROJECT_ROOT"):\n            assert isinstance(n.value, ast.Call), ast.dump(n.value)[:120]\n            fn = getattr(n.value.func, "id", None) or getattr(n.value.func, "attr", None)\n            assert fn == "resolve_project_root", fn\n            return\n    raise AssertionError("PROJECT_ROOT is not assigned at module level")\n'

T1_NEW = 'def test_project_root_equals_the_one_runtime_path_authority():\n    """A SEMANTIC contract, not an implementation one.\n\n    This test previously required `PROJECT_ROOT = resolve_project_root()` by\n    syntax tree -- the exact call and the exact import. That was an appropriate\n    migration guard for PROJECT-ROOT-HARDCODED-1, and it is the milder form of\n    the /probe_anchor mistake: encoding HOW something is done as the contract\n    for WHAT it must be.\n\n    OUTPUT-ROOT-CONFLATION-1 changed the how -- one resolve_runtime_paths()\n    call replacing resolve_project_root() -- without changing anything the\n    commit existed to guarantee. The property is that PROJECT_ROOT is the\n    validated project root of the ONE runtime-path authority this module\n    resolved.\n    """\n    assert C.PROJECT_ROOT == C._RUNTIME_PATHS.project_root\n\n\ndef test_there_is_exactly_ONE_runtime_path_authority():\n    """Two resolver calls would be two authorities for one process, and each\n    performs a full discovery walk. The module resolves once."""\n    import ast\n    tree = ast.parse(io.open(_CONFIG, encoding="utf-8").read())\n    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)\n             and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))\n             in ("resolve_runtime_paths", "resolve_project_root")]\n    assert len(calls) == 1, (\n        "config.py makes {} resolver call(s); exactly one authority is "\n        "resolved per process".format(len(calls)))\n'

T2_OLD = 'def test_the_resolver_is_imported():\n    tree = ast.parse(io.open(_CONFIG, encoding="utf-8").read())\n    found = set()\n    for n in ast.walk(tree):\n        if isinstance(n, ast.ImportFrom) and n.module:\n            for a in n.names:\n                found.add("{}.{}".format(n.module, a.name))\n    assert ("genomic_variant_classifier.paths.runtime_paths.resolve_project_root"\n            in found), sorted(found)\n'

T2_NEW = 'def test_no_absolute_path_literal_and_no_env_fallback_survives():\n    """What the resolver import was standing in for.\n\n    The old test named the imported symbol. The property is that PROJECT_ROOT\n    is not read from an environment variable with a literal default -- the\n    shape that made one author\'s absolute path the value on every machine.\n    """\n    import ast\n    tree = ast.parse(io.open(_CONFIG, encoding="utf-8").read())\n    for n in tree.body:\n        if (isinstance(n, ast.Assign) and n.targets\n                and getattr(n.targets[0], "id", None) == "PROJECT_ROOT"):\n            for sub in ast.walk(n.value):\n                if isinstance(sub, ast.Call):\n                    fn = (getattr(sub.func, "id", None)\n                          or getattr(sub.func, "attr", None))\n                    assert fn != "getenv", (\n                        "PROJECT_ROOT is read from the environment with a "\n                        "default; resolution must verify identity and RAISE")\n            return\n    raise AssertionError("PROJECT_ROOT is not assigned at module level")\n'

T3_OLD = 'def test_derived_constants_track_the_resolved_root():\n    """Not merely absolute -- rooted at the RESOLVED root, so a change in\n    resolution moves them together."""\n    assert C.CHECKPOINT_DIR == C.MODELS_DIR / "checkpoints"\n    assert C.MODELS_DIR == C.PROJECT_ROOT / "models"\n    assert C.SHAP_REPORT_DIR == C.PROJECT_ROOT / "reports" / "shap"\n    assert C.LITERATURE_DIGEST_DIR == C.PROJECT_ROOT / "reports" / "literature"\n'

T3_NEW = 'def test_repository_paths_derive_from_PROJECT_ROOT():\n    """Repository-owned paths stay repository-owned. This is half of\n    OUTPUT-ROOT-CONFLATION-1: the repair is an OWNERSHIP correction, not a\n    blanket move of every path under artifact_root."""\n    assert C.MODELS_DIR == C.PROJECT_ROOT / "models"\n    assert C.CHECKPOINT_DIR == C.MODELS_DIR / "checkpoints"\n\n\ndef test_report_paths_derive_from_REPORTS_ROOT_not_project_root():\n    """The other half. These two were computed from repository identity while\n    being artifact destinations -- and the previous version of this test\n    asserted exactly that, so it literally specified the defect."""\n    assert C.SHAP_REPORT_DIR == C._RUNTIME_PATHS.reports_root / "shap"\n    assert C.LITERATURE_DIGEST_DIR == C._RUNTIME_PATHS.reports_root / "literature"\n\n\ndef test_the_two_root_domains_can_DIVERGE(tmp_path):\n    """THE RELEASE-BLOCKING TEST for this unit.\n\n    MEASURED 2026-08-19: on this workstation artifact_root == project_root, so\n    the defect is INVISIBLE under the default configuration. Testing only here\n    can never validate the boundary -- the same lesson the sentinel repair\n    taught, where a rule valid inside the checkout was impossible inside the\n    trainer image.\n\n        An artifact path contract must be tested in an environment where\n        artifact identity DIFFERS from repository identity.\n\n    Uses the supported RuntimePaths injection mechanism (GVC_ARTIFACT_ROOT),\n    not a new test-only vocabulary. Both directions are asserted: report paths\n    follow the artifact root, and checkpoints do NOT.\n    """\n    repo = tmp_path / "repo"\n    (repo / "src" / "genomic_variant_classifier").mkdir(parents=True)\n    (repo / "pyproject.toml").write_text(\n        \'[project]\\nname = "genomic-variant-classifier"\\nversion = "0.1.0"\\n\',\n        encoding="utf-8")\n    artifacts = tmp_path / "artifacts"\n    artifacts.mkdir()\n\n    code = (\n        "from genomic_variant_classifier.agent_layer import config as C\\n"\n        "print(C.PROJECT_ROOT)\\n"\n        "print(C.SHAP_REPORT_DIR)\\n"\n        "print(C.LITERATURE_DIGEST_DIR)\\n"\n        "print(C.CHECKPOINT_DIR)\\n"\n    )\n    import os\n    env = dict(os.environ, GVC_PROJECT_ROOT=str(repo),\n               GVC_ARTIFACT_ROOT=str(artifacts))\n    out = subprocess.run([sys.executable, "-B", "-c", code], env=env,\n                         capture_output=True, text=True, timeout=300)\n    assert out.returncode == 0, out.stderr[-800:]\n    root, shap, lit, ckpt = [Path(l) for l in out.stdout.strip().splitlines()]\n\n    assert root == repo.resolve(), (root, repo)\n    assert root != artifacts.resolve()\n\n    # Artifact-owned: they follow artifact identity.\n    assert shap == artifacts.resolve() / "reports" / "shap", shap\n    assert lit == artifacts.resolve() / "reports" / "literature", lit\n\n    # Repository-owned: they do NOT.\n    assert ckpt == repo.resolve() / "models" / "checkpoints", ckpt\n    assert artifacts.resolve() not in ckpt.parents, ckpt\n\n\ndef test_runtime_paths_are_a_SNAPSHOT_not_a_live_lookup(tmp_path):\n    """Runtime path configuration is immutable for the lifetime of a process.\n\n    Mutating the environment after import must NOT relocate a running process:\n    a program whose artifact identity silently moved mid-execution would be far\n    harder to reason about than one that requires a restart.\n\n        fresh process    -> fresh resolution\n        existing process -> stable paths\n    """\n    code = (\n        "import os\\n"\n        "from genomic_variant_classifier.agent_layer import config as C\\n"\n        "before = str(C.SHAP_REPORT_DIR)\\n"\n        "os.environ[\'GVC_ARTIFACT_ROOT\'] = %r\\n"\n        "import importlib\\n"\n        "print(before)\\n"\n        "print(str(C.SHAP_REPORT_DIR))\\n"\n    ) % str(tmp_path)\n    out = subprocess.run([sys.executable, "-B", "-c", code],\n                         capture_output=True, text=True, timeout=300)\n    assert out.returncode == 0, out.stderr[-800:]\n    before, after = out.stdout.strip().splitlines()\n    assert before == after, (before, after)\n'

CONFIG = "src/genomic_variant_classifier/agent_layer/config.py"
ROOT_TEST = "tests/unit/test_config_root.py"

EDITS = (
    (CONFIG, CFG_IMPORT_OLD, CFG_IMPORT_NEW, "import resolve_runtime_paths"),
    (CONFIG, CFG_ROOT_OLD, CFG_ROOT_NEW, "_RUNTIME_PATHS = resolve_runtime_paths()"),
    (CONFIG, CFG_SHAP_OLD, CFG_SHAP_NEW, "_RUNTIME_PATHS.reports_root / \"shap\""),
    (CONFIG, CFG_LIT_OLD, CFG_LIT_NEW, "_RUNTIME_PATHS.reports_root / \"literature\""),
    (ROOT_TEST, T1_OLD, T1_NEW, "test_there_is_exactly_ONE_runtime_path_authority"),
    (ROOT_TEST, T2_OLD, T2_NEW, "test_no_absolute_path_literal_and_no_env_fallback_survives"),
    (ROOT_TEST, T3_OLD, T3_NEW, "test_the_two_root_domains_can_DIVERGE"),
)

#: Repository-owned. These must NOT move to artifact_root -- the repair is an
#: ownership correction, not a blanket transformation.
REPOSITORY_OWNED = ("MODELS_DIR", "CHECKPOINT_DIR", "VAL_PARQUET")

#: Artifact-owned. These must derive from reports_root.
ARTIFACT_OWNED = ("SHAP_REPORT_DIR", "LITERATURE_DIGEST_DIR")

#: MEASURED: the seven stale path constants. NOT removed here -- that is
#: CONFIG-DEAD-PATHS-1 and a scope decision. Asserted so removal is deliberate.
STALE_RETAINED = ("SHARED_STATE_PATH", "AUDIT_LOG_DIR", "RAW_DATA_DIR",
                  "CORPUS_MANIFEST_PATH", "PROCESSED_DATA_DIR",
                  "TRAIN_PARQUET", "REPLAY_BUFFER_PARQUET")


def _assignments(tree) -> dict:
    out = {}
    for n in tree.body:
        name = None
        if isinstance(n, ast.Assign) and n.targets and isinstance(n.targets[0], ast.Name):
            name = n.targets[0].id
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            name = n.target.id
        if name is not None and n.value is not None:
            out[name] = n.value
    return out


def _root_name(value) -> str:
    """The first Name or Attribute base an expression reads from."""
    for sub in ast.walk(value):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
            return "{}.{}".format(sub.value.id, sub.attr)
        if isinstance(sub, ast.Name):
            return sub.id
    return "?"


def _verify_config(source: str) -> tuple:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "config.py: syntax error after patch: {}".format(exc)

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
             in ("resolve_runtime_paths", "resolve_project_root")]
    if len(calls) != 1:
        return False, ("config.py: {} resolver call(s); exactly ONE authority "
                       "is resolved per process".format(len(calls)))
    fn = getattr(calls[0].func, "id", None) or getattr(calls[0].func, "attr", None)
    if fn != "resolve_runtime_paths":
        return False, "config.py: the authority is {!r}".format(fn)

    a = _assignments(tree)
    if "_RUNTIME_PATHS" not in a:
        return False, "config.py: _RUNTIME_PATHS is missing"
    if "PROJECT_ROOT" not in a:
        return False, "config.py: PROJECT_ROOT is missing"
    if _root_name(a["PROJECT_ROOT"]) != "_RUNTIME_PATHS.project_root":
        return False, ("config.py: PROJECT_ROOT derives from {}, not the "
                       "authority".format(_root_name(a["PROJECT_ROOT"])))

    for name in ARTIFACT_OWNED:
        if name not in a:
            return False, "config.py: {} is missing".format(name)
        base = _root_name(a[name])
        if base != "_RUNTIME_PATHS.reports_root":
            return False, ("config.py: {} derives from {}; artifact "
                           "destinations derive from reports_root"
                           .format(name, base))

    for name in REPOSITORY_OWNED:
        if name not in a:
            return False, "config.py: {} is missing".format(name)
        base = _root_name(a[name])
        if base.startswith("_RUNTIME_PATHS.artifact_root"):
            return False, ("config.py: {} was swept under artifact_root; the "
                           "repair is an OWNERSHIP correction, not a blanket "
                           "transformation".format(name))

    gone = [c for c in STALE_RETAINED if c not in a]
    if gone:
        return False, ("config.py: stale constant(s) removed without a "
                       "recorded decision: {}".format(gone))

    # No getenv fallback may reappear for PROJECT_ROOT.
    for sub in ast.walk(a["PROJECT_ROOT"]):
        if isinstance(sub, ast.Call):
            f = getattr(sub.func, "id", None) or getattr(sub.func, "attr", None)
            if f == "getenv":
                return False, "config.py: PROJECT_ROOT reads getenv again"

    return True, ("config.py: one authority; {} artifact-owned from "
                  "reports_root; {} repository-owned unchanged; {} stale "
                  "intact".format(len(ARTIFACT_OWNED), len(REPOSITORY_OWNED),
                                  len(STALE_RETAINED)))


def _verify_root_test(source: str) -> tuple:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "root test: syntax error after patch: {}".format(exc)
    tests = {n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    for required in ("test_project_root_equals_the_one_runtime_path_authority",
                     "test_there_is_exactly_ONE_runtime_path_authority",
                     "test_report_paths_derive_from_REPORTS_ROOT_not_project_root",
                     "test_repository_paths_derive_from_PROJECT_ROOT",
                     "test_the_two_root_domains_can_DIVERGE",
                     "test_runtime_paths_are_a_SNAPSHOT_not_a_live_lookup"):
        if required not in tests:
            return False, "root test: {} is missing".format(required)
    for gone in ("test_project_root_is_assigned_from_the_resolver",
                 "test_the_resolver_is_imported",
                 "test_derived_constants_track_the_resolved_root"):
        if gone in tests:
            return False, ("root test: {} survives; it encodes an "
                           "IMPLEMENTATION contract".format(gone))
    return True, "root test: {} test(s), implementation contracts replaced".format(
        len(tests))


VERIFIERS = {CONFIG: _verify_config, ROOT_TEST: _verify_root_test}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root)

    sources = {}
    for rel in (CONFIG, ROOT_TEST):
        p = repo / rel
        if not p.exists():
            print("  ERROR: not found: {}".format(rel))
            return 2
        raw = p.read_bytes()          # NOT read_text: newline= is 3.13+, and
        sources[rel] = raw.decode("utf-8")   # the default translates CRLF.
        print("  {:<48} {} bytes, CRLF {}, non-ASCII {}".format(
            rel.split("/")[-1], len(raw), raw.count(b"\r\n"),
            sum(1 for b in raw if b > 0x7F)))

    pending, already = {}, 0
    for rel, old, new, marker in EDITS:
        if marker in sources[rel]:
            already += 1
            print("  {:<52} already applied".format(marker[:50]))
            continue
        cur = pending.get(rel, sources[rel])
        n = cur.count(old)
        if n != 1:
            print("  ERROR: anchor for {!r} occurs {} time(s) in {}, expected "
                  "1; NOTHING written.".format(marker[:34], n, rel))
            return 1
        print("  {:<52} anchor OK".format(marker[:50]))
        pending[rel] = cur.replace(old, new, 1)

    if not pending:
        print("\n  All {} edit(s) already applied.".format(len(EDITS)))
        return 0

    for rel, patched in pending.items():
        ok, msg = VERIFIERS[rel](patched)
        if not ok:
            print("  ERROR: verification failed BEFORE writing ({}); "
                  "NOTHING written.".format(msg))
            return 1
        print("  pre-write  {}".format(msg))

    if args.check:
        print("\n  --check: {} file(s) pending, {} already applied. Nothing "
              "written.".format(len(pending), already))
        return 0

    written = []
    for rel, patched in pending.items():
        p = repo / rel
        before = p.read_bytes()
        bak = p.with_suffix(p.suffix + ".pre_ownership.bak")
        if not bak.exists():
            bak.write_bytes(before)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(patched)
        written.append((p, before))
        after = p.read_bytes()
        b_na = sum(1 for b in before if b > 0x7F)
        a_na = sum(1 for b in after if b > 0x7F)
        if a_na != b_na:
            for p2, b2 in written:
                p2.write_bytes(b2)
            print("  ERROR: {} non-ASCII {} -> {}; ROLLED BACK.".format(
                rel, b_na, a_na))
            return 1
        print("  wrote {}  ({} non-ASCII preserved, CRLF {})".format(
            rel.split("/")[-1], a_na, after.count(b"\r\n")))

    for p, before in written:
        rel = str(p).replace("\\", "/")
        key = CONFIG if rel.endswith("config.py") else ROOT_TEST
        ok, msg = VERIFIERS[key](p.read_bytes().decode("utf-8"))
        if not ok:
            for p2, b2 in written:
                p2.write_bytes(b2)
            print("  ERROR: POST-WRITE failed ({}); ROLLED BACK all {} file(s)."
                  .format(msg, len(written)))
            return 1
        print("  post-write {}".format(msg))

    print("\n  {} file(s) patched; {} edit(s) already were.".format(
        len(written), already))
    return 0


if __name__ == "__main__":
    sys.exit(main())
