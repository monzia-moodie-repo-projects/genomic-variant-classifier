#!/usr/bin/env python3
"""apply_config_root_resolver.py -- Author: Monzia Moodie

PROJECT-ROOT-HARDCODED-1: config.py adopts the runtime-path resolver, and the
one test that encoded the defect as a contract is repaired with it.

THE DEFECT
    PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT",
                                   r"C:\\Projects\\genomic-variant-classifier"))

MEASURED 2026-08-14: GVC_PROJECT_ROOT is set NOWHERE -- not in continuous
integration, not in the Dockerfile, not in any script, not in the shell. So the
fallback was the value EVERY consumer received, and on the Linux runner it
named a path that cannot exist.

BLAST RADIUS, MEASURED BY SYNTAX TREE 2026-08-17
    4  definitions of PROJECT_ROOT. THREE WERE ALREADY CORRECT: c3_inventory.py
       :21, c3_sweep.py:29 and run11_preflight.py:27 use
       Path(__file__).resolve().parent.parent. Only config.py:17 was defective.
    17 load-context references to it, 0 attribute accesses, 13 importers.

WHY THIS SCRIPT ALSO EDITS A TEST
A first attempt patched config.py alone and the suite gate caught it:

    FAILED test_agent_root_anchor.py::test_the_default_TRACKS_the_environment_not_the_cwd
    4786 passed, 10 skipped, 1 failed   (expected 4787 / 10 / 0)

That test sets GVC_PROJECT_ROOT to a bare "/probe_anchor" -- a path that does
not exist -- and asserts the agent follows it. Under the old line os.getenv
accepted ANY string. Under the resolver a non-repository RAISES, which is the
point. The test encoded the DEFECT as a contract, and its own docstring said
so: "config.py reads GVC_PROJECT_ROOT at import time".

PROVEN BY A THREE-WAY MATRIX, not argued:
    OLD test + OLD config    2 passed
    OLD test + NEW config    1 FAILED    <- the gate's failure, reproduced
    NEW test + NEW config    3 passed

The repaired test keeps its property -- a variable moves the anchor, the
working directory does not -- with a fixture that builds a REAL repository, and
adds the assertion the old one could not make: that "/probe_anchor" is refused.

TWO FILES, TWO LINE-ENDING CONVENTIONS
config.py is CRLF (401 of 402 lines). test_agent_root_anchor.py is LF-only.
Each anchor was measured against its own target; an anchor with the wrong
terminator matches NOTHING, silently.

THE REPLACEMENT USES ONLY NAMES THE TEST FILE HAS
Measured: that file imports subprocess, sys and pytest -- no io, no Path, no
os. A first draft used io.open and would have raised NameError. The
replacement has ZERO free names, importing os locally as the original does.

A DEFECT FOUND BY RUNNING THIS SCRIPT
Path.read_text(newline=...) is Python 3.13+. This environment is 3.12, so the
call raised TypeError before any output. Worse, WITHOUT that argument read_text
performs universal-newline translation -- converting CRLF to LF and making
every config.py anchor match nothing. This reads bytes and decodes explicitly.

Idempotent, ast-verifies before AND after writing, backs up each file, and
rolls BOTH back if any post-write check fails.

Usage:  python scripts/apply_config_root_resolver.py --repo-root . --check
        python scripts/apply_config_root_resolver.py --repo-root .
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

CONFIG_IMPORT_OLD = 'import os\r\nfrom pathlib import Path\r\n'

CONFIG_IMPORT_NEW = 'import os\r\nfrom pathlib import Path\r\n\r\nfrom genomic_variant_classifier.paths.runtime_paths import resolve_project_root\r\n'

CONFIG_ROOT_OLD = 'PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT", r"C:\\Projects\\genomic-variant-classifier"))\r\n'

CONFIG_ROOT_NEW = '#\r\n# PROJECT-ROOT-HARDCODED-1. This line read:\r\n#\r\n#     PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT",\r\n#                                    r"C:\\Projects\\genomic-variant-classifier"))\r\n#\r\n# MEASURED 2026-08-14: GVC_PROJECT_ROOT is set NOWHERE -- not in continuous\r\n# integration, not in the Dockerfile, not in any script, not in the shell. So\r\n# the fallback was the value EVERY consumer received, and on the Linux runner\r\n# it named a path that cannot exist.\r\n#\r\n# resolve_project_root() takes an explicit argument, then GVC_PROJECT_ROOT,\r\n# then discovery anchored to __file__, then RAISES. Discovery verifies\r\n# IDENTITY -- three sentinels in conjunction plus the declared project name\r\n# from pyproject.toml -- because any directory can contain a src/ folder.\r\n#\r\n# It RAISES rather than falling back. A resolver that guesses on failure is\r\n# how one absolute path became the value on every machine in the world.\r\nPROJECT_ROOT = resolve_project_root()\r\n'

ANCHOR_TEST_OLD = 'def test_the_default_TRACKS_the_environment_not_the_cwd():\n    """Measured 2026-08-14: config.py reads GVC_PROJECT_ROOT at import time, so\n    a fresh interpreter with the variable set gets a different anchor -- while\n    changing the working directory does NOT move it.\n\n    That distinction is the whole point: an ambient cwd is accidental, an\n    explicit environment variable is a decision.\n    """\n    code = (\n        "import os\\n"\n        "from genomic_variant_classifier.agent_layer.agents."\n        "agent_ops_monitor_agent import AgentOpsMonitorAgent\\n"\n        "class S:\\n"\n        "    def __getattr__(self, n): return lambda *a, **k: None\\n"\n        "print(AgentOpsMonitorAgent(S())._root)\\n"\n    )\n    import os\n    env = {**os.environ, "GVC_PROJECT_ROOT": os.path.join(os.sep, "probe_anchor")}\n    out = subprocess.run([sys.executable, "-B", "-c", code],\n                         capture_output=True, text=True, env=env, timeout=300)\n    assert out.returncode == 0, out.stderr[-800:]\n    assert "probe_anchor" in out.stdout, out.stdout.strip()\n'

ANCHOR_TEST_NEW = 'def test_the_default_TRACKS_the_environment_not_the_cwd(tmp_path):\n    """An ambient working directory is accidental; an explicit environment\n    variable is a decision. That property is unchanged. Its FIXTURE was not.\n\n    WRITTEN 2026-08-14, when config.py read:\n\n        PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT", r"C:\\\\Projects\\\\..."))\n\n    That line accepted ANY string, so this test pointed GVC_PROJECT_ROOT at a\n    bare "/probe_anchor" -- a path that does not exist -- and asserted the\n    agent followed it. The test encoded the DEFECT as a contract, and its own\n    docstring said so: "config.py reads GVC_PROJECT_ROOT at import time".\n\n    Since PROJECT-ROOT-HARDCODED-1 the resolver VERIFIES IDENTITY (sentinels in\n    conjunction plus the declared project name) and RAISES on anything that is\n    not this repository. The override must now name a real one.\n\n    The property is still asserted in both directions: a variable set in a\n    fresh interpreter MOVES the anchor, and the working directory does NOT.\n    """\n    import os\n\n    fake = tmp_path / "repo"\n    (fake / "src" / "genomic_variant_classifier").mkdir(parents=True)\n    (fake / "pyproject.toml").write_text(\n        \'[project]\\nname = "genomic-variant-classifier"\\nversion = "0.1.0"\\n\',\n        encoding="utf-8")\n\n    code = (\n        "from genomic_variant_classifier.agent_layer.agents."\n        "agent_ops_monitor_agent import AgentOpsMonitorAgent\\n"\n        "class S:\\n"\n        "    def __getattr__(self, n): return lambda *a, **k: None\\n"\n        "print(AgentOpsMonitorAgent(S())._root)\\n"\n    )\n    env = {**os.environ, "GVC_PROJECT_ROOT": str(fake)}\n    out = subprocess.run([sys.executable, "-B", "-c", code],\n                         capture_output=True, text=True, env=env, timeout=300)\n    assert out.returncode == 0, out.stderr[-800:]\n    assert str(fake.resolve()) in out.stdout, out.stdout.strip()\n\n    # The cwd does NOT move it: same code, no variable, launched elsewhere.\n    env2 = {k: v for k, v in os.environ.items() if k != "GVC_PROJECT_ROOT"}\n    out2 = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(tmp_path),\n                          capture_output=True, text=True, env=env2, timeout=300)\n    assert out2.returncode == 0, out2.stderr[-800:]\n    assert str(tmp_path) not in out2.stdout, out2.stdout.strip()\n\n\ndef test_a_NONEXISTENT_env_override_now_RAISES():\n    """The half the previous fixture could not assert.\n\n    "/probe_anchor" -- the value it used -- must now be REFUSED. A resolver\n    that accepts any string is how one author\'s absolute path became the value\n    on every machine in the world.\n    """\n    import os\n\n    code = "import genomic_variant_classifier.agent_layer.config"\n    env = {**os.environ, "GVC_PROJECT_ROOT": os.path.join(os.sep, "probe_anchor")}\n    out = subprocess.run([sys.executable, "-B", "-c", code],\n                         capture_output=True, text=True, env=env, timeout=300)\n    assert out.returncode != 0, "a nonexistent override was accepted"\n    assert ("RuntimePathError" in out.stderr\n            or "not a directory" in out.stderr), out.stderr[-500:]\n'

CONFIG = "src/genomic_variant_classifier/agent_layer/config.py"
ANCHOR_TEST = "tests/unit/test_agent_root_anchor.py"

#: (relative path, old, new, marker) -- marker present means already applied.
EDITS = (
    (CONFIG, CONFIG_IMPORT_OLD, CONFIG_IMPORT_NEW,
     "runtime_paths import resolve_project_root"),
    (CONFIG, CONFIG_ROOT_OLD, CONFIG_ROOT_NEW,
     "PROJECT_ROOT = resolve_project_root()"),
    (ANCHOR_TEST, ANCHOR_TEST_OLD, ANCHOR_TEST_NEW,
     "test_a_NONEXISTENT_env_override_now_RAISES"),
)

#: MEASURED by reachability over config.py: of twelve path constants only these
#: four are imported by another module. They must survive.
LIVE_DERIVED = ("CHECKPOINT_DIR", "SHAP_REPORT_DIR",
                "LITERATURE_DIGEST_DIR", "VAL_PARQUET")

#: MEASURED: the seven stale path constants. They are NOT removed here --
#: reachability found 35 unreachable assignments of which 28 are unwired
#: roadmap configuration, and that split is a scope decision. Asserted present
#: so a later deletion is deliberate.
STALE_RETAINED = ("SHARED_STATE_PATH", "AUDIT_LOG_DIR", "RAW_DATA_DIR",
                  "CORPUS_MANIFEST_PATH", "PROCESSED_DATA_DIR",
                  "TRAIN_PARQUET", "REPLAY_BUFFER_PARQUET")


def _module_assigned(tree) -> set:
    out = set()
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            out.add(n.target.id)
    return out


def _verify_config(source: str) -> tuple:
    """Structural checks by AST, per ROOTFIX-VERIFY-TEXTUAL-1.

    A substring check for "resolve_project_root" would be satisfied by the
    comment block this edit inserts, which names the function in prose. This
    reads the ASSIGNMENT'S VALUE.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "config.py: syntax error after patch: {}".format(exc)

    root = None
    for n in tree.body:
        if (isinstance(n, ast.Assign) and n.targets
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == "PROJECT_ROOT"):
            root = n
    if root is None:
        return False, "config.py: PROJECT_ROOT is not assigned at module level"
    if not isinstance(root.value, ast.Call):
        return False, ("config.py: PROJECT_ROOT is assigned {}, not a call"
                       .format(type(root.value).__name__))
    fn = getattr(root.value.func, "id", None) or getattr(root.value.func, "attr", None)
    if fn != "resolve_project_root":
        return False, "config.py: PROJECT_ROOT is assigned from {!r}".format(fn)
    if root.value.args or root.value.keywords:
        return False, "config.py: resolve_project_root is called with arguments"

    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                imported.add(n.module + "." + a.name)
    if ("genomic_variant_classifier.paths.runtime_paths.resolve_project_root"
            not in imported):
        return False, "config.py: the resolver is not imported"

    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = getattr(n, "body", None)
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)):
                docs.add(id(b[0].value))
    drive = re.compile(r"^[A-Za-z]:[\\/]")
    live = [c.value for c in ast.walk(tree)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and id(c) not in docs]
    bad = [s for s in live if drive.match(s)]
    if bad:
        return False, "config.py: absolute path literal(s) survive: {}".format(bad)

    assigned = _module_assigned(tree)
    missing = [c for c in LIVE_DERIVED if c not in assigned]
    if missing:
        return False, "config.py: live derived constant(s) lost: {}".format(missing)
    gone = [c for c in STALE_RETAINED if c not in assigned]
    if gone:
        return False, ("config.py: stale constant(s) removed without a recorded "
                       "decision: {}".format(gone))
    return True, ("config.py: PROJECT_ROOT resolved; no path literal; {} live "
                  "and {} stale constant(s) intact"
                  .format(len(LIVE_DERIVED), len(STALE_RETAINED)))


def _verify_anchor_test(source: str) -> tuple:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, "anchor test: syntax error after patch: {}".format(exc)

    tests = {n.name for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    for required in ("test_the_default_TRACKS_the_environment_not_the_cwd",
                     "test_a_NONEXISTENT_env_override_now_RAISES"):
        if required not in tests:
            return False, "anchor test: {} is missing".format(required)

    # The stale fixture must be GONE: no bare "/probe_anchor" asserted as
    # ACCEPTED. It may still appear in the refusal test, where returncode != 0.
    for n in tree.body:
        if (isinstance(n, ast.FunctionDef)
                and n.name == "test_the_default_TRACKS_the_environment_not_the_cwd"):
            # Skip the DOCSTRING: the repaired test explains the old fixture in
            # prose, and an ast.dump of the whole function would match that
            # explanation. A check that fires on a description of the defect is
            # the same shape as an idempotence guard matching its own comment.
            body = n.body[1:] if (n.body and isinstance(n.body[0], ast.Expr)
                                  and isinstance(getattr(n.body[0], "value", None),
                                                 ast.Constant)) else n.body
            live = [c.value for stmt in body for c in ast.walk(stmt)
                    if isinstance(c, ast.Constant) and isinstance(c.value, str)]
            if any("probe_anchor" in s for s in live):
                return False, ("anchor test: the tracking test still uses the "
                               "nonexistent /probe_anchor fixture")
    # Only names the file actually provides may be used.
    module_imports = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            module_imports.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            module_imports.update(a.asname or a.name for a in n.names)
    for n in tree.body:
        if not (isinstance(n, ast.FunctionDef)
                and n.name in ("test_the_default_TRACKS_the_environment_not_the_cwd",
                               "test_a_NONEXISTENT_env_override_now_RAISES")):
            continue
        local = {a.asname or a.name.split(".")[0]
                 for sub in ast.walk(n) if isinstance(sub, ast.Import)
                 for a in sub.names}
        params = {a.arg for a in n.args.args}
        assigned = set()
        for sub in ast.walk(n):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    for nm in ast.walk(t):
                        if isinstance(nm, ast.Name):
                            assigned.add(nm.id)
        # A comprehension target may be a Name OR a Tuple -- `for k, v in
        # d.items()` unpacks into ast.Tuple, and reading only .id misses BOTH
        # names. Same shape as the docstring oversight above: incomplete node
        # handling in a check, not a defect in what it checks.
        comps = set()
        for sub in ast.walk(n):
            if isinstance(sub, (ast.DictComp, ast.ListComp, ast.SetComp,
                                ast.GeneratorExp)):
                for g in sub.generators:
                    for nm in ast.walk(g.target):
                        if isinstance(nm, ast.Name):
                            comps.add(nm.id)
        # Assignment targets can also be tuples: `a, b = ...`.
        for sub in ast.walk(n):
            if isinstance(sub, (ast.For, ast.With, ast.withitem)):
                tgt = getattr(sub, "target", None) or getattr(sub, "optional_vars", None)
                if tgt is not None:
                    for nm in ast.walk(tgt):
                        if isinstance(nm, ast.Name):
                            comps.add(nm.id)
        avail = module_imports | local | params | assigned | comps | set(dir(__builtins__))
        used = {x.id for x in ast.walk(n)
                if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
        free = sorted(used - avail)
        if free:
            return False, ("anchor test: {} uses undefined name(s) {}"
                           .format(n.name, free))
    return True, "anchor test: both tests present, fixture repaired, no free names"


VERIFIERS = {CONFIG: _verify_config, ANCHOR_TEST: _verify_anchor_test}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    repo = Path(args.repo_root)

    # Path.read_text(newline=...) is 3.13+; the default would translate CRLF to
    # LF here and every config.py anchor would match nothing.
    sources = {}
    for rel in (CONFIG, ANCHOR_TEST):
        p = repo / rel
        if not p.exists():
            print("  ERROR: not found: {}".format(rel))
            return 2
        raw = p.read_bytes()
        sources[rel] = raw.decode("utf-8")
        print("  {:<52} {} bytes, CRLF {}, non-ASCII {}".format(
            rel, len(raw), raw.count(b"\r\n"),
            sum(1 for b in raw if b > 0x7F)))

    pending = {}
    already = 0
    for rel, old, new, marker in EDITS:
        if marker in sources[rel]:
            already += 1
            print("  {:<44} already applied".format(marker[:42]))
            continue
        cur = pending.get(rel, sources[rel])
        n = cur.count(old)
        if n != 1:
            print("  ERROR: anchor {!r} occurs {} time(s) in {}, expected 1; "
                  "NOTHING written.".format(marker[:30], n, rel))
            return 1
        print("  {:<44} anchor OK".format(marker[:42]))
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
        print("\n  --check: {} file(s) pending, {} edit(s) already applied. "
              "Nothing written.".format(len(pending), already))
        return 0

    written = []
    for rel, patched in pending.items():
        p = repo / rel
        before = p.read_bytes()
        backup = p.with_suffix(p.suffix + ".pre_cfgroot.bak")
        if not backup.exists():
            backup.write_bytes(before)
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(patched)
        written.append((p, before))
        after = p.read_bytes()
        b_na = sum(1 for b in before if b > 0x7F)
        a_na = sum(1 for b in after if b > 0x7F)
        if a_na != b_na:
            for p2, b2 in written:
                p2.write_bytes(b2)
            print("  ERROR: {} non-ASCII {} -> {}; ROLLED BACK all {} file(s)."
                  .format(rel, b_na, a_na, len(written)))
            return 1
        print("  wrote {}  ({} non-ASCII byte(s) preserved, CRLF {})".format(
            rel, a_na, after.count(b"\r\n")))

    for p, before in written:
        rel = str(p).replace("\\", "/")
        key = CONFIG if rel.endswith("config.py") else ANCHOR_TEST
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
