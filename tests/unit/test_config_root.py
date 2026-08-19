"""config.PROJECT_ROOT must be resolved, never guessed.

PROJECT-ROOT-HARDCODED-1
========================
`config.py:17` read:

    PROJECT_ROOT = Path(os.getenv("GVC_PROJECT_ROOT",
                                   r"C:\\Projects\\genomic-variant-classifier"))

MEASURED 2026-08-14: GVC_PROJECT_ROOT is set NOWHERE -- not in continuous
integration, not in the Dockerfile, not in any script, not in the shell. So the
fallback was the value EVERY consumer received, and on the Linux runner it
named a path that cannot exist.

BLAST RADIUS, MEASURED BY SYNTAX TREE 2026-08-17
    4  definitions of PROJECT_ROOT in the tree. THREE ARE ALREADY CORRECT:
       c3_inventory.py:21, c3_sweep.py:29 and run11_preflight.py:27 all use
       Path(__file__).resolve().parent.parent. Only config.py:17 was defective.
    17 load-context references to the defective one: 9 inside config.py, 6 in
       agents as `root if root is not None else str(PROJECT_ROOT)`, 2 in
       test_agent_root_anchor.py.
    0  attribute accesses of the `config.PROJECT_ROOT` form.
    13 modules import from agent_layer.config; 7 take PROJECT_ROOT.

WHY THIS FILE EXISTS
config.py is imported at MODULE SCOPE by thirteen modules. If resolution
fails, every one of those imports fails. That is the intended behaviour --
failing loudly at import beats returning a path that cannot exist -- but it
makes the resolution path load-bearing for the entire agent layer, so it is
tested directly rather than inferred from RuntimePaths' own suite.

Author: Monzia Moodie
"""
from __future__ import annotations

import ast
import io
import subprocess
import sys
from pathlib import Path

import pytest

from genomic_variant_classifier.agent_layer import config as C

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "src" / "genomic_variant_classifier" / "agent_layer" / "config.py"


# ---- the defect, as a source-level assertion ---------------------------
def test_no_absolute_path_literal_survives_in_config():
    """THE DEFECT. A drive-letter or POSIX absolute path in an executable
    constant reintroduces it. Docstrings and comments quote the old line
    deliberately, so only LIVE constants are checked."""
    import re
    src = io.open(_CONFIG, encoding="utf-8").read()
    tree = ast.parse(src)
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            b = getattr(n, "body", None)
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)):
                docs.add(id(b[0].value))
    live = [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs]
    drive = re.compile(r"^[A-Za-z]:[\\/]")
    bad = [s for s in live if drive.match(s)]
    assert not bad, (
        "absolute path literal(s) in executable config: {}".format(bad))


def test_project_root_equals_the_one_runtime_path_authority():
    """A SEMANTIC contract, not an implementation one.

    This test previously required `PROJECT_ROOT = resolve_project_root()` by
    syntax tree -- the exact call and the exact import. That was an appropriate
    migration guard for PROJECT-ROOT-HARDCODED-1, and it is the milder form of
    the /probe_anchor mistake: encoding HOW something is done as the contract
    for WHAT it must be.

    OUTPUT-ROOT-CONFLATION-1 changed the how -- one resolve_runtime_paths()
    call replacing resolve_project_root() -- without changing anything the
    commit existed to guarantee. The property is that PROJECT_ROOT is the
    validated project root of the ONE runtime-path authority this module
    resolved.
    """
    assert C.PROJECT_ROOT == C._RUNTIME_PATHS.project_root


def test_there_is_exactly_ONE_runtime_path_authority():
    """Two resolver calls would be two authorities for one process, and each
    performs a full discovery walk. The module resolves once."""
    import ast
    tree = ast.parse(io.open(_CONFIG, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and (getattr(n.func, "id", None) or getattr(n.func, "attr", None))
             in ("resolve_runtime_paths", "resolve_project_root")]
    assert len(calls) == 1, (
        "config.py makes {} resolver call(s); exactly one authority is "
        "resolved per process".format(len(calls)))


def test_no_absolute_path_literal_and_no_env_fallback_survives():
    """What the resolver import was standing in for.

    The old test named the imported symbol. The property is that PROJECT_ROOT
    is not read from an environment variable with a literal default -- the
    shape that made one author's absolute path the value on every machine.
    """
    import ast
    tree = ast.parse(io.open(_CONFIG, encoding="utf-8").read())
    for n in tree.body:
        if (isinstance(n, ast.Assign) and n.targets
                and getattr(n.targets[0], "id", None) == "PROJECT_ROOT"):
            for sub in ast.walk(n.value):
                if isinstance(sub, ast.Call):
                    fn = (getattr(sub.func, "id", None)
                          or getattr(sub.func, "attr", None))
                    assert fn != "getenv", (
                        "PROJECT_ROOT is read from the environment with a "
                        "default; resolution must verify identity and RAISE")
            return
    raise AssertionError("PROJECT_ROOT is not assigned at module level")


# ---- what the resolved value must satisfy ------------------------------
def test_project_root_is_an_existing_directory():
    assert C.PROJECT_ROOT.is_absolute(), C.PROJECT_ROOT
    assert C.PROJECT_ROOT.is_dir(), C.PROJECT_ROOT


def test_project_root_is_THIS_repository():
    """Identity, not mere existence. Any directory can contain a src/ folder."""
    from genomic_variant_classifier.paths.runtime_paths import (
        looks_like_project_root,
    )
    assert looks_like_project_root(C.PROJECT_ROOT), C.PROJECT_ROOT


def test_project_root_agrees_with_the_script_convention():
    """MEASURED: c3_inventory.py, c3_sweep.py and run11_preflight.py all use
    Path(__file__).resolve().parent.parent. Those three were ALREADY correct,
    and config.py must now agree with them."""
    script_anchor = (_REPO / "scripts" / "c3_sweep.py").resolve().parent.parent
    assert C.PROJECT_ROOT == script_anchor, (C.PROJECT_ROOT, script_anchor)


def test_project_root_does_not_depend_on_the_working_directory(tmp_path):
    """The whole point. Resolution is anchored to __file__, so a subprocess
    launched from an unrelated directory must resolve identically."""
    code = (
        "from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT;"
        "print(PROJECT_ROOT)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=str(tmp_path),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == str(C.PROJECT_ROOT), (
        out.stdout.strip(), str(C.PROJECT_ROOT))


# ---- the derived constants still compute -------------------------------
@pytest.mark.parametrize("name", [
    "CHECKPOINT_DIR", "SHAP_REPORT_DIR", "LITERATURE_DIGEST_DIR", "VAL_PARQUET",
])
def test_the_four_LIVE_derived_constants_still_resolve(name):
    """MEASURED by reachability over config.py: of twelve path constants, only
    these four are imported by another module. They must survive the swap."""
    value = getattr(C, name)
    assert isinstance(value, Path), (name, type(value))
    assert value.is_absolute(), (name, value)
    assert str(value).startswith(str(C.PROJECT_ROOT)), (name, value)


def test_repository_paths_derive_from_PROJECT_ROOT():
    """Repository-owned paths stay repository-owned. This is half of
    OUTPUT-ROOT-CONFLATION-1: the repair is an OWNERSHIP correction, not a
    blanket move of every path under artifact_root."""
    assert C.MODELS_DIR == C.PROJECT_ROOT / "models"
    assert C.CHECKPOINT_DIR == C.MODELS_DIR / "checkpoints"


def test_report_paths_derive_from_REPORTS_ROOT_not_project_root():
    """The other half. These two were computed from repository identity while
    being artifact destinations -- and the previous version of this test
    asserted exactly that, so it literally specified the defect."""
    assert C.SHAP_REPORT_DIR == C._RUNTIME_PATHS.reports_root / "shap"
    assert C.LITERATURE_DIGEST_DIR == C._RUNTIME_PATHS.reports_root / "literature"


def test_the_two_root_domains_can_DIVERGE(tmp_path):
    """THE RELEASE-BLOCKING TEST for this unit.

    MEASURED 2026-08-19: on this workstation artifact_root == project_root, so
    the defect is INVISIBLE under the default configuration. Testing only here
    can never validate the boundary -- the same lesson the sentinel repair
    taught, where a rule valid inside the checkout was impossible inside the
    trainer image.

        An artifact path contract must be tested in an environment where
        artifact identity DIFFERS from repository identity.

    Uses the supported RuntimePaths injection mechanism (GVC_ARTIFACT_ROOT),
    not a new test-only vocabulary. Both directions are asserted: report paths
    follow the artifact root, and checkpoints do NOT.
    """
    repo = tmp_path / "repo"
    (repo / "src" / "genomic_variant_classifier").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "genomic-variant-classifier"\nversion = "0.1.0"\n',
        encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    code = (
        "from genomic_variant_classifier.agent_layer import config as C\n"
        "print(C.PROJECT_ROOT)\n"
        "print(C.SHAP_REPORT_DIR)\n"
        "print(C.LITERATURE_DIGEST_DIR)\n"
        "print(C.CHECKPOINT_DIR)\n"
    )
    import os
    env = dict(os.environ, GVC_PROJECT_ROOT=str(repo),
               GVC_ARTIFACT_ROOT=str(artifacts))
    out = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    root, shap, lit, ckpt = [Path(l) for l in out.stdout.strip().splitlines()]

    assert root == repo.resolve(), (root, repo)
    assert root != artifacts.resolve()

    # Artifact-owned: they follow artifact identity.
    assert shap == artifacts.resolve() / "reports" / "shap", shap
    assert lit == artifacts.resolve() / "reports" / "literature", lit

    # Repository-owned: they do NOT.
    assert ckpt == repo.resolve() / "models" / "checkpoints", ckpt
    assert artifacts.resolve() not in ckpt.parents, ckpt


def test_runtime_paths_are_a_SNAPSHOT_not_a_live_lookup(tmp_path):
    """Runtime path configuration is immutable for the lifetime of a process.

    Mutating the environment after import must NOT relocate a running process:
    a program whose artifact identity silently moved mid-execution would be far
    harder to reason about than one that requires a restart.

        fresh process    -> fresh resolution
        existing process -> stable paths
    """
    code = (
        "import os\n"
        "from genomic_variant_classifier.agent_layer import config as C\n"
        "before = str(C.SHAP_REPORT_DIR)\n"
        "os.environ['GVC_ARTIFACT_ROOT'] = %r\n"
        "import importlib\n"
        "print(before)\n"
        "print(str(C.SHAP_REPORT_DIR))\n"
    ) % str(tmp_path)
    out = subprocess.run([sys.executable, "-B", "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    before, after = out.stdout.strip().splitlines()
    assert before == after, (before, after)


# ---- the environment override still works ------------------------------
def test_GVC_PROJECT_ROOT_still_overrides(tmp_path):
    """The variable is set nowhere today, but the branch must keep working --
    it is the documented escape hatch for a container or a Colab mount."""
    import shutil
    fake = tmp_path / "repo"
    (fake / "src" / "genomic_variant_classifier").mkdir(parents=True)
    (fake / "tests").mkdir()
    io.open(fake / "pyproject.toml", "w", encoding="utf-8", newline="\n").write(
        '[project]\nname = "genomic-variant-classifier"\nversion = "0.1.0"\n')
    io.open(fake / "tests" / "EXPECTED_SUITE_SIZE", "w", newline="\n").write("1\n")
    code = (
        "from genomic_variant_classifier.agent_layer.config import PROJECT_ROOT;"
        "print(PROJECT_ROOT)"
    )
    import os
    env = dict(os.environ, GVC_PROJECT_ROOT=str(fake))
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip() == str(fake.resolve()), out.stdout.strip()


def test_a_WRONG_GVC_PROJECT_ROOT_raises_rather_than_being_used(tmp_path):
    """Fail closed. A directory that exists but is not this repository must not
    silently become the root."""
    other = tmp_path / "not-the-repo"
    (other / "src").mkdir(parents=True)
    code = "import genomic_variant_classifier.agent_layer.config"
    import os
    env = dict(os.environ, GVC_PROJECT_ROOT=str(other))
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode != 0, "a non-repository was accepted as the root"
    assert "RuntimePathError" in out.stderr or "not this repository" in out.stderr, (
        out.stderr[-400:])


# ---- the dead constants are recorded, not silently kept ----------------
def test_the_seven_unreadable_path_constants_are_recorded():
    """MEASURED by reachability 2026-08-17: of 71 module-level assignments, 35
    are unreachable from any external import. Seven of those are stale PATH
    constants; the remaining 28 are unwired configuration for roadmap
    capabilities (EWC, ResNet, replay, SHAP tuning, endpoints) and are NOT
    dead code.

    This test asserts the seven still EXIST -- their removal is a scope
    decision, not a measurement -- so that deleting them is a deliberate act
    that breaks a test rather than a silent edit.
    """
    stale = ["SHARED_STATE_PATH", "AUDIT_LOG_DIR", "RAW_DATA_DIR",
             "CORPUS_MANIFEST_PATH", "PROCESSED_DATA_DIR", "TRAIN_PARQUET",
             "REPLAY_BUFFER_PARQUET"]
    present = [n for n in stale if hasattr(C, n)]
    assert present == stale, (
        "these stale constants were removed without a recorded decision: {}"
        .format(sorted(set(stale) - set(present))))


def test_the_two_dead_agent_layer_paths_still_point_at_a_nonexistent_dir():
    """CONFIG-DEAD-PATHS-1, as an observation rather than a repair.

    SHARED_STATE_PATH and AUDIT_LOG_DIR resolve to PROJECT_ROOT/"agent_layer",
    but agent_layer lives under src/genomic_variant_classifier/. Nothing reads
    either. This records the fact so a future repair has a failing test to fix
    rather than a claim to re-derive.
    """
    assert not C.SHARED_STATE_PATH.parent.exists(), (
        "PROJECT_ROOT/agent_layer now exists -- re-examine CONFIG-DEAD-PATHS-1")
    assert not C.AUDIT_LOG_DIR.exists()
