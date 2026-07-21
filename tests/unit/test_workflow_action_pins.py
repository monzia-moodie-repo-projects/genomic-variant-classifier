"""Every GitHub Actions reference is pinned by commit SHA, and every pin is Node 24.

WHY THIS FILE EXISTS
====================
On 2026-07-21 the first run of the new CI failure alert succeeded and carried a
warning:

    Node.js 20 is deprecated. The following actions target Node.js 20 but are
    being forced to run on Node.js 24: actions/github-script@v7

Being forced onto Node 24 is a countdown, not a reprieve.

WHAT THE AUDIT FOUND, AND WHAT IT CORRECTED
--------------------------------------------
My first pass listed `actions/checkout@v4` and `actions/setup-python@v5` as
exposed. That was WRONG. ci.yml and drift_monitor.yml already pinned those by
commit SHA, and resolving the SHAs showed they were checkout v6.0.2,
setup-python v6.2.0 and upload-artifact v7.0.0 -- all node24 already. Only
data_freshness.yml still carried floating tags.

Two version guesses were also wrong, and both would have shipped a "fix" that
fixed nothing while the commit message claimed otherwise:

  * actions/upload-artifact v5 is STILL node20. v6 is the first node24 release.
  * docker/build-push-action v6 is STILL node20. v7 is the first node24 release.

Reading each action's own `action.yml` at the exact pinned SHA is what caught
both. The tag is not what ships; the SHA is.

WHAT THIS FILE PINS
-------------------
Two invariants, both checkable offline:

  1. Every non-local `uses:` is a full 40-character commit SHA. A tag is
     mutable -- `v7` can be repointed at any commit by its owner -- so a tag is
     a trust decision renewed silently on every run.

  2. Every pinned SHA appears in EXPECTED_PINS below with its resolved version
     and runtime, and every runtime is node24. Adding a pin without recording
     what it resolves to fails the suite, which forces the lookup rather than
     hoping someone does it.

The second invariant is what stops this file becoming decoration. A test that
only checked "is it a SHA?" would happily accept a node20 SHA and report green.

Author: written for Monzia Moodie, 2026-07-21.
"""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml   # HARD dependency: pyyaml==6.0.3, imported at runtime by
              # data/pipeline.py and utils/helpers.py. NOT importorskip -- a
              # module-level importorskip collapses every test here into ONE
              # skip entry, which is how the graph-neural-network branch went
              # untested for 508 Continuous Integration runs.

WORKFLOW_DIR = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows"

USES = re.compile(r"uses:\s*([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)@([A-Za-z0-9_.\-]+)")
SHA40 = re.compile(r"\A[0-9a-f]{40}\Z")

# Resolved 2026-07-21 by reading action.yml at each SHA over the network. The
# runtime column is the point: a SHA alone says nothing about whether it is
# deprecated.
EXPECTED_PINS = {
    "ed597411d8f924073f98dfc5c65a23a2325f34cd":
        ("actions/github-script", "v8.0.0", "node24"),
    "de0fac2e4500dabe0009e67214ff5f5447ce83dd":
        ("actions/checkout", "v6.0.2", "node24"),
    "a309ff8b426b58ec0e2a45f0f869d46889d02405":
        ("actions/setup-python", "v6.2.0", "node24"),
    "bbbca2ddaa5d8feaa63e36b76fdaad77386f024f":
        ("actions/upload-artifact", "v7.0.0", "node24"),
    "af1e73f918a031802d376d3c8bbc3fe56130a9b0":
        ("docker/login-action", "v4", "node24"),
    "dc802804100637a589fabce1cb79ff13a1411302":
        ("docker/metadata-action", "v6.2.0", "node24"),
    "53b7df96c91f9c12dcc8a07bcb9ccacbed38856a":
        ("docker/build-push-action", "v7", "node24"),
}


def _workflow_files():
    files = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    assert files, f"no workflow files found under {WORKFLOW_DIR}"
    return files


def _references():
    out = []
    for path in _workflow_files():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            m = USES.search(line)
            if m:
                out.append((path.name, line_no, m.group(1), m.group(2)))
    return out


# --------------------------------------------------------------------------- #
# 1. the workflows are real and parseable
# --------------------------------------------------------------------------- #
def test_the_workflow_directory_is_where_we_think():
    assert WORKFLOW_DIR.is_dir(), WORKFLOW_DIR


@pytest.mark.parametrize("path", _workflow_files(), ids=lambda p: p.name)
def test_every_workflow_parses_as_yaml(path):
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(doc, dict) and doc.get("jobs"), f"{path.name} has no jobs"


def test_at_least_one_action_reference_exists():
    """Guards the guard: if the regular expression stops matching, every test
    below would pass over an empty list and report green."""
    assert len(_references()) >= 7


# --------------------------------------------------------------------------- #
# 2. every reference is pinned by commit SHA
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ref", _references(),
                         ids=lambda r: f"{r[0]}:{r[1]}:{r[2]}")
def test_every_action_is_pinned_by_commit_sha(ref):
    """A tag is mutable. `v7` can be repointed at any commit by its owner, so a
    tag re-makes a trust decision silently on every run."""
    fname, line_no, repo, version = ref
    assert SHA40.match(version), (
        f"{fname}:{line_no} uses {repo}@{version}, which is a tag, not a "
        "40-character commit SHA")


# --------------------------------------------------------------------------- #
# 3. every pin is known, and known to be Node 24
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ref", _references(),
                         ids=lambda r: f"{r[0]}:{r[1]}:{r[2]}")
def test_every_pin_is_recorded_with_its_resolved_version(ref):
    """Adding a pin without recording what it resolves to fails here, which
    forces the lookup rather than hoping someone does it."""
    fname, line_no, repo, sha = ref
    if not SHA40.match(sha):
        pytest.skip("covered by test_every_action_is_pinned_by_commit_sha")
    assert sha in EXPECTED_PINS, (
        f"{fname}:{line_no} pins {repo}@{sha}, which is not in EXPECTED_PINS. "
        "Resolve it: read action.yml at that SHA and record repo, version and "
        "runs.using here.")
    recorded_repo, _version, _runtime = EXPECTED_PINS[sha]
    assert recorded_repo == repo, (
        f"{fname}:{line_no} pins {repo}@{sha}, but EXPECTED_PINS records that "
        f"SHA as {recorded_repo}")


@pytest.mark.parametrize("sha,meta", sorted(EXPECTED_PINS.items()),
                         ids=lambda x: x if isinstance(x, str) else x[0])
def test_every_recorded_pin_targets_node24(sha, meta):
    """The invariant that matters. Node.js 20 is deprecated and GitHub already
    forces those actions onto Node 24; a SHA alone says nothing about that."""
    repo, version, runtime = meta
    assert runtime == "node24", f"{repo} {version} ({sha[:12]}) is {runtime}"


def test_no_recorded_pin_is_unused():
    """A stale entry is a claim about a dependency that is no longer there, and
    it would keep a removed action looking audited."""
    used = {sha for _f, _l, _r, sha in _references() if SHA40.match(sha)}
    unused = sorted(set(EXPECTED_PINS) - used)
    assert not unused, (
        "EXPECTED_PINS records pins no workflow uses: " +
        ", ".join(f"{EXPECTED_PINS[s][0]} {s[:12]}" for s in unused))


# --------------------------------------------------------------------------- #
# 4. the specific traps found on 2026-07-21
# --------------------------------------------------------------------------- #
def test_github_script_is_not_on_the_node20_v7():
    """The reference that produced the original warning."""
    for _f, _l, repo, sha in _references():
        if repo == "actions/github-script":
            assert sha != "v7"
            assert SHA40.match(sha) and EXPECTED_PINS[sha][2] == "node24"


def test_upload_artifact_is_past_v5_not_merely_past_v4():
    """upload-artifact v5 is STILL node20; v6 is the first node24 release. The
    obvious one-major bump would have resolved nothing."""
    for sha, (repo, version, runtime) in EXPECTED_PINS.items():
        if repo == "actions/upload-artifact":
            major = int(version.lstrip("v").split(".")[0])
            assert major >= 6, f"{version} is node20"
            assert runtime == "node24"


def test_build_push_action_is_past_v6_not_merely_past_v5():
    """docker/build-push-action v6 is STILL node20; v7 is the first node24
    release. Same trap, different action."""
    for sha, (repo, version, runtime) in EXPECTED_PINS.items():
        if repo == "docker/build-push-action":
            major = int(version.lstrip("v").split(".")[0])
            assert major >= 7, f"{version} is node20"
            assert runtime == "node24"
