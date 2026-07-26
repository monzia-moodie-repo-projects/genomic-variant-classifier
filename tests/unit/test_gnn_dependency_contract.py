"""The graph-neural-network branch is covered in Continuous Integration, and the
comments in its test files say so truthfully.

WHY THIS FILE EXISTS
====================
Four GNN test files carried the comment "skip where absent (e.g. CI)" on their
`pytest.importorskip("torch_geometric")` line. That was FALSE as of 2026-07-21:
torch-geometric is pinned at 2.7.0 in requirements.txt, and ci.yml carries a
hard gate ("Assert the coverage-critical dependencies are present") that FAILS
the build if torch_geometric is not importable. The GNN branch is fully covered
in CI; the skip fires only on an under-provisioned local machine.

That comment was not merely stale. It asserted the exact belief that let the
graph-neural-network branch go untested for 508 runs -- that CI is allowed to
skip these tests. A false comment next to a real safety net is worse than no
comment, because it tells a debugging reader the wrong thing with apparent
authority.

WHAT THIS FILE PINS
-------------------
Two contracts, both checkable without importing torch:

  1. No GNN test file reintroduces the "skip where absent (e.g. CI)" phrasing,
     or any phrasing that says CI may skip the branch.

  2. ci.yml's REQUIRED-dependency gate still lists torch_geometric. If someone
     removes it, the branch could silently stop being covered again, and this
     test turns that red.

It does NOT assert torch_geometric is importable HERE -- this file must pass in
any environment, including one deliberately without the training stack. It
asserts the CONTRACT that guarantees coverage where it matters, which is CI.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TESTS_UNIT = REPO / "tests" / "unit"
CI = REPO / ".github" / "workflows" / "ci.yml"

# The four files whose comments were corrected on 2026-07-21. Named explicitly
# rather than globbed, so that deleting a file surfaces here instead of shrinking
# the check silently.
CORRECTED_FILES = (
    "test_gnn_shared_graph.py",
    "test_gnn_optim.py",
    "test_gnn_tier2_denoise.py",
    "test_gnn_gps.py",
)

STALE = re.compile(r"skip where absent.*CI", re.IGNORECASE)
# There is deliberately NO broad paraphrase net here. An earlier version matched
# any comment containing both "CI" and "skip" and then tried to subtract the true
# statements ("never skips in CI") and the quote-to-refute ones ("do not read
# this as 'CI may skip'") with a growing exemption list. That is a losing game:
# distinguishing an ASSERTION of a falsehood from its NEGATION or its REFUTATION
# is natural-language work a regex cannot do cleanly, and every fix spawned the
# next false positive -- including on this project's own corrected comments.
#
# The precise STALE check above catches the one dead phrasing that ever actually
# occurred. The real guarantee that the GNN branch is covered is not any comment
# at all: it is the ci.yml REQUIRED gate, pinned by the two tests below. A
# comment is documentation; the gate is the contract. We guard the contract.


def test_the_four_corrected_files_still_exist():
    for name in CORRECTED_FILES:
        assert (TESTS_UNIT / name).is_file(), f"{name} vanished"


@pytest.mark.parametrize("name", CORRECTED_FILES)
def test_the_corrected_files_keep_their_guard(name):
    """The comment was corrected; the importorskip itself must remain, so a
    developer without PyG still gets a clean skip rather than a collection error."""
    text = (TESTS_UNIT / name).read_text(encoding="utf-8")
    assert 'pytest.importorskip("torch_geometric")' in text


@pytest.mark.parametrize("name", CORRECTED_FILES)
def test_the_stale_phrase_is_gone(name):
    text = (TESTS_UNIT / name).read_text(encoding="utf-8")
    assert not STALE.search(text), (
        f"{name} still claims CI may skip the GNN branch; it is pinned and "
        "CI-gated")


def _required_gate_body() -> str:
    """The heredoc body of ci.yml's "Assert the coverage-critical dependencies
    are present" step. Scoped precisely, because torch_geometric also appears in
    the pip-check gate lower down, and a whole-file search cannot tell a mention
    in the REQUIRED set from a mention anywhere else."""
    ci = CI.read_text(encoding="utf-8")
    assert "coverage-critical dependencies are present" in ci, (
        "the REQUIRED-imports gate is gone; the corrected GNN comments now "
        "over-promise")
    start = ci.index("coverage-critical dependencies")
    opener = ci.index("<<'PY'", start) + len("<<'PY'")
    m = re.search(r"\n[ \t]*PY\b", ci[opener:])
    assert m, "could not find the closing PY delimiter of the dependency gate"
    return ci[opener:opener + m.start()]


def test_ci_still_requires_torch_geometric_in_the_required_set():
    """The other half of the contract. The corrected comments PROMISE that CI
    fails without torch_geometric. The promise lives specifically in the
    REQUIRED gate body -- a mention in the pip-check gate does not make the
    build fail on a missing import, so this must check the right region."""
    gate = _required_gate_body()
    assert '"torch_geometric":' in gate, (
        "torch_geometric dropped from the REQUIRED set; the GNN branch can now "
        "skip in CI and report green -- the 508-run failure mode")


def test_ci_gate_actually_fails_the_build_on_a_missing_dependency():
    """A gate that logs and exits 0 is decoration. This one must sys.exit(1),
    inside the REQUIRED gate body specifically."""
    gate = _required_gate_body()
    assert "sys.exit(1)" in gate, (
        "the dependency gate logs but does not fail the build -- a check that "
        "cannot fail manufactures confidence")
