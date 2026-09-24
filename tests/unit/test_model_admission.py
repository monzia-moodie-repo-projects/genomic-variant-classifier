"""The ONE admission route (model_admission; owner ruling 2026-09-23, point 2): every step, and that none of
them deserializes anything.

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pytest

import genomic_variant_classifier.model_admission as A
from genomic_variant_classifier import quarantine_policy
from genomic_variant_classifier.containment import ContainmentError
from genomic_variant_classifier.quarantine_policy import QUARANTINED_FEATURES

sys.path.insert(0, str(Path(__file__).resolve().parent))   # tests/unit helpers (_admission_support)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from _admission_support import AllowForTests, register  # noqa: E402

UNPICKLED = []


def _record_unpickling():
    UNPICKLED.append(1)
    return "unpickled"


class _Tripwire:
    """Unpickling this object records a marker: proof that admission deserializes nothing."""

    def __reduce__(self):
        return (_record_unpickling, ())


@pytest.fixture
def artifact(tmp_path):
    UNPICKLED.clear()
    path = tmp_path / "model.joblib"
    path.write_bytes(pickle.dumps(_Tripwire()))
    return path


@pytest.fixture
def registry(tmp_path, artifact):
    return register(tmp_path / "registry.v1.json", artifact, feature_names=["a", "b"], roster=["m"])


class _Returns:
    def __init__(self, value):
        self.value = value

    def decide(self, record, *, consumer, policy_sha256):
        return self.value(record, consumer, policy_sha256) if callable(self.value) else self.value


def _decision(sha=None, consumer="c", policy=None, outcome=A.Outcome.ALLOW, reason="ok"):
    return lambda r, c, p: A.Decision(sha or r.artifact.sha256, consumer if consumer != "same" else c,
                                      policy or p, outcome, reason)


def test_the_tripwire_really_records_unpickling(artifact):
    pickle.loads(artifact.read_bytes())
    assert UNPICKLED == [1]


@pytest.mark.parametrize("consumer", ["", " c", 7, None])
def test_a_named_consumer_is_required(artifact, registry, consumer):
    with pytest.raises(ContainmentError, match="named consumer"):
        A.admit_artifact(artifact, consumer=consumer, registry_path=registry, authority=AllowForTests())


def test_a_missing_artifact_is_not_a_containment_question(tmp_path, registry):
    with pytest.raises(FileNotFoundError):
        A.admit_artifact(tmp_path / "absent.joblib", consumer="c", registry_path=registry)


def test_a_missing_registry_is_no_trusted_binding_not_an_empty_one(tmp_path, artifact):
    with pytest.raises(ContainmentError, match="No trusted binding"):
        A.admit_artifact(artifact, consumer="c", registry_path=tmp_path / "none.json", authority=AllowForTests())
    assert UNPICKLED == []


def test_an_unregistered_artifact_is_unbound_and_never_unpickled(tmp_path, artifact):
    other = tmp_path / "other.bin"
    other.write_bytes(b"x")
    reg = register(tmp_path / "r.json", other, feature_names=["a"], roster=["m"])
    with pytest.raises(ContainmentError, match="Unbound artifact"):
        A.admit_artifact(artifact, consumer="c", registry_path=reg, authority=AllowForTests())
    assert UNPICKLED == []


def test_a_record_naming_a_quarantined_feature_is_refused(tmp_path, artifact):
    reg = register(tmp_path / "r.json", artifact, feature_names=["a", QUARANTINED_FEATURES[0]], roster=["m"])
    with pytest.raises(ContainmentError, match="suspended"):
        A.admit_artifact(artifact, consumer="c", registry_path=reg, authority=AllowForTests())
    assert UNPICKLED == []


def test_positive_control_an_explicit_allow_admits_without_unpickling(artifact, registry):
    admitted = A.admit_artifact(artifact, consumer="c", registry_path=registry, authority=AllowForTests())
    assert admitted.decision.outcome is A.Outcome.ALLOW and admitted.record.feature_names == ("a", "b")
    assert UNPICKLED == []


def test_the_default_authority_denies_everything_citing_c10(artifact, registry):
    with pytest.raises(ContainmentError, match="C10"):
        A.admit_artifact(artifact, consumer="c", registry_path=registry)


@pytest.mark.parametrize("returned", [False, None, True, {"outcome": "allow"}])
def test_only_a_typed_decision_counts(artifact, registry, returned):
    with pytest.raises(ContainmentError, match="typed Decision"):
        A.admit_artifact(artifact, consumer="c", registry_path=registry, authority=_Returns(returned))


@pytest.mark.parametrize("decision", [
    _decision(sha="0" * 64), _decision(consumer="someone-else"), _decision(policy="1" * 64),
    _decision(reason="  "),
], ids=["other-artifact", "other-consumer", "other-policy", "blank-reason"])
def test_a_decision_must_be_bound_to_this_artifact_consumer_and_policy(artifact, registry, decision):
    with pytest.raises(ContainmentError, match="not bound"):
        A.admit_artifact(artifact, consumer="c", registry_path=registry, authority=_Returns(decision))


def test_an_explicit_deny_refuses_with_its_reason(artifact, registry):
    deny = _decision(consumer="same", outcome=A.Outcome.DENY, reason="not for you")
    with pytest.raises(ContainmentError, match="not for you"):
        A.admit_artifact(artifact, consumer="c", registry_path=registry, authority=_Returns(deny))


def test_an_allow_computed_under_the_previous_policy_is_stale(artifact, registry, monkeypatch):
    old_policy = A.quarantine_policy_sha256()
    monkeypatch.setattr(quarantine_policy, "QUARANTINED_FEATURES", QUARANTINED_FEATURES + ("new_quarantine",))
    assert A.quarantine_policy_sha256() != old_policy
    stale = _decision(consumer="same", policy=old_policy)
    with pytest.raises(ContainmentError, match="not bound"):
        A.admit_artifact(artifact, consumer="c", registry_path=registry, authority=_Returns(stale))


def test_export_refuses_an_ensemble_reduction_and_raises_the_refusal(tmp_path):
    import export_model
    args = argparse.Namespace(input=str(tmp_path), output=str(tmp_path / "o.joblib"),
                              exclude_models=["cnn_1d"], skip_smoke_test=True)
    with pytest.raises(ContainmentError, match="different model"):
        export_model.cmd_export(args)


# --- a STATIC guarantee (2026-09-23): no loader call escapes admission, whatever an environment can run ---
# A test the development sandbox skipped (it needs PyTorch) still called VariantEnsemble.load without a
# consumer and failed only on the owner's machine. This check reads the code, so it does not depend on
# which tests an environment is able to execute.
_LOADERS = ("InferencePipeline.load", "VariantEnsemble.load")


def _loader_violations(root: Path) -> list:
    import ast
    found = []
    for folder in ("src", "scripts", "tests"):
        for path in sorted((root / folder).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Attribute) and ast.unparse(node).endswith(_LOADERS)):
                    continue
                parent = parents.get(node)
                if isinstance(parent, ast.Call) and parent.func is node:
                    call = parent                                   # InferencePipeline.load(...)
                elif (isinstance(parent, ast.Call) and ast.unparse(parent.func) in ("functools.partial", "partial")
                      and parent.args and parent.args[0] is node):
                    call = parent                                   # functools.partial(InferencePipeline.load, ...)
                else:
                    found.append(f"{path.relative_to(root)}:{node.lineno} passes {ast.unparse(node)} as a bare value")
                    continue
                if "consumer" not in {k.arg for k in call.keywords}:
                    found.append(f"{path.relative_to(root)}:{node.lineno} {ast.unparse(node)} without consumer=")
    return found


def test_every_loader_call_and_reference_names_a_consumer():
    root = Path(__file__).resolve().parents[2]
    violations = _loader_violations(root)
    assert not violations, "loads that bypass the admission route's consumer binding:\n" + "\n".join(violations)
