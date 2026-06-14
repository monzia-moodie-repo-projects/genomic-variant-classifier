"""test_interpretability_agent.py -- Monzia Moodie

Liveness tests for InterpretabilityAgent -- previously the only registered agent with zero tests. Hermetic:
a fresh SharedState with no checkpoint and no CHECKPOINT_READY message means the SHAP audit is not due, so
run() returns the 'skipped' contract without needing torch/shap. Covers dry-run, a real run on empty state,
and idempotent re-run.
"""
from genomic_variant_classifier.agent_layer.agents.interpretability_agent import InterpretabilityAgent
from genomic_variant_classifier.agent_layer.shared_state import SharedState


def _agent(tmp_path):
    return InterpretabilityAgent(SharedState(state_file=tmp_path / "state.json"))


def test_dry_run_skips_cleanly_with_no_checkpoint(tmp_path):
    res = _agent(tmp_path).run(dry_run=True)
    assert isinstance(res, dict) and "action" in res
    assert res["action"] == "skipped" and res.get("audited") is False


def test_real_run_does_not_raise_on_empty_inbox(tmp_path):
    res = _agent(tmp_path).run(dry_run=False)
    assert res["action"] == "skipped"  # not due -> no heavy SHAP path, no exception


def test_second_run_idempotent(tmp_path):
    a = _agent(tmp_path)
    a.run(dry_run=True)
    assert a.run(dry_run=True)["action"] == "skipped"
