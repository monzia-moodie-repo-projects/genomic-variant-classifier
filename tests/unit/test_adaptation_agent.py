"""
test_adaptation_agent.py - Monzia Moodie

Unit tests for AdaptationAgent. The heavy isolated-venv evaluation is exercised
via a stubbed _evaluate_candidate (the real one builds venvs + installs, which
belongs to opt-in integration runs, not the unit suite). Everything else - the
candidate parsing, the pytest-output verdict parser, plan mode, append-only
ledger dedup, and the dry_run contract - is tested against the real BaseAgent
and SharedState.
"""
import json
from pathlib import Path

import pytest

from genomic_variant_classifier.agent_layer.shared_state import SharedState
from genomic_variant_classifier.agent_layer.agents.adaptation_agent import (
    AdaptationAgent,
    AdaptationConfig,
    Candidate,
    parse_candidates,
    parse_pytest_output,
)


# -- fixtures ---------------------------------------------------------------
@pytest.fixture
def state(tmp_path) -> SharedState:
    ss = SharedState(state_file=str(tmp_path / "state.json"))
    ss.update_section("version_monitor", {
        "deps_major_bumps": ["pandas 2.3.3 -> 3.0.3", "numpy 1.26.4 -> 2.0.0"],
        "python_alert": "",
        "pyg_abi_alert": "",
        "python_running": "3.12.10",
    })
    return ss


def _cfg(tmp_path, **kw) -> AdaptationConfig:
    base = dict(
        evaluate=False,
        ledger_path=tmp_path / "ledger.jsonl",
        venv_root=tmp_path / "venvs",
        project_root=tmp_path,
        max_candidates_per_run=1,
    )
    base.update(kw)
    return AdaptationConfig(**base)


# -- candidate parsing ------------------------------------------------------
def test_parse_candidates_deps_python_abi():
    vm = {
        "deps_major_bumps": ["pandas 2.3.3 -> 3.0.3", "garbled-entry"],
        "python_alert": "newer series 3.13.2 available; running 3.12.10",
        "pyg_abi_alert": "torch_scatter failed to import (OSError 0xc0000139)",
        "python_running": "3.12.10",
    }
    cands = parse_candidates(vm)
    kinds = sorted(c.kind for c in cands)
    assert kinds == ["deps_major", "deps_major", "pyg_abi", "python"]
    pandas = next(c for c in cands if c.name == "pandas")
    assert pandas.from_version == "2.3.3" and pandas.to_version == "3.0.3"
    assert pandas.key == "deps_major:pandas:3.0.3"
    # unparseable bump is preserved, not dropped
    assert any(c.raw == "garbled-entry" for c in cands)


def test_parse_candidates_empty():
    assert parse_candidates({}) == []
    assert parse_candidates({"deps_major_bumps": [], "python_alert": "",
                             "pyg_abi_alert": ""}) == []


# -- pytest output parser ---------------------------------------------------
@pytest.mark.parametrize("out,exp", [
    ("960 passed, 6 skipped, 41 warnings in 213.91s (0:03:33)",
     {"passed": 960, "skipped": 6, "failed": 0, "error": 0, "first_failure": ""}),
    ("4 passed in 4.87s",
     {"passed": 4, "failed": 0, "error": 0, "skipped": 0, "first_failure": ""}),
    ("FAILED tests/unit/test_x.py::test_y - assert 1 == 2\n2 failed, 3 passed in 1.2s",
     {"passed": 3, "failed": 2, "error": 0,
      "first_failure": "tests/unit/test_x.py::test_y"}),
    ("ERROR tests/unit/test_z.py::test_w\n1 error in 0.50s",
     {"error": 1, "passed": 0, "failed": 0,
      "first_failure": "tests/unit/test_z.py::test_w"}),
])
def test_parse_pytest_output(out, exp):
    parsed = parse_pytest_output(out)
    for k, v in exp.items():
        assert parsed[k] == v, f"{k}: {parsed[k]} != {v}"


# -- plan mode (default, safe) ----------------------------------------------
def test_plan_mode_writes_ledger_and_alerts_without_evaluating(state, tmp_path, monkeypatch):
    agent = AdaptationAgent(state, _cfg(tmp_path, evaluate=False))
    # ensure the heavy path is NOT taken
    monkeypatch.setattr(agent, "_evaluate_candidate",
                        lambda c: pytest.fail("evaluate must not run in plan mode"))
    result = agent.run(dry_run=False)

    assert result["action"] == "plan"
    assert result["n_candidates"] == 2 and result["n_new"] == 2
    assert result["n_evaluated"] == 0

    ledger = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines() if l.strip()]
    assert len(ledger) == 2
    assert all(e["action"] == "planned" and e["verdict"] == "planned" for e in ledger)

    # the human is still alerted
    assert len(state.unresolved_review_items()) == 1
    # state section reflects the run
    assert state.get_section("adaptation")["n_new"] == 2


# -- append-only ledger dedup -----------------------------------------------
def test_evaluated_candidate_is_not_reprocessed(state, tmp_path):
    led = tmp_path / "ledger.jsonl"
    led.parent.mkdir(parents=True, exist_ok=True)
    # pre-seed: pandas already EVALUATED; numpy only PLANNED (should re-appear)
    led.write_text(
        json.dumps({"candidate_key": "deps_major:pandas:3.0.3", "action": "evaluated"}) + "\n"
        + json.dumps({"candidate_key": "deps_major:numpy:2.0.0", "action": "planned"}) + "\n"
    )
    agent = AdaptationAgent(state, _cfg(tmp_path, evaluate=False))
    result = agent.run(dry_run=False)

    # pandas evaluated -> skipped; numpy only planned before -> still "new"
    assert result["n_new"] == 1
    assert result["latest"][0]["candidate"] == "deps_major:numpy:2.0.0"


# -- evaluate mode (heavy path stubbed) -------------------------------------
def test_evaluate_mode_records_verdict(state, tmp_path, monkeypatch):
    agent = AdaptationAgent(state, _cfg(tmp_path, evaluate=True, max_candidates_per_run=1))

    def fake_eval(cand: Candidate) -> dict:
        return {"venv_path": "X", "install_ok": True, "install_error": "",
                "test_returncode": 1, "passed": 950, "failed": 10, "error": 0,
                "skipped": 6, "first_failure": "tests/unit/test_api.py::test_z",
                "duration_s": 12.3, "verdict": "incompatible"}

    monkeypatch.setattr(agent, "_evaluate_candidate", fake_eval)
    result = agent.run(dry_run=False)

    assert result["action"] == "evaluate"
    assert result["n_evaluated"] == 1            # capped at max_candidates_per_run
    assert result["n_incompatible"] == 1
    ledger = [json.loads(l) for l in (tmp_path / "ledger.jsonl").read_text().splitlines() if l.strip()]
    evaluated = [e for e in ledger if e["action"] == "evaluated"]
    assert len(evaluated) == 1
    assert evaluated[0]["verdict"] == "incompatible"
    assert evaluated[0]["first_failure"] == "tests/unit/test_api.py::test_z"
    # the second candidate beyond the cap is still recorded as planned
    assert any(e["action"] == "planned" for e in ledger)


# -- dry_run contract -------------------------------------------------------
def test_dry_run_writes_no_external_state(state, tmp_path):
    agent = AdaptationAgent(state, _cfg(tmp_path, evaluate=True))
    result = agent.run(dry_run=True)

    # no ledger file, no review items (external) ...
    assert not (tmp_path / "ledger.jsonl").exists()
    assert state.unresolved_review_items() == []
    # ... but the internal section IS updated, and nothing was evaluated
    assert state.get_section("adaptation")["n_new"] == 2
    assert result["n_evaluated"] == 0
    assert result["action"] == "plan"   # dry_run forces plan (no evaluation)
