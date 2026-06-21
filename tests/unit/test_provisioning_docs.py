"""test_provisioning_docs.py  --  Author: Monzia Moodie
Validates the provisioning section schema, the read-modify-write history recorder
(with cap), loud input validation, and the markdown run-doc writer.
Imports from the installed package (src-layout).
"""
from genomic_variant_classifier.agent_layer.provisioning import provisioning_docs as P


def _evt(**kw):
    base = dict(phase="select", provider="vast", offer_id="40057706",
                gpu_name="RTX_4090", num_gpus=1, vram_gb=24.0, price_per_hr=0.3347,
                budget_cap_per_hr=0.77, est_hours=15.0, dry_run=True)
    base.update(kw)
    return P.new_event(**base)


def test_new_event_computes_cost_and_within_budget():
    e = _evt()
    assert e["est_cost_usd"] == round(0.3347 * 15.0, 4)
    assert e["within_budget"] is True
    assert e["provider"] == "vast" and e["offer_id"] == "40057706"


def test_new_event_within_budget_false_when_over_cap():
    e = _evt(price_per_hr=1.50, budget_cap_per_hr=0.77)
    assert e["within_budget"] is False


def test_new_event_loud_validation():
    import pytest
    with pytest.raises(ValueError):
        P.new_event(phase="bogus", provider="vast", offer_id="1")
    with pytest.raises(ValueError):
        P.new_event(phase="select", provider="aws", offer_id="1")      # bad provider
    with pytest.raises(ValueError):
        P.new_event(phase="select", provider="vast", offer_id="")      # missing id
    with pytest.raises(ValueError):
        P.new_event(phase="select", provider="vast", offer_id="1", price_per_hr=-1)


class _FakeState:
    """Mimics SharedState's get_section / update_section (shallow-merge)."""
    def __init__(self):
        self.sections: dict[str, dict] = {}

    def get_section(self, name):
        return dict(self.sections.get(name, {}))

    def update_section(self, name, updates):
        self.sections.setdefault(name, {}).update(updates)


def test_record_appends_history_and_scalars():
    st = _FakeState()
    P.record_provisioning(st.get_section, st.update_section, _evt(offer_id="A"))
    P.record_provisioning(st.get_section, st.update_section, _evt(offer_id="B", phase="provision",
                                                                  instance_id="inst-9"))
    sec = st.sections[P.PROVISIONING_SECTION]
    assert sec["n_events"] == 2
    assert [h["offer_id"] for h in sec["history"]] == ["A", "B"]
    assert sec["last_offer_id"] == "B" and sec["last_phase"] == "provision"
    assert sec["last_instance_id"] == "inst-9"


def test_record_caps_history():
    st = _FakeState()
    for i in range(60):
        P.record_provisioning(st.get_section, st.update_section, _evt(offer_id=str(i)), cap=50)
    sec = st.sections[P.PROVISIONING_SECTION]
    assert len(sec["history"]) == 50
    assert sec["history"][0]["offer_id"] == "10"   # oldest 10 dropped
    assert sec["history"][-1]["offer_id"] == "59"


def test_write_doc_contains_all_audit_fields(tmp_path):
    e = _evt(phase="provision", instance_id="contract-123", ssh_host="1.2.3.4", ssh_port=22001,
             image="pytorch/pytorch:2.4.0", disk_gb=120, approved=True,
             candidate_offers=[{"provider": "vast", "offer_id": "X", "gpu_name": "RTX_4090",
                                "price_per_hr": 0.40}],
             scp_legs=[{"direction": "up", "path": "data/", "size": "1.1 GB"}],
             preflight_gates={"preflight_run17": "GO", "string_artifacts": "present"},
             agent_liveness="all ACTIVE (21/21)", teardown_confirmed=False,
             reason="cheapest verified 4090 within $0.77 cap")
    path = P.write_provisioning_doc(tmp_path, e)
    assert path.exists() and path.suffix == ".md"
    body = path.read_text(encoding="utf-8")
    for needle in ["contract-123", "1.2.3.4", "22001", "pytorch/pytorch:2.4.0",
                   "cheapest verified 4090", "preflight_run17", "all ACTIVE (21/21)",
                   "Teardown confirmed", "SCP legs", "Candidate offers considered"]:
        assert needle in body, needle


def test_doc_filename_is_filesystem_safe(tmp_path):
    e = _evt(ts="2026-06-20T02:30:00+00:00")
    path = P.write_provisioning_doc(tmp_path, e)
    assert ":" not in path.name and path.name.startswith("PROVISION_")
