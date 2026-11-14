"""Stage 1D-1b.4 (set "as"): the claims authority and the within-unit statistics. Probes T2, T4, E1, E2 and E5 of the
2026-11-13 reference audit are tests here. Owner ruling 68ca322f; reference packages 7b277d7c and 7ba10b3b.
Author: Monzia Moodie."""
from __future__ import annotations

import json

import numpy as np
import pytest

from genomic_variant_classifier.information_sources import claims as C


def exp(**kw):
    d = dict(experiment_lock="lock", fit="fit", model="m", population="popA", split="spA", identity_policy="hgnc",
             source_snapshot="src", label_snapshot="lab", endpoint="e", regime="strict")
    d.update(kw)
    return C.Experiment(**d)


def ev(kind, e, status="supported", issuer="iss"):
    scope = e.scope_for(kind)
    return C.Evidence(kind, scope, json.dumps({"status": status, "scope": scope}, sort_keys=True).encode(), issuer)


def _rule(enum):
    def evaluate(payload, scope):
        body = json.loads(payload)
        if body["scope"] != dict(scope):       # the payload must describe the scope it is pinned to
            return enum("unknown") if enum is C.EvidenceStatus else enum("not_evaluated")
        return enum(body["status"])
    return C.Rule(evaluate, "rel-1")


def authority(items, **override):
    pins = {(a.kind, a.content_sha256): C.TrustedPin(a.kind, a.issuer, a.content_sha256, a.scope) for a in items}
    rules = {k: _rule(C.Finding if k == C.RESULT_KIND else C.EvidenceStatus) for k in C.EVIDENCE_SCOPE}
    rules.update(override)
    return C.ClaimsAuthority(pins=pins, rules=rules)


def _full(claim, tier, e):
    items = {k: ev(k, e) for k in C.required_evidence(claim, tier) if e.scope_for(k)}
    items[C.RESULT_KIND] = ev(C.RESULT_KIND, e)
    return items


LR, EXPL = C.ClaimKind.LABEL_REPRODUCTION, C.Tier.EXPLORATORY


def test_label_reproduction_needs_no_assay_dimension_and_publishes_only_with_a_pinned_result():
    e = exp()                                                     # assay_snapshot is None: no placeholder (E2)
    items = _full(LR, EXPL, e)
    out = authority(items.values()).assess(LR, EXPL, e, items)
    assert out["eligibility"] == "eligible" and out["scientific_finding"] == "supported" and out["publish_supported"]
    assert out["policy_digest"] == C.POLICY_DIGEST
    no_result = {k: v for k, v in items.items() if k != C.RESULT_KIND}
    out = authority(no_result.values()).assess(LR, EXPL, e, no_result)
    assert out["eligibility"] == "eligible" and out["scientific_finding"] == "not_evaluated" and not out["publish_supported"]


def test_a_claim_needing_an_undefined_dimension_is_not_evaluated():
    e = exp()
    items = _full(C.ClaimKind.MOLECULAR_EFFECT, EXPL, e)
    out = authority(items.values()).assess(C.ClaimKind.MOLECULAR_EFFECT, EXPL, e, items)
    assert out["facts"]["assay"]["status"] == "not_evaluated" and out["eligibility"] == "unknown"
    assert "assay_snapshot" in out["facts"]["assay"]["reason"]


@pytest.mark.parametrize("kind", ["upstream", "statistical_result", "evaluation", "criterion"])
def test_evidence_for_population_a_never_certifies_population_b(kind):
    """T2 (TaskScoped's upstream scope) and E1 (EvidenceScope's statistical-result key) are closed."""
    a, b = exp(), exp(population="popB", split="spB")
    claim = C.ClaimKind.INDEPENDENT_CLINICAL_EVIDENCE
    items_b = _full(claim, EXPL, b)
    items_b[kind] = ev(kind, a)                                   # this one kind is population A's
    out = authority(items_b.values()).assess(claim, EXPL, b, items_b)
    got = out["statistical_result"] if kind == C.RESULT_KIND else out["facts"][kind]
    assert got["status"] == "unknown" and not out["publish_supported"]


def test_a_split_fact_is_model_independent_and_scopes_are_complete():
    assert exp().scope_for("split") == exp(model="other").scope_for("split")
    with pytest.raises(ValueError, match="exactly"):
        C.Evidence("split", {"population": "p", "split": "s"}, b"x", "iss")              # identity_policy missing
    with pytest.raises(ValueError, match="exactly"):
        C.Evidence("split", {"population": "p", "split": "s", "identity_policy": "h", "model": "m"}, b"x", "iss")
    with pytest.raises(ValueError, match="unregistered"):
        C.Evidence("vibes", {}, b"x", "iss")
    with pytest.raises(ValueError, match="nonempty"):
        exp(population=" ")


def test_tampered_bytes_another_issuer_or_a_mislabelled_payload_are_unknown():
    e = exp()
    items = _full(LR, EXPL, e)
    auth = authority(items.values())
    o = items["split"]
    for bad in (C.Evidence("split", o.scope, b"tampered", "iss"), C.Evidence("split", o.scope, o.payload, "other")):
        out = auth.assess(LR, EXPL, e, dict(items, split=bad))
        assert out["facts"]["split"]["status"] == "unknown"
    lying = C.Evidence("split", o.scope, json.dumps({"status": "supported", "scope": {"population": "x"}}).encode(), "iss")
    out = authority(list(items.values()) + [lying]).assess(LR, EXPL, e, dict(items, split=lying))
    assert out["facts"]["split"]["status"] == "unknown"           # E5: the rule checks the payload's own scope


def test_tiers_add_requirements_and_a_violation_blocks():
    assert {"preregistration", "holdout_release"} <= C.required_evidence(LR, C.Tier.CONFIRMATORY)
    assert "external_population" in C.required_evidence(C.ClaimKind.MECHANISM, C.Tier.EXTERNAL)
    assert "external_population" not in C.required_evidence(C.ClaimKind.MECHANISM, C.Tier.CONFIRMATORY)
    e = exp()
    items = _full(LR, EXPL, e)
    items["split"] = ev("split", e, status="violated")
    out = authority(items.values()).assess(LR, EXPL, e, items)
    assert out["eligibility"] == "blocked" and not out["publish_supported"]


def test_a_rule_returning_a_plain_string_or_the_wrong_enum_is_refused():
    e = exp()
    items = _full(LR, EXPL, e)
    with pytest.raises(TypeError, match="split"):
        authority(items.values(), split=C.Rule(lambda p, s: "supported", "rel")).assess(LR, EXPL, e, items)
    with pytest.raises(TypeError, match="statistical_result"):
        authority(items.values(), statistical_result=_rule(C.EvidenceStatus)).assess(LR, EXPL, e, items)


def test_the_authority_validates_its_pins_and_rules_and_is_fixed():
    e = exp()
    item = ev("split", e)
    pin = C.TrustedPin("split", "iss", item.content_sha256, item.scope)
    with pytest.raises(ValueError, match="keyed"):
        C.ClaimsAuthority(pins={("split", "0" * 64): pin}, rules={})
    with pytest.raises(ValueError, match="registered"):
        C.ClaimsAuthority(pins={}, rules={"vibes": _rule(C.EvidenceStatus)})
    with pytest.raises(ValueError, match="SHA-256"):
        C.TrustedPin("split", "iss", "x", item.scope)
    auth = authority([item])
    with pytest.raises(AttributeError):
        auth._rules = {}                                           # no instance dictionary: the rules are fixed
    with pytest.raises(ValueError, match="unregistered"):
        auth.assess(LR, EXPL, e, {"vibes": item})


def test_the_assessment_digest_binds_the_facts():
    e = exp()
    items = _full(LR, EXPL, e)
    other = dict(items, evaluation=C.Evidence("evaluation", items["evaluation"].scope,
                                              items["evaluation"].payload + b" ", "iss"))
    a = authority(items.values()).assess(LR, EXPL, e, items)
    b = authority(other.values()).assess(LR, EXPL, e, other)
    assert a["eligibility"] == b["eligibility"] == "eligible"
    assert a["assessment_digest"] != b["assessment_digest"]


# ------------------------------------------------------------------------------------------------ statistics
def test_two_estimands_and_exclusions():
    g = ["big"] * 200
    y = [0] * 100 + [1] * 100
    s = [0.0] * 100 + [1.0] * 100
    for k in range(50):
        g += [f"s{k}"] * 2
        y += [0, 1]
        s += [1.0, 0.0]
    g += ["one"] * 3
    y += [1, 1, 1]
    s += [0.1, 0.2, 0.3]
    d = C.within_unit_auc(np.array(y, dtype=np.int64), s, g)
    assert d["unit_macro"] == pytest.approx(1 / 51) and d["pair_weighted"] == pytest.approx(10000 / 10050)
    assert d["excluded_one_class_units"] == 1 and d["eligible_pairs"] == 10050 and d["per_unit"]["big"] == 1.0


def test_ties_take_mid_ranks_and_bad_inputs_are_refused():
    d = C.within_unit_auc(np.array([0, 1, 0, 1], dtype=np.int64), [0.5, 0.5, 0.5, 0.5], ["u"] * 4)
    assert d["unit_macro"] == 0.5
    for y, s, u in ((np.array([True, False]), [0.1, 0.2], ["u", "u"]), (np.array([0, 1]), [0.1, np.nan], ["u", "u"]),
                    (np.array([0, 1]), [0.1, 0.2], ["u", ""]), (np.array([0, 1]), [0.1], ["u", "u"])):
        with pytest.raises((TypeError, ValueError)):
            C.within_unit_auc(y, s, u)


def test_bootstrap_is_deterministic_and_zero_for_identical_scorers():
    m = np.array([0.6, 0.7, 0.8, 0.9])
    r = C.paired_unit_bootstrap(m, m, replicates=200)
    assert r["interval"] == (0.0, 0.0) and r == C.paired_unit_bootstrap(m, m, replicates=200)
    shifted = C.paired_unit_bootstrap(m + 0.1, m, replicates=200)
    assert shifted["estimate"] == pytest.approx(0.1) and shifted["interval"][0] == pytest.approx(0.1)
    for kw in (dict(replicates=50), dict(seed=1.0), dict(alpha=1)):
        with pytest.raises(ValueError):
            C.paired_unit_bootstrap(m, m, **kw)
    with pytest.raises(ValueError):
        C.paired_unit_bootstrap(m[:1], m[:1])


def test_one_authority_trusts_many_pins_of_one_kind():
    """T3: TaskScoped kept ONE (digest, issuer) per evidence kind, so two models' evaluations could not both be trusted."""
    a, b = exp(), exp(model="other", fit="fit2")
    items_a, items_b = _full(LR, EXPL, a), _full(LR, EXPL, b)
    auth = authority(list(items_a.values()) + list(items_b.values()))
    assert auth.assess(LR, EXPL, a, items_a)["publish_supported"]
    assert auth.assess(LR, EXPL, b, items_b)["publish_supported"]
    crossed = dict(items_b, evaluation=items_a["evaluation"])
    assert auth.assess(LR, EXPL, b, crossed)["facts"]["evaluation"]["status"] == "unknown"
