"""The package-1 prediction contract: identities, payloads, replay, the selected blend and its consumers.
Author: Monzia Moodie

Frozen acceptance matrix PACKAGE1_ACCEPTANCE_MATRIX_2026-11-07.md, set 1a: gates G1, G3, G4, G5.2, G6 and G9. Every
refusal test names the gate it proves. "Refused" means a typed exception before any value is published.
"""
from __future__ import annotations

import math
from fractions import Fraction

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from genomic_variant_classifier.models.combiner import (
    BLEND_KERNEL,
    ConvexBlendSpec,
    MemberSlot,
    combine_stored_blend,
    DOT2_CANDIDATE_KERNEL,
    HISTORICAL_KERNEL,
    dot2,
    exact_blend,
    historical_parity,
    require_supported_modalities,
    score_once,
    validate_stored_weights,
)
from genomic_variant_classifier.prediction_contract import (
    Binding,
    ContractError,
    DrawPlan,
    DrawSpec,
    Observation,
    ObservationSet,
    PlannedReplay,
    ProbabilityBlock,
    Quantity,
    ReplayVerifier,
    ScientificScope,
    canonical_digest,
    integer_positions,
    positive_column,
    select_original_rows,
)
from tests.unit._prediction_support import FixedProba, WidthSensitive, blend_pipeline, blend_spec

D = canonical_digest("test.dataset", {"n": 1})
U = canonical_digest("test.unit", {"unit": "whole_variant"})
U2 = canonical_digest("test.unit", {"unit": "transcript_A"})


def _sha(tag: str) -> str:
    return canonical_digest("test.sha", tag)


def _rows(n=3, dataset=D, context=U):
    return ObservationSet(tuple(Observation(dataset, f"position:{i}", f"1:{i + 1}:A:T", context) for i in range(n)))


def _binding(producer="p"):
    return Binding(_sha(producer), _sha("ic"), _sha("split"), _sha("sched"), _sha("in"), _sha("exec"))


def _block(rows, p, draw="d0", binding=None, classes=(0, 1)):
    p = np.asarray(p, dtype=np.float64)
    matrix = np.column_stack([1 - p, p]) if classes == (0, 1) else np.column_stack([p, 1 - p])
    return ProbabilityBlock.from_array(rows=rows, class_order=classes, positive_class=1,
                                       binding=binding or _binding(), draw_id=draw, probabilities=matrix)


# ---------------------------------------------------------------------------------------------------------------------
# G1. Identity of prediction units
# ---------------------------------------------------------------------------------------------------------------------

class TestPredictionUnits:

    def test_the_two_identities_are_distinct_namespaces_even_one_to_one(self):
        row = Observation(D, "position:0", "1:1:A:T", U)
        assert row.source_observation_id != row.prediction_unit_id

    def test_one_source_row_may_expand_into_several_units(self):
        rows = ObservationSet((Observation(D, "position:0", "1:1:A:T", U), Observation(D, "position:0", "1:1:A:T", U2)))
        assert rows.source_count == 1 and rows.max_units_per_source == 2

    def test_a_repeated_prediction_unit_is_refused(self):
        row = Observation(D, "position:0", "1:1:A:T", U)
        with pytest.raises(ContractError, match="duplicate prediction units"):
            ObservationSet((row, Observation(D, "position:0", "1:1:A:T", U)))

    def test_a_changed_dataset_changes_both_identities_and_split_is_not_part_of_either(self):
        a = Observation(D, "position:0", "1:1:A:T", U)
        b = Observation(canonical_digest("test.dataset", {"n": 2}), "position:0", "1:1:A:T", U)
        assert a.source_observation_id != b.source_observation_id and a.prediction_unit_id != b.prediction_unit_id
        assert not {"split", "fold"} & set(vars(a))

    @pytest.mark.parametrize("raw", [np.array([True, False]), np.array(["0", "1"]), np.array([0.0, 1.0]),
                                     [0, True], [0.0], ["1"]])
    def test_positions_other_than_integers_are_refused(self, raw):
        with pytest.raises(ContractError):
            integer_positions(raw)

    def test_numpy_positions_enter_through_one_recorded_conversion(self):
        positions, conversion = integer_positions(np.array([2, 0], dtype=np.int32))
        assert positions == [2, 0] and type(positions[0]) is int and conversion == "numpy.int32.tolist_to_python_int"
        picked = select_original_rows(_rows(3), positions)
        assert [r.source_row_key for r in picked.rows] == ["position:2", "position:0"]

    def test_the_ordered_set_is_digested_once_and_order_matters(self):
        rows = _rows(3)
        reversed_rows = ObservationSet(tuple(reversed(rows.rows)))
        assert rows.digest != reversed_rows.digest
        with pytest.raises(Exception):
            rows.digest = "x"          # frozen


# ---------------------------------------------------------------------------------------------------------------------
# G4. Identity-bound payloads and class handling
# ---------------------------------------------------------------------------------------------------------------------

class TestPayloads:

    @pytest.mark.parametrize("matrix", [np.array([[0.4, 0.4], [0.5, 0.5], [0.1, 0.9]]),     # row sum 0.8
                                        np.array([[1.2, -0.2], [0.5, 0.5], [0.1, 0.9]]),    # outside [0, 1]
                                        np.array([[0, 1], [1, 0], [0, 1]])])               # integers
    def test_invalid_payloads_are_refused_never_repaired(self, matrix):
        with pytest.raises(ContractError):
            ProbabilityBlock.from_array(rows=_rows(3), class_order=(0, 1), positive_class=1, binding=_binding(),
                                        draw_id="d", probabilities=matrix)

    def test_the_payload_is_immutable(self):
        block = _block(_rows(3), [0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            block.probabilities()[0, 0] = 0.5

    def test_class_order_selects_the_positive_column_and_the_unused_column_is_validated(self):
        model = FixedProba(0.8, classes=(1, 0))
        values = model.predict_proba(np.zeros((2, 1)))
        positive, classes, _ = positive_column(model, values, 2)
        assert classes == (1, 0) and np.allclose(positive, 0.8)
        bad = values.copy()
        bad[:, 1] = 0.7                          # the unused column alone breaks the row sums
        with pytest.raises(ContractError):
            positive_column(model, bad, 2)

    @pytest.mark.parametrize("classes", [np.array(["0", "1"]), np.array([0.0, 1.0]), np.array([1, 2]),
                                         np.array([0, 1, 2])])
    def test_classes_other_than_integer_zero_and_one_are_refused(self, classes):
        model = FixedProba(0.5)
        model.classes_ = classes
        with pytest.raises(ContractError):
            positive_column(model, np.full((2, 2), 0.5), 2)

    def test_quantities_carry_unit_and_status(self):
        assert Quantity(0.0, "nats", "defined").as_dict() == {"value": 0.0, "unit": "nats", "status": "defined",
                                                              "reason": None}
        with pytest.raises(ContractError):
            Quantity(None, "nats", "unavailable")          # a reason is required
        with pytest.raises(ContractError):
            Quantity(0.0, "nats", "unavailable", "x")      # an unavailable value is null, never zero
        with pytest.raises(ContractError):
            Quantity(1.0, "bits", "defined")               # unknown unit


# ---------------------------------------------------------------------------------------------------------------------
# G9. Replay and the declared draw plan
# ---------------------------------------------------------------------------------------------------------------------

SCOPE = ScientificScope(_sha("task"), _sha("method"), _sha("numerics"))


def _plan(seeds=(11, 22, 33)):
    member = _sha("mc_member")
    return DrawPlan(owner_sha256=member, execution_sha256=_sha("exec"), scope=SCOPE, mechanism="mc_dropout",
                    draws=tuple(DrawSpec(f"pass:{i}", member, s) for i, s in enumerate(seeds)))


def _plan_block(plan, i, p, rows):
    binding = Binding(plan.owner_sha256, _sha("ic"), _sha("split"), plan.digest, _sha("in"), plan.execution_sha256)
    return _block(rows, p, draw=plan.draws[i].draw_id, binding=binding)


class TestReplay:

    def test_an_identical_replay_completes(self):
        rows, plan = _rows(2), _plan()
        values = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        replay = PlannedReplay(plan)
        for i, v in enumerate(values):
            replay.observe(_plan_block(plan, i, v, rows), actual_draw=plan.draws[i], actual_scope=SCOPE)
        replay.seal()
        for i, v in enumerate(values):
            replay.check(_plan_block(plan, i, v, rows), actual_draw=plan.draws[i], actual_scope=SCOPE)
        evidence = replay.finish()
        assert evidence["draws_verified"] == 3 and evidence["schedule_status"] == "matches_declared_execution_plan"

    def test_a_changed_value_or_order_is_refused(self):
        rows, plan = _rows(2), _plan()
        replay = PlannedReplay(plan)
        for i, v in enumerate([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]):
            replay.observe(_plan_block(plan, i, v, rows), actual_draw=plan.draws[i], actual_scope=SCOPE)
        replay.seal()
        with pytest.raises(ContractError, match="differs"):
            replay.check(_plan_block(plan, 0, [0.1, 0.2000001], rows), actual_draw=plan.draws[0],
                         actual_scope=SCOPE)

    def test_a_replayed_prefix_is_incomplete(self):
        verifier = ReplayVerifier()
        rows = _rows(2)
        blocks = [_block(rows, [0.1, 0.2], draw=f"d{i}") for i in range(3)]
        for b in blocks:
            verifier.observe(b)
        verifier.seal()
        verifier.check(blocks[0])
        with pytest.raises(ContractError, match="incomplete"):
            verifier.finish()

    def test_an_executed_draw_outside_the_plan_is_refused(self):
        rows, plan = _rows(2), _plan()
        replay = PlannedReplay(plan)
        with pytest.raises(ContractError, match="planned schedule"):
            replay.observe(_plan_block(plan, 0, [0.1, 0.2], rows),
                           actual_draw=DrawSpec("pass:0", plan.owner_sha256, 99), actual_scope=SCOPE)

    def test_duplicate_mc_seeds_are_refused_by_the_plan(self):
        with pytest.raises(ContractError, match="distinct declared MC seeds"):
            _plan(seeds=(5, 5, 6))


# ---------------------------------------------------------------------------------------------------------------------
# G3. The stored blend and its kernel
# ---------------------------------------------------------------------------------------------------------------------

def _spec(weights, names=("a", "b")):
    return ConvexBlendSpec("in_memory_fit_record_sha256", _sha("src"), _sha("scope"),
                           tuple(MemberSlot(n, _sha(n), _sha(n + "stage")) for n in names), tuple(weights),
                           _sha("fit"), BLEND_KERNEL)


class TestStoredBlend:

    @pytest.mark.parametrize("weights", [(0.5, float("nan")), (1.5, -0.5), (0.5, 0.5 + 2e-12), (0.5, 0.4)])
    def test_defective_weights_are_refused_never_normalised(self, weights):
        with pytest.raises(ContractError):
            _spec(weights)

    def test_weights_within_the_declared_legacy_tolerance_are_kept_exactly(self):
        w = (0.3, 0.7 + 4.99e-17)   # Run 15's measured mass defect is 4.99e-17
        assert _spec(w).weights == w

    def test_missing_weights_are_refused(self):
        with pytest.raises(ContractError, match="no stored blend weights"):
            validate_stored_weights(None, 2)

    def test_the_weights_are_bound_by_their_exact_binary64_form(self):
        a, b = _spec((0.25, 0.75)), _spec((0.25 + 2.0 ** -54, 0.75 - 2.0 ** -54))
        assert a.digest != b.digest

    def test_an_unqualified_kernel_is_refused(self):
        with pytest.raises(ContractError, match="unqualified combiner kernel"):
            ConvexBlendSpec("in_memory_fit_record_sha256", _sha("src"), _sha("scope"),
                            (MemberSlot("a", _sha("a"), _sha("s")),), (1.0,), _sha("fit"),
                            DOT2_CANDIDATE_KERNEL)

    def test_the_exact_kernel_equals_exact_rational_arithmetic_rounded_once(self):
        rng = np.random.default_rng(1107)
        rows = [rng.uniform(0.0, 1.0, size=13) for _ in range(300)]
        rows += [np.ldexp(rng.random(13), rng.integers(-1074, 1, 13)) for _ in range(300)]   # log-uniform, subnormal
        p = np.array(rows)
        w = rng.dirichlet(np.ones(13))
        w = tuple(float(x) for x in w)
        w = w[:-1] + (float(Fraction(1) - sum(Fraction(x) for x in w[:-1])),)   # mass exactly representable near 1
        got = exact_blend(p, w)
        for i in range(len(p)):
            exact = sum((Fraction(float(p[i, j])) * Fraction(w[j]) for j in range(13)), Fraction(0))
            assert got[i] == float(exact), i            # Fraction -> float is correctly rounded, ties to even

    def test_the_owner_counterexamples_separate_exact_rounding_from_dot2(self):
        p, w = np.array([[1.0, 2.0 ** -52, 2.0 ** -106]]), (0.5, 0.25, 0.25)
        assert exact_blend(p, w)[0] == math.nextafter(0.5, 1.0) and exact_blend(p, w)[0] > 0.5
        assert dot2(p, w)[0] == 0.5
        tiny = math.ulp(0.0)
        assert exact_blend(np.array([[tiny, tiny]]), (0.5, 0.5))[0] == tiny
        with np.errstate(under="ignore"):
            assert dot2(np.array([[tiny, tiny]]), (0.5, 0.5))[0] == 0.0

    def test_ties_round_to_even_and_zero_is_canonical(self):
        assert exact_blend(np.array([[1.0, 2.0 ** -53]]), (0.5, 0.5))[0] == 0.5         # exact 0.5 + 2^-54: a tie
        import struct
        assert struct.pack(">d", exact_blend(np.array([[-0.0, 0.0]]), (0.5, 0.5))[0]) == bytes(8)

    def test_a_result_above_one_is_refused_not_clipped(self):
        with pytest.raises(ContractError, match="above one"):
            exact_blend(np.ones((1, 2)), (0.5, 0.5 + 1e-13))

    def test_dot2_is_a_candidate_never_a_selectable_kernel(self):
        with pytest.raises(ContractError, match="unqualified combiner kernel"):
            ConvexBlendSpec("in_memory_fit_record_sha256", _sha("src"), _sha("scope"),
                            (MemberSlot("a", _sha("a"), _sha("s")),), (1.0,), _sha("fit"), DOT2_CANDIDATE_KERNEL)

    def test_a_kernel_change_is_a_predictor_identity_change_and_history_replays_bitwise(self):
        from dataclasses import replace
        exact_spec = _spec((0.25, 0.75))
        historical_spec = replace(exact_spec, kernel=HISTORICAL_KERNEL)
        assert historical_spec.digest != exact_spec.digest and historical_spec.weights == exact_spec.weights
        p = np.random.default_rng(9).uniform(size=(50, 2))
        replayed = combine_stored_blend(historical_spec, p, observed_members=historical_spec.members,
                                        ordered_rows_sha256=_sha("r"), rendered_input_sha256=_sha("i")).scores()
        assert replayed.tobytes() == (p @ np.array([0.25, 0.75])).tobytes()

    def test_historical_parity_is_reported_not_gated(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0.0, 1.0, size=(2000, 13))
        w = tuple(float(x) for x in rng.dirichlet(np.ones(13)))
        w = w[:-1] + (float(Fraction(1) - sum(Fraction(x) for x in w[:-1])),)
        report = historical_parity(p, w, edges=(0.1, 0.5, 0.9))
        assert report["kernel"] == BLEND_KERNEL and report["historical_kernel"] == HISTORICAL_KERNEL
        assert set(report) >= {"rows_bitwise_different", "ulp_histogram", "decision_changes_ge",
                               "decision_changes_gt", "sorted_positions_differing", "max_abs_difference"}
        assert sum(report["ulp_histogram"].values()) == 2000

    def test_unsupported_modalities_are_refused_before_any_member_is_called(self):
        spec = _spec((0.5, 0.5), names=("lightgbm", "cnn_1d"))
        with pytest.raises(ContractError, match="unsupported fitted predictor capabilities"):
            require_supported_modalities(spec, required_by_member={"lightgbm": frozenset({"tabular"}),
                                                                   "cnn_1d": frozenset({"sequence_windows"})},
                                         available=frozenset({"tabular"}))
        a, cnn = FixedProba(0.4), FixedProba(0.6)
        with pytest.raises(ContractError, match="unsupported fitted predictor capabilities"):
            blend_pipeline({"lightgbm": a, "cnn_1d": cnn})
        assert a.calls == [] and cnn.calls == []


class TestScoreOnce:

    def _setup(self):
        spec = _spec((0.25, 0.75))
        rows = _rows(3)
        bindings = (Binding(spec.members[0].artifact_sha256, _sha("ic"), _sha("split"), _sha("s"), _sha("in"),
                            _sha("e")),
                    Binding(spec.members[1].artifact_sha256, _sha("ic"), _sha("split"), _sha("s"), _sha("in"),
                            _sha("e")))
        return spec, rows, bindings

    def _run(self, spec, rows, bindings, predict):
        return score_once(spec, models={"a": object(), "b": object()}, predict_member=predict, rows=rows,
                          member_bindings=bindings, member_draw_ids=("a:0", "b:0"), result_draw_id="r",
                          combiner_execution_sha256=_sha("ce"))

    def test_the_result_binds_the_member_blocks(self):
        spec, rows, bindings = self._setup()
        values = {"a": [0.1, 0.2, 0.3], "b": [0.9, 0.8, 0.7]}

        def predict(name, model):
            i = spec.member_names.index(name)
            return spec.members[i], _block(rows, values[name], draw=f"{name}:0", binding=bindings[i])

        result, members = self._run(spec, rows, bindings, predict)
        expected = exact_blend(np.column_stack([values["a"], values["b"]]), spec.weights)
        assert np.array_equal(result.positive_probability(), expected)
        assert result.binding.ordered_input_sha256 == canonical_digest("gvc.combiner-input-blocks.v1",
                                                                        [m.digest for m in members])

    @pytest.mark.parametrize("defect", ["reversed_rows", "other_units", "binding", "bare_array", "string_classes",
                                        "slot"])
    def test_inconsistent_member_outputs_are_refused(self, defect):
        spec, rows, bindings = self._setup()

        def predict(name, model):
            i = spec.member_names.index(name)
            use_rows, binding, slot = rows, bindings[i], spec.members[i]
            if defect == "reversed_rows" and name == "b":
                use_rows = ObservationSet(tuple(reversed(rows.rows)))
            if defect == "other_units" and name == "b":
                use_rows = _rows(3, context=U2)
            if defect == "binding" and name == "b":
                binding = bindings[0]
            if defect == "slot" and name == "b":
                slot = MemberSlot("b", _sha("other bytes"), spec.members[1].prediction_stage_sha256)
            if defect == "bare_array" and name == "b":
                return slot, np.column_stack([[0.5] * 3, [0.5] * 3])
            if defect == "string_classes" and name == "b":
                return slot, ProbabilityBlock.from_array(
                    rows=rows, class_order=("0", "1"), positive_class="1", binding=binding, draw_id="b:0",
                    probabilities=np.full((3, 2), 0.5))
            return slot, _block(use_rows, [0.5] * 3, draw=f"{name}:0", binding=binding)

        with pytest.raises(ContractError):
            self._run(spec, rows, bindings, predict)

    def test_the_full_roster_is_required_including_zero_weight_members(self):
        spec = _spec((1.0, 0.0))
        with pytest.raises(ContractError, match="full fitted roster"):
            score_once(spec, models={"a": object()}, predict_member=None, rows=_rows(1),
                       member_bindings=(_binding("a"), _binding("b")), member_draw_ids=("a", "b"),
                       result_draw_id="r", combiner_execution_sha256=_sha("ce"))

    def test_combine_refuses_reordered_members(self):
        spec = _spec((0.25, 0.75))
        with pytest.raises(ContractError, match="fitted order"):
            combine_stored_blend(spec, np.full((2, 2), 0.5), observed_members=tuple(reversed(spec.members)),
                                 ordered_rows_sha256=_sha("r"), rendered_input_sha256=_sha("i"))


# ---------------------------------------------------------------------------------------------------------------------
# G3.6 and G4.4. The golden fixture: blend and logistic differ substantially; every authoritative consumer returns the
# blend, each member is called once per logical request, and the logistic candidate is never called.
# ---------------------------------------------------------------------------------------------------------------------

class _Forbidden:
    """A logistic 'candidate' that fails the test if any authoritative path calls it."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raise AssertionError("the logistic research candidate was called on an authoritative path")


class _Counting:
    """A fitted member proxy counting calls (picklable; classes_ delegated)."""

    def __init__(self, inner):
        self.inner = inner
        self.n_calls = 0

    @property
    def classes_(self):
        return self.inner.classes_

    def predict_proba(self, X):
        self.n_calls += 1
        return self.inner.predict_proba(X)


def _fitted_two_member_ensemble(tmp_path=None):
    from genomic_variant_classifier.models.variant_ensemble import (
        EnsembleConfig, TABULAR_FEATURES, VariantEnsemble,
    )
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.random((80, len(TABULAR_FEATURES))), columns=TABULAR_FEATURES)
    y = pd.Series([0] * 40 + [1] * 40)
    ens = VariantEnsemble(EnsembleConfig(n_folds=2, model_dir=tmp_path))
    ens.base_estimators = {"logistic_regression": LogisticRegression(max_iter=200),
                           "random_forest": RandomForestClassifier(n_estimators=15, random_state=0)}
    ens.fit(X, None, y)
    return ens, X, y


class TestGoldenFixture:

    def test_every_authoritative_consumer_returns_the_blend(self):
        from genomic_variant_classifier.api.pipeline import InferencePipeline
        from genomic_variant_classifier.models.variant_ensemble import SELECTED_ENSEMBLE_ROW

        ens, X, y = _fitted_two_member_ensemble()
        stacker = ens.meta_learner
        # The fixture is only meaningful if the two combiners really disagree.
        names, base = ens._member_probability_matrix(X, None)
        blend = exact_blend(base, tuple(float(w) for w in ens.blend_weights_))
        logistic = stacker.predict_proba(base)[:, 1]
        assert np.max(np.abs(blend - logistic)) > 0.05, "fixture: blend and logistic must differ substantially"

        ens.meta_learner = _Forbidden()
        assert np.array_equal(ens.predict_proba(X)[:, 1], blend)
        row = ens.evaluate(X, None, y).loc[SELECTED_ENSEMBLE_ROW]
        from sklearn.metrics import roc_auc_score
        # evaluate() reports metrics rounded by pandas (half-even on the binary value), so compare the same way.
        assert row["auroc"] == pd.Series([roc_auc_score(y, blend)]).round(4).iloc[0]

        pipe = InferencePipeline.from_variant_ensemble(ens)
        assert isinstance(pipe.research_candidates["logistic_stacker"], _Forbidden)
        raw = pd.DataFrame({"chrom": ["1", "2"], "pos": [10, 20], "ref": ["A", "C"], "alt": ["T", "G"]})
        served = pipe.predict_proba(raw)
        rows = pipe.score(raw)
        assert np.array_equal(served, rows.scores())
        member_matrix = rows.member_matrix()
        assert np.array_equal(served, exact_blend(member_matrix, pipe.combiner.weights))

    def test_evaluate_calls_each_member_once(self):
        ens, X, y = _fitted_two_member_ensemble()
        ens.trained_models_ = {n: _Counting(m) for n, m in ens.trained_models_.items()}
        ens.evaluate(X, None, y)
        assert {n: m.n_calls for n, m in ens.trained_models_.items()} == {n: 1 for n in ens.trained_models_}

    def test_the_pipeline_calls_each_member_once_per_request(self):
        a, b = _Counting(FixedProba(0.2)), _Counting(FixedProba(0.6))
        pipe = blend_pipeline({"lightgbm": a, "xgboost": b}, weights=[0.25, 0.75])
        records = pipe.predict_batch([{"chrom": "1", "pos": i + 1, "ref": "A", "alt": "T"} for i in range(4)])
        assert a.n_calls == 1 and b.n_calls == 1
        # MEASURED: the naive float expression 0.25*0.2 + 0.75*0.6 is 0.49999999999999994; the exact value of the
        # stored binary64 inputs is 0.5000000000000000061..., whose correct rounding (the kernel's result) is 0.5.
        exact = Fraction(0.25) * Fraction(0.2) + Fraction(0.75) * Fraction(0.6)
        assert [r["pathogenicity_score"] for r in records] == [float(exact)] * 4 == [0.5] * 4

    @pytest.mark.parametrize("weights", [None, np.array([np.nan, 1.0]), np.array([-0.1, 1.1]),
                                         np.array([0.5, 0.6])])
    def test_the_ensemble_refuses_an_uncomputable_blend_and_never_substitutes(self, weights):
        ens, X, _ = _fitted_two_member_ensemble()
        ens.meta_learner = _Forbidden()
        ens.blend_weights_ = weights
        with pytest.raises(ContractError):
            ens.predict_proba(X)

    def test_an_undeclared_combiner_is_refused(self):
        ens, X, _ = _fitted_two_member_ensemble()
        ens.selected_combiner_ = None
        with pytest.raises(ContractError, match="declares the combiner None"):
            ens.predict_proba(X)

    def test_a_width_sensitive_member_proves_the_rendered_width(self):
        """Negative control for the width-sensitive fixture: a member fitted on another width refuses."""
        pipe = blend_pipeline({"lightgbm": WidthSensitive(n_features=1)})
        with pytest.raises(ValueError, match="features, but the member was fitted with 1"):
            pipe.predict_proba(pd.DataFrame({"chrom": ["1"], "pos": [1], "ref": ["A"], "alt": ["T"]}))


# ---------------------------------------------------------------------------------------------------------------------
# G3.1 and G2-adjacent persistence: the declared combiner, the missing-value policy and member identity survive save
# ---------------------------------------------------------------------------------------------------------------------

class TestPersistence:

    def _register(self, tmp_path, path, ens):
        from tests.unit._admission_support import register
        return register(tmp_path / "registry.v1.json", path, feature_names=tuple(ens.tabular_columns_),
                        roster=list(ens.trained_models_))

    def _load(self, path, registry, **kw):
        from genomic_variant_classifier.models.variant_ensemble import VariantEnsemble
        from tests.unit._admission_support import AllowForTests
        return VariantEnsemble.load(path, consumer="test", registry_path=registry, authority=AllowForTests(), **kw)

    def test_save_persists_the_declared_combiner_policy_and_member_identity(self, tmp_path):
        from genomic_variant_classifier.monitoring.model_registry import ArtifactIdentity

        ens, X, _ = _fitted_two_member_ensemble()
        spec_before = ens.convex_blend_spec(SCOPE)
        path = tmp_path / "ens.joblib"
        ens.save(path)
        orchestrator = joblib.load(path)
        assert orchestrator["selected_combiner"] == "convex_probability_blend"
        assert orchestrator["preprocessor_"] is not None
        assert orchestrator["blend_fit_report"]["convergence"] in ("converged",
                                                                     "not_converged_weights_kept_and_recorded")
        for name, digest in orchestrator["saved_model_sha256"].items():
            assert digest == ArtifactIdentity.measure(path.parent / "ens_models" / f"{name}.joblib").sha256
            # The first serialisation in this process IS the saved file: the identity recorded before the save holds.
            assert digest == {m.name: m.artifact_sha256 for m in spec_before.members}[name]
        back = self._load(path, self._register(tmp_path, path, ens))
        assert back.selected_combiner_ == "convex_probability_blend"
        assert back.combiner_declaration_source_ == "orchestrator"
        assert back.preprocessor_ is not None
        assert back.member_identities() == orchestrator["saved_model_sha256"]
        assert np.array_equal(back.predict_proba(X), ens.predict_proba(X))
        spec_after = back.convex_blend_spec(SCOPE)
        assert spec_after.source_kind == "saved_orchestrator_sha256"
        assert spec_after.weights == spec_before.weights

    def test_a_legacy_orchestrator_needs_an_explicit_declaration(self, tmp_path):
        ens, X, _ = _fitted_two_member_ensemble()
        path = tmp_path / "ens.joblib"
        ens.save(path)
        orchestrator = joblib.load(path)
        for key in ("selected_combiner", "blend_fit_report", "preprocessor_"):
            del orchestrator[key]                     # exactly what an orchestrator saved before 2026-11-07 holds
        joblib.dump(orchestrator, path)
        registry = self._register(tmp_path, path, ens)
        with pytest.raises(ContractError, match="declares no combiner"):
            self._load(path, registry)
        back = self._load(path, registry, legacy_combiner_declaration="convex_probability_blend")
        assert back.combiner_declaration_source_ == "caller_declared_for_legacy_orchestrator"
        assert back.preprocessor_ is None                 # absent, recorded as absent, never refitted
        assert np.array_equal(back.predict_proba(X), ens.predict_proba(X))
        with pytest.raises(ContractError, match="unsupported declared combiner"):
            self._load(path, registry, legacy_combiner_declaration="logistic_stacker")

    def test_a_declaring_orchestrator_refuses_a_caller_declaration(self, tmp_path):
        ens, _, _ = _fitted_two_member_ensemble()
        path = tmp_path / "ens.joblib"
        ens.save(path)
        with pytest.raises(ContractError, match="declares its own combiner"):
            self._load(path, self._register(tmp_path, path, ens),
                       legacy_combiner_declaration="convex_probability_blend")

    def test_tampered_stored_weights_are_refused_at_load(self, tmp_path):
        ens, _, _ = _fitted_two_member_ensemble()
        path = tmp_path / "ens.joblib"
        ens.save(path)
        orchestrator = joblib.load(path)
        orchestrator["blend_weights_"] = np.array([0.7, 0.7])
        joblib.dump(orchestrator, path)
        with pytest.raises(ContractError, match="weight mass"):
            self._load(path, self._register(tmp_path, path, ens))
