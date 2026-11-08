"""src/genomic_variant_classifier/models/combiner.py

Author: Monzia Moodie

The selected predictor (gate G3 of the frozen package-1 acceptance matrix): Run 15's stored convex blend, exactly.

Owner rulings of 2026-11-07:
  * The stored convex blend is the single authoritative score: exact binary64 weights, the full ordered fitted roster
    (zero-weight members included) and the member prediction stages. The logistic stacker is a separately identified
    research candidate and is never called on the authoritative path.
  * Unsupported projections are refused. No normalisation, fallback, abs() or pruning.
  * The optimiser contract is repaired in TRAINING only; the Run 15 optimiser is never rerun.

Numerical contract (owner ruling of 2026-11-07, review bundle revision 5): `exact_binary64_products_sum_rne_v1`.
The score of row i is the EXACT real sum of the products of the stored binary64 member probabilities and the stored
binary64 weights, rounded ONCE to binary64, nearest with ties to even:

    q_i = RN_64( sum_j exact(p_ij) * exact(w_j) )

"exact" means the value the stored bits represent, not that the biological probability is known exactly. The
implementation accumulates integer products in units of 2**-2148 (the product of two minimum subnormals) and builds
the rounded bits directly: no linear-algebra library, no extended-precision type, no fused multiply-add and no
summation tree can change the result. Canonical positive zero; a result above one is refused; nothing is normalised,
clipped, reordered or omitted. A different kernel is a different predictor identity.

The historical operation `P @ w` keeps its own identity (HISTORICAL_KERNEL) for replay of historical evidence only.
Dot2 (Ogita, Rump and Oishi 2005) is retained as an optimisation candidate and is NOT selectable: it is not correctly
rounded in general (the owner's counterexamples: p = (1, 2**-52, 2**-106), w = (1/2, 1/4, 1/4) has the exact value
1/2 + 2**-54 + 2**-108, which rounds to the successor of 1/2, while Dot2 returns 1/2; and p = (2**-1074, 2**-1074),
w = (1/2, 1/2) gives 2**-1074 exactly but 0 through Dot2's underflowing products).

MEASURED 2026-11-07 on Run 15's real 883,127 x 13 out-of-fold member matrix (scratchpad exact_kernel_run15.json):
the integer kernel took 8.47 s; an independent exact method (math.fsum over error-free TwoProduct pairs) agreed
bitwise on every row; the historical `P @ w` differed in 356,050 rows by at most 3 units in the last place, with zero
call changes at 0.1, 0.3, 0.5, 0.7 and 0.9 under both the >= and the > convention and zero rank changes; Dot2 agreed
with the exact result on every row of THAT matrix (which proves nothing about other matrices).

Adapted from the owner's combiner_contract_reference.py and numerical_score_reference.py (revision 5); the fit-input
stage and source kind are recorded explicitly.
"""
from __future__ import annotations

import hashlib
import logging
import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from genomic_variant_classifier.prediction_contract import (
    Binding,
    ContractError,
    ObservationSet,
    ProbabilityBlock,
    ScientificScope,
    canonical_digest,
    require_sha256,
    require_text,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BLEND_KERNEL",
    "COMBINER_KIND",
    "DOT2_CANDIDATE_KERNEL",
    "HISTORICAL_KERNEL",
    "exact_blend",
    "WEIGHT_MASS_ATOL",
    "BlendScore",
    "ConvexBlendSpec",
    "MemberSlot",
    "combine_stored_blend",
    "dot2",
    "historical_parity",
    "member_artifact_sha256",
    "member_draw_schedule",
    "member_modalities",
    "SELECTED_PREDICTOR_SCOPE",
    "prediction_stage_sha256",
    "require_supported_modalities",
    "score_once",
    "validate_stored_weights",
]

COMBINER_KIND = "convex_probability_blend"
#: The authoritative numerical contract of the selected predictor.
BLEND_KERNEL = "exact_binary64_products_sum_rne_v1"
#: The historical operation, identified for replay and parity reports only.
HISTORICAL_KERNEL = "numpy_float64_matmul_stored_weights_v1"
#: An optimisation candidate; never selectable (not correctly rounded in general).
DOT2_CANDIDATE_KERNEL = "dot2_dekker_fixed_order_candidate_v1"
_SELECTABLE_KERNELS = frozenset({BLEND_KERNEL, HISTORICAL_KERNEL})
_UNIT_EXPONENT = -2148   # the product of two minimum binary64 subnormals
WEIGHT_MASS_ATOL = 1e-12
_SOURCE_KINDS = frozenset({"saved_orchestrator_sha256", "in_memory_fit_record_sha256"})

_SPLIT = 134217729.0   # 2**27 + 1: the Veltkamp split constant for binary64

#: References to the authoritative definitions of the task, the method and the numerical policy of the SELECTED
#: predictor (not restatements). ONE definition for serving (api.pipeline.SERVING_SCOPE is this object) and for the
#: offline split scorer (VariantEnsemble.score_split), so the combiner identity a run evaluates is the identity a
#: server would carry for the same fitted ensemble. Moved here unchanged from api/pipeline.py in set 1b.
SELECTED_PREDICTOR_SCOPE = ScientificScope(
    task_sha256=canonical_digest("gvc.task.v1", {
        "endpoint": "binary_pathogenicity_probability_of_a_submitted_variant",
        "classes": [0, 1], "positive_class": 1,
        "label_definition": "the training label column of the fitted ensemble; defined by its training run"}),
    method_sha256=canonical_digest("gvc.method.v1", {
        "combiner": "convex_probability_blend", "kernel": BLEND_KERNEL,
        "members": "full_fitted_roster_in_fitted_order", "member_stage": "as_fitted_and_served"}),
    numerical_policy_sha256=canonical_digest("gvc.numerical-policy.v1", {
        "storage": "little_endian_binary64", "row_sum_atol": 1e-7, "weight_mass_atol": 1e-12,
        "clipping": "forbidden", "units": ["nats", "probability_squared"], "variance_ddof": 0}),
)

#: What each fitted member consumes. Every member not named here consumes the tabular matrix.
_MEMBER_MODALITIES: dict[str, frozenset] = {"cnn_1d": frozenset({"sequence_windows"})}


def member_modalities(names) -> dict[str, frozenset]:
    """The modalities each named member needs (ONE definition, shared by serving and the offline split scorer)."""
    return {name: _MEMBER_MODALITIES.get(name, frozenset({"tabular"})) for name in names}


# ---------------------------------------------------------------------------------------------------------------------
# Member identity
# ---------------------------------------------------------------------------------------------------------------------

class _HashingSink:
    """A write-only file object for joblib.dump: hashes the bytes without holding them (tell() is required)."""

    def __init__(self) -> None:
        self._h = hashlib.sha256()
        self._n = 0

    def write(self, data: bytes) -> int:
        self._h.update(data)
        self._n += len(data)
        return len(data)

    def tell(self) -> int:
        return self._n

    def flush(self) -> None:
        return None

    def hexdigest(self) -> str:
        return self._h.hexdigest()


def member_artifact_sha256(model: Any) -> str:
    """SHA-256 of the joblib serialisation of a member, measured ONCE when its identity is first recorded.

    MEASURED 2026-11-07 (scratchpad/p1a_probe/digest_probe.py): these bytes are identical when dumped twice in one
    process, but differ after a dump/load round trip for every tabular member (pickle memo sharing changes), so an
    identity is never recomputed from a reloaded object: a saved ensemble's members carry their saved file digests.
    """
    import joblib

    sink = _HashingSink()
    joblib.dump(model, sink)
    return sink.hexdigest()


#: Members whose served predictions pass through the isotonic wrapper fitted on the calibration fold
#: (VariantEnsemble.fit, _RECALIBRATE). The stored blend weights were fitted on RAW out-of-fold predictions, so for
#: these members the served stage differs from the fit-input stage; that mismatch is recorded, never hidden.
ISOTONIC_STAGE_MEMBERS = frozenset({"xgboost", "lightgbm", "random_forest"})


def prediction_stage_sha256(name: str, model: Any) -> str:
    """Identity of the prediction stage a member serves: wrapper class and calibration stage."""
    require_text(name, "member name")
    stage = "isotonic_calibrated_on_calibration_fold" if type(model).__name__ == "_IsotonicCalibrator" else "raw"
    return canonical_digest("gvc.member-prediction-stage.v1", {
        "member": name, "served_class": f"{type(model).__module__}.{type(model).__qualname__}", "stage": stage})


@dataclass(frozen=True)
class MemberSlot:
    name: str
    artifact_sha256: str
    prediction_stage_sha256: str

    def __post_init__(self) -> None:
        require_text(self.name, "member name")
        require_sha256(self.artifact_sha256, "member artifact")
        require_sha256(self.prediction_stage_sha256, "member prediction stage")


# ---------------------------------------------------------------------------------------------------------------------
# The stored blend
# ---------------------------------------------------------------------------------------------------------------------

def validate_stored_weights(weights: Any, n_members: int) -> tuple[float, ...]:
    """Exact stored weights as Python floats; refused (never repaired) when missing, non-finite, negative or
    mass-defective beyond the declared legacy round-off tolerance."""
    if weights is None:
        raise ContractError("no stored blend weights: the declared blend cannot be computed and no other combiner "
                            "is substituted")
    w = np.asarray(weights)
    if w.ndim != 1 or w.dtype.kind != "f" or len(w) != n_members:
        raise ContractError(f"one float weight per fitted member required (members {n_members}, weights {w.shape})")
    values = tuple(float(x) for x in w.tolist())
    if any(not math.isfinite(x) or x < 0 or x > 1 for x in values):
        raise ContractError("finite stored weights in [0, 1] required; no abs() and no clipping")
    if abs(math.fsum(values) - 1.0) > WEIGHT_MASS_ATOL:
        raise ContractError(f"stored weight mass {math.fsum(values)!r} is outside the declared tolerance "
                            f"{WEIGHT_MASS_ATOL}; no normalisation")
    return values


@dataclass(frozen=True)
class ConvexBlendSpec:
    """The authoritative combiner: exact weights bound by their binary64 hexadecimal form."""

    source_kind: str
    source_sha256: str
    scientific_scope_sha256: str
    members: tuple[MemberSlot, ...]
    weights: tuple[float, ...]
    fit_input_stage_sha256: str
    kernel: str = BLEND_KERNEL
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.source_kind not in _SOURCE_KINDS:
            raise ContractError(f"unknown source kind {self.source_kind!r}")
        for name in ("source_sha256", "scientific_scope_sha256", "fit_input_stage_sha256"):
            require_sha256(getattr(self, name), name)
        if (type(self.members) is not tuple or not self.members
                or any(type(m) is not MemberSlot for m in self.members)):
            raise ContractError("nonempty immutable ordered member slots required")
        if len({m.name for m in self.members}) != len(self.members):
            raise ContractError("duplicate member name")
        if type(self.weights) is not tuple or len(self.weights) != len(self.members):
            raise ContractError("one immutable weight per fitted member required")
        if any(type(w) is not float for w in self.weights):
            raise ContractError("Python float weights required")
        validate_stored_weights(np.asarray(self.weights, dtype=np.float64), len(self.members))
        if self.kernel not in _SELECTABLE_KERNELS:
            raise ContractError(f"unqualified combiner kernel {self.kernel!r}")
        value = {k: getattr(self, k) for k in ("source_kind", "source_sha256", "scientific_scope_sha256",
                                                "fit_input_stage_sha256", "kernel")}
        value.update({"kind": COMBINER_KIND, "positive_class": 1,
                      "members": [asdict(m) for m in self.members],
                      "weights_binary64_hex": [w.hex() for w in self.weights],
                      "weight_policy": "preserve_stored_no_normalization",
                      "weight_mass_atol": WEIGHT_MASS_ATOL})
        object.__setattr__(self, "digest", canonical_digest("gvc.combiner.convex-blend.v1", value))

    @property
    def member_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self.members)

    def describe(self) -> dict[str, Any]:
        return {"kind": COMBINER_KIND, "combiner_sha256": self.digest, "kernel": self.kernel,
                "members": list(self.member_names),
                "weights_binary64_hex": [w.hex() for w in self.weights],
                "fit_input_stage_sha256": self.fit_input_stage_sha256,
                "source_kind": self.source_kind, "source_sha256": self.source_sha256}


def _two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = a + b
    bp = s - a
    return s, (a - (s - bp)) + (b - bp)


def _split(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def _two_product(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, al * bl - (((p - ah * bh) - al * bh) - ah * bl)


def _round_right_ties_even(n: int, shift: int) -> int:
    quotient, remainder = divmod(n, 1 << shift)
    half = 1 << (shift - 1)
    return quotient + int(remainder > half or (remainder == half and quotient & 1))


def _rounded_nonnegative_accumulator(n: int) -> float:
    """n * 2**-2148 rounded once to nonnegative binary64 (nearest, ties to even); 0 -> canonical +0."""
    if n == 0:
        return 0.0
    shift = max(1074, n.bit_length() - 53)     # subnormal spacing 2**-1074; normal results keep 53 bits
    q = _round_right_ties_even(n, shift)
    if shift == 1074 and q < (1 << 52):
        bits = q
    else:
        if q == (1 << 53):
            q >>= 1
            shift += 1
        exponent = _UNIT_EXPONENT + shift + 52
        if exponent > 1023:
            raise ArithmeticError("exact sum overflows binary64")
        bits = ((exponent + 1023) << 52) | (q - (1 << 52))
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _weight_terms(weights: tuple[float, ...]) -> list[tuple[int, int]]:
    terms = []
    for w in weights:
        if type(w) is not float or not math.isfinite(w) or w < 0:
            raise ContractError("finite nonnegative Python float weights required")
        wn, wd = w.as_integer_ratio()                    # binary64 denominators are powers of two
        terms.append((wn, -_UNIT_EXPONENT - (wd.bit_length() - 1)))
    return terms


_EXACT_CHUNK_ROWS = 65536


def exact_blend(matrix: Any, weights: tuple[float, ...]) -> np.ndarray:
    """The authoritative kernel: per row, the exact sum of stored-value products rounded once (see module text).

    Inputs: a validated (rows, members) binary64 matrix in [0, 1] and the exact stored weights. A result above one is
    refused, never clipped.
    """
    p = np.asarray(matrix)
    if p.ndim != 2 or p.dtype != np.float64 or p.shape[1] != len(weights) or not len(p):
        raise ContractError("nonempty binary64 (rows, members) matrix with one weight per member required")
    terms = _weight_terms(tuple(weights))
    out = np.empty(len(p), dtype=np.float64)
    for start in range(0, len(p), _EXACT_CHUNK_ROWS):
        for i, row in enumerate(p[start:start + _EXACT_CHUNK_ROWS].tolist(), start):
            acc = 0
            for value, (wn, wshift) in zip(row, terms):
                if value:                                   # +0 and -0 contribute exactly nothing
                    pn, pd = value.as_integer_ratio()
                    acc += (pn * wn) << (wshift - (pd.bit_length() - 1))
            out[i] = _rounded_nonnegative_accumulator(acc)
    if np.any(out > 1.0):
        raise ContractError("the exact stored-weight result lies above one; nothing is clipped")
    return out


def dot2(matrix: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    """Dot2 CANDIDATE (DOT2_CANDIDATE_KERNEL): compensated dot product in the fixed member order.

    Not selectable and not correctly rounded in general (see the module text). Kept for qualification studies.
    """
    n = matrix.shape[0]
    s, c = _two_product(matrix[:, 0], np.full(n, weights[0]))
    for j in range(1, matrix.shape[1]):
        h, r = _two_product(matrix[:, j], np.full(n, weights[j]))
        s, q = _two_sum(s, h)
        c = c + (q + r)
    return s + c


def _matrix(values: Any, n_members: int) -> np.ndarray:
    p = np.asarray(values)
    if (p.ndim != 2 or p.shape[0] == 0 or p.shape[1] != n_members
            or p.dtype.kind != "f" or p.dtype.itemsize not in (4, 8)):
        raise ContractError("nonempty float32/64 (rows, fitted members) matrix required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ContractError("invalid member probability; no clipping")
    return np.array(p, dtype=np.float64, order="C", copy=True)


@dataclass(frozen=True, eq=False)
class BlendScore:
    combiner_sha256: str
    ordered_rows_sha256: str
    rendered_input_sha256: str
    n_rows: int
    score_bytes: bytes
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for name in ("combiner_sha256", "ordered_rows_sha256", "rendered_input_sha256"):
            require_sha256(getattr(self, name), name)
        if type(self.n_rows) is not int or self.n_rows <= 0:
            raise ContractError("positive row count required")
        if type(self.score_bytes) is not bytes or len(self.score_bytes) != self.n_rows * 8:
            raise ContractError("immutable binary64 score payload required")
        p = self.scores()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ContractError("computed blend outside the probability domain; no repair")

    def scores(self) -> np.ndarray:
        return np.frombuffer(self.score_bytes, dtype="<f8")

    def prediction_matrix(self) -> np.ndarray:
        p = self.scores()
        return np.column_stack((1.0 - p, p))


def combine_stored_blend(spec: ConvexBlendSpec, base_probabilities: Any, *, observed_members: tuple[MemberSlot, ...],
                         ordered_rows_sha256: str, rendered_input_sha256: str) -> BlendScore:
    if type(spec) is not ConvexBlendSpec:
        raise ContractError("explicit convex blend spec required; no implicit fallback")
    if type(observed_members) is not tuple or observed_members != spec.members:
        raise ContractError("member identity, prediction stage, or fitted order mismatch")
    p = _matrix(base_probabilities, len(spec.members))
    if spec.kernel == BLEND_KERNEL:
        score = exact_blend(p, spec.weights)
    else:   # HISTORICAL_KERNEL: replay of historical evidence only; the constructor admits no other kernel
        score = p @ np.asarray(spec.weights, dtype=np.float64)
    return BlendScore(spec.digest, ordered_rows_sha256, rendered_input_sha256, len(p),
                      score.astype("<f8").tobytes())


def score_once(spec: ConvexBlendSpec, *, models: Mapping[str, Any],
               predict_member: Callable[[str, Any], tuple[MemberSlot, ProbabilityBlock]], rows: ObservationSet,
               member_bindings: tuple[Binding, ...], member_draw_ids: tuple[str, ...], result_draw_id: str,
               combiner_execution_sha256: str) -> tuple[ProbabilityBlock, tuple[ProbabilityBlock, ...]]:
    """Each member called ONCE and the selected combiner applied ONCE (G4.4).

    predict_member returns (executed MemberSlot, ProbabilityBlock) from the producer adapter. Expected bindings come
    from the immutable request plan. These checks detect inconsistency, not dishonesty about what was executed.
    Returns the result block (whose input identity binds the member block digests) and the member blocks.
    """
    if type(spec) is not ConvexBlendSpec:
        raise ContractError("explicit blend required")
    if type(rows) is not ObservationSet:
        raise ContractError("validated immutable request rows required")
    if (type(member_bindings) is not tuple or len(member_bindings) != len(spec.members)
            or any(type(b) is not Binding for b in member_bindings)):
        raise ContractError("one expected input/execution binding per member required")
    if type(member_draw_ids) is not tuple or len(member_draw_ids) != len(spec.members):
        raise ContractError("one expected invocation identity per member required")
    for name in member_draw_ids:
        require_text(name, "member draw identity")
    require_text(result_draw_id, "result draw identity")
    require_sha256(combiner_execution_sha256, "actual combiner execution identity")
    if len({b.split_sha256 for b in member_bindings}) != 1:
        raise ContractError("members must share the declared prediction split")
    if any(b.producer_sha256 != m.artifact_sha256 for b, m in zip(member_bindings, spec.members)):
        raise ContractError("expected producer does not match the fitted member")
    names = spec.member_names
    if set(models) != set(names):
        raise ContractError("full fitted roster required, including zero-weight members")
    vectors, receipts, blocks = [], [], []
    for i, name in enumerate(names):
        receipt, block = predict_member(name, models[name])
        if type(receipt) is not MemberSlot or receipt != spec.members[i]:
            raise ContractError("executed member or prediction stage differs from the fitted spec")
        if type(block) is not ProbabilityBlock:
            raise ContractError("identity-bound ProbabilityBlock required; bare arrays refused")
        if block.rows.digest != rows.digest:
            raise ContractError("member rows differ from the ordered request rows")
        if block.binding != member_bindings[i] or block.draw_id != member_draw_ids[i]:
            raise ContractError("member input, split, schedule, execution or invocation mismatch")
        if (any(type(c) is not int for c in block.class_order) or set(block.class_order) != {0, 1}
                or type(block.positive_class) is not int or block.positive_class != 1):
            raise ContractError("the selected blend requires canonical integer classes {0, 1}, positive 1")
        vectors.append(block.positive_probability())
        receipts.append(receipt)
        blocks.append(block)
    input_digest = canonical_digest("gvc.combiner-input-blocks.v1", [b.digest for b in blocks])
    score = combine_stored_blend(spec, np.column_stack(vectors), observed_members=tuple(receipts),
                                 ordered_rows_sha256=rows.digest, rendered_input_sha256=input_digest)
    result_binding = Binding(
        producer_sha256=spec.digest,
        input_contract_sha256=canonical_digest("gvc.combiner-input-contract.v1", {
            "scope": spec.scientific_scope_sha256, "members": [asdict(m) for m in spec.members]}),
        split_sha256=member_bindings[0].split_sha256,
        draw_schedule_sha256=canonical_digest("gvc.combiner-member-invocations.v1", [
            {"draw_id": d, "schedule": b.draw_schedule_sha256} for d, b in zip(member_draw_ids, member_bindings)]),
        ordered_input_sha256=input_digest,
        execution_sha256=canonical_digest("gvc.combiner-execution.v1", {
            "kernel": spec.kernel, "combiner_execution": combiner_execution_sha256,
            "member_executions": [b.execution_sha256 for b in member_bindings]}),
    )
    result = ProbabilityBlock.from_array(rows=rows, class_order=(0, 1), positive_class=1, binding=result_binding,
                                         draw_id=result_draw_id, probabilities=score.prediction_matrix())
    return result, tuple(blocks)


def member_draw_schedule(name: str, model: Any) -> dict[str, Any]:
    """The declared draw schedule of one member's prediction (ONE definition, shared by serving and the offline
    split scorer): the MC dropout seed schedule, the fitted deep-ensemble member seeds, or a single evaluation."""
    require_text(name, "member name")
    if hasattr(model, "draw_seeds") and getattr(model, "_supports_mc", False):
        return {"mechanism": "mc_dropout_mean", "seeds": model.draw_seeds()}
    if hasattr(model, "members_"):
        return {"mechanism": "fitted_ensemble_mean",
                "member_seeds": list(getattr(model, "member_seeds_", ["unrecorded"]))}
    return {"mechanism": "deterministic_single_evaluation"}


def require_supported_modalities(spec: ConvexBlendSpec, *, required_by_member: Mapping[str, frozenset],
                                 available: frozenset) -> None:
    """Construction-time capability check, before any member is called (G3.5)."""
    if type(spec) is not ConvexBlendSpec:
        raise ContractError("explicit blend required")
    if set(required_by_member) != set(spec.member_names):
        raise ContractError("complete member capability declaration required")
    if any(type(v) is not frozenset for v in required_by_member.values()):
        raise ContractError("immutable modality sets required")
    if type(available) is not frozenset:
        raise ContractError("immutable available modality set required")
    missing = {name: sorted(needs - available) for name, needs in required_by_member.items()
               if not needs <= available}
    if missing:
        raise ContractError(f"unsupported fitted predictor capabilities: {missing}. The stored blend needs every "
                            "fitted member; a projection that drops one is a different, unevaluated predictor and "
                            "is refused")


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Integer distance of nonnegative binary64 values (signed zero canonicalised first)."""
    ua = np.where(a == 0, 0.0, a).astype("<f8").view("<u8")
    ub = np.where(b == 0, 0.0, b).astype("<f8").view("<u8")
    return np.maximum(ua, ub) - np.minimum(ua, ub)


def historical_parity(base_probabilities: Any, weights: tuple[float, ...],
                      edges: tuple[float, ...] = (0.5,)) -> dict[str, Any]:
    """The numerical migration report (owner ruling 2026-11-07): the historical `P @ w` against the exact kernel.

    Reported, never gated, and never turned into a tolerance: bits, ULP distances, absolute differences, call changes
    at every declared edge under BOTH equality conventions, and changes of score order.
    """
    p = _matrix(base_probabilities, len(weights))
    historical = p @ np.asarray(weights, dtype=np.float64)
    exact = exact_blend(p, tuple(weights))
    ulps = _ulp_distance(historical, exact)
    cuts = np.asarray(edges, dtype=np.float64)
    ge = lambda x: np.searchsorted(cuts, x, side="right")   # noqa: E731 -- positive_if_p_ge: equal counts above
    gt = lambda x: np.searchsorted(cuts, x, side="left")    # noqa: E731 -- a strict > convention
    changed_ge = np.flatnonzero(ge(historical) != ge(exact))
    changed_gt = np.flatnonzero(gt(historical) != gt(exact))
    order_changes = int((np.argsort(historical, kind="stable") != np.argsort(exact, kind="stable")).sum())
    values, counts = np.unique(ulps, return_counts=True)
    return {"rows": int(len(p)), "rows_bitwise_different": int((ulps != 0).sum()), "max_ulp": int(ulps.max()),
            "ulp_histogram": {int(v): int(c) for v, c in zip(values, counts)},
            "max_abs_difference": float(np.max(np.abs(historical - exact))),
            "decision_changes_ge": int(changed_ge.size), "decision_change_rows_ge": changed_ge.tolist()[:100],
            "decision_changes_gt": int(changed_gt.size), "decision_change_rows_gt": changed_gt.tolist()[:100],
            "sorted_positions_differing": order_changes,
            "edges": [float(e) for e in cuts], "historical_kernel": HISTORICAL_KERNEL, "kernel": BLEND_KERNEL}
