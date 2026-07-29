"""The writer and the reader must agree about absence.

WHAT WAS WRONG
==============
`MetricResult.to_dict` emitted the raw `NaN` that a non-OK result is REQUIRED to
carry in memory, and `dump_strict_json` refuses a non-finite number by design --
so every refused result was UNPERSISTABLE through `to_dict` alone.

`from_dict` had always documented the opposite:

    "Round-trip from to_dict(). NaN does not survive strict JSON, so a null
     value is read back as NaN rather than rejected."

**The reader was right the whole time; only the writer disagreed with it.** The
fix aligns the writer, and this module pins the agreement.

THE INTERNAL REPRESENTATION IS UNCHANGED
-----------------------------------------
`test_metric_result_relocation` pins that a non-OK result must carry `NaN` in
memory, and it still does. `NaN` is a perfectly good in-memory sentinel; it is
only in an ARTIFACT that it becomes an absent estimate wearing a number's
clothes.

THE RULE IS STATUS-AWARE
------------------------
Absence is authorised by the STATUS, never inferred from the value. An OK result
whose value is somehow non-finite is a defect, and nulling it here would disguise
that defect as a legitimate absence -- so it is left for the strict writer to
refuse.
"""
from __future__ import annotations

import math

import pytest

from genomic_variant_classifier.evaluation.capabilities import (
    MetricResult,
    MetricStatus,
)
from genomic_variant_classifier.evaluation.serialization import dump_strict_json

REFUSED_STATUSES = [s for s in MetricStatus if s is not MetricStatus.OK]


def _refused(status):
    return MetricResult(float("nan"), status, "binary_class_support_required", {})


@pytest.mark.parametrize("status", REFUSED_STATUSES, ids=lambda s: s.value)
def test_a_refused_result_serialises_as_null(status):
    """THE DEFECT. Every refused result was unpersistable before this fix."""
    assert _refused(status).to_dict()["value"] is None


@pytest.mark.parametrize("status", REFUSED_STATUSES, ids=lambda s: s.value)
def test_a_refused_result_survives_strict_json(status):
    dump_strict_json(_refused(status).to_dict(), artifact="probe")


@pytest.mark.parametrize("status", REFUSED_STATUSES, ids=lambda s: s.value)
def test_the_round_trip_restores_the_in_memory_sentinel(status):
    """`null` on the way out, `NaN` on the way back -- which is exactly what
    `from_dict` documented before the writer agreed with it."""
    original = _refused(status)
    restored = MetricResult.from_dict(original.to_dict())

    assert math.isnan(restored.value), "the in-memory sentinel must return"
    assert restored.status is original.status
    assert restored.reason == original.reason


def test_an_ok_result_keeps_its_value():
    """Guards the guard: a blanket non-finite sweep would pass every test above
    while erasing real measurements."""
    payload = MetricResult(0.87, MetricStatus.OK, None, {}).to_dict()
    assert payload["value"] == 0.87
    assert MetricResult.from_dict(payload).value == 0.87


def test_an_ok_result_with_a_non_finite_value_is_not_disguised():
    """STATUS-AWARE, NOT VALUE-AWARE.

    An OK result cannot normally hold a NaN -- the constructor forbids it -- so
    this asserts the serialiser's RULE rather than a reachable state: absence is
    authorised by the status. Were the rule value-based instead, a defect that
    produced a non-finite OK value would be silently relabelled as a legitimate
    absence, and the strict writer would never get the chance to refuse it.
    """
    import inspect

    source = inspect.getsource(MetricResult.to_dict)
    assert "self.status is not MetricStatus.OK" in source, (
        "the null rule must be gated on STATUS; a blanket non-finite sweep would "
        "disguise a defect as an absence")


def test_the_reader_still_accepts_a_legacy_nan_payload():
    """Artifacts written before this fix carry a raw NaN. They must still load,
    or the fix would strand every report already on disk."""
    restored = MetricResult.from_dict({
        "value": float("nan"), "status": "undefined",
        "reason": "binary_class_support_required", "metadata": {}})
    assert math.isnan(restored.value)
    assert restored.status is MetricStatus.UNDEFINED
