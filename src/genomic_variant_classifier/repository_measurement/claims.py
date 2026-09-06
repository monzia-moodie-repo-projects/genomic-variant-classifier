"""What a measurement establishes, and what it explicitly does not.

ADR-0005.

A measurement without scope is a future fabricated inference waiting for an
opportunity. `does_not_prove` is not a disclaimer: it is the field that stops a
later reader promoting a bounded result into a general one.

There is exactly ONE such field. A renderer may LABEL it "LIMITATIONS", but
storing both `limitations` and `does_not_prove` would duplicate authority --
the defect this whole programme exists to remove.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MeasurementClaim:
    """The propositions a measurement establishes and refuses.

    Order is the author's presentation order and is NOT sorted. These are
    semantic prose, and silently reordering them would change emphasis the
    author chose.
    """

    proves: Tuple[str, ...]
    does_not_prove: Tuple[str, ...]
    method: str

    def __post_init__(self) -> None:
        if not self.proves:
            raise ValueError(
                "measurement must state at least one proposition it "
                "establishes; a measurement that proves nothing is not a "
                "measurement")
        if not self.method.strip():
            raise ValueError("measurement method must be non-empty")
        for field, values in (("proves", self.proves),
                              ("does_not_prove", self.does_not_prove)):
            blank = [v for v in values if not v.strip()]
            if blank:
                raise ValueError(
                    "{} contains {} empty proposition(s)".format(
                        field, len(blank)))
        overlap = set(self.proves) & set(self.does_not_prove)
        if overlap:
            raise ValueError(
                "claim contradiction: proposition appears under both proves "
                "and does_not_prove: {!r}".format(sorted(overlap)))
