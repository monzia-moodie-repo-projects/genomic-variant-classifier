"""Reference migration controls. No source-authority or independence certification."""
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

PARTITIONS = ("train", "validation", "test")
EXPOSURES = frozenset({"training", "tuning", "calibration", "exploration", "test_feedback"})


def identity(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def valid_id(x):
    return isinstance(x, str) and bool(x) and x == x.strip()


@dataclass(frozen=True)
class SplitRegistry:
    policy_id: str
    salt: str
    weights: tuple
    assignments: object

    def __post_init__(self):
        if not valid_id(self.policy_id) or not valid_id(self.salt):
            raise ValueError("Require a frozen policy ID and salt")
        if len(self.weights) != 3 or any(type(w) is not int or w <= 0 for w in self.weights):
            raise ValueError("Require three positive integer allocation weights")
        mapping = dict(self.assignments)
        if any(not valid_id(k) or v not in PARTITIONS for k, v in mapping.items()):
            raise ValueError("Invalid existing group assignment")
        object.__setattr__(self, "weights", tuple(self.weights))
        object.__setattr__(self, "assignments", MappingProxyType(mapping))

    def extend(self, groups):
        result = dict(self.assignments)
        for group in groups:
            if not valid_id(group): raise ValueError("Invalid group ID")
            if group in result: continue
            h = hashlib.sha256(json.dumps([self.policy_id, self.salt, group],
                                         separators=(",", ":")).encode()).digest()
            bucket = int.from_bytes(h, "big") % sum(self.weights)
            boundary = 0
            for partition, weight in zip(PARTITIONS, self.weights):
                boundary += weight
                if bucket < boundary:
                    result[group] = partition
                    break
        return SplitRegistry(self.policy_id, self.salt, self.weights, result)

    def manifest(self):
        return {"policy_id": self.policy_id, "salt": self.salt,
                "weights": list(self.weights), "assignments": dict(self.assignments)}

    @property
    def sha256(self): return identity(self.manifest())


def confirmation_screen(candidates, ledger):
    seen_variants, seen_groups = set(), set()
    for item in ledger:
        if item["use"] not in EXPOSURES:
            raise ValueError("Unknown exposure type")
        if not valid_id(item["variant_id"]) or not valid_id(item["group_id"]):
            raise ValueError("Invalid ledger identity")
        seen_variants.add(item["variant_id"])
        seen_groups.add(item["group_id"])
    output, ids = [], set()
    for row in candidates:
        key, group = row["variant_id"], row["group_id"]
        if not valid_id(key) or not valid_id(group) or key in ids:
            raise ValueError("Missing or duplicated candidate identity")
        ids.add(key)
        blockers = []
        if key in seen_variants: blockers.append("previously_exposed_variant")
        if group in seen_groups: blockers.append("previously_exposed_group")
        output.append({"variant_id": key, "group_id": group,
                       "passes_recorded_exposure_screen": not blockers, "blockers": blockers})
    return output


def cohort_cells(legacy_members, corrected_members, universe):
    sets = []
    for values in (legacy_members, corrected_members, universe):
        values = list(values)
        if any(not valid_id(x) for x in values) or len(values) != len(set(values)):
            raise ValueError("Invalid or duplicated population identity")
        sets.append(set(values))
    old, new, all_rows = sets
    if not (old | new) <= all_rows: raise ValueError("Membership outside universe")
    return {name: sorted(values) for name, values in {
        "common": old & new, "added": new - old, "removed": old - new,
        "excluded_both": all_rows - (old | new)}.items()}


def brier_factorial(rows):
    sums = {c: {m: 0.0 for m in ("legacy", "corrected")} for c in ("common", "added")}
    counts = dict.fromkeys(sums, 0)
    ids = set()
    for r in rows:
        key, c, y = r["variant_id"], r["evaluation_cell"], r["label"]
        if not valid_id(key) or key in ids: raise ValueError("Invalid prediction identity")
        ids.add(key)
        if c not in sums or type(y) is not int or y not in (0, 1):
            raise ValueError("Invalid cell or label")
        counts[c] += 1
        for m in ("legacy", "corrected"):
            p = r[m + "_probability"]
            if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Invalid probability")
            sums[c][m] += (p - y) ** 2
    if not all(counts.values()): raise ValueError("Both common and added populations required")
    means = {c: {m: sums[c][m] / counts[c] for m in sums[c]} for c in sums}
    delta_common = means["common"]["corrected"] - means["common"]["legacy"]
    delta_added = means["added"]["corrected"] - means["added"]["legacy"]
    return {"counts": counts, "mean_brier": means, "delta_common": delta_common,
            "delta_added": delta_added, "interaction": delta_added - delta_common,
            "interpretation": "Negative delta is lower Brier loss; descriptive, not causal."}
