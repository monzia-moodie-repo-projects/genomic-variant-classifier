"""Source-only lockfile repair admission (owner ruling 2026-10-06, options C + D).

Admits a candidate lockfile whose ONLY differences from the current one are provenance fields of explicitly selected packages: no
package added, removed or re-versioned, no top-level change (the R version included), no field outside SOURCE_FIELDS. It is a
CHANGE-SCOPE check, not artifact verification -- allowing `Hash` to change does not establish that the new value is correct;
legitimate records come from the pinned tooling and their diff is inspected independently.

The code is the owner's reference (ruling generation 027757ee, lines 153-211), transformed mechanically: ValueError -> AdmissionError
(a ValueError subclass, same messages). Measured 2026-10-06 against the real 87-package lockfile: an S4Arrays Repository change was
admitted; version, R, removal, non-target, forbidden-field and unknown-target changes were refused.

Author: Monzia Moodie
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

from genomic_variant_classifier.environment_qualification.r_runtime import AdmissionError

logger = logging.getLogger(__name__)

__all__ = ["SOURCE_FIELDS", "admit_source_repair"]

SOURCE_FIELDS = {
    "Source", "Repository",
    "RemoteType", "RemoteHost", "RemoteUsername", "RemoteRepo",
    "RemoteUrl", "RemoteRef", "RemoteSha", "RemoteSubdir",
    "Hash",
}

def admit_source_repair(
    old: Mapping,
    new: Mapping,
    target_packages: set[str],
) -> dict[str, list[str]]:
    if set(old) != set(new):
        raise AdmissionError("lock.top_level_changed")

    for key in old:
        if key != "Packages" and old[key] != new[key]:
            raise AdmissionError(f"lock.metadata_changed:{key}")

    before = old["Packages"]
    after = new["Packages"]

    if set(before) != set(after):
        raise AdmissionError("lock.package_set_changed")

    if not target_packages <= set(before):
        raise AdmissionError("lock.unknown_repair_target")

    changes = {}

    for package, previous in before.items():
        candidate = after[package]

        changed = {
            key for key in set(previous) | set(candidate)
            if (
                (key in previous) != (key in candidate)
                or previous.get(key) != candidate.get(key)
            )
        }

        if not changed:
            continue

        if package not in target_packages:
            raise AdmissionError(f"lock.unapproved_package:{package}")

        forbidden = changed - SOURCE_FIELDS
        if forbidden:
            raise AdmissionError(
                f"lock.forbidden_fields:{package}:"
                + ",".join(sorted(forbidden))
            )

        changes[package] = sorted(changed)

    return changes
