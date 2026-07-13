#!/usr/bin/env python3
"""patch_runbook_kan_and_offer.py -- SPENT AND SUPERSEDED. DO NOT RUN. (2026-07-13)

ORIGINAL PURPOSE
    Make two edits to docs/launch/RUN16_RUNBOOK.md: (1) replace the vague imodelsx KAN
    placeholder in Section E with the exact self-guarding `sed` patch (from Run 11/13/14,
    commit bf2f665); (2) set a RAM-aware default offer in Section A. It did its job.

WHY IT IS NEUTRALISED
    Its KAN_BLOCK payload injects, into the runbook, an instruction to `sed -i` the INSTALLED
    imodelsx package inside site-packages:

        sed -i 's/test_size=test_size/test_size=self.test_size/g' "$IMODELSX_KAN"

    That approach was REMOVED on 2026-07-13 and must not come back. The `sed` ran only in the
    Run 11 / Run 16 launch scripts and on the developer's laptop -- never in Continuous
    Integration, never in Docker, and never in scripts/vm_bootstrap_run.sh (the Run 17 path).
    Consequently the Kolmogorov-Arnold Network raised NameError in every Continuous
    Integration run, the ensemble's bare `except Exception` swallowed it, and a TWELVE-model
    ensemble trained and reported as healthy. It also left the developer's virtual environment
    holding a library no clean machine had.

    The repair is now IN-PROCESS: models/kan.py::_repair_imodelsx_kan_bare_names(), applied
    identically in every environment, covered by tests/unit/test_kan_actually_fits.py.

    Re-running this script would re-inject the sed instruction into the runbook and re-create
    the exact environment divergence that hid the defect for two months. It therefore refuses
    to run.

Author: Monzia Moodie.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.exit(
    "patch_runbook_kan_and_offer.py is SUPERSEDED and will not run (2026-07-13).\n"
    "It injects a `sed -i` of the installed imodelsx package into RUN16_RUNBOOK.md.\n"
    "That mechanism was removed: the KAN repair is now in-process in models/kan.py\n"
    "(_repair_imodelsx_kan_bare_names), applied in EVERY environment. Re-running this\n"
    "would re-create the environment divergence that silently dropped KAN from every\n"
    "Continuous Integration run for two months. See\n"
    "docs/status/REMEDIATION_2026-07-13_warnings-and-silent-model-drop.md."
)

TARGET = Path("docs/launch/RUN16_RUNBOOK.md")

KAN_ANCHOR = (
    "# Apply your standing imodelsx v1.0.13 KAN workaround here if it is not already in\n"
    "# requirements (else the kan base estimator errors at fit()).\n"
)
KAN_BLOCK = (
    "# imodelsx v1.0.13 KAN bug fix (bare-name refs in KANClassifier.fit). The kan.py\n"
    "# attribute fix is already in the repo; this patches the INSTALLED package file.\n"
    "# Self-guarding (only patches if the bug is present) and idempotent.\n"
    "IMODELSX_KAN=$(python -c \"import imodelsx.kan.kan_sklearn as m; print(m.__file__)\" 2>/dev/null)\n"
    "if [ -n \"$IMODELSX_KAN\" ] && grep -q \"test_size=test_size\" \"$IMODELSX_KAN\"; then\n"
    "  sed -i 's/test_size=test_size/test_size=self.test_size/g' \"$IMODELSX_KAN\"\n"
    "  sed -i 's/random_state=random_state/random_state=self.random_state/g' \"$IMODELSX_KAN\"\n"
    "  sed -i 's/shuffle=shuffle/shuffle=self.shuffle/g' \"$IMODELSX_KAN\"\n"
    "  echo \"imodelsx_patch: fixed 3 bare-name refs in $IMODELSX_KAN\"\n"
    "else\n"
    "  echo \"imodelsx_patch: already patched or not installed\"\n"
    "fi\n"
)

OFFER_ANCHOR = ('$OfferId = "37194516"   # the ID column from YOUR fresh search '
                "(this is just an example)\n")
OFFER_NEW = ('$OfferId = "38381901"   # EXAMPLE ONLY -- offers expire; pick a FRESH '
             "verified offer with >= 64 GB RAM (prefer ~128) to avoid data-prep OOM\n")

MARKER = "imodelsx_patch: fixed 3 bare-name refs"


def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found (run from repo root).")
        return 1
    raw = TARGET.open("r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")

    if MARKER in text:
        print("Already patched (KAN sed block present). No-op.")
        return 0

    changed = 0
    if text.count(KAN_ANCHOR) == 1:
        text = text.replace(KAN_ANCHOR, KAN_BLOCK)
        changed += 1
        print("  KAN: replaced placeholder with concrete sed block")
    else:
        print(f"  KAN: ABORT -- anchor found {text.count(KAN_ANCHOR)} times (expected 1)")
        return 1

    if text.count(OFFER_ANCHOR) == 1:
        text = text.replace(OFFER_ANCHOR, OFFER_NEW)
        changed += 1
        print("  OFFER: set RAM-aware default example")
    else:
        print(f"  OFFER: WARN -- anchor found {text.count(OFFER_ANCHOR)} times; left as-is")

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_kanpatch.bak")
    backup.write_bytes(TARGET.read_bytes())
    TARGET.open("w", encoding="utf-8", newline="").write(text.replace("\n", nl))
    print(f"OK: {changed} edit(s); backup {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
