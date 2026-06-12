#!/usr/bin/env python3
"""patch_runbook_kan_and_offer.py -- make two concrete edits to
docs/launch/RUN16_RUNBOOK.md: (1) replace the vague imodelsx KAN placeholder in Sec.E
with the exact self-guarding sed patch (from Run-11/13/14, commit bf2f665); (2) set a
RAM-aware default offer in Sec.A. Count-guarded, idempotent, backup-first, CRLF/LF-
preserving, ASCII-only. Run from repo root. Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

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
