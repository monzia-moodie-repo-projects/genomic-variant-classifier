#!/usr/bin/env python3
"""patch_stage_skip_present.py -- make stage_run16.py's manifest upload idempotent: stat the
remote file first and skip the scp when a byte-identical copy is already present. Turns a
re-stage (after a timeout/abort) into a cheap resume instead of re-pushing ~1.76 GB.
Count-guarded, idempotent, backup-first, CRLF/LF-preserving, ASCII-only. Run from repo root.
Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("scripts/stage_run16.py")
MARKER = "idempotent resume: skip scp"

ANCHOR = (
    "        rc, out, err = t.scp(str(local), rp)\n"
    "        if not t.dry and rc != 0:\n"
)
REPLACEMENT = (
    "        lsize = local.stat().st_size\n"
    "        # idempotent resume: skip scp if the box already holds a byte-identical copy\n"
    "        if not t.dry:\n"
    "            _, rpre, _ = t.ssh(f\"stat -c %s {rp} 2>/dev/null || echo MISSING\")\n"
    "            if rpre.isdigit() and int(rpre) == lsize:\n"
    "                print(f\"  [skip] {rel}  ({lsize/1048576:.1f} MB already present)\")\n"
    "                continue\n"
    "        rc, out, err = t.scp(str(local), rp)\n"
    "        if not t.dry and rc != 0:\n"
)
# the later verify block recomputes lsize; that line becomes redundant but harmless. Remove it
# to avoid a duplicate-assignment lint by collapsing 'lsize = local.stat().st_size' there.
VERIFY_OLD = (
    "            rc2, rout, _ = t.ssh(f\"stat -c %s {rp} 2>/dev/null || echo MISSING\")\n"
    "            lsize = local.stat().st_size\n"
)
VERIFY_NEW = (
    "            rc2, rout, _ = t.ssh(f\"stat -c %s {rp} 2>/dev/null || echo MISSING\")\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found (run from repo root).")
        return 1
    raw = TARGET.open("r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")

    if MARKER in text:
        print("Already patched (skip-if-present present). No-op.")
        return 0
    if text.count(ANCHOR) != 1:
        print(f"ABORT: upload anchor found {text.count(ANCHOR)} times (expected 1).")
        return 1
    if text.count(VERIFY_OLD) != 1:
        print(f"ABORT: verify anchor found {text.count(VERIFY_OLD)} times (expected 1).")
        return 1

    text = text.replace(ANCHOR, REPLACEMENT)
    text = text.replace(VERIFY_OLD, VERIFY_NEW)

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_skip.bak")
    backup.write_bytes(TARGET.read_bytes())
    TARGET.open("w", encoding="utf-8", newline="").write(text.replace("\n", nl))
    print(f"OK: stage_run16.py now skips byte-present files; backup {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
