#!/usr/bin/env python3
"""patch_test_esm2_llr_ci_offline.py -- make the ESM-2 LLR end-to-end test CI-safe.

test_llr_long_protein_scores_finite_without_oom calls conn.annotate_llr(df), which
fetches the 8M weights from HuggingFace Hub on first use. CI runners have no local
cache and get rate-limited (429 -> OSError), so the test flakes red even on
docs-only commits while passing locally (cached weights). This wraps the single
live-load call in try/except OSError -> pytest.skip, leaving the test fully active
wherever the model loads (local, and Vast.ai where 650M is cached). The windowing
index math stays covered offline by test_windowed_logit_row_reads_correct_residue.

Idempotent, py_compile-gated, newline-preserving, ASCII. Author: Monzia Moodie."""
from __future__ import annotations

import py_compile
import shutil
import sys
from pathlib import Path

TARGET = Path("tests/unit/test_esm2_llr_windowing.py")
MARKER = "ESM-2 8M not loadable offline"

ANCHOR = "    out = conn.annotate_llr(df)\n"
REPLACEMENT = (
    "    try:\n"
    "        out = conn.annotate_llr(df)\n"
    "    except OSError as exc:\n"
    "        # The 8M weights are fetched from HF Hub on first use; CI runners have\n"
    "        # no local cache and get rate-limited (429 -> OSError). Skip on that\n"
    "        # network condition rather than red the suite -- the windowing index\n"
    "        # math is fully covered by test_windowed_logit_row_reads_correct_residue\n"
    "        # (which mocks the model and needs no download).\n"
    "        pytest.skip(f\"ESM-2 8M not loadable offline (HF Hub unavailable): {exc}\")\n"
)


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root).")
        return 2
    with TARGET.open("r", encoding="utf-8", newline="") as f:
        raw = f.read()
    if MARKER in raw:
        print("already patched (marker present); no change.")
        return 0
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if text.count(ANCHOR) != 1:
        print(f"ERROR: expected exactly 1 occurrence of the anchor, found {text.count(ANCHOR)}.")
        return 2
    text = text.replace(ANCHOR, REPLACEMENT, 1)

    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    with TARGET.open("w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", nl))

    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET)
        print(f"ERROR: py_compile failed, reverted: {exc}")
        return 2
    print(f"patched {TARGET} (backup at {bak.name}); py_compile OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
