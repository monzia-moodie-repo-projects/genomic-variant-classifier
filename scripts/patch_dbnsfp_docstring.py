#!/usr/bin/env python3
"""patch_dbnsfp_docstring.py -- fix the dbNSFP doc-code drift: the docstring says the
cache filename is dbnsfp_full_index.parquet, but _cache_path() hard-codes
dbnsfp_clinvar_index.parquet. This corrects the docstring to match the code.

Count-guarded (the wrong token must appear exactly once), idempotent (no-op if already
fixed), backup-first, CRLF/LF-preserving, ASCII-only. Author: Monzia Moodie."""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/dbnsfp.py")
WRONG = "dbnsfp_full_index.parquet"
RIGHT = "dbnsfp_clinvar_index.parquet"


def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found (run from repo root).")
        return 1
    raw = TARGET.open("r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")

    n_wrong = text.count(WRONG)
    # _cache_path legitimately contains RIGHT once (line ~387); the docstring adds the
    # only WRONG occurrence. After fix, WRONG=0 and RIGHT=2.
    if n_wrong == 0 and text.count(RIGHT) >= 2:
        print("Already fixed (no dbnsfp_full_index.parquet in docstring). No-op.")
        return 0
    if n_wrong != 1:
        print(f"ABORT: expected exactly 1 '{WRONG}' (the docstring), found {n_wrong}. "
              "Manual review required.")
        return 1

    backup = TARGET.with_suffix(TARGET.suffix + ".pre_docstringfix.bak")
    backup.write_bytes(TARGET.read_bytes())
    new = text.replace(WRONG, RIGHT)
    TARGET.open("w", encoding="utf-8", newline="").write(new.replace("\n", nl))
    print(f"OK: docstring fixed ({WRONG} -> {RIGHT}); backup {backup.name}")
    print(f"  occurrences now: WRONG={new.count(WRONG)}  RIGHT={new.count(RIGHT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
