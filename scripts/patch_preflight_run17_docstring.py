#!/usr/bin/env python3
"""
patch_preflight_run17_docstring.py  --  Monzia Moodie

Behavior-neutral docstring refresh for scripts/preflight_run17.py. The MODULE docstring still
describes the pre-Run-17 reality:
  * "<1000G Phase-3 AF parquet>"  -> the kg parquet is the 1000G 30x high-coverage GRCh38 callset,
    not Phase-3.
  * "the 81-column baseline" / "n_columns must be 81" / "regress the baseline 81 -> 78"
    -> the schema baseline is 87 columns (EXPECTED_SCHEMA_COLS = 87 in code; schema_baseline.json
    n_columns = 87). The 78 is the DEFAULT_MATRIX footgun target and is left intact.

No code path changes. Exact-string, idempotent, EOL/BOM-safe. Run from repo root.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("scripts/preflight_run17.py")

# (old, new) -- each old must appear exactly once when unpatched; skipped if already new.
EDITS = [
    ("<1000G Phase-3 AF parquet>", "<1000G 30x high-coverage GRCh38 AF parquet>"),
    ("the 81-column baseline", "the 87-column baseline"),
    ("n_columns must be 81)", "n_columns must be 87)"),
    ("regress the baseline 81 -> 78", "regress the baseline 87 -> 78"),
]


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr)
        return 2
    raw = TARGET.read_bytes()
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf else "\n"
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")

    applied, already = [], []
    for old, new in EDITS:
        if new in text and old not in text:
            already.append(new); continue
        c = text.count(old)
        if c == 0:
            print(f"ERROR: anchor not found and not already patched: {old!r}", file=sys.stderr)
            return 3
        if c != 1:
            print(f"ERROR: anchor {old!r} found {c}x (expected 1); not patching", file=sys.stderr)
            return 4
        text = text.replace(old, new); applied.append(new)

    if not applied:
        print("[skip] preflight_run17.py module docstring already current (81->87, high-coverage)")
        return 0

    out = text.replace("\n", eol)
    data = out.encode("utf-8")
    if had_bom:
        data = b"\xef\xbb\xbf" + data
    TARGET.write_bytes(data)
    print(f"[patched] preflight_run17.py docstring: {len(applied)} edit(s) "
          f"(eol={'CRLF' if eol != chr(10) else 'LF'}, bom={'yes' if had_bom else 'no'})")
    for a in applied:
        print(f"    -> {a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
