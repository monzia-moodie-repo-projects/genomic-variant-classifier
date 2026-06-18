#!/usr/bin/env python3
"""
patch_preflight_schema_cols_87.py  --  Monzia Moodie

Bump the Run-17 preflight schema-count guard 82 -> 87 after the baseline gained the 5
rnaseq_* columns (extend_schema_baseline_rnaseq.py). Updates the preflight constant +
docstring and the test fixtures/assertions that encode 82. Per-edit occurrence counts are
asserted (a wrong count ABORTS rather than mis-editing). EOL-agnostic, idempotent,
py_compile-checked.
"""
from __future__ import annotations
import py_compile, sys, tempfile
from pathlib import Path

# (old, new, expected_count)
EDITS = {
    "scripts/preflight_run17.py": [
        ("EXPECTED_SCHEMA_COLS = 82", "EXPECTED_SCHEMA_COLS = 87", 1),
        ('"""The 82-col baseline must be intact (guards the build_schema_baseline DEFAULT_MATRIX footgun)."""',
         '"""The 87-col baseline must be intact (82 base + 5 rnaseq_*; guards the build_schema_baseline DEFAULT_MATRIX footgun)."""', 1),
    ],
    "tests/unit/test_preflight_run17.py": [
        ("def test_schema_gate_82_ok(tmp_path):", "def test_schema_gate_87_ok(tmp_path):", 1),
        ("_baseline(b, 82)", "_baseline(b, 87)", 3),
        ('"n_columns=82"', '"n_columns=87"', 1),
    ],
}


def _eol(raw: bytes) -> bytes:
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    return b"\r\n" if crlf > lf else b"\n"


def patch_file(path: Path, edits) -> bool:
    raw = path.read_bytes(); eol = _eol(raw)
    text = raw.replace(b"\r\n", b"\n").decode("utf-8")
    changed = False
    for old, new, exp in edits:
        if new in text and old not in text:
            print(f"  [skip] already patched: {old[:42]!r}"); continue
        c = text.count(old)
        if c != exp:
            print(f"  [ABORT] {path.name}: {old[:46]!r} found {c}x (need {exp})"); return False
        text = text.replace(old, new); changed = True
        print(f"  [ok] x{c} {old[:42]!r} -> {new[:42]!r}")
    if not changed:
        return True
    out = text.encode("utf-8").replace(b"\n", eol)
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".py") as tf:
        tf.write(out); tmp = tf.name
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"  [ABORT] py_compile failed: {e}"); return False
    path.write_bytes(out)
    print(f"  [written] {path}  (eol={'CRLF' if eol == b'\\r\\n' else 'LF'})")
    return True


def main(root="."):
    root = Path(root); ok = True
    for rel, edits in EDITS.items():
        p = root / rel
        print(f"== {rel} ==")
        if not p.exists():
            print(f"  [ABORT] not found: {p}"); ok = False; continue
        ok = patch_file(p, edits) and ok
    print("\n" + ("[PASS] preflight + tests bumped 82 -> 87" if ok else "[FAIL] see aborts"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
