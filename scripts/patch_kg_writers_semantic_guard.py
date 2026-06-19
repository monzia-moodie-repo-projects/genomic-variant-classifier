#!/usr/bin/env python3
"""
patch_kg_writers_semantic_guard.py  --  Monzia Moodie

Wire scripts/kg_semantic_hash.py into the 1000G parquet writers:
  * scripts/merge_1kg_parquets.py -- REAL skip-guard: if the output already exists and is semantically
    identical (sorted key + AF columns; parquet container bytes ignored), skip the rewrite and print
    "1KGP AF semantic hash unchanged; not rewriting parquet". Prevents redundant ~6 MB re-commits
    (cf. 26342e9 -> 988439c). Atomic tmp->replace path preserved for the changed case.
  * scripts/build_1kg_parquet.py -- INFORMATIONAL provenance log of the semantic hash after the
    streamed write (no behavior change; the streaming ParquetWriter can't pre-compare without buffering,
    and re-running build re-streams the VCFs regardless, so the meaningful skip lives in merge).

Requires scripts/kg_semantic_hash.py present. Idempotent, exact-anchor, EOL/BOM-safe. Run from repo root.
"""
from __future__ import annotations
import sys
from pathlib import Path

EDITS = {
    "scripts/merge_1kg_parquets.py": [
        ("skip if already imported", "from kg_semantic_hash import semantic_hash, KGSchemaError",
         "import pandas as pd\n\nSUPERPOP =",
         "import pandas as pd\n\nsys.path.insert(0, str(Path(__file__).resolve().parent))\n"
         "from kg_semantic_hash import semantic_hash, KGSchemaError\n\nSUPERPOP ="),
        ("skip if guard wired", 'semantic hash unchanged; not rewriting',
         '    out = Path(args.out)\n    out.parent.mkdir(parents=True, exist_ok=True)\n'
         '    tmp = out.with_suffix(out.suffix + ".tmp")\n',
         '    out = Path(args.out)\n    out.parent.mkdir(parents=True, exist_ok=True)\n'
         '    if out.exists():\n        try:\n'
         '            if semantic_hash(out) == semantic_hash(merged):\n'
         '                print("1KGP AF semantic hash unchanged; not rewriting parquet")\n'
         '                return 0\n'
         '        except KGSchemaError as e:\n'
         '            print(f"[kg-guard] existing parquet not comparable ({e}); rewriting", file=sys.stderr)\n'
         '    tmp = out.with_suffix(out.suffix + ".tmp")\n'),
    ],
    "scripts/build_1kg_parquet.py": [
        ("skip if hash log wired", "1KGP AF semantic hash:",
         '    logger.info("Wrote %d variants -> %s", total, out_path)\n'
         '    logger.info("Non-zero super-pop AF counts: %s", nonzero)\n',
         '    logger.info("Wrote %d variants -> %s", total, out_path)\n'
         '    try:\n'
         '        import sys as _sys\n'
         '        _sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
         '        from kg_semantic_hash import semantic_hash as _sh\n'
         '        logger.info("1KGP AF semantic hash: %s", _sh(out_path))\n'
         '    except Exception as e:  # noqa: BLE001\n'
         '        logger.warning("kg semantic hash log skipped: %s", e)\n'
         '    logger.info("Non-zero super-pop AF counts: %s", nonzero)\n'),
    ],
}


def patch_file(rel: str, edits) -> int:
    p = Path(rel)
    if not p.exists():
        print(f"ERROR: {rel} not found (run from repo root)", file=sys.stderr); return 2
    raw = p.read_bytes()
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf else "\n"
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    changed = False
    for label, marker, old, new in edits:
        if marker in text:
            print(f"[skip] {rel}: {label}"); continue
        c = text.count(old)
        if c != 1:
            print(f"ERROR: {rel}: anchor for '{label}' found {c}x (expected 1); aborting", file=sys.stderr)
            return 3
        text = text.replace(old, new); changed = True
        print(f"[patched] {rel}: {label}")
    if changed:
        data = text.replace("\n", eol).encode("utf-8")
        if had_bom:
            data = b"\xef\xbb\xbf" + data
        p.write_bytes(data)
        print(f"[ok] wrote {rel} (eol={'CRLF' if eol != chr(10) else 'LF'})")
    return 0


def main() -> int:
    if not Path("scripts/kg_semantic_hash.py").exists():
        print("ERROR: scripts/kg_semantic_hash.py missing -- place it before wiring", file=sys.stderr)
        return 2
    rc = 0
    for rel, edits in EDITS.items():
        rc |= patch_file(rel, edits)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
