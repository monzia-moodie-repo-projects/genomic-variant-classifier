#!/usr/bin/env python3
"""
patch_build_1kg_atomic_write.py  --  Monzia Moodie

Make scripts/build_1kg_parquet.py's build() write ATOMICALLY: stream to <out>.tmp, then os.replace into
place only after the coverage gates pass -- so a mid-stream crash/interrupt can never leave a corrupt
out_path (the prior good file stays intact). Folds in the same semantic-hash skip-guard as
merge_1kg_parquets.py (skip the publish if an existing out_path is semantically identical) and cleans up
the tmp on every failure path. Span-replaces the whole build() function (def build -> just before
def _sources); idempotent, EOL/BOM-safe. Run from repo root.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("scripts/build_1kg_parquet.py")
START = "def build(sources: list, out_path: str, cohort_keys=None, chunk_size: int = 2_000_000) -> None:"
END = "\n\ndef _sources(args) -> list:"

NEW_BUILD = '''def build(sources: list, out_path: str, cohort_keys=None, chunk_size: int = 2_000_000) -> None:
    cols = ["variant_id", "allele_freq", *_POP_OUT]
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path + ".tmp"  # atomic: stream to tmp, os.replace into place only on success
    writer = None
    total = 0
    nonzero = {c: 0 for c in _POP_OUT}
    buf: list = []

    def flush():
        nonlocal writer, total, buf
        if not buf:
            return
        df = pd.DataFrame.from_records(buf)
        df = df.dropna(subset=["variant_id"]).drop_duplicates(subset=["variant_id"])
        df = df[cols].astype({c: "float64" for c in cols[1:]})
        for c in _POP_OUT:
            nonzero[c] += int((df[c] > 0).sum())
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp_path, table.schema)
        writer.write_table(table)
        total += len(df)
        buf = []

    def _discard_tmp():
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

    try:
        try:
            for src in sources:
                for line in _iter_lines(src):
                    if line.startswith("#"):
                        continue
                    buf.extend(rows_from_vcf_line(line, cohort_keys))
                    if len(buf) >= chunk_size:
                        flush()
                flush()
                logger.info("  %s -> running total %d (kept)", os.path.basename(src.rstrip('/')), total)
        finally:
            if writer is not None:
                writer.close()
    except BaseException:
        _discard_tmp()  # never leave a partial tmp on crash/interrupt; prior out_path stays intact
        raise

    if total == 0:
        _discard_tmp()
        raise SystemExit("No records written (cohort filter matched nothing, or no parseable lines).")
    if all(nonzero[c] == 0 for c in _POP_OUT):
        _discard_tmp()
        raise SystemExit(
            "COVERAGE GATE FAILED: every super-pop AF column is all-zero -- the INFO field names did not "
            "match any candidate. Inspect the VCF header (inspect_1kg_header.py) and extend _POP_CANDIDATES."
        )

    # Atomic publish + semantic-hash skip-guard (mirrors merge_1kg_parquets.py).
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kg_semantic_hash import semantic_hash as _sh
        if os.path.exists(out_path) and _sh(tmp_path) == _sh(out_path):
            _discard_tmp()
            logger.info("1KGP AF semantic hash unchanged; not rewriting parquet")
            logger.info("Non-zero super-pop AF counts: %s", nonzero)
            return
        os.replace(tmp_path, out_path)  # atomic publish
        logger.info("Wrote %d variants -> %s", total, out_path)
        logger.info("1KGP AF semantic hash: %s", _sh(out_path))
    except Exception as e:  # noqa: BLE001
        # hash/import failure must neither strand tmp nor lose the build
        if os.path.exists(tmp_path):
            os.replace(tmp_path, out_path)
            logger.info("Wrote %d variants -> %s", total, out_path)
        logger.warning("kg semantic hash step skipped: %s", e)
    logger.info("Non-zero super-pop AF counts: %s", nonzero)
'''


def main() -> int:
    if not TARGET.exists():
        print(f"ERROR: {TARGET} not found (run from repo root)", file=sys.stderr); return 2
    raw = TARGET.read_bytes()
    crlf = raw.count(b"\r\n"); lf = raw.count(b"\n") - crlf
    eol = "\r\n" if crlf >= lf else "\n"
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    if "tmp_path = out_path" in text:
        print("[skip] build() already writes atomically"); return 0
    i = text.find(START)
    if i < 0:
        print("ERROR: build() signature not found", file=sys.stderr); return 3
    j = text.find(END, i)
    if j < 0:
        print("ERROR: end marker (def _sources) not found after build()", file=sys.stderr); return 4
    new_text = text[:i] + NEW_BUILD.rstrip("\n") + text[j:]
    data = new_text.replace("\n", eol).encode("utf-8")
    if had_bom:
        data = b"\xef\xbb\xbf" + data
    TARGET.write_bytes(data)
    print(f"[patched] build() now writes atomically (tmp->replace) + semantic skip "
          f"(eol={'CRLF' if eol != chr(10) else 'LF'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
