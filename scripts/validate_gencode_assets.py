from pathlib import Path
import gzip
import sys

base = Path(r"C:\Projects\genomic-variant-classifier\data\external\gencode")

files = [
    "gencode.v50.annotation.gtf.gz",
    "gencode.v50.annotation.gff3.gz",
    "gencode.v50.transcripts.fa.gz",
    "gencode.v50.pc_transcripts.fa.gz",
    "gencode.v50.lncRNA_transcripts.fa.gz",
]

failures = []

for name in files:
    path = base / name
    print(f"Checking {path}")

    if not path.exists():
        failures.append(f"MISSING: {path}")
        continue

    if path.stat().st_size <= 0:
        failures.append(f"ZERO_BYTES: {path}")
        continue

    try:
        with gzip.open(path, "rb") as handle:
            total = 0
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)

        if total <= 0:
            failures.append(f"NO_READABLE_CONTENT: {path}")
        else:
            print(f"OK: {name} uncompressed_read_bytes={total}")

    except Exception as exc:
        failures.append(f"GZIP_ERROR: {path}: {exc!r}")

if failures:
    print("\nVALIDATION FAILED")
    for failure in failures:
        print(failure)
    sys.exit(2)

print("\nGENCODE validation passed.")
