from __future__ import annotations

import hashlib
import json
from pathlib import Path


FEATURE_PATH = Path("data/processed/gtex/gtex_v11_gene_median_tpm_features.tsv")
MANIFEST_PATH = Path("data/processed/gtex/gtex_v11_gene_median_tpm_features.manifest.json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main() -> int:
    if not FEATURE_PATH.exists():
        raise SystemExit(f"ERROR: missing feature table: {FEATURE_PATH}")

    record = {
        "artifact": str(FEATURE_PATH).replace("\\", "/"),
        "source": "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_median_tpm.gct.gz",
        "builder": "scripts/build_gtex_median_tpm_features.py",
        "validator": "scripts/validate_gtex_median_tpm_features.py",
        "rows": 74628,
        "columns": 8,
        "bytes": FEATURE_PATH.stat().st_size,
        "sha256": sha256(FEATURE_PATH),
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"OK wrote: {MANIFEST_PATH}")
    print(f"sha256={record['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
