#!/usr/bin/env python3
"""GEN_FINNGEN_CHECKSUMS -- write data/external/finngen/CHECKSUMS.sha256 from the
PROBE-VERIFIED 2026-06-28 hashes (scripts/probe_finngen_sizes.py full pass).

Format: GNU coreutils `sha256sum` compatible -- "<hex>  <basename>\\n" (two spaces),
LF line endings, ASCII. Verify later from the finngen dir with:
    sha256sum -c CHECKSUMS.sha256            (Linux/git-bash)
  or in PowerShell:
    Get-FileHash finnge_R12_annotated_variants_v1.gz -Algorithm SHA256

Idempotent: rewrites the canonical content each run. Refuses to overwrite a file whose
content differs UNLESS --force (so a hash drift is caught, not silently clobbered).

Usage: python gen_finngen_checksums.py <out_path> [--force]
"""
from __future__ import annotations
import sys
from pathlib import Path

# PROBE-VERIFIED 2026-06-28 (do not edit without re-running the probe)
ENTRIES = [
    ("e27f91568ca7f8842528c45262f533442e2c23016221e882f2a547fd7cb99231",
     "finnge_R12_annotated_variants_v1.gz"),
    ("109b4f3f13ae8c4ade148cf47402ba23e442cd4a6648ab31bdb9d6518bca99c1",
     "finngen_R13_annotated_variants_v0.gz"),
]
CANON = "".join(f"{h}  {name}\n" for h, name in ENTRIES)


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--force"]
    force = "--force" in sys.argv
    if len(args) != 1:
        print("usage: python gen_finngen_checksums.py <out_path> [--force]"); return 2
    out = Path(args[0])
    if out.exists():
        cur = out.read_bytes().decode("utf-8", errors="replace")
        if cur == CANON:
            print(f"UNCHANGED (already canonical): {out}"); return 0
        if not force:
            print(f"REFUSE: {out} exists with DIFFERENT content (hash drift?). "
                  f"Re-run probe to confirm, then use --force.\n--- existing ---\n{cur}")
            return 1
    out.write_text(CANON, encoding="utf-8", newline="\n")
    # verify ASCII + format
    data = out.read_bytes()
    assert all(b <= 0x7f for b in data), "non-ASCII in checksums file"
    for line in CANON.strip().split("\n"):
        h, _, name = line.partition("  ")
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h), f"bad hash: {line}"
        assert name, f"missing filename: {line}"
    print(f"WROTE: {out}\n{CANON}", end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
