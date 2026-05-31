#!/usr/bin/env python3
"""Pre-flight gate: STRING DB presence for the GNN modality. Exit 1 if GNN cannot run."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
LINKS_GLOB="*protein.links.detailed*.txt.gz"; INFO_GLOB="*protein.info*.txt.gz"
MIN_LINKS=10_000_000; MIN_INFO=100_000
def _one(d,p):
    h=sorted(d.glob(p)); return h[0] if h else None
def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--string-dir",default="data/external/string")
    a=ap.parse_args(); d=Path(a.string_dir)
    if not d.is_dir(): print(f"PREFLIGHT-GNN FAIL: dir not found {d}"); return 1
    ok=True; links=_one(d,LINKS_GLOB); info=_one(d,INFO_GLOB)
    if links is None: print(f"PREFLIGHT-GNN FAIL: no links file ({LINKS_GLOB})"); ok=False
    else:
        s=links.stat().st_size; g=s>=MIN_LINKS; print(f"  links {links.name} {s:,}B [{'OK' if g else 'FAIL small'}]"); ok=ok and g
    if info is None: print(f"PREFLIGHT-GNN FAIL: no info file ({INFO_GLOB})"); ok=False
    else:
        s=info.stat().st_size; g=s>=MIN_INFO; print(f"  info  {info.name} {s:,}B [{'OK' if g else 'FAIL small'}]"); ok=ok and g
    print("PREFLIGHT-GNN " + ("PASS" if ok else "FAIL")); return 0 if ok else 1
if __name__=="__main__": sys.exit(main())
