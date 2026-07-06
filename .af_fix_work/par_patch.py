import io, sys
from pathlib import Path
B=Path(sys.argv[1]); W=Path(sys.argv[2])
def r(p):
    with io.open(p,"r",encoding="utf-8",newline="\n") as fh: return fh.read()
def w(p,s):
    with io.open(p,"w",encoding="utf-8",newline="\n") as fh: fh.write(s)
def blk(n): return (W/n).read_text(encoding="utf-8").rstrip("\n")
def g(src,old,new,l):
    c=src.count(old)
    if c!=1: raise SystemExit("ABORT [%s]: expected 1 match, found %d. No changes written." % (l,c))
    return src.replace(old,new)
s=r(B)
if "_fetch_results" in s:
    print("already parallel; skipping"); sys.exit(0)
s=g(s, blk("pb1o"), blk("pb1n"), "atomic-cif")
s=g(s, blk("pb3o"), blk("pb3n"), "argparse-workers")
s=g(s, blk("pb4o"), blk("pb4n"), "phase1-insert")
s=g(s, blk("pb5o"), blk("pb5n"), "rewire-download")
s=g(s, blk("pb6o"), blk("pb6n"), "rewire-sites")
s=g(s, blk("pb7o"), blk("pb7n"), "determinism-sort")
w(B,s)
print("6 guarded edits applied")
