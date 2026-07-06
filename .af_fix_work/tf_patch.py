from __future__ import annotations
import io, sys
from pathlib import Path
TESTS = Path(sys.argv[1]); WORK = Path(sys.argv[2])
def r(p):
    with io.open(p,"r",encoding="utf-8",newline="\n") as fh: return fh.read()
def w(p,s):
    with io.open(p,"w",encoding="utf-8",newline="\n") as fh: fh.write(s)
def blk(n): return (WORK/n).read_text(encoding="utf-8").rstrip("\n")
def guard(src,old,new,label):
    c=src.count(old)
    if c!=1: raise SystemExit("ABORT [%s]: expected 1 match, found %d. No changes written." % (label,c))
    return src.replace(old,new)
src=r(TESTS)
if '_download_cif("P04637", tmp_path, _CANON)' in src:
    print("test-fix already applied; skipping"); sys.exit(0)
src=guard(src, blk("tf_t5o"), blk("tf_t5n"), "canon-const")
src=guard(src, blk("tf_t1o"), blk("tf_t1n"), "caller-1")
src=guard(src, blk("tf_t2o"), blk("tf_t2n"), "caller-2")
src=guard(src, blk("tf_t3o"), blk("tf_t3n"), "caller-3")
src=guard(src, blk("tf_t4o"), blk("tf_t4n"), "caller-4")
w(TESTS, src)
print("test-fix: 5 guarded edits applied")
