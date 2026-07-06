from __future__ import annotations
import io, sys
from pathlib import Path
BUILD = Path(sys.argv[1]); TESTS = Path(sys.argv[2]); WORK = Path(sys.argv[3])
def r(p):
    with io.open(p,"r",encoding="utf-8",newline="\n") as fh: return fh.read()
def w(p,s):
    with io.open(p,"w",encoding="utf-8",newline="\n") as fh: fh.write(s)
def blk(n): return (WORK/n).read_text(encoding="utf-8").rstrip("\n")
def guard(src,old,new,label):
    c=src.count(old)
    if c!=1: raise SystemExit("ABORT [%s]: expected 1 match, found %d. No changes written." % (label,c))
    return src.replace(old,new)
src=r(BUILD)
if "canonical_seq: str" in src and "_load_acc_sequences" in src:
    print("builder: canonical selection already present; skipping builder edits")
else:
    src=guard(src, blk("b1o"), blk("b1n"), "resolve-sig")
    src=guard(src, blk("b2o"), blk("b2n"), "record-pick")
    src=guard(src, blk("b3o"), blk("b3n"), "dl-sig")
    src=guard(src, blk("b4o"), blk("b4n"), "dl-call")
    src=guard(src, blk("b5o"), blk("b5n"), "helper")
    src=guard(src, blk("b6o"), blk("b6n"), "accmap")
    src=guard(src, blk("b7o"), blk("b7n"), "callsite")
    src=guard(src, blk("b8o"), blk("b8n"), "write-gate")
    if "data[0] if isinstance(data, list)" in src:
        raise SystemExit("ABORT: data[0] isoform-pick still present after patch.")
    w(BUILD, src)
    print("builder: 8 guarded edits applied")
tsrc=r(TESTS)
if "test_resolve_cif_url_selects_canonical_record" in tsrc:
    print("tests: canonical guards already present; skipping")
else:
    if not tsrc.endswith("\n"): tsrc += "\n"
    w(TESTS, tsrc + blk("btests"))
    print("tests: canonical guards appended")
