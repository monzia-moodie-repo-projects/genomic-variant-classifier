from __future__ import annotations
import io, sys
from pathlib import Path

FEAT = Path(sys.argv[1]); BUILD = Path(sys.argv[2]); TESTS = Path(sys.argv[3]); WORK = Path(sys.argv[4])

def r(p):
    with io.open(p, "r", encoding="utf-8", newline="\n") as fh: return fh.read()
def w(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh: fh.write(s)
def blk(name): return (WORK / name).read_text(encoding="utf-8")
def rstrip_nl(s): return s.rstrip("\n")

def guard(src, old, new, label):
    c = src.count(old)
    if c != 1:
        raise SystemExit("ABORT [%s]: expected 1 match, found %d. No changes written." % (label, c))
    return src.replace(old, new)

fsrc = r(FEAT)
if "import numpy as np" in fsrc and "from scipy.spatial import cKDTree" in fsrc:
    print("features: numpy/scipy already imported; skipping")
else:
    fsrc = guard(fsrc, rstrip_nl(blk("rv_imports_old.txt")), rstrip_nl(blk("rv_imports_new.txt")), "imports")
    print("features: imports inserted")
if "tree = cKDTree(coords)" in fsrc:
    print("features: RSA already vectorized; skipping")
else:
    fsrc = guard(fsrc, rstrip_nl(blk("rv_rsa_old.txt")), rstrip_nl(blk("rv_rsa_new.txt")), "rsa-loop")
    print("features: RSA loop vectorized")
w(FEAT, fsrc)

bsrc = r(BUILD)
if "for _i_gene, (gene, acc) in enumerate(acc_map.items()" in bsrc:
    print("builder: progress log already present; skipping")
else:
    bsrc = guard(bsrc, rstrip_nl(blk("rv_loop_old.txt")), rstrip_nl(blk("rv_loop_new.txt")), "progress-log")
    w(BUILD, bsrc)
    print("builder: per-gene progress logging added")

tsrc = r(TESTS)
if "test_rsa_vectorized_matches_naive_reference" in tsrc:
    print("tests: RSA guards already present; skipping")
else:
    if not tsrc.endswith("\n"): tsrc += "\n"
    w(TESTS, tsrc + blk("rv_tests.txt"))
    print("tests: RSA guards appended")