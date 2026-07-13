#!/usr/bin/env python
"""audit_split_protocol_v2.py (2026-07-11) -- confirm split_protocol_v2 is exactly as we left it,
BEFORE wiring it into real_data_prep + train.py (W2). Read-only except running its own tests.

Reports:
  1. The module's public API: every top-level def/class signature + any dataclass fields, so W2
     wires against the REAL current interface (not a remembered one).
  2. The exact RETURN shape of the main split function (what W2 must unpack + route).
  3. Runs tests/test_split_protocol_v2.py and reports pass/fail count.
  4. The canonical _gene_hash import it reuses (confirm the shared-hash contract is intact).
ASCII-safe.
"""
from __future__ import annotations
import ast
import io
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


def a(s): return s.encode("ascii", "replace").decode("ascii")
def line(c="-", n=78): print(c * n)


def dump_api(path: Path, label: str):
    if not path.exists():
        print(a(f"  ABSENT: {path}")); return
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except Exception as e:
        print(a(f"  parse error: {e}")); return
    print(a(f"--- {label}: {path.name} ({len(src.splitlines())} lines) ---"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            args = [ar.arg for ar in node.args.args]
            ret = ast.unparse(node.returns) if node.returns else ""
            print(a(f"  def {node.name}({', '.join(args)}){(' -> ' + ret) if ret else ''}"))
        elif isinstance(node, ast.ClassDef):
            bases = [ast.unparse(b) for b in node.bases]
            print(a(f"  class {node.name}({', '.join(bases)})"))
            # dataclass fields (AnnAssign) + methods
            for sub in node.body:
                if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    ann = ast.unparse(sub.annotation)
                    dflt = f" = {ast.unparse(sub.value)}" if sub.value is not None else ""
                    print(a(f"      {sub.target.id}: {ann}{dflt}"))
                elif isinstance(sub, ast.FunctionDef):
                    args = [ar.arg for ar in sub.args.args if ar.arg != "self"]
                    print(a(f"      def {sub.name}({', '.join(args)})"))
    # imports of _gene_hash / splits
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom,)):
            names = ", ".join(n.name for n in node.names)
            if "gene_hash" in names or "splits" in (node.module or "") or "_gene_hash" in names:
                print(a(f"  imports: from {node.module} import {names}"))
    line()


def main() -> int:
    print("=" * 78)
    print("SPLIT_PROTOCOL_V2 RE-AUDIT (confirm current API + tests before W2 wiring)")
    print("=" * 78)
    base = Path("src/genomic_variant_classifier/data")
    dump_api(base / "split_protocol_v2.py", "public API")

    # main function return shape: find the return annotation / a representative return stmt
    sp = base / "split_protocol_v2.py"
    if sp.exists():
        src = sp.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
        print("main split function(s) + return statements:")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and ("split" in node.name.lower()):
                rets = [ast.unparse(n.value) for n in ast.walk(node)
                        if isinstance(n, ast.Return) and n.value is not None]
                print(a(f"  {node.name}: returns {rets[:3]}"))
        line()

    # run the tests
    tpath = Path("tests/test_split_protocol_v2.py")
    if tpath.exists():
        print("running tests/test_split_protocol_v2.py ...")
        r = subprocess.run([sys.executable, "-m", "pytest", str(tpath), "-q"],
                           capture_output=True, text=True, timeout=300)
        tail = (r.stdout + r.stderr).strip().splitlines()[-4:]
        for ln in tail:
            print(a(f"  {ln}"))
        print(a(f"  pytest rc = {r.returncode}"))
    else:
        print("tests/test_split_protocol_v2.py ABSENT")
    line("=")
    print("SPLIT_PROTOCOL_V2 RE-AUDIT COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
