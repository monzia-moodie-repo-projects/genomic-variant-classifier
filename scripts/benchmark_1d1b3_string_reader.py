"""Stage 1D-1b.3: benchmark of the exact-pair STRING reader (graph_identity.read_string_pairs) on a real links file.

Owner ruling f2ec92cd, section III: "benchmark scratch-space requirements, peak memory, and runtime before making it a
production requirement." Each configuration (partitions x records per chunk) runs in its OWN child process, so peak
resident memory is that configuration's alone; the reader's semantic-closure digest is recorded before and after, and the
pair table's content digest must be identical in every configuration (the result may not depend on the layout).

    python scripts/benchmark_1d1b3_string_reader.py --links L --sha256 PIN --scratch DIR --out JSON

Peak memory comes from ``resource`` (Linux, macOS); elsewhere it is null. Author: Monzia Moodie.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from genomic_variant_classifier.information_sources import census as C  # noqa: E402

READER = "src/genomic_variant_classifier/information_sources/graph_identity.py::read_string_pairs"
CONFIGS = ((16, 1 << 18), (64, 1 << 18), (256, 1 << 18), (64, 1 << 16), (64, 1 << 20))
CHILD = r"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from genomic_variant_classifier.information_sources import graph_identity as G
t0 = time.perf_counter()
t = G.read_string_pairs(Path(sys.argv[2]), expected_sha256=sys.argv[3], scratch_dir=Path(sys.argv[4]),
                        partitions=int(sys.argv[5]), chunk_records=int(sys.argv[6]))
seconds = time.perf_counter() - t0
try:
    import resource
    peak = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
except ImportError:
    peak = None
print(json.dumps({"partitions": t.partitions, "chunk_records": int(sys.argv[6]), "pairs": len(t), "records": t.n_records,
                  "proteins": len(t.proteins), "content_sha256": t.content_sha256, "source_sha256": t.source_sha256,
                  "max_partition_records": t.max_partition_records, "scratch_bytes": t.scratch_bytes,
                  "seconds": round(seconds, 1), "peak_rss_mib": peak}))
"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--links", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    before = C.symbol_closure_digest(REPO, READER)
    runs = []
    for parts, chunk in CONFIGS:
        out = subprocess.run([sys.executable, "-I", "-c", CHILD, str(REPO / "src"), str(a.links), a.sha256,
                              str(a.scratch), str(parts), str(chunk)], capture_output=True, text=True, check=False,
                             env={**os.environ, "PYTHONHASHSEED": "0"})
        if out.returncode != 0:
            print(out.stderr, file=sys.stderr)
            raise SystemExit(f"configuration {parts} x {chunk} failed")
        runs.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(json.dumps(runs[-1]), flush=True)
    if C.symbol_closure_digest(REPO, READER) != before:
        raise SystemExit("the reader changed during the benchmark")
    if len({r["content_sha256"] for r in runs}) != 1 or len({r["pairs"] for r in runs}) != 1:
        raise SystemExit("the pair table depends on the partition layout")
    rec = {"schema": "gvc.string-reader-benchmark.1d1b3/1", "author": "Monzia Moodie", "reader": READER,
           "reader_sha256": before, "links": {"file": a.links.name, "sha256": a.sha256, "bytes": a.links.stat().st_size},
           "platform": {"system": platform.system(), "machine": platform.machine(), "python": platform.python_version(),
                        "cpus": os.cpu_count()},
           "runs": runs}
    a.out.write_bytes((json.dumps(rec, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
