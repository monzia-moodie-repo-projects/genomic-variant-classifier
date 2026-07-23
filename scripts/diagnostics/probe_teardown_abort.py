#!/usr/bin/env python3
"""probe_teardown_abort.py -- pin the culprit behind a native teardown abort.

WHY THIS EXISTS
===============
On 2026-07-22, Continuous Integration run 29962715186 (run number 585, commit
821a990) failed one test on Python 3.12 and only on Python 3.12:

    tests/unit/test_rnaseq_ablation_tools.py::test_full_is_unchanged
    child returncode -6 (SIGABRT)
    child stderr     "terminate called without an active exception"
    child stdout     "[ok] mode=full seed=0 -> ... (120 genes, 120 with any
                      non-zero rnaseq feature)"

The child printed its success line, which is the LAST statement of main(), after
the parquet was written. So the work completed and the process aborted afterwards,
during interpreter finalization. "terminate called without an active exception" is
a libstdc++ message from std::terminate(), classically raised when a std::thread is
destroyed while still joinable -- a native-library thread-teardown fault.

A re-run of the identical commit went green, which proves the fault is
NONDETERMINISTIC. 730 local reproduction attempts at the exact pinned library
versions (numpy 2.4.4, pandas 2.3.3, pyarrow 23.0.1) produced zero aborts, so the
rate is low and environment-specific.

This probe does NOT fix anything and does NOT change production behaviour. It
exists to answer one question with evidence rather than inference:

    WHICH native library aborts, and under what conditions?

HOW IT ANSWERS THAT
-------------------
Two independent lines of attack, run as separate "arms":

  1. BISECT BY CONSTRUCTION. Each arm runs a child process that loads a different
     subset of the stack -- numpy alone, pandas without parquet, a pyarrow read, a
     pyarrow write, pyarrow without pandas -- and the real production script. The
     smallest arm that aborts names the culprit by construction, with no symbol
     lookup required.

  2. BISECT BY MITIGATION. Further arms run the real script with the native thread
     pools constrained (Arrow input/output threads, Arrow central processing unit
     count, OpenMP thread count). An arm whose abort rate collapses to zero while
     the baseline aborts identifies the pool responsible, causally.

Statistics are reported honestly. A zero-abort arm does not prove the arm is safe;
it bounds the rate. With no events in n trials the one-sided 95% upper bound is
approximately 3/n (the rule of three), and that bound is reported so a negative
arm can be read for what it is.

Author: written for Monzia Moodie, 2026-07-22.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# Child programs for the bisect-by-construction arms. Each is written to a
# temporary file and executed as its own process, so the teardown being measured
# is a real interpreter finalization, not a function return.
_RNASEQ_COLS = (
    "rnaseq_mean_log_tpm", "rnaseq_detection_rate", "rnaseq_log2_cv",
    "rnaseq_log2fc", "rnaseq_de_neglog10p",
)

_CHILD_NUMPY_ONLY = """
import numpy as np
a = np.random.default_rng(0).random((120, 5))
print("[ok] numpy_only", float(a.sum()))
"""

_CHILD_PANDAS_NO_PARQUET = """
import numpy as np, pandas as pd
df = pd.DataFrame(np.random.default_rng(0).random((120, 5)))
print("[ok] pandas_no_parquet", int(df.shape[0]))
"""

_CHILD_PYARROW_READ = """
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1])
print("[ok] pyarrow_read", int(df.shape[0]))
"""

_CHILD_PYARROW_WRITE = """
import sys, numpy as np, pandas as pd
df = pd.DataFrame(np.random.default_rng(0).random((120, 5)))
df.to_parquet(sys.argv[1], index=False)
print("[ok] pyarrow_write", int(df.shape[0]))
"""

_CHILD_PYARROW_DIRECT = """
import sys
import pyarrow.parquet as pq
t = pq.read_table(sys.argv[1])
pq.write_table(t, sys.argv[2])
print("[ok] pyarrow_direct", t.num_rows)
"""


# --------------------------------------------------------------------------- #
# Round-two children (2026-07-23). Round one localised the abort to
# pandas.read_parquet: 45/5000 there, 0/5000 for pyarrow's own read_table plus
# write_table. The only step that distinguishes them is the Arrow-to-pandas
# conversion, so these children test that conversion directly, and test it with
# its thread use switched off -- the sharpest available test of the hypothesis.
# --------------------------------------------------------------------------- #

_CHILD_TO_PANDAS = """
import sys
import pyarrow.parquet as pq
table = pq.read_table(sys.argv[1])
frame = table.to_pandas()
print("[ok] to_pandas_explicit", int(frame.shape[0]))
"""

_CHILD_TO_PANDAS_NO_THREADS = """
import sys
import pyarrow.parquet as pq
table = pq.read_table(sys.argv[1])
frame = table.to_pandas(use_threads=False)
print("[ok] to_pandas_no_threads", int(frame.shape[0]))
"""

_CHILD_READ_CPU_COUNT_1 = """
import sys
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
import pandas as pd
frame = pd.read_parquet(sys.argv[1])
print("[ok] read_cpu_count_1", int(frame.shape[0]))
"""

# arm name -> (child source or None to use the real script, environment overlay)
ARMS: dict[str, tuple[str | None, dict[str, str]]] = {
    # bisect by construction
    "numpy_only":            (_CHILD_NUMPY_ONLY, {}),
    "pandas_no_parquet":     (_CHILD_PANDAS_NO_PARQUET, {}),
    "pyarrow_read":          (_CHILD_PYARROW_READ, {}),
    "pyarrow_write":         (_CHILD_PYARROW_WRITE, {}),
    "pyarrow_direct":        (_CHILD_PYARROW_DIRECT, {}),
    # the real production script, unmodified
    "baseline_real_script":  (None, {}),
    # bisect by mitigation, all against the real script
    "arrow_io_threads_1":    (None, {"ARROW_IO_THREADS": "1"}),
    "omp_num_threads_1":     (None, {"OMP_NUM_THREADS": "1"}),
    "all_threads_1":         (None, {"ARROW_IO_THREADS": "1", "OMP_NUM_THREADS": "1",
                                     "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}),

    # Round two. These are referenced against pyarrow_read (45/5000 = 0.90%), not
    # against the real script (1/5000 = 0.02%). At 5000 iterations the read arm
    # expects about 45 events, so an arm that drops to zero is decisive, whereas
    # the same zero measured against the real script proves nothing at all.
    "to_pandas_explicit":     (_CHILD_TO_PANDAS, {}),
    "to_pandas_no_threads":   (_CHILD_TO_PANDAS_NO_THREADS, {}),
    "read_cpu_count_1":       (_CHILD_READ_CPU_COUNT_1, {}),
    "read_arrow_io_threads_1": (_CHILD_PYARROW_READ, {"ARROW_IO_THREADS": "1"}),
    "read_all_threads_1":     (_CHILD_PYARROW_READ, {"ARROW_IO_THREADS": "1",
                                                     "OMP_NUM_THREADS": "1",
                                                     "OPENBLAS_NUM_THREADS": "1",
                                                     "MKL_NUM_THREADS": "1"}),
}

# Arms whose child takes the source parquet as its single argument.
_ARMS_NEEDING_SOURCE = {
    "pyarrow_read", "to_pandas_explicit", "to_pandas_no_threads",
    "read_cpu_count_1", "read_arrow_io_threads_1", "read_all_threads_1",
}


def _rule_of_three_upper_bound(n_events: int, n_trials: int) -> float:
    """One-sided 95% upper bound on the event rate. For zero events this is the
    classical 3/n; for non-zero events fall back to the observed rate, which is
    reported alongside the raw counts so nothing is hidden behind a formula."""
    if n_trials <= 0:
        return float("nan")
    if n_events == 0:
        return 3.0 / n_trials
    return n_events / n_trials


def _write_source_parquet(path: Path) -> None:
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {"gene_symbol": [f"G{i}" for i in range(120)],
         **{col: rng.random(120) for col in _RNASEQ_COLS}})
    frame.to_parquet(path, index=False)


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {"python": sys.version.split()[0],
                           "platform": platform.platform()}
    for name in ("numpy", "pandas", "pyarrow"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[name] = f"import-failed: {type(exc).__name__}"
    return out


def _build_command(arm: str, workdir: Path, real_script: Path, index: int) -> list[str]:
    child_source, _env = ARMS[arm]
    src = workdir / "src.parquet"
    if child_source is None:
        return [sys.executable, "-X", "faulthandler", str(real_script),
                "--src", str(src), "--out", str(workdir / f"out_{index % 4}.parquet"),
                "--mode", "full", "--seed", "0"]
    child_path = workdir / f"child_{arm}.py"
    if not child_path.exists():
        child_path.write_text(child_source, encoding="utf-8", newline="\n")
    argv = [sys.executable, "-X", "faulthandler", str(child_path)]
    if arm in _ARMS_NEEDING_SOURCE:
        argv.append(str(src))
    elif arm in ("pyarrow_write",):
        argv.append(str(workdir / f"w_{index % 4}.parquet"))
    elif arm in ("pyarrow_direct",):
        argv += [str(src), str(workdir / f"d_{index % 4}.parquet")]
    return argv


def _signature_of(stderr_text: str) -> str:
    """Group failures by the line that actually names the fault. The C++ abort
    message is what identifies this class, and with faulthandler enabled it is not
    necessarily the last line, so prefer the diagnostic line over stderr's tail."""
    lines = [line.strip() for line in stderr_text.strip().splitlines() if line.strip()]
    if not lines:
        return "<empty stderr>"
    for marker in ("terminate called", "Fatal Python error", "Segmentation fault",
                   "Aborted", "double free", "corrupted", "Assertion"):
        for line in lines:
            if marker in line:
                return line[:200]
    return lines[-1][:200]


def run_arm(arm: str, iterations: int, workdir: Path, real_script: Path) -> dict:
    _child_source, env_overlay = ARMS[arm]
    env = dict(os.environ)
    env.update(env_overlay)

    codes: Counter[int] = Counter()
    signatures: Counter[str] = Counter()
    first_failure: dict | None = None

    started = time.time()
    for index in range(iterations):
        argv = _build_command(arm, workdir, real_script, index)
        completed = subprocess.run(argv, capture_output=True, text=True, env=env)
        codes[completed.returncode] += 1
        if completed.returncode != 0:
            signatures[_signature_of(completed.stderr)] += 1
            if first_failure is None:
                first_failure = {
                    "iteration": index,
                    "returncode": completed.returncode,
                    "argv": argv,
                    "stdout_tail": completed.stdout.strip()[-400:],
                    "stderr_full": completed.stderr.strip()[-4000:],
                }
    elapsed = time.time() - started

    n_failures = sum(count for code, count in codes.items() if code != 0)
    return {
        "arm": arm,
        "environment_overlay": env_overlay,
        "iterations": iterations,
        "returncode_counts": {str(k): v for k, v in sorted(codes.items())},
        "n_failures": n_failures,
        "observed_rate": (n_failures / iterations) if iterations else float("nan"),
        "upper_bound_95_one_sided": _rule_of_three_upper_bound(n_failures, iterations),
        "stderr_signatures": dict(signatures),
        "first_failure": first_failure,
        "elapsed_seconds": round(elapsed, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--out", required=True, help="path for the JSON result")
    parser.add_argument("--repo-root", default=".",
                        help="repository root, used to locate the real script")
    args = parser.parse_args(argv)

    if args.iterations < 1:
        print("ERROR: --iterations must be at least 1", file=sys.stderr)
        return 2

    real_script = (Path(args.repo_root).resolve()
                   / "scripts" / "make_rnaseq_ablation_parquet.py")
    if not real_script.is_file():
        print(f"ERROR: real script not found at {real_script}", file=sys.stderr)
        return 3

    with tempfile.TemporaryDirectory(prefix="teardown_probe_") as tmp:
        workdir = Path(tmp)
        _write_source_parquet(workdir / "src.parquet")
        result = run_arm(args.arm, args.iterations, workdir, real_script)

    result["library_versions"] = _library_versions()
    result["real_script"] = str(real_script)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8", newline="\n")

    print(f"[arm] {result['arm']}")
    print(f"  iterations        : {result['iterations']}")
    print(f"  returncode counts : {result['returncode_counts']}")
    print(f"  failures          : {result['n_failures']}")
    print(f"  observed rate     : {result['observed_rate']:.6f}")
    print(f"  95% upper bound   : {result['upper_bound_95_one_sided']:.6f}")
    print(f"  elapsed           : {result['elapsed_seconds']}s")
    for signature, count in result["stderr_signatures"].items():
        print(f"  stderr x{count}: {signature}")
    # A diagnostic reports; it does not gate. Finding an abort is a SUCCESS for
    # this probe, so the exit status stays 0 and the workflow always publishes.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
