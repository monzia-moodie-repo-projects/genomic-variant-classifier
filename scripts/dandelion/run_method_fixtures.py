"""Run the predetermined DANDELION method fixtures and write the layered report (owner rulings 2026-10-08e, 2026-10-08f section 6,
2026-10-08g).

Every fixture runs TWICE, each time in a FRESH R process: recorders off, then recorders on (scripts/dandelion/method_fixtures.R; the
backend recorder for every fixture, the per-exposure outcome recorder for the trace fixtures). The
specification is bound by its SHA-256 before anything runs; the runner sees inputs only, never predictions. The judge
(inference/method_trace.py) reports every layer; a failed prediction is a finding (exit 1 with the report written), malformed or
incomplete evidence refuses (exit 1, no report). The child R environment is the caller's minus every R_* / RENV_* variable, plus
R_LIBS when --r-libs names the libraries to use (os.pathsep-separated, each must exist; the qualified library in the isolated replay);
R_LIBS_USER and R_LIBS_SITE point at one EMPTY directory, so no package can load from a user or site library the run did not declare
(R would otherwise add an existing default user library silently). Each run records .libPaths().
The checkout's src/ is put first on the import path, so `python -I run_method_fixtures.py ...` uses this checkout's judge.

Usage: python run_method_fixtures.py --spec tests/fixtures/dandelion/method_fixtures_v2.json --spec-sha256 <digest>
       --rscript <path to Rscript> [--r-libs <library>] --out <NEW directory>

Author: Monzia Moodie
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--spec-sha256", required=True)
    ap.add_argument("--rscript", required=True)
    ap.add_argument("--r-libs", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args(argv)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from genomic_variant_classifier.inference.exact_confirmation import InferenceError
    from genomic_variant_classifier.inference.method_trace import judge, load_spec, prepare_scenarios, render_report
    spec_bytes = Path(a.spec).read_bytes()
    if hashlib.sha256(spec_bytes).hexdigest() != a.spec_sha256:
        print("REFUSED: the specification is not the frozen one (sha256 {})".format(hashlib.sha256(spec_bytes).hexdigest()))
        return 1
    spec = load_spec(spec_bytes)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    ids = prepare_scenarios(spec, out / "scenarios")
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("R_", "RENV_"))}
    if a.r_libs:
        env["R_LIBS"] = os.pathsep.join(str(Path(x).resolve(strict=True)) for x in a.r_libs.split(os.pathsep))
    empty = out / "empty_library"
    empty.mkdir()
    env["R_LIBS_USER"] = env["R_LIBS_SITE"] = str(empty.resolve())
    rscript = str(Path(a.rscript).resolve(strict=True))
    for fixture in ids:
        for mode in ("off", "on"):
            run_dir = out / "runs" / fixture / mode
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            done = subprocess.run([rscript, "--vanilla", str(HERE / "method_fixtures.R"), str(HERE / "dandelion_backend_recorder.R"),
                                   str(HERE / "dandelion_exposure_recorder.R"), str(out / "scenarios" / fixture), mode, str(run_dir)],
                                  capture_output=True, env=env, timeout=a.timeout, stdin=subprocess.DEVNULL)
            log = out / "logs"
            log.mkdir(exist_ok=True)
            (log / "{}_{}.stdout".format(fixture, mode)).write_bytes(done.stdout)
            (log / "{}_{}.stderr".format(fixture, mode)).write_bytes(done.stderr)
            if done.returncode != 0:
                print("REFUSED: fixture {} ({}) exited {}; see {}".format(fixture, mode, done.returncode, log))
                return 1
    try:
        report = judge(spec_bytes, a.spec_sha256, out / "runs")
    except InferenceError as exc:
        print("REFUSED: " + str(exc))
        return 1
    raw = render_report(report)
    (out / "report.json").write_bytes(raw)
    print("REPORT {} sha256 {}".format(out / "report.json", hashlib.sha256(raw).hexdigest()))
    print("FIXTURES PASSED: {}".format(report["fixtures_passed"]))
    for failure in report["failures"]:
        print("  finding: " + failure)
    return 0 if report["fixtures_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
