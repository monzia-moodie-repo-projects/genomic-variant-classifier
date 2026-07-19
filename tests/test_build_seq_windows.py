"""tests/test_build_seq_windows.py -- coverage for the SURVIVING window builder and driver.

WHY THIS FILE EXISTS (2026-07-18)
---------------------------------
Phase 3 retires `data/seq_windows.py` and `data/populate_fasta_seq.py`, which are the
superseded builder pair. Deleting them also deletes their 21 tests. An audit on 2026-07-18
found that the surviving code does NOT carry equivalent coverage in two respects:

  1. `tests/test_delta_window_builder.py` exercises `build_window` ONLY through synthetic
     dictionary fetchers. Nothing verified the injected-`fetch` contract against a real
     genome file. `test_seq_windows.py::test_pyfaidx_adapter_agrees_with_dict` was the only
     such test in the suite, and it dies with its module.
  2. `scripts/build_seq_windows.py` -- the driver that survives -- had NO tests at all.
     `test_populate_fasta_seq.py` covered the DYING driver's guards.

This file ports both before anything is deleted. Coverage moves; it does not evaporate.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import random
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from genomic_variant_classifier.data.delta_window_builder import POLY, build_window

_ROOT = Path(__file__).resolve().parents[1]
WINDOW = 101
HALF = WINDOW // 2

_rng = random.Random(20260718)
GENOME = "".join(_rng.choice("ACGT") for _ in range(600))  # non-periodic on purpose


def _dict_fetch(genome: str):
    """Fetcher backed by a plain string, mirroring test_delta_window_builder.py."""
    def fetch(contig, start0, length):
        if start0 < 0:
            return None
        return genome[start0:start0 + length]
    return fetch


def _write_fasta(path: Path, genome: str, contig: str = "c1") -> Path:
    """Write the FASTA once per path.

    The resume and corrupt-part tests invoke the driver twice against the same tmp_path.
    Rewriting the FASTA on the second call leaves pyfaidx's .fai index older than the
    file it indexes, which emits `RuntimeWarning: Index file ... is older than FASTA
    file`. That warning is a fixture artifact, not a finding -- so it is removed at the
    source rather than filtered, since a suite that tolerates known noise stops being
    able to see new noise.
    """
    if not path.exists():
        path.write_text(">" + contig + "\n" + textwrap.fill(genome, width=60) + "\n")
    return path


def _pyfaidx_fetch(fa):
    """Fetcher backed by pyfaidx, byte-for-byte the adapter used in
    scripts/build_seq_windows.py::main(). That adapter is a closure inside main() and is
    therefore not directly importable; reproducing its exact shape here is what makes this
    test cover the production path rather than a paraphrase of it."""
    def fetch(contig, start0, length):
        try:
            if start0 < 0:
                return None
            return str(fa[str(contig)][start0:start0 + length])
        except Exception:
            return None
    return fetch


# =======================================================================================
# PART A -- build_window against a REAL genome file
# =======================================================================================

def test_pyfaidx_fetch_agrees_with_dict_fetch(tmp_path):
    """PORTED from test_seq_windows.py::test_pyfaidx_adapter_agrees_with_dict.

    The builder takes `fetch` by injection, which keeps it pure and unit-testable -- but
    it also means every existing test proves only that the builder agrees with a fetcher
    written by the same hand. This opens a real FASTA through pyfaidx and requires the two
    to agree exactly, so a drift between the injected contract and reality is caught.
    """
    pytest.importorskip("pyfaidx")
    from genomic_variant_classifier.data.seq_window_manifest import KEY_COLS  # noqa: F401
    from pyfaidx import Fasta

    fa_path = _write_fasta(tmp_path / "tiny.fa", GENOME)
    fa = Fasta(str(fa_path), rebuild=True)
    f_dict = _dict_fetch(GENOME)
    f_fa = _pyfaidx_fetch(fa)

    checked = 0
    for pos in (60, 150, 300, 450):
        ref_b = GENOME[pos - 1]
        alt_b = next(c for c in "ACGT" if c != ref_b)
        r_dict = build_window(f_dict, "c1", pos, ref_b, alt_b)
        r_fa = build_window(f_fa, "c1", pos, ref_b, alt_b)
        assert r_fa.ok is True, "real-genome build failed at pos {}: {}".format(pos, r_fa.reason)
        assert r_fa.ref_window == r_dict.ref_window, "ref window differs at pos {}".format(pos)
        assert r_fa.alt_window == r_dict.alt_window, "alt window differs at pos {}".format(pos)
        assert r_fa.ok == r_dict.ok and r_fa.reason == r_dict.reason
        checked += 1
    assert checked == 4


def test_real_genome_window_equals_genome_slice(tmp_path):
    """The reference window must BE the genome, not merely agree with another function."""
    pytest.importorskip("pyfaidx")
    from pyfaidx import Fasta

    fa = Fasta(str(_write_fasta(tmp_path / "tiny.fa", GENOME)), rebuild=True)
    pos = 300
    ref_b = GENOME[pos - 1]
    alt_b = next(c for c in "ACGT" if c != ref_b)
    r = build_window(_pyfaidx_fetch(fa), "c1", pos, ref_b, alt_b)
    assert r.ok is True
    assert r.ref_window == GENOME[pos - 1 - HALF: pos - 1 - HALF + WINDOW]
    assert r.ref_window[HALF] == ref_b
    assert r.alt_window[HALF] == alt_b
    assert sum(1 for x, y in zip(r.ref_window, r.alt_window) if x != y) == 1


def test_insertion_pushes_downstream_right():
    """PORTED from test_seq_windows.py::test_insertion_pushes_downstream_right.

    The surviving suite checked only that insertion windows keep their LENGTH. Length is
    preserved by any implementation that truncates; it does not pin WHERE the bases went.
    """
    pos = 200
    ref = GENOME[pos - 1]
    alt = ref + "TTT"
    r = build_window(_dict_fetch(GENOME), "c1", pos, ref, alt)
    assert r.ok is True, r.reason
    assert r.alt_window[HALF:HALF + len(alt)] == alt
    tail = WINDOW - HALF - len(alt)
    assert r.alt_window[HALF + len(alt):] == r.ref_window[HALF + len(ref):HALF + len(ref) + tail]


def test_deletion_pulls_downstream_left():
    """PORTED from test_seq_windows.py::test_deletion_pulls_downstream_left."""
    pos = 200
    ref = GENOME[pos - 1:pos + 2]
    alt = ref[0]
    r = build_window(_dict_fetch(GENOME), "c1", pos, ref, alt)
    assert r.ok is True, r.reason
    assert r.alt_window[HALF:HALF + len(alt)] == alt
    n = 20
    assert (r.alt_window[HALF + len(alt):HALF + len(alt) + n]
            == r.ref_window[HALF + len(ref):HALF + len(ref) + n])


def test_placeholder_is_N_not_A():
    """The regression that started all of this: the placeholder must not be a real base.

    'N' is absent from encode_sequence's BASES = 'ACGT', so it one-hot-encodes to an
    all-zero vector -- an honest 'unknown'. 'A' encodes to a confident adenine.
    """
    assert POLY == "N"
    r = build_window(_dict_fetch(GENOME), "c1", 200, "A", "-")
    assert r.ok is False
    assert r.reason == "non_acgt_allele"
    assert r.ref_window == "N" * WINDOW and r.alt_window == "N" * WINDOW
    assert "A" * WINDOW not in (r.ref_window, r.alt_window)


# =======================================================================================
# PART B -- the SURVIVING driver, scripts/build_seq_windows.py
# =======================================================================================
# It had no tests. test_populate_fasta_seq.py covered the guards of the DYING driver:
# row-order passthrough, refusal on a missing required column, abort-and-clean-temp, and
# agreement between grouped and per-row processing. The surviving driver is chunked and
# resumable, so its equivalents are: row order across chunk boundaries, provenance
# recorded per row, resumption from a .done marker, and rebuild of a corrupt part.

@contextlib.contextmanager
def _stdout_guard():
    """Isolate the global stdout mutation that scripts/build_seq_windows.py performs.

    At import time (line ~40) the script executes, at MODULE level:

        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                          errors="replace")
        except Exception:
            pass

    Under pytest, sys.stdout is the capture object. Wrapping its .buffer and later
    discarding the wrapper makes the wrapper CLOSE that buffer on garbage collection,
    and pytest then dies at teardown with `ValueError: I/O operation on closed file`
    while flushing captured output. Restoring sys.stdout alone does NOT prevent this --
    the wrapper must be DETACHED from the buffer it adopted.

    Measured 2026-07-18: 35 scripts under scripts/ carry this same module-level mutation
    (none in src/, none in tests/). It is very likely why this driver had no tests: it
    could not be imported under pytest's default capture without breaking the run. The
    guard below isolates it so coverage can exist today; the systemic pattern is recorded
    separately rather than quietly worked around.
    """
    saved = sys.stdout
    try:
        yield
    finally:
        installed = sys.stdout
        sys.stdout = saved
        if installed is not saved:
            try:
                installed.detach()      # sever it from pytest's buffer before it is GC'd
            except Exception:
                pass


with _stdout_guard():
    _SPEC = importlib.util.spec_from_file_location(
        "build_seq_windows", _ROOT / "scripts" / "build_seq_windows.py")
    bsw = importlib.util.module_from_spec(_SPEC)
    sys.modules["build_seq_windows"] = bsw
    _SPEC.loader.exec_module(bsw)


def _cohort(tmp_path: Path, n: int = 40) -> Path:
    """A small cohort with a deliberate mix: SNVs, an insertion, a deletion, and two rows
    the builder must REFUSE (a non-ACGT allele and an empty allele)."""
    rows = []
    for i in range(n):
        pos = 120 + i * 3
        ref = GENOME[pos - 1]
        alt = next(c for c in "ACGT" if c != ref)
        rows.append({"chrom": "c1", "pos": pos, "ref": ref, "alt": alt})
    rows[5]["alt"] = rows[5]["ref"] + "TT"          # insertion
    rows[6]["ref"] = GENOME[rows[6]["pos"] - 1:rows[6]["pos"] + 1]   # deletion
    rows[6]["alt"] = rows[6]["ref"][0]
    rows[7]["alt"] = "N"                            # non_acgt_allele -> ok=False
    rows[8]["alt"] = ""                             # empty_allele    -> ok=False
    p = tmp_path / "cohort.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _run_driver(tmp_path, monkeypatch, chunk_size=13, extra=None):
    fa = _write_fasta(tmp_path / "ref.fa", GENOME)
    cohort = _cohort(tmp_path)
    out = tmp_path / "out"
    argv = ["build_seq_windows.py",
            "--cohort", str(cohort),
            "--reference", str(fa),
            "--out-dir", str(out),
            "--chunk-size", str(chunk_size),
            "--verify-per-class", "5"]
    if extra:
        argv += extra
    monkeypatch.setattr(sys, "argv", argv)
    with _stdout_guard():
        rc = bsw.main()
    return rc, out, cohort


def test_driver_builds_parts_manifest_and_ok_column(tmp_path, monkeypatch):
    pytest.importorskip("pyfaidx")
    rc, out, cohort = _run_driver(tmp_path, monkeypatch)
    assert rc == 0, "driver exited {}".format(rc)

    merged = out / "seq_windows.parquet"
    assert merged.exists(), "merged output not written"
    df = pd.read_parquet(merged)
    assert len(df) == 40
    for c in ("chrom", "pos", "ref", "alt", "fasta_seq_ref", "fasta_seq_alt", "ok", "reason"):
        assert c in df.columns, "missing column {}".format(c)
    assert all(len(s) == WINDOW for s in df["fasta_seq_ref"])
    assert all(len(s) == WINDOW for s in df["fasta_seq_alt"])

    man = json.loads((out / "seq_windows.manifest.json").read_text())
    assert man["n_rows_built"] == 40
    assert man["window"] == WINDOW
    assert man["n_ok"] + man["n_poly"] == 40
    assert man["builder_version"]


def test_driver_records_provenance_not_content(tmp_path, monkeypatch):
    """The two unbuildable rows must be flagged via `ok`, with a machine-readable reason --
    and their windows must be poly-N, never poly-A."""
    pytest.importorskip("pyfaidx")
    rc, out, _ = _run_driver(tmp_path, monkeypatch)
    assert rc == 0
    df = pd.read_parquet(out / "seq_windows.parquet")

    bad = df.loc[~df["ok"].astype(bool)]
    assert len(bad) == 2, "expected exactly the 2 planted unbuildable rows, got {}".format(len(bad))
    assert set(bad["reason"]) == {"non_acgt_allele", "empty_allele"}
    for s in list(bad["fasta_seq_ref"]) + list(bad["fasta_seq_alt"]):
        assert s == "N" * WINDOW
        assert s != "A" * WINDOW
    # and every ok row must carry real sequence, not a placeholder
    good = df.loc[df["ok"].astype(bool)]
    assert not (good["fasta_seq_ref"] == "N" * WINDOW).any()


def test_driver_preserves_row_order_across_chunks(tmp_path, monkeypatch):
    """PORTED in spirit from test_populate_fasta_seq.py::test_passthrough_order_*.

    chunk_size=13 over 40 rows forces 4 chunks, so this fails if the merge reorders.
    """
    pytest.importorskip("pyfaidx")
    rc, out, cohort = _run_driver(tmp_path, monkeypatch, chunk_size=13)
    assert rc == 0
    src = pd.read_parquet(cohort)
    df = pd.read_parquet(out / "seq_windows.parquet")
    assert list(df["pos"]) == list(src["pos"])
    assert list(df["alt"]) == list(src["alt"])


def test_driver_resumes_from_done_marker(tmp_path, monkeypatch):
    """A .done marker must cause the chunk to be SKIPPED, not silently rebuilt."""
    pytest.importorskip("pyfaidx")
    rc, out, _ = _run_driver(tmp_path, monkeypatch, chunk_size=13)
    assert rc == 0
    parts = sorted(out.glob("part_*.parquet"))
    assert len(parts) >= 2, "expected multiple chunks, got {}".format(len(parts))
    before = parts[0].read_bytes()

    rc2, out2, _ = _run_driver(tmp_path, monkeypatch, chunk_size=13)
    assert rc2 == 0
    assert sorted(out2.glob("part_*.parquet"))[0].read_bytes() == before, \
        "a completed chunk was rebuilt despite its .done marker"


def test_driver_rebuilds_corrupt_part(tmp_path, monkeypatch):
    """A .done marker beside an UNREADABLE part must not be trusted.

    This is the failure mode that matters: trusting a marker over the artifact it claims
    to describe is exactly how a silently truncated build gets treated as complete.
    """
    pytest.importorskip("pyfaidx")
    rc, out, _ = _run_driver(tmp_path, monkeypatch, chunk_size=13)
    assert rc == 0
    part = sorted(out.glob("part_*.parquet"))[0]
    good = pd.read_parquet(part)
    part.write_bytes(b"not a parquet file")          # corrupt it, keep the .done marker

    rc2, out2, _ = _run_driver(tmp_path, monkeypatch, chunk_size=13)
    assert rc2 == 0, "driver did not recover from a corrupt part"
    rebuilt = pd.read_parquet(part)
    assert len(rebuilt) == len(good)
    assert list(rebuilt["pos"]) == list(good["pos"])
