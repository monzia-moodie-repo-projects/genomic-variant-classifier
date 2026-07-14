"""Unit tests for scripts/preflight_run17.py (Run-17 Gate-F preflight). Author: Monzia Moodie."""
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# THE single source of truth for the tabular feature count. Imported, never transcribed --
# a second copy here is what turned `test_schema_gate_87_ok` into a stale test that failed a
# correct gate on 2026-07-13. See that test's replacement below.
from genomic_variant_classifier.models.variant_ensemble import EXPECTED_TABULAR_FEATURE_COUNT

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load("preflight_run17")


def _parse(cmd):
    import sys
    sys.path.insert(0, str(_SCRIPTS))
    import preflight_gate as gate
    return gate._parse_candidate(cmd)


def _kg_parquet(path, cols):
    tbl = pa.table({c: pa.array([0.1, 0.2], type=pa.float64()) for c in cols})
    pq.write_table(tbl, str(path))


def _baseline(path, n):
    path.write_text(json.dumps({"n_columns": n, "run_label": "test", "expected_schema_hash": "abc123def456"}))


# ---- KG gate ----
def test_kg_defer_is_conscious_warn():
    rows = P.kg_gate(_parse("python scripts/run_phase2_eval.py --clinvar x"), defer_kg=True, data_root="data")
    assert any(lv == "WARN" and "DEFERRED" in m for lv, m in rows)
    assert not any(lv == "FAIL" for lv, m in rows)


def test_kg_omitted_without_defer_fails():
    rows = P.kg_gate(_parse("python scripts/run_phase2_eval.py --clinvar x"), defer_kg=False, data_root="data")
    assert any(lv == "FAIL" and "silently stay constant" in m for lv, m in rows)


def test_kg_defer_and_kg_both_fails():
    rows = P.kg_gate(_parse("python scripts/run_phase2_eval.py --kg k.parquet"), defer_kg=True, data_root="data")
    assert any(lv == "FAIL" and "choose ONE" in m for lv, m in rows)


def test_kg_healthy_parquet_ok(tmp_path):
    kg = tmp_path / "kg.parquet"
    _kg_parquet(kg, ["variant_id", "allele_freq", "AFR_AF", "EUR_AF", "EAS_AF", "SAS_AF", "AMR_AF"])
    rows = P.kg_gate(_parse(f"python scripts/run_phase2_eval.py --kg {kg}"), defer_kg=False, data_root="data")
    assert any(lv == "OK" and "all 5 per-superpop" in m for lv, m in rows)
    assert not any(lv == "FAIL" for lv, m in rows)


def test_kg_parquet_missing_pop_cols_fails(tmp_path):
    kg = tmp_path / "kg.parquet"
    _kg_parquet(kg, ["variant_id", "allele_freq", "AFR_AF"])  # only 1 of 5 superpops
    rows = P.kg_gate(_parse(f"python scripts/run_phase2_eval.py --kg {kg}"), defer_kg=False, data_root="data")
    assert any(lv == "FAIL" and "missing per-superpop" in m for lv, m in rows)


def test_kg_parquet_absent_path_fails(tmp_path):
    rows = P.kg_gate(_parse(f"python scripts/run_phase2_eval.py --kg {tmp_path}/nope.parquet"),
                     defer_kg=False, data_root="data")
    assert any(lv == "FAIL" and "not found" in m for lv, m in rows)


# ---- schema gate ----
def test_schema_gate_accepts_a_baseline_that_matches_the_feature_contract(tmp_path):
    """The gate must pass a baseline whose column count equals the CONTRACT.

    This test was `test_schema_gate_87_ok`, and it hard-coded 87: it built a synthetic
    87-column baseline and demanded the gate call it OK.

    That made the TEST a second hand-kept copy of the feature count -- the very defect the gate
    exists to catch. On 2026-07-13, when the contract moved to 95 and
    `preflight_run17.EXPECTED_SCHEMA_COLS` was changed from a hard-coded 87 to an import of
    EXPECTED_TABULAR_FEATURE_COUNT, this test went red: the gate correctly rejected an
    87-column baseline, and the test insisted it should have been accepted.

    The gate was right. The test was the stale copy. Rewriting the literal 87 as a literal 95
    would simply have moved the rot, and it would have gone stale again at 96.

    So the test derives the number too. Now there is exactly ONE place in the entire repository
    where the tabular feature count is written down -- EXPECTED_TABULAR_FEATURE_COUNT in
    variant_ensemble.py -- and the gate, the committed baseline, the METHODS.md prose, and this
    test all re-derive it on every run. Root pattern (a) has nowhere left to hide here.
    """
    b = tmp_path / "schema_baseline.json"
    _baseline(b, EXPECTED_TABULAR_FEATURE_COUNT)
    rows = P.schema_gate(b)
    assert rows[0][0] == "OK", rows
    assert f"n_columns={EXPECTED_TABULAR_FEATURE_COUNT}" in rows[0][1]


def test_schema_gate_rejects_a_baseline_that_does_not_match_the_contract(tmp_path):
    """FAIL in BOTH directions -- a shrunken baseline AND an over-grown one.

    Was `test_schema_gate_78_fails`, which only ever exercised the SHRINK direction (the
    build_schema_baseline.py DEFAULT_MATRIX footgun, which has silently cut the baseline
    81 -> 78 in the past).

    But the other direction is a real failure too, and it is the one that was actually
    happening: a feature is added to TABULAR_FEATURES and the baseline is not regenerated, so
    the schema-drift gate is blind to that column for an entire run. The committed baseline was
    TEN COLUMNS behind the contract on 2026-07-13 and nothing said a word.

    87 is kept in the list on purpose: it is the exact stale value this gate carried for weeks,
    and it must now be rejected.
    """
    for n in (78, 87, EXPECTED_TABULAR_FEATURE_COUNT - 1, EXPECTED_TABULAR_FEATURE_COUNT + 1):
        b = tmp_path / f"schema_baseline_{n}.json"
        _baseline(b, n)
        rows = P.schema_gate(b)
        assert rows[0][0] == "FAIL", (
            f"a baseline with n_columns={n} was ACCEPTED against a contract of "
            f"{EXPECTED_TABULAR_FEATURE_COUNT}. The gate is not gating."
        )
        assert str(EXPECTED_TABULAR_FEATURE_COUNT) in rows[0][1], (
            "the failure message must name the count the gate actually expects, or the next "
            "person cannot act on it"
        )


def test_schema_gate_missing_fails(tmp_path):
    rows = P.schema_gate(tmp_path / "nope.json")
    assert rows[0][0] == "FAIL"


# ---- scripts gate ----
def test_scripts_gate_present_against_real_repo():
    rows = P.scripts_gate(_SCRIPTS)
    assert all(lv == "OK" for lv, m in rows)  # real repo has all three


def test_scripts_gate_missing_fails(tmp_path):
    rows = P.scripts_gate(tmp_path)
    assert all(lv == "FAIL" for lv, m in rows)


# ---- emit + integration ----
def test_emit_defer_has_no_kg_and_required_flags():
    cmd = P.emit_command(None, "outputs/run17", None)
    assert "--kg " not in cmd  # af_1kg parquet flag; --kg-edges is always present (no-defer)
    assert "--string-db auto" in cmd and "--unseen-gene-holdout" in cmd and "--min-review-tier 3" in cmd


def test_emit_kg_includes_kg_and_smoke_maxtrain():
    cmd = P.emit_command("data/external/1000g/kg.parquet", "outputs/run17_smoke", 3000)
    assert "--kg data/external/1000g/kg.parquet" in cmd and "--max-train 3000" in cmd


def test_emitted_command_passes_command_level_gate(tmp_path, monkeypatch):
    # the emitted command must satisfy preflight_gate's flag-level rules (values/flags/forbidden)
    import sys
    sys.path.insert(0, str(_SCRIPTS))
    import preflight_gate as gate
    cmd = P.emit_command("kg.parquet", "outputs/run17", None)
    ns = gate._parse_candidate(cmd)
    rows = gate.validate(ns, str(tmp_path), n_train=1000, ack={"kg", "finngen"})
    # required values/flags present, no forbidden -> the only FAILs would be missing data paths (tmp empty)
    flag_fails = [m for lv, m in rows if lv == "FAIL" and ("missing" in m or "absent" in m or "diminish" in m)]
    assert not flag_fails, f"command-level flag failures: {flag_fails}"


def test_run_all_flags_forbidden_skip(tmp_path):
    b = tmp_path / "schema_baseline.json"; _baseline(b, 87)
    kg = tmp_path / "kg.parquet"
    _kg_parquet(kg, ["variant_id", "allele_freq", "AFR_AF", "EUR_AF", "EAS_AF", "SAS_AF", "AMR_AF"])
    cmd = (f"python scripts/run_phase2_eval.py --clinvar c --string-db auto --min-review-tier 3 "
           f"--n-folds 5 --unseen-gene-holdout --skip-nn --kg {kg} --output outputs/run17")
    rows = P.run_all(cmd, str(tmp_path), 1000, defer_kg=False, baseline_path=b, scripts_dir=_SCRIPTS)
    assert any(lv == "FAIL" and "--skip-nn" in m for lv, m in rows)  # forbidden skip caught
    assert any(lv == "OK" and "per-superpop" in m for lv, m in rows)  # kg still validated


def test_parse_candidate_preserves_windows_backslash_paths():
    # Regression: POSIX shlex eats backslashes, which silently mangled Windows --kg/--output paths and
    # failed 3 kg tests on Windows (this Linux sandbox uses forward slashes, so it passed here -- the
    # blind spot). Platform-independent: uses a literal backslash string so it runs identically anywhere.
    import sys
    sys.path.insert(0, str(_SCRIPTS))
    import preflight_gate as gate
    cmd = r"python scripts/run_phase2_eval.py --kg C:\data\1kg\kg.parquet --output outputs\run17"
    ns = gate._parse_candidate(cmd)
    assert vars(ns)["kg"] == r"C:\data\1kg\kg.parquet"
    assert vars(ns)["output"] == r"outputs\run17"


# ---- STRING-DB gate (gnn_score prerequisite) ----
def test_string_gate_cached_graph_ok(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "string_graph_700.pkl").write_text("x")
    rows = P.string_db_gate(700, cache, tmp_path / "nolinks.txt.gz")
    assert rows[0][0] == "OK" and "cached graph" in rows[0][1]


def test_string_gate_cached_links_ok(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "string_links.parquet").write_text("x")
    rows = P.string_db_gate(700, cache, tmp_path / "nolinks.txt.gz")
    assert rows[0][0] == "OK" and "cached links" in rows[0][1]


def test_string_gate_local_txtgz_ok(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    local = tmp_path / "9606.protein.links.detailed.v12.0.txt.gz"; local.write_text("x")
    rows = P.string_db_gate(700, cache, local)
    assert rows[0][0] == "OK" and "local links file" in rows[0][1]


def test_string_gate_none_present_warns_about_download(tmp_path):
    cache = tmp_path / "cache"; cache.mkdir()
    rows = P.string_db_gate(700, cache, tmp_path / "absent.txt.gz")
    assert rows[0][0] == "WARN" and "DOWNLOAD STRING" in rows[0][1]


def test_string_threshold_from_auto_is_700():
    ns = _parse("python scripts/run_phase2_eval.py --string-db auto")
    assert P._string_threshold_from_ns(ns) == 700


def test_string_threshold_from_numeric():
    ns = _parse("python scripts/run_phase2_eval.py --string-db 400")
    assert P._string_threshold_from_ns(ns) == 400


def test_string_gate_threshold_picks_matching_pkl(tmp_path):
    # a 700 pkl must NOT satisfy a 400-threshold run (different cache file)
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "string_graph_700.pkl").write_text("x")
    rows = P.string_db_gate(400, cache, tmp_path / "absent.txt.gz")
    assert rows[0][0] == "WARN"


def test_run_all_includes_string_gate(tmp_path):
    b = tmp_path / "schema_baseline.json"; _baseline(b, 87)
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "string_graph_700.pkl").write_text("x")
    cmd = P.emit_command(None, "outputs/run17", None)
    rows = P.run_all(cmd, str(tmp_path), 1000, defer_kg=True, baseline_path=b,
                     scripts_dir=_SCRIPTS, cache_dir=cache, local_links=tmp_path / "absent.txt.gz")
    assert any(lv == "OK" and "cached graph" in m for lv, m in rows)
