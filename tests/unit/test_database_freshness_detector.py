"""test_database_freshness_detector.py -- Monzia Moodie

Unit tests for the registry-driven freshness detector. Network is mocked via the injectable `probe` so the
tests are hermetic. Covers: changed / unchanged / first-observation / unreachable / manual_skip upstream
classification, and present / missing / cruft local classification, plus a full scan over the real registry.
"""
import pytest

from genomic_variant_classifier.monitoring import registry as R
from genomic_variant_classifier.agent_layer.agents import database_freshness_detector as D


def _src(key="alphamissense"):
    return R.by_key(key)  # a real HTTP_ETAG (probeable) source


def test_upstream_manual_skip():
    s = R.by_key("dbnsfp")  # MANUAL
    r = D.check_upstream(s, last_seen="x")
    assert r.status == D.MANUAL_SKIP and r.current is None


def test_upstream_first_observation_is_changed():
    r = D.check_upstream(_src(), last_seen=None, probe=lambda s: ("etag-1", "ok"))
    assert r.status == D.CHANGED and r.current == "etag-1" and r.previous is None


def test_upstream_unchanged():
    r = D.check_upstream(_src(), last_seen="etag-1", probe=lambda s: ("etag-1", "ok"))
    assert r.status == D.UNCHANGED


def test_upstream_changed():
    r = D.check_upstream(_src(), last_seen="etag-1", probe=lambda s: ("etag-2", "ok"))
    assert r.status == D.CHANGED and r.previous == "etag-1" and r.current == "etag-2"


def test_upstream_unreachable_on_probe_exception():
    def boom(s):
        raise OSError("network down")
    r = D.check_upstream(_src(), last_seen="etag-1", probe=boom)
    assert r.status == D.UNREACHABLE and "OSError" in r.detail  # never raises


def test_upstream_unreachable_on_none_fingerprint():
    r = D.check_upstream(_src(), last_seen="x", probe=lambda s: (None, "empty"))
    assert r.status == D.UNREACHABLE


def test_local_missing(tmp_path):
    r = D.check_local(R.by_key("spliceai"), root=str(tmp_path))
    assert r.status == D.MISSING


def test_local_present(tmp_path):
    s = R.by_key("spliceai")
    p = tmp_path / s.local_path
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x" * 10)
    r = D.check_local(s, root=str(tmp_path))
    assert r.status == D.PRESENT and r.size_bytes == 10


def test_local_cruft_detected(tmp_path):
    s = R.by_key("dbnsfp")
    p = tmp_path / s.local_path
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    (p.parent / "dbnsfp_full_index.parquet.OOMbak").write_bytes(b"stale")  # the real cruft we found
    r = D.check_local(s, root=str(tmp_path))
    assert r.status == D.CRUFT and "OOMbak" in r.detail


def test_scan_over_real_registry_no_network():
    # mock every probe so no real network; first-observation -> all probeable become 'changed'
    rep = D.scan({}, root="/nonexistent", probe=lambda s: ("fp", "mock"))
    assert len(rep["upstream"]) == len(R.all_sources())
    probeable_keys = {s.key for s in R.probeable()}
    changed_keys = {u.key for u in rep["changes"]}
    assert changed_keys == probeable_keys  # exactly the probeable sources flagged on first obs
    manual = [u for u in rep["upstream"] if u.status == D.MANUAL_SKIP]
    assert len(manual) == len(R.all_sources()) - len(probeable_keys)
