"""test_registry.py -- Monzia Moodie

Integrity tests for monitoring/registry.py (the data-source single-source-of-truth). These enforce the
invariants that keep it honest: keys unique+lowercase; a probe is configured iff a real URL exists (no
fabricated URLs -- the no-guessing rule); an ACTIVE source has a local asset; critical_assets() is exactly
the ACTIVE local paths and all are repo-relative.
"""
from genomic_variant_classifier.monitoring import registry as R


def test_keys_unique_and_lowercase():
    keys = [s.key for s in R.all_sources()]
    assert len(keys) == len(set(keys)), "duplicate registry keys"
    assert all(k == k.lower() and k.strip() == k for k in keys)


def test_corpus_is_comprehensive():
    # the registry must span the whole corpus, not the old 4-source hardcoded set
    assert len(R.all_sources()) >= 20
    must_have = {"clinvar", "dbnsfp", "spliceai", "alphamissense", "gnomad",
                 "gnomad_constraint", "lovd", "string", "uniprot", "finngen", "dbsnp"}
    assert must_have.issubset({s.key for s in R.all_sources()})


def test_probe_invariant_no_fabricated_urls():
    # check == MANUAL  <=>  upstream_url is None  (a MANUAL source must NOT carry a guessed URL)
    for s in R.all_sources():
        if s.check is R.Check.MANUAL:
            assert s.upstream_url is None, f"{s.key}: MANUAL but has a URL"
        else:
            assert s.upstream_url, f"{s.key}: {s.check} but no URL"


def test_probeable_are_exactly_non_manual_with_url():
    pr = R.probeable()
    assert pr, "expected at least one probeable source"
    assert all(s.check is not R.Check.MANUAL and s.upstream_url for s in pr)
    assert "clinvar" in {s.key for s in pr} and "alphamissense" in {s.key for s in pr}


def test_active_sources_have_local_assets():
    for s in R.by_verdict(R.Verdict.ACTIVE):
        assert s.local_path, f"{s.key}: ACTIVE but no local_path"


def test_critical_assets_are_active_local_paths_repo_relative():
    ca = R.critical_assets()
    assert ca == [s.local_path for s in R.by_verdict(R.Verdict.ACTIVE) if s.local_path]
    for p in ca:
        assert p.startswith("data"), f"{p}: not repo-relative"
        assert "\\" not in p and not p.startswith("/"), f"{p}: must use forward slashes / be relative"


def test_by_key_roundtrip_and_keyerror():
    assert R.by_key("clinvar").name == "ClinVar"
    try:
        R.by_key("no_such_source")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_enum_membership_valid():
    for s in R.all_sources():
        assert isinstance(s.category, R.Category)
        assert isinstance(s.verdict, R.Verdict)
        assert isinstance(s.check, R.Check)


def test_known_stale_items_are_flagged_in_notes():
    # the audit found these; the registry must record them so they aren't silently forgotten
    assert "v4.0" in R.by_key("gnomad").notes and "STALE" in R.by_key("gnomad").notes
    assert "TYPO" in R.by_key("finngen").notes and "R12" in R.by_key("finngen").notes
    assert "DUP" in R.by_key("spliceai").notes
