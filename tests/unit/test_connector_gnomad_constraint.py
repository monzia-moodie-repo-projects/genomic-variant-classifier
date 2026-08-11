"""Tests for the repaired gnomAD constraint connector -- DUPLICATE-1A Part B.

Every fixture reproduces a structure MEASURED in
`gnomad.v4.1.constraint_metrics.tsv` on 2026-08-09, not an invented one.

Author: Monzia Moodie
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import genomic_variant_classifier.data.connectors.connector_gnomad_constraint as C
from genomic_variant_classifier.data.connectors.connector_gnomad_constraint import (
    CONSTRAINT_COLS, CONSTRAINT_DEFAULTS, CONSTRAINT_INDEX_SCHEMA_VERSION,
    ConstraintScores, GnomADConstraintConnector,
)
from genomic_variant_classifier.data.constraint_canonicalize import (
    ConstraintSourceError,
)


def _src_row(gene, gene_id, transcript, pli, oe, loeuf, obs, exp,
             mane="true", canonical=None, syn=0.5, mis=0.5):
    return {"gene": gene, "gene_id": gene_id, "transcript": transcript,
            "mane_select": mane,
            "canonical": mane if canonical is None else canonical,
            "lof.pLI": pli, "lof.oe": oe, "lof.oe_ci.upper": loeuf,
            "lof.obs": obs, "lof.exp": exp,
            "syn.z_score": syn, "mis.z_score": mis}


def write_source(tmp_path, rows=None):
    """A miniature of the real file: namespace pairs, a canonical-only gene,
    a gene with no declared transcript, and a gene whose point estimate
    exceeds its reported upper bound (DNMT3A is real: lof.exp 106.98)."""
    if rows is None:
        rows = [
            _src_row("ZZZ3", "26009", "NM_015534.6", 0.10, 0.5, 0.72, 40.0, 80.0),
            _src_row("ZZZ3", "ENSG00000036549", "ENST00000370801",
                     0.10, 0.5, 0.72, 40.0, 80.0),
            _src_row("DNMT3A", "1788", "NM_022552.5",
                     0.99, 2.4957, 1.998, 267.0, 106.98),
            _src_row("DNMT3A", "ENSG00000119772", "ENST00000321117",
                     0.99, 2.4957, 1.998, 267.0, 106.98),
            _src_row("CANONONLY", "ENSG00000000002", "ENST2",
                     0.20, 0.5, 0.90, 10.0, 20.0, mane="false", canonical="true"),
            _src_row("NOTIER", "ENSG00000000004", "ENST3",
                     0.30, 0.5, 1.10, 5.0, 10.0, mane="false", canonical="false"),
        ]
    p = tmp_path / "gnomad.v4.1.constraint_metrics.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


def _cohort(genes):
    return pd.DataFrame({"gene_symbol": list(genes),
                         "chrom": ["chr1"] * len(genes)})


# ---- the defect this commit exists to repair -----------------------------
def test_gene_constraint_oe_is_the_POINT_ESTIMATE_not_loeuf(tmp_path):
    """DUPLICATE-1. The two columns were bit-identical because nothing ever
    produced gene_constraint_oe and engineer_features fell back to loeuf."""
    c = GnomADConstraintConnector(write_source(tmp_path))
    out = c.annotate_dataframe(_cohort(["ZZZ3"]))
    assert float(out.loc[0, "gene_constraint_oe"]) == 0.5      # lof.oe
    assert float(out.loc[0, "loeuf"]) == 0.72                  # lof.oe_ci.upper
    assert out.loc[0, "gene_constraint_oe"] != out.loc[0, "loeuf"]


def test_the_two_columns_are_not_identical_across_the_whole_index(tmp_path):
    c = GnomADConstraintConnector(write_source(tmp_path))
    out = c.annotate_dataframe(_cohort(["ZZZ3", "DNMT3A", "CANONONLY"]))
    assert not out["gene_constraint_oe"].equals(out["loeuf"]), (
        "gene_constraint_oe is still an alias of loeuf")


# ---- missingness is preserved, never fabricated ---------------------------
def test_defaults_are_NaN_not_biological_values():
    assert all(np.isnan(v) for v in CONSTRAINT_DEFAULTS.values())
    s = ConstraintScores()
    assert all(np.isnan(getattr(s, c)) for c in CONSTRAINT_COLS)


def test_unmatched_gene_gets_NaN_not_one_point_zero(tmp_path):
    """A LOEUF of 1.0 asserts complete tolerance of loss of function. An
    unmatched gene has made no such measurement."""
    c = GnomADConstraintConnector(write_source(tmp_path))
    out = c.annotate_dataframe(_cohort(["NOT_IN_GNOMAD"]))
    for col in CONSTRAINT_COLS:
        assert pd.isna(out.loc[0, col]), "{} was fabricated".format(col)


def test_stub_mode_yields_NaN_for_every_feature(tmp_path):
    for connector in (GnomADConstraintConnector(None),
                      GnomADConstraintConnector(tmp_path / "absent.tsv")):
        out = connector.annotate_dataframe(_cohort(["ZZZ3"]))
        for col in CONSTRAINT_COLS:
            assert pd.isna(out.loc[0, col])


def test_missing_gene_symbol_column_yields_NaN_columns(tmp_path):
    c = GnomADConstraintConnector(write_source(tmp_path))
    out = c.annotate_dataframe(pd.DataFrame({"chrom": ["chr1"]}))
    for col in CONSTRAINT_COLS:
        assert col in out.columns and pd.isna(out.loc[0, col])


# ---- transcript selection -------------------------------------------------
def test_canonical_only_gene_is_annotated(tmp_path):
    """696 real genes reach the index only through the canonical tier."""
    c = GnomADConstraintConnector(write_source(tmp_path), allow_canonical_fallback=True)
    out = c.annotate_dataframe(_cohort(["CANONONLY"]))
    assert float(out.loc[0, "loeuf"]) == 0.90


def test_gene_with_no_declared_transcript_is_NOT_annotated(tmp_path):
    """21 real genes have neither flag. Missing beats an arbitrary value."""
    c = GnomADConstraintConnector(write_source(tmp_path))
    out = c.annotate_dataframe(_cohort(["NOTIER"]))
    assert pd.isna(out.loc[0, "loeuf"])


def test_disabling_canonical_fallback_drops_those_genes(tmp_path):
    c = GnomADConstraintConnector(write_source(tmp_path), allow_canonical_fallback=False)
    out = c.annotate_dataframe(_cohort(["CANONONLY", "ZZZ3"]))
    assert pd.isna(out.loc[0, "loeuf"])
    assert float(out.loc[1, "loeuf"]) == 0.72


def test_selection_does_not_depend_on_source_row_order(tmp_path):
    """`drop_duplicates(keep='first')` -- what this replaces -- fails outright.
    Measured: first-row selection disagrees with MANE on 31.3% of genes."""
    rows = [
        _src_row("ZZZ3", "26009", "NM_1", 0.1, 0.5, 0.72, 40.0, 80.0),
        _src_row("ZZZ3", "ENSG00000036549", "ENST1", 0.1, 0.5, 0.72, 40.0, 80.0),
    ]
    a = GnomADConstraintConnector(write_source(tmp_path / "a", rows)) \
        if False else None
    (tmp_path / "fwd").mkdir(); (tmp_path / "rev").mkdir()
    p1 = write_source(tmp_path / "fwd", rows)
    p2 = write_source(tmp_path / "rev", list(reversed(rows)))
    o1 = GnomADConstraintConnector(p1).annotate_dataframe(_cohort(["ZZZ3"]))
    o2 = GnomADConstraintConnector(p2).annotate_dataframe(_cohort(["ZZZ3"]))
    for col in CONSTRAINT_COLS:
        assert o1.loc[0, col] == o2.loc[0, col], "{} depends on row order".format(col)


# ---- values are validated, never coerced ----------------------------------
def test_point_estimate_above_two_is_NOT_clipped(tmp_path):
    """DNMT3A: lof.oe 2.4957 against a reported upper bound of 1.998, with
    lof.exp 106.98 -- well powered, arithmetically sound (267/106.98). The old
    parser clipped loeuf to [0, 5], a coercion that had never fired."""
    c = GnomADConstraintConnector(write_source(tmp_path))
    out = c.annotate_dataframe(_cohort(["DNMT3A"]))
    assert abs(float(out.loc[0, "gene_constraint_oe"]) - 2.4957) < 1e-9
    assert float(out.loc[0, "gene_constraint_oe"]) > float(out.loc[0, "loeuf"])


def test_out_of_range_pli_RAISES_rather_than_clipping(tmp_path):
    rows = [_src_row("BAD", "ENSG00000000005", "ENST5", 1.4, 0.5, 0.7, 4.0, 8.0)]
    c = GnomADConstraintConnector(write_source(tmp_path, rows))
    try:
        c.annotate_dataframe(_cohort(["BAD"]))
    except ConstraintSourceError as exc:
        assert "outside the published range" in str(exc)
        return
    raise AssertionError("an out-of-range pLI was silently clipped")


def test_broken_obs_over_exp_identity_RAISES(tmp_path):
    rows = [_src_row("BAD", "ENSG00000000006", "ENST6", 0.5, 0.9, 1.0, 40.0, 80.0)]
    c = GnomADConstraintConnector(write_source(tmp_path, rows))
    try:
        c.annotate_dataframe(_cohort(["BAD"]))
    except ConstraintSourceError as exc:
        assert "obs/exp identity" in str(exc)
        return
    raise AssertionError("lof.oe inconsistent with lof.obs/lof.exp was accepted")


# ---- CACHEIDENTITY-1 ------------------------------------------------------
def test_cache_is_written_with_an_identity_sidecar(tmp_path):
    p = write_source(tmp_path)
    GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    meta = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.meta.json"
    assert meta.exists()
    d = json.loads(meta.read_text())
    assert d["schema_version"] == CONSTRAINT_INDEX_SCHEMA_VERSION
    assert len(d["source_sha256"]) == 64
    assert "audit" in d


def test_a_cache_from_a_DIFFERENT_source_is_rejected(tmp_path):
    """The heart of CACHEIDENTITY-1. The old key was the source FILENAME, so a
    sidecar built by the defective parser was preferred to a repaired one --
    correct source code, old semantics."""
    p = write_source(tmp_path)
    GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    cache = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.parquet"
    # Poison the cache, then change the source so its digest no longer matches.
    poisoned = pd.read_parquet(cache)
    poisoned["loeuf"] = 99.0
    poisoned.to_parquet(cache, index=False)
    rows = [_src_row("ZZZ3", "ENSG00000036549", "ENST1", 0.1, 0.5, 0.72, 40.0, 80.0),
            _src_row("NEW", "ENSG00000000007", "ENST7", 0.2, 0.4, 0.66, 8.0, 20.0)]
    write_source(tmp_path, rows)
    out = GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    assert float(out.loc[0, "loeuf"]) == 0.72, "a stale cache was served"


def test_a_cache_with_no_sidecar_is_rejected(tmp_path):
    p = write_source(tmp_path)
    GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    meta = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.meta.json"
    cache = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.parquet"
    poisoned = pd.read_parquet(cache); poisoned["loeuf"] = 99.0
    poisoned.to_parquet(cache, index=False)
    meta.unlink()
    out = GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    assert float(out.loc[0, "loeuf"]) == 0.72, "an unverified cache was served"


def test_a_cache_from_an_older_schema_version_is_rejected(tmp_path):
    p = write_source(tmp_path)
    GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    meta = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.meta.json"
    cache = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.parquet"
    d = json.loads(meta.read_text()); d["schema_version"] = 1
    meta.write_text(json.dumps(d))
    poisoned = pd.read_parquet(cache); poisoned["loeuf"] = 99.0
    poisoned.to_parquet(cache, index=False)
    out = GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    assert float(out.loc[0, "loeuf"]) == 0.72, "an old-schema cache was served"


def test_a_matching_cache_IS_reused(tmp_path, monkeypatch):
    """The rejection tests would pass trivially if the cache were never used.

    REUSE IS PROVEN BY COUNTING PARSER INVOCATIONS, not by editing the cache.
    An earlier version wrote loeuf=0.4242 into the parquet and asserted the
    edited value came back -- which proved reuse and simultaneously proved that
    a corrupted cache with valid metadata was trusted. Byte integrity now
    rejects that, correctly, so the property must be demonstrated without
    tampering.
    """
    calls = {"n": 0}
    real = C.canonicalize_mane_constraint

    def counting(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(C, "canonicalize_mane_constraint", counting)
    p = write_source(tmp_path)

    first = GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    assert calls["n"] == 1, "the first build did not invoke the canonicaliser"

    second = GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    assert calls["n"] == 1, (
        "a VALID cache was ignored -- the canonicaliser ran twice; the "
        "rejection tests prove nothing if the cache is never reused")
    assert float(second.loc[0, "loeuf"]) == float(first.loc[0, "loeuf"])


def test_cache_identity_includes_the_canonical_fallback_POLICY(tmp_path):
    """THE BLOCKER, measured 2026-08-09.

    allow_canonical_fallback changes which genes exist in the index, so it is
    part of the cache IDENTITY. Without it: same source digest, same schema
    version, same policy string -- and a connector explicitly requesting
    MANE-only received the MANE-plus-canonical index, returning 0.9 for a
    canonical-tier gene where the requested policy demands NaN.

    The pre-existing test_disabling_canonical_fallback_drops_those_genes does
    NOT catch this: it builds a fresh source per call and never exercises the
    TRANSITION between policies.
    """
    p = write_source(tmp_path)

    enabled = GnomADConstraintConnector(p, allow_canonical_fallback=True)
    out_enabled = enabled.annotate_dataframe(_cohort(["CANONONLY"]))
    assert float(out_enabled.loc[0, "loeuf"]) == 0.90

    disabled = GnomADConstraintConnector(p, allow_canonical_fallback=False)
    out_disabled = disabled.annotate_dataframe(_cohort(["CANONONLY"]))
    assert pd.isna(out_disabled.loc[0, "loeuf"]), (
        "a cache built under allow_canonical_fallback=True was reused under "
        "the MANE-only policy: two different transcript-selection sciences "
        "sharing one cached index")

    # ... and the reverse transition, so neither direction can pass by luck.
    back = GnomADConstraintConnector(p, allow_canonical_fallback=True)
    assert float(back.annotate_dataframe(_cohort(["CANONONLY"])).loc[0, "loeuf"]) == 0.90


def test_a_cache_whose_BYTES_were_edited_is_rejected(tmp_path):
    """Identity and integrity are different contracts. A cache with valid
    metadata and altered bytes passed the identity check alone."""
    p = write_source(tmp_path)
    GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    cache = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.parquet"
    tampered = pd.read_parquet(cache)
    tampered["loeuf"] = 0.4242
    tampered.to_parquet(cache, index=False)
    out = GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    assert float(out.loc[0, "loeuf"]) == 0.72, (
        "a cache with edited bytes was served despite valid metadata")


def test_the_sidecar_records_integrity_fields(tmp_path):
    p = write_source(tmp_path)
    GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    meta = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.meta.json"
    d = json.loads(meta.read_text())
    assert len(d["cache_sha256"]) == 64
    assert d["n_rows"] > 0
    assert "loeuf" in d["columns"] and "gene_constraint_oe" in d["columns"]
    assert d["allow_canonical_fallback"] is True
    assert "canonicalization_policy" in d


def test_an_interrupted_publication_never_leaves_a_manifest_without_its_data(tmp_path,
                                                                             monkeypatch):
    """CRASH ORDER, simulated. Data is moved into place FIRST, manifest LAST.

    Swapping the two os.replace calls is invisible in any run that completes,
    which is why a sabotage mutation of the order went undetected. The property
    only shows under interruption: with the correct order a crash can leave a
    cache with NO manifest -- which the loader rejects -- but never a manifest
    vouching for bytes that were never written.
    """
    real_replace = C.os.replace
    state = {"n": 0}

    def crash_after_first(src, dst):
        state["n"] += 1
        if state["n"] == 1:
            return real_replace(src, dst)
        raise OSError("simulated interruption between publication steps")

    monkeypatch.setattr(C.os, "replace", crash_after_first)
    p = write_source(tmp_path)
    try:
        GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    except OSError:
        pass
    monkeypatch.undo()

    cache = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.parquet"
    meta = tmp_path / "gnomad.v4.1.constraint_metrics.constraint_index.meta.json"
    assert not (meta.exists() and not cache.exists()), (
        "an interrupted publication left a MANIFEST with no cache: the "
        "publication order is wrong")

    # And whatever survived, the next run must produce correct values.
    out = GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    assert float(out.loc[0, "loeuf"]) == 0.72


def test_publication_leaves_no_temporary_files(tmp_path):
    """Data is moved into place first and the manifest last, so an interrupted
    write can leave a cache with no manifest -- rejected -- but never a
    manifest vouching for bytes that were never written."""
    p = write_source(tmp_path)
    GnomADConstraintConnector(p).annotate_dataframe(_cohort(["ZZZ3"]))
    leftovers = sorted(x.name for x in tmp_path.iterdir() if x.name.endswith(".tmp"))
    assert leftovers == [], "temporary publication files survived: {}".format(leftovers)


# ---- coverage accounting --------------------------------------------------
def test_get_scores_returns_NaN_for_an_unknown_gene(tmp_path):
    c = GnomADConstraintConnector(write_source(tmp_path))
    s = c.get_scores("NOT_A_GENE")
    assert all(np.isnan(getattr(s, col)) for col in CONSTRAINT_COLS)


def test_audit_records_the_policy_INVARIANT_source_facts(tmp_path):
    """AUDITCOUNT-1. `tier_counts["no_declared_transcript"]` is a SELECTION
    quantity: it reported 1 with the canonical fallback enabled and 2 with it
    disabled, for the same source, because a gene that HAS a canonical
    transcript was folded in whenever policy declined to use it.

    A test asserting that count would pin the defect as correct in a second
    file. The source facts are what a connector consumer should depend on.
    """
    c = GnomADConstraintConnector(write_source(tmp_path))
    c.annotate_dataframe(_cohort(["ZZZ3"]))
    facts = c.audit["source_facts"]
    assert facts["n_gene_symbols"] == 4          # ZZZ3, DNMT3A, CANONONLY, NOTIER
    assert facts["n_with_mane_select"] == 2      # ZZZ3, DNMT3A
    assert facts["n_without_declared_transcript"] == 1   # NOTIER, both policies

    sel = c.audit["selection"]
    assert sel["allow_canonical_fallback"] is True
    assert sel["n_selected_canonical"] == 1      # CANONONLY


def test_audit_records_the_tier_counts(tmp_path):
    c = GnomADConstraintConnector(write_source(tmp_path))
    c.annotate_dataframe(_cohort(["ZZZ3"]))
    assert c.audit is not None
    tiers = c.audit["tier_counts"]
    assert tiers["mane_select"] == 2          # ZZZ3, DNMT3A
    assert tiers["canonical"] == 1            # CANONONLY
    # RETAINED, but read as what it is: a SELECTION count under the policy this
    # call ran with, not a property of the gnomAD file. The invariant assertion
    # lives in test_audit_records_the_policy_INVARIANT_source_facts above.
    assert tiers["no_declared_transcript"] == 1   # NOTIER, under fallback=True


def test_coverage_is_measured_by_INDEX_MEMBERSHIP_not_by_metric_value(tmp_path):
    """The old form counted `pli_score != 0.0`, so a gene with a GENUINE pLI of
    exactly zero was logged as a miss. The cached index holds 259 such genes,
    so every historical coverage figure was understated.

    ZEROPLI has pli_score exactly 0.0 and IS in the index. It must count as a
    match, and an absent gene must not.
    """
    # ASYMMETRIC BY CONSTRUCTION. An earlier fixture had one zero-pLI gene and
    # one absent gene, and the two errors CANCELLED: `pli_score != 0.0` excludes
    # the genuine zero and -- because NaN != 0.0 is True in pandas -- includes
    # the absent one. Net count identical, mutation undetected. Two zero-pLI
    # genes against one absent breaks the symmetry.
    rows = [
        _src_row("ZEROPLI_A", "ENSG00000000008", "ENST8", 0.0, 0.5, 0.72, 40.0, 80.0),
        _src_row("ZEROPLI_B", "ENSG00000000009", "ENST9", 0.0, 0.4, 0.66, 8.0, 20.0),
        _src_row("ZZZ3", "ENSG00000036549", "ENST1", 0.9, 0.5, 0.72, 40.0, 80.0),
    ]
    c = GnomADConstraintConnector(write_source(tmp_path, rows))
    out = c.annotate_dataframe(_cohort(["ZEROPLI_A", "ZEROPLI_B", "ZZZ3", "ABSENT"]))
    assert float(out.loc[0, "pli_score"]) == 0.0
    assert pd.isna(out.loc[3, "pli_score"])
    cov = c.coverage
    assert cov["n_rows"] == 4
    assert cov["n_matched"] == 3, (
        "genes with a genuine pLI of 0.0 were counted as unmatched, or an "
        "absent gene was counted as matched: {}".format(cov))
    assert cov["n_unmatched"] == 1
    assert cov["n_genes_in_index"] == 3


def main() -> int:
    import tempfile, pathlib
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = []
    for name, fn in tests:
        try:
            if "tmp_path" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(pathlib.Path(td))
            else:
                fn()
            print("  PASS  {}".format(name))
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print("  FAIL  {}  {}".format(name, exc))
    print("\n  {} passed, {} failed, {} total".format(
        len(tests) - len(failures), len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
