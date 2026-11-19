"""Stage 1D-1c.2A (set "aw"): the computational-predictor source audit (owner ruling e755519d, 2026-11-17/18).

Synthetic tests of what scripts/measure_predictor_sources_1d1c2.py does with the real files (docs/census/
PREDICTOR_SOURCES_1D1c2.json is tested in test_predictor_sources_record.py), and of the append-only observation state
SOURCE_ROW_WITHOUT_VALUE. Author: Monzia Moodie."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]


def _script():
    spec = importlib.util.spec_from_file_location("measure_predictor_sources_1d1c2",
                                                  REPO / "scripts" / "measure_predictor_sources_1d1c2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _script()


# ------------------------------------------------------------------------------------- the appended observation state
def test_state_codes_are_append_only():
    from genomic_variant_classifier.information_sources import columns as K
    from genomic_variant_classifier.information_sources.observation import ObservationState as O
    from genomic_variant_classifier.information_sources.observation import SourceIdentity

    assert [s.value for s in K.STATE_CODES] == ["observed", "source_absent", "not_in_source", "unsupported_variant",
                                                "bad_mapping", "computation_failed", "input_unobserved",
                                                "source_row_without_value"]
    assert K.CODE[O.SOURCE_ROW_WITHOUT_VALUE] == 7
    # An append changes no sealed record: counts name codes 0-6 always and an appended state only where it occurs. Measured
    # 2026-11-18 without this rule: the re-measured AF_REPAIR_1D1c1.json and QUALIFIED_PREPARATION_1D1c1C.json gained a zero
    # count, and the canonical identity 9619b172 changed.
    sealed = [s.value for s in K.STATE_CODES[:7]]
    assert [s.value for s in K.COUNTED_STATES] == sealed
    col = K.ObservationColumn.from_rows("revel_score", SourceIdentity("dataset:revel_v1_3", "c" * 64),
                                        K.Population("p", ("a", "b")), {"a": (O.OBSERVED, 0.1), "b": (O.NOT_IN_SOURCE, None)})
    assert list(col.counts()) == sealed and sum(col.counts().values()) == 2

    def state_count_keys(x):
        if isinstance(x, dict):
            yield from ([sorted(x["state_counts"])] if "state_counts" in x else [])
            for v in x.values():
                yield from state_count_keys(v)
        elif isinstance(x, list):
            for v in x:
                yield from state_count_keys(v)

    for name in ("AF_REPAIR_1D1c1.json", "QUALIFIED_PREPARATION_1D1c1C.json", "PILOTS_1D1b2.json"):
        found = list(state_count_keys(json.loads((REPO / "docs" / "census" / name).read_text(encoding="utf-8"))))
        assert found and all(k == sorted(sealed) for k in found), name


def test_source_row_without_value_requires_the_digest_and_carries_no_value():
    from genomic_variant_classifier.information_sources import observation as O

    st = O.ObservationState.SOURCE_ROW_WITHOUT_VALUE
    assert O.SOURCE_RULE[st] is O.SourceRequirement.REQUIRED and st not in O.DERIVED_STATES
    with pytest.raises(ValueError, match="SHA-256"):
        O.absent("revel_score", "r1", st, O.SourceIdentity("dataset:revel_v1_3"))
    ok = O.absent("revel_score", "r1", st, O.SourceIdentity("dataset:revel_v1_3", "a" * 64))
    assert ok.value is None and not ok.is_observed
    with pytest.raises(ValueError, match="sentinel"):
        O.NumericObservation("revel_score", "r1", st, 0.5, O.SourceIdentity("dataset:revel_v1_3", "a" * 64))
    with pytest.raises(ValueError, match="derived"):
        O.absent("revel_pathogenic", "r1", st, O.SourceIdentity("derived:revel_pathogenic"))


def test_column_with_the_new_state_reconciles():
    from genomic_variant_classifier.information_sources import columns as K
    from genomic_variant_classifier.information_sources import observation as O

    pop = K.Population("p", ("a", "b", "c"))
    src = O.SourceIdentity("dataset:revel_v1_3", "b" * 64)
    col = K.ObservationColumn.from_rows("revel_score", src, pop, {
        "a": (O.ObservationState.OBSERVED, 0.5), "b": (O.ObservationState.SOURCE_ROW_WITHOUT_VALUE, None),
        "c": (O.ObservationState.NOT_IN_SOURCE, None)})
    assert col.counts()["source_row_without_value"] == 1 and col.counts()["observed"] == 1
    assert col.values[0] == 0.5 and np.isnan(col.values[1])
    d = K.derive("revel_pathogenic", [col])          # an unobserved root is never a positive flag
    assert d.column.values[0] == 1.0 and np.isnan(d.column.values[1]) and np.isnan(d.column.values[2])


# ------------------------------------------------------------------------------------------------- the parts' bytes
def _windows_text_writes(monkeypatch) -> None:
    """Path.write_text and every text-mode open end lines with CR LF when no newline is given, exactly as on Windows."""
    import builtins

    orig_write_text, orig_open = Path.write_text, builtins.open

    def write_text(self, data, encoding=None, errors=None, newline=None):
        return orig_write_text(self, data, encoding=encoding, errors=errors, newline="\r\n" if newline is None else newline)

    def text_open(file, mode="r", buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
        if any(c in mode for c in "wax+") and "b" not in mode and newline is None:
            newline = "\r\n"
        return orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(builtins, "open", text_open)
    monkeypatch.setattr(io, "open", text_open)


def test_parts_are_the_same_bytes_on_every_platform(tmp_path, monkeypatch):
    """The error class of the owner's V1 of 2026-11-18: every value re-measured identically, but the parts were written in
    text mode, so on Windows their lines ended in CR LF and the SHA-256 the record binds for each part differed."""
    _windows_text_writes(monkeypatch)
    probe = tmp_path / "probe.txt"
    probe.write_text("a\n", encoding="ascii")
    assert probe.read_bytes() == b"a\r\n"                       # the emulation is in force
    rec = {"part": "mane", "b": [1, 2.5], "a": {"y": "z"}}
    out = tmp_path / "mane.json"
    S._write(out, rec)
    raw = out.read_bytes()
    assert raw == S._canonical(rec) and b"\r" not in raw and raw.endswith(b"}\n")
    assert S._read_part(out) == (rec, raw)


@pytest.mark.parametrize("damage", ["crlf", "indent", "unsorted", "no_final_newline"])
def test_a_part_that_is_not_canonical_is_refused_before_it_is_used(tmp_path, damage):
    rec = {"part": "pilot", "inputs": {"rows": 3, "file": "x"}}
    good = S._canonical(rec)
    bad = {"crlf": good.replace(b"\n", b"\r\n"),
           "indent": (json.dumps(rec, indent=2, sort_keys=True) + "\n").encode("ascii"),
           "unsorted": (json.dumps(rec, indent=1) + "\n").encode("ascii"),
           "no_final_newline": good[:-1]}[damage]
    assert bad != good and json.loads(bad) == rec
    p = tmp_path / "pilot.json"
    p.write_bytes(bad)
    with pytest.raises(SystemExit, match="not a canonical part"):
        S.collect(REPO, [p], tmp_path / "record.json")
    with pytest.raises(SystemExit, match="not a canonical part"):            # the mane and revel steps read it first
        S._load_frame(p, tmp_path / "pilot_frame.parquet")
    assert not (tmp_path / "record.json").exists()


# ---------------------------------------------------------------------------------------------- the publisher scan
H = S.REVEL_COLUMNS
GOOD = [["1", "100", "200", "A", "G", "K", "R", "0.5", "ENST00000000001;ENST00000000002"],
        ["1", "101", ".", "C", "T", "S", "P", "0.901", "ENST00000000001"],
        ["1", "102", "202", "A", "T", "K", "I", ".", "ENST00000000003"],
        ["X", "5", "6", "G", "C", "A", "P", "1", "."],
        ["1", "100", "200", "A", "G", "K", "R", "0.2", "ENST00000000009"]]


def _zip(rows, header=H, member="revel_with_transcript_ids") -> bytes:
    b = io.StringIO(newline="")
    w = csv.writer(b, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as f:
        f.writestr(member, b.getvalue())
    return out.getvalue()


def test_scan_census_keeps_a_genuine_half_and_a_missing_score_distinct():
    census, chroms, tx, dec, bad, matched, unresolved37 = S.scan_revel(_zip(GOOD), {"1:200:A:G", "1:202:A:T", "X:6:G:C"},
                                                                       {"1:101:C:T", "1:100:A:G"})
    assert census["rows"] == 5 and census["score_missing"] == 1 and census["score_exactly_0_5"] == 1
    assert census["grch38_unresolved"] == 1 and census["no_transcript"] == 1
    assert census["transcript_ids_unversioned"] == 5 and census.get("transcript_ids_versioned", 0) == 0
    assert dict(chroms) == {"1": 4, "X": 1} and dict(tx) == {0: 1, 1: 3, 2: 1} and dict(dec) == {0: 1, 1: 2, 3: 1}
    assert [r[0] for r in matched["1:200:A:G"]] == [2, 6]          # row numbers: the header is row 1
    assert [r[1] for r in matched["1:200:A:G"]] == ["0.5", "0.2"]   # exact tokens, both rows kept
    assert matched["1:202:A:T"][0][1] is None and matched["X:6:G:C"][0][4] == ()
    assert "1:201:C:T" not in matched                               # an unresolved GRCh38 row never matches
    # ... but is found at its GRCh37 allele; a row whose GRCh38 position IS resolved is never reported there
    assert unresolved37 == {"1:101:C:T": ["0.901"]}


@pytest.mark.parametrize("field,value", [(7, "0.5x"), (7, "1.1"), (7, "-0.1"), (7, "nan"), (7, "1e-3"), (0, "M"),
                                         (1, "0"), (2, "x"), (3, "N"), (4, "AA"), (4, "A"), (8, "ENST1"),
                                         (8, "ENST00000000001;ENST00000000001")])
def test_scan_refuses_every_deviation(field, value):
    rows = [list(r) for r in GOOD]
    rows[0][field] = value
    with pytest.raises(SystemExit):
        S.scan_revel(_zip(rows), set())


def test_scan_refuses_a_changed_header_a_missing_member_and_a_ragged_row():
    with pytest.raises(SystemExit):
        S.scan_revel(_zip(GOOD, header=H[:-1] + ("x",)), set())
    with pytest.raises(SystemExit):
        S.scan_revel(_zip(GOOD, member="other"), set())
    rows = [list(r) for r in GOOD]
    rows[1] = rows[1] + ["extra"]
    with pytest.raises(Exception):
        S.scan_revel(_zip(rows), set())


# ------------------------------------------------------------------------------------------- the declared selection
def _rows():
    return [(2, "0.5", "K", "R", ("ENST00000000001",)), (3, "0.8", "K", "E", ("ENST00000000002",)),
            (4, None, "K", "R", ("ENST00000000003",))]


def test_selection_is_never_a_maximum_or_a_first_row():
    rows, amb = _rows(), "ambiguous_named_by_the_caller"
    assert S.select_scalar([], lambda r: True, amb) == "not_in_publisher_grch38_view"
    assert S.select_scalar(rows, lambda r: "ENST00000000001" in r[4], amb) == S.CANDIDATE == "candidate"  # a genuine 0.5
    assert S.select_scalar(rows, lambda r: "ENST00000000009" in r[4], amb) == "no_applicable_source_row"
    assert S.select_scalar(rows, lambda r: True, amb) == amb                                # never max(0.5, 0.8)
    assert S.select_scalar(rows, lambda r: (r[2], r[3]) == ("K", "R"), amb) == amb          # 0.5 against no value
    assert S.select_scalar(rows, lambda r: "ENST00000000003" in r[4], amb) == "applicable_row_without_value"
    same = [(2, "0.3", "K", "R", ("ENST00000000001",)), (5, "0.3", "K", "R", ("ENST00000000004",))]
    assert S.select_scalar(same, lambda r: True, amb) == S.CANDIDATE                       # agreeing rows
    # what a disagreement IS depends on the selection (owner ruling ace92e09, section 8)
    assert S.MANE_AMBIGUOUS == "source_conflict_same_transcript"
    assert S.CLINVAR_AMBIGUOUS == "context_unresolved_same_amino_acid_pair"


def test_clinvar_selection_matches_the_allele_and_the_amino_acid_pair_only():
    """Owner ruling ace92e09, section 4: the ClinVar-name selection uses the allele and the amino-acid pair; the RefSeq
    accession and the residue position are not matched (the publisher has neither), so a G->R row on ANY transcript is
    selected -- a candidate, never a verified transcript context -- and two such rows that disagree are an unresolved
    context, not a source conflict."""
    gr = lambda r: (r[2], r[3]) == ("G", "R")                                       # what the selection compares
    one_tx = [(2, "0.21", "G", "R", ("ENST00000000001",))]
    other_tx = [(2, "0.21", "G", "R", ("ENST00000000002",))]                        # Gly987Arg on another isoform
    assert S.select_scalar(one_tx, gr, S.CLINVAR_AMBIGUOUS) == S.select_scalar(other_tx, gr, S.CLINVAR_AMBIGUOUS) \
        == S.CANDIDATE
    two = one_tx + [(3, "0.88", "G", "R", ("ENST00000000002",))]
    assert S.select_scalar(two, gr, S.CLINVAR_AMBIGUOUS) == "context_unresolved_same_amino_acid_pair"


def _gtf(records) -> bytes:
    lines = []
    for chrom, kind, start, end, gene, gene_id, tx, tags in records:
        attrs = f'gene_id "{gene_id}"; transcript_id "{tx}"; gene_name "{gene}";' + "".join(f' tag "{t}";' for t in tags)
        lines.append("\t".join([chrom, "HAVANA", kind, str(start), str(end), ".", "+", ".", attrs]))
    return gzip.compress(("\n".join(lines) + "\n").encode("ascii"), mtime=0)


def test_locus_resolution_of_mane_is_independent_of_the_cohort_gene():
    """Owner ruling ace92e09, sections 4 and 8: the gene is resolved from the GRCh38 locus -- every MANE Select coding
    sequence covering the position, overlapping genes kept -- while the cohort-gene policy reads the cohort's
    gene_symbol (a ClinVar-derived field) and finds nothing for a symbol list."""
    gtf = _gtf([("chr1", "transcript", 100, 300, "A", "G1.1", "ENST00000000001.2", ("MANE_Select",)),
                ("chr1", "CDS", 100, 200, "A", "G1.1", "ENST00000000001.2", ("MANE_Select",)),
                ("chr1", "transcript", 150, 400, "B", "G2.1", "ENST00000000002.5", ("MANE_Select",)),
                ("chr1", "CDS", 150, 300, "B", "G2.1", "ENST00000000002.5", ("MANE_Select",)),
                ("chr1", "CDS", 120, 130, "B", "G2.1", "ENST00000000003.1", ("basic",)),
                ("chr2", "transcript", 400, 500, "C", "G3.1", "ENST00000000004.1", ("MANE_Select",)),
                ("chr2", "CDS", 400, 500, "C", "G3.1", "ENST00000000004.1", ("MANE_Select",))])
    tx = S._gtf_transcripts(gtf)
    assert tx["sequences"] == {"chr1", "chr2"}
    assert S.mane_cds_covering(tx, "1", 120) == {"ENST00000000001": "A"}          # a non-MANE CDS never counts
    assert S.mane_cds_covering(tx, "1", 170) == {"ENST00000000001": "A", "ENST00000000002": "B"}
    assert S.mane_cds_covering(tx, "1", 250) == {"ENST00000000002": "B"}           # the cohort may say "A" here
    assert S.mane_cds_covering(tx, "1", 350) == {} and S.mane_cds_covering(tx, "X", 1) == {}
    assert S.mane_cds_covering(tx, "2", 400) == S.mane_cds_covering(tx, "2", 500) == {"ENST00000000004": "C"}
    assert S.stable_mane_of_gene(tx, "A") == {"ENST00000000001"}
    assert S.stable_mane_of_gene(tx, "A;B") == set() == S.stable_mane_of_gene(tx, "") == S.stable_mane_of_gene(tx, "Z")


PARTITION = {"hist_absent_pub_absent": 1126527, "hist_absent_pub_present": 42861, "hist_0_5_pub_absent__fabricated": 115538,
             "hist_0_5_pub_has_0_5__genuine": 66, "hist_0_5_pub_other_values": 14175, "hist_value_pub_absent": 2,
             "hist_value_equals_pub_max": 188525, "hist_value_differs_from_pub_max": 2320}


def test_absence_semantics_are_derived_from_the_partition_and_reconcile():
    """Owner ruling ace92e09, section 3: 1,299,101 is NOT the number without a publisher score; 1,242,065 have none in
    the GRCh38 view and 57,036 have one. Every quantity is derived from the partition, and a partition that does not
    reconcile is refused."""
    from collections import Counter

    by_domain = {k: {"single_nucleotide_other_consequence": v} for k, v in PARTITION.items()}
    a = S.absence_semantics(Counter(PARTITION), by_domain, {}, 0)
    assert a["default_triggered_positives"] == 1299167 and a["pilot_unobserved"] == 1169388
    assert a["default_triggered_without_index_score_or_without_a_publisher_0_5_at_the_allele"] == 1299101
    assert a["of_which_no_publisher_score_in_the_grch38_view"] == 1242065
    assert a["of_which_a_publisher_score_exists_at_the_allele"] == 57036
    assert a["of_which_publisher_has_0_5_at_allele"] == 66
    assert a["nondefault_index_value_without_publisher_score_positivity_not_recorded"] == 2
    assert sum(a["no_publisher_score_in_the_grch38_view_by_domain"].values()) == 1242065
    bad = {**by_domain, "hist_absent_pub_absent": {"single_nucleotide_missense": 1}}
    with pytest.raises(SystemExit, match="domain partition"):
        S.absence_semantics(Counter(PARTITION), bad, {}, 0)
    with pytest.raises(SystemExit, match="unexpected partition"):
        S.absence_semantics(Counter({**PARTITION, "new_category": 1}), by_domain, {}, 0)


def test_clinvar_declared_substitution_from_the_name():
    lines = ["#AlleleID\tName\tAssembly\tChromosome\tVariationID\tPositionVCF\tReferenceAlleleVCF\tAlternateAlleleVCF",
             "1\tNM_000001.2(G1):c.10A>G (p.Lys4Arg)\tGRCh38\t1\t11\t200\tA\tG",
             "1\tNM_000001.2(G1):c.10A>G (p.Lys4Arg)\tGRCh37\t1\t11\t150\tA\tG",
             "2\tNM_000002.1(G2):c.5del (p.Leu2fs)\tGRCh38\t1\t12\t300\tCA\tC",
             "3\tNM_000003.4(G3):c.7C>T (p.Arg3Ter)\tGRCh38\t1\t13\t400\tC\tT",
             "4\tNC_000001.11:g.500A>G\tGRCh38\t1\t14\t500\tA\tG",
             "5\tNM_000005.1(G5):c.9G>A (p.Gly3Asp)\tGRCh38\t1\t15\t600\tG\tA",
             "5\tNM_000005.2(G5):c.9G>A (p.Gly3Asp)\tGRCh38\t1\t15\t600\tG\tA",
             "2\tNM_000002.1(G2):c.5del (p.Leu2fs)\tGRCh37\t1\t12\t250\tCA\tC",
             "2\tNM_000002.1(G2):c.5del (p.Leu2fs)\tGRCh37\t1\t12\t251\tCA\tC"]
    raw = gzip.compress(("\n".join(lines) + "\n").encode("ascii"))
    frame = pd.DataFrame({"source_id": ["11", "12", "13", "14", "15", "16"], "chrom": ["1"] * 6,
                          "pos": [200, 300, 400, 500, 600, 700], "ref": ["A", "CA", "C", "A", "G", "A"],
                          "alt": ["G", "C", "T", "G", "A", "C"]})
    status, transcript, substitution, grch37, census = S.clinvar_declared(raw, frame)
    assert status == ["declared", "no_single_residue_substitution", "no_single_residue_substitution",
                      "name_not_transcript_hgvs", "no_unique_grch38_record", "no_unique_grch38_record"]
    assert transcript[0] == "NM_000001.2" and substitution[0] == ("K", "R") and substitution[2] is None
    assert grch37 == ["1:150:A:G", None, None, None, None, None]     # VariationID 12 has two GRCh37 records: none
    assert census == {"rows": 9, "grch38_records_of_population_variants": 6, "keys_with_several_records": 1,
                      "grch37_records_of_population_variants": 3, "variation_ids_with_one_grch37_record": 1,
                      "variation_ids_with_several_grch37_records": 1}
