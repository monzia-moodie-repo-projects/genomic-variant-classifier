"""The dbNSFP historical-index reader after the containment merge (pull request #82): one normalisation, chromosome reuse,
Arrow representations, row multiplicity and source failures, each through the public interfaces.

Added 2026-11-22 (the read follow-up the owner's containment ruling of 2026-11-21 asked for as a separate commit):

* ONE normalisation. A stored row belongs to a requested chromosome exactly when ``_index_chrom`` (the function both public
  interfaces apply to the query) maps its stored label to the requested name. The reader applies it to every DISTINCT
  stored label and reads the matching rows with a LITERAL set of stored spellings, which row-group statistics can prune.
* Chromosome reuse: one connector answers chr1 -> chr2 -> chr1 and the reverse, with stored and absent alleles interleaved,
  exactly as an independent oracle built from the fixture's own rows.
* The four Arrow chromosome representations (string, large_string, and dictionaries of each), with and without column
  statistics and across several row groups.
* A frame lookup preserves the caller's order, duplicates and index (also a non-unique one), and equals the scalar lookup.
* Every failure of a configured source raises ``DbNSFPSourceError`` with the underlying exception as its cause -- a missing
  file, a directory, a malformed or truncated parquet, a read that fails part-way, a non-numeric score -- while a valid
  absent allele returns the defaults, through both interfaces and through the pipeline's annotation step.
"""
from __future__ import annotations

import gzip

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from genomic_variant_classifier.data import dbnsfp
from genomic_variant_classifier.data.dbnsfp import DbNSFPConnector, DbNSFPSourceError

SCORES = ["sift_score", "polyphen2_score", "revel_score", "cadd_phred", "phylop_score", "gerp_score"]

# (stored label, canonical chromosome, position, ref, alt): every stored spelling of chromosome 1, 2, X and MT, spread over
# several row groups. The REVEL value of row i is i / 100, so every answer names the row it came from.
ROWS = [
    ("1", "1", 100, "A", "C"), ("chr1", "1", 200, "G", "T"), ("cHr1", "1", 300, "C", "A"), (" 1 ", "1", 400, "T", "G"),
    ("2", "2", 100, "A", "C"), ("chr2", "2", 250, "G", "A"), ("X", "X", 100, "A", "G"), ("chrX", "X", 500, "C", "T"),
    ("MT", "MT", 100, "A", "G"), ("chrM", "MT", 200, "T", "C"), ("M", "MT", 300, "G", "A"), ("1", "1", 500, "C", "G"),
]


def _revel(i: int) -> float:
    return round(i / 100, 2)


ORACLE = {(canon, pos, ref, alt): _revel(i) for i, (_, canon, pos, ref, alt) in enumerate(ROWS)}


def _chrom_array(labels: list[str], kind: str) -> pa.Array:
    if kind == "string":
        return pa.array(labels, type=pa.string())
    if kind == "large_string":
        return pa.array(labels, type=pa.large_string())
    values = sorted(set(labels))
    indices = pa.array([values.index(label) for label in labels], type=pa.int32())
    value_type = pa.string() if kind == "dictionary_string" else pa.large_string()
    return pa.DictionaryArray.from_arrays(indices, pa.array(values, type=value_type))


def _write_index(path, kind: str = "string", rows=ROWS, *, row_group_size: int = 4, statistics: bool = True,
                 revel=None) -> None:
    n = len(rows)
    table = pa.table({
        "chrom": _chrom_array([r[0] for r in rows], kind),
        "pos": pa.array([r[2] for r in rows], type=pa.int64()),
        "ref": pa.array([r[3] for r in rows], type=pa.string()),
        "alt": pa.array([r[4] for r in rows], type=pa.string()),
        "sift_score": pa.array([0.25] * n), "polyphen2_score": pa.array([0.75] * n),
        "revel_score": pa.array(revel if revel is not None else [_revel(i) for i in range(n)]),
        "cadd_phred": pa.array([20.0] * n), "phylop_score": pa.array([1.5] * n), "gerp_score": pa.array([-0.5] * n),
    })
    pq.write_table(table, path, row_group_size=row_group_size, write_statistics=statistics)


def _query(keys):
    return pd.DataFrame({"chrom": [k[0] for k in keys], "pos": [k[1] for k in keys],
                         "ref": [k[2] for k in keys], "alt": [k[3] for k in keys]})


def _expected(canon_keys):
    return [ORACLE.get(k, dbnsfp.DEFAULT_REVEL) for k in canon_keys]


# ---------------------------------------------------------------------------
# Chromosome reuse, every Arrow representation, with and without statistics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["string", "large_string", "dictionary_string", "dictionary_large_string"])
@pytest.mark.parametrize("statistics", [True, False])
def test_one_connector_answers_every_chromosome_order_like_the_oracle(tmp_path, kind, statistics):
    path = tmp_path / "index.parquet"
    _write_index(path, kind, statistics=statistics)
    conn = DbNSFPConnector(path)
    stored_1 = [("1", 100, "A", "C"), ("1", 200, "G", "T"), ("1", 300, "C", "A"), ("1", 400, "T", "G"), ("1", 500, "C", "G")]
    absent_1 = [("1", 100, "A", "T"), ("1", 999, "C", "G")]
    stored_2 = [("2", 100, "A", "C"), ("2", 250, "G", "A")]
    absent_2 = [("2", 100, "A", "G")]
    interleaved_1 = [stored_1[0], absent_1[0], stored_1[1], absent_1[1], stored_1[2], stored_1[3], stored_1[4]]
    interleaved_2 = [absent_2[0], stored_2[0], stored_2[1]]
    for order in ([interleaved_1, interleaved_2, interleaved_1], [interleaved_2, interleaved_1, interleaved_2]):
        for keys in order:
            got = conn.annotate_dataframe(_query(keys))["revel_score"].tolist()
            assert got == _expected(keys)
            assert [conn.get_scores(*k).revel_score for k in keys] == _expected(keys)


@pytest.mark.parametrize("kind", ["string", "large_string", "dictionary_string", "dictionary_large_string"])
def test_every_stored_spelling_answers_every_query_spelling(tmp_path, kind):
    path = tmp_path / "index.parquet"
    _write_index(path, kind)
    conn = DbNSFPConnector(path)
    spellings = {"1": ["1", "chr1", "CHR1", " 1 "], "X": ["X", "x", "chrX", "chrx"], "MT": ["MT", "M", "chrM", "chrMT", "mt"]}
    for canon, queries in spellings.items():
        keys = [k for k in ORACLE if k[0] == canon]
        for q in queries:
            frame = _query([(q, p, r, a) for _, p, r, a in keys])
            assert conn.annotate_dataframe(frame)["revel_score"].tolist() == [ORACLE[k] for k in keys], (kind, q)
            assert [conn.get_scores(q, p, r, a).revel_score for _, p, r, a in keys] == [ORACLE[k] for k in keys], (kind, q)


def test_a_frame_spanning_all_chromosomes_reads_each_row_once(tmp_path):
    path = tmp_path / "index.parquet"
    _write_index(path, "dictionary_large_string", row_group_size=3)
    keys = list(ORACLE)
    got = DbNSFPConnector(path).annotate_dataframe(_query(keys))
    assert got["revel_score"].tolist() == [ORACLE[k] for k in keys]
    for col, value in (("sift_score", 0.25), ("polyphen2_score", 0.75), ("cadd_phred", 20.0), ("phylop_score", 1.5),
                       ("gerp_score", -0.5)):
        assert (got[col] == value).all()


def test_the_read_skips_row_groups_of_other_chromosomes(tmp_path, monkeypatch):
    """The literal spelling set is prunable: of three row groups only the one whose label range holds "2" is planned.

    Each row group holds TWO labels, so its statistics are a range (as in the real index, whose three groups span '10'-'19',
    '1'-'5' and '5'-'Y'): an expression over the normalised column cannot be decided on a range and keeps all three groups
    (measured); with a single label per group Arrow would decide even the expression, and the test would prove nothing."""
    import pyarrow.dataset as ds

    rows = [(c, c, 100 + i, "A", "C") for c in ("1", "10", "2", "20", "X", "Y") for i in range(2)]
    path = tmp_path / "index.parquet"
    _write_index(path, "string", rows=rows, row_group_size=4)
    seen = []
    real = pq.read_table

    def spy(source, *args, **kwargs):
        seen.append(kwargs.get("filters"))
        return real(source, *args, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    assert DbNSFPConnector(path).get_scores("chr2", 101, "A", "C").revel_score == _revel(5)
    assert pq.read_metadata(path).num_row_groups == 3
    predicate = [f for f in seen if f is not None]
    assert len(predicate) == 1
    fragment = next(ds.dataset(path, format="parquet").get_fragments())
    assert len(fragment.split_by_row_group(predicate[0])) == 1


def test_a_label_the_normaliser_does_not_leave_fixed_is_found_by_its_own_spelling(tmp_path):
    """"chrchr1" normalises to "CHR1", which normalises to "1". With ONE normalisation (the read filter, the stored keys and the
    query each through _index_chrom once) the row answers its own spelling in both interfaces and never answers "1". On main the
    read filter (an Arrow expression), the twice-normalised requested names and the frame join's twice-normalised stored keys
    disagreed, so this stored row was unreachable: queried as "chrchr1", both interfaces returned the defaults (measured)."""
    rows = [("chrchr1", "CHR1", 100, "A", "C"), ("1", "1", 100, "A", "C")]
    path = tmp_path / "index.parquet"
    _write_index(path, "string", rows=rows, revel=[0.11, 0.22])
    conn = DbNSFPConnector(path)
    assert conn.get_scores("chrchr1", 100, "A", "C").revel_score == 0.11
    assert conn.get_scores("1", 100, "A", "C").revel_score == 0.22
    assert conn.annotate_dataframe(_query([("chrchr1", 100, "A", "C"), ("1", 100, "A", "C")]))["revel_score"].tolist() == [0.11, 0.22]


# ---------------------------------------------------------------------------
# Order, duplicates and the caller's index: frame equals scalar
# ---------------------------------------------------------------------------

def test_frame_keeps_order_duplicates_and_a_non_unique_index_and_equals_scalar(tmp_path):
    path = tmp_path / "index.parquet"
    _write_index(path, "large_string")
    keys = [("chr2", 250, "g", "a"), ("1", 100, "A", "C"), ("chr2", 250, "G", "A"), ("MT", 999, "A", "G"),
            ("chrM", 200, "T", "C"), ("1", 100, "A", "C")]
    frame = _query(keys)
    frame.index = pd.Index([7, 7, 3, 3, 0, 7], name="row")
    frame["carried"] = list("abcdef")
    before = frame.copy()
    conn = DbNSFPConnector(path)
    got = conn.annotate_dataframe(frame)
    assert got.index.equals(frame.index)
    assert got["carried"].tolist() == list("abcdef")
    scalar = [conn.get_scores(*k).to_dict() for k in keys]
    for col in SCORES:
        assert got[col].tolist() == [s[col] for s in scalar]
    canon = [("2", 250, "G", "A"), ("1", 100, "A", "C"), ("2", 250, "G", "A"), ("MT", 999, "A", "G"), ("MT", 200, "T", "C"),
             ("1", 100, "A", "C")]
    assert got["revel_score"].tolist() == _expected(canon)
    pd.testing.assert_frame_equal(frame, before)                 # the caller's frame is not changed


# ---------------------------------------------------------------------------
# Source failures: one type, the cause kept, distinguishable from a lookup miss
# ---------------------------------------------------------------------------

def _both_interfaces_raise(conn, cause=None, match=None):
    for call in (lambda: conn.get_scores("1", 100, "A", "C"),
                 lambda: conn.annotate_dataframe(_query([("1", 100, "A", "C")]))):
        with pytest.raises(DbNSFPSourceError, match=match) as err:
            call()
        if cause is not None:
            assert isinstance(err.value.__cause__, cause), repr(err.value.__cause__)


def test_a_valid_absent_allele_is_a_miss_not_a_failure(tmp_path):
    path = tmp_path / "index.parquet"
    _write_index(path)
    conn = DbNSFPConnector(path)
    assert conn.get_scores("1", 100, "A", "T") == dbnsfp._DEFAULT_SCORES
    assert conn.get_scores("17", 1, "A", "T") == dbnsfp._DEFAULT_SCORES
    got = conn.annotate_dataframe(_query([("1", 100, "A", "T"), ("17", 1, "A", "T")]))
    assert got["revel_score"].tolist() == [dbnsfp.DEFAULT_REVEL] * 2


@pytest.mark.parametrize("name", ["index.parquet", "raw.tsv", "raw.tsv.gz"])
def test_a_missing_configured_source_raises_with_its_cause(tmp_path, name):
    _both_interfaces_raise(DbNSFPConnector(tmp_path / name), FileNotFoundError, "not a readable file")


def test_a_directory_is_not_a_source(tmp_path):
    folder = tmp_path / "index.parquet"
    folder.mkdir()
    _both_interfaces_raise(DbNSFPConnector(folder), FileNotFoundError, "not a readable file")


def test_a_malformed_parquet_raises(tmp_path):
    path = tmp_path / "index.parquet"
    path.write_bytes(b"this is not a parquet file at all" * 10)
    _both_interfaces_raise(DbNSFPConnector(path), pa.ArrowException, "Cannot read configured historical parquet")


def test_a_truncated_parquet_raises(tmp_path):
    good = tmp_path / "good.parquet"
    _write_index(good)
    blob = good.read_bytes()
    for cut in (len(blob) // 2, len(blob) - 1, 4):
        path = tmp_path / f"cut_{cut}.parquet"
        path.write_bytes(blob[:cut])
        _both_interfaces_raise(DbNSFPConnector(path), pa.ArrowException, "Cannot read configured historical parquet")


def test_a_read_that_fails_after_a_valid_footer_raises(tmp_path):
    """The footer and schema are intact; the column data are not. The read fails part-way, after the footer was accepted."""
    good = tmp_path / "good.parquet"
    _write_index(good, row_group_size=len(ROWS))
    blob = bytearray(good.read_bytes())
    meta = pq.read_metadata(good)
    column = meta.row_group(0).column(4)                      # sift_score's data page
    start = column.data_page_offset
    blob[start:start + column.total_compressed_size] = b"\xff" * column.total_compressed_size
    path = tmp_path / "corrupt.parquet"
    path.write_bytes(bytes(blob))
    assert pq.read_metadata(path).num_rows == len(ROWS)      # the footer still reads
    _both_interfaces_raise(DbNSFPConnector(path), (OSError, pa.ArrowException), "Cannot read configured historical parquet")


def test_a_non_numeric_historical_score_raises(tmp_path):
    path = tmp_path / "index.parquet"
    _write_index(path)
    frame = pd.read_parquet(path)
    frame["revel_score"] = frame["revel_score"].astype(str)
    frame.loc[0, "revel_score"] = "not-a-number"
    frame.to_parquet(path, index=False)
    _both_interfaces_raise(DbNSFPConnector(path), ValueError, "Non-numeric historical score in revel_score")


def test_a_truncated_raw_source_raises(tmp_path):
    content = ("#chr\tpos(1-based)\tref\talt\tREVEL_score\n" + "".join(f"1\t{100 + i}\tA\tC\t0.{i % 10}\n" for i in range(2000)))
    blob = gzip.compress(content.encode())
    path = tmp_path / "raw.tsv.gz"
    path.write_bytes(blob[: len(blob) // 2])
    _both_interfaces_raise(DbNSFPConnector(path), (OSError, EOFError), "Cannot parse configured dbNSFP source")


def test_the_pipeline_annotation_step_surfaces_a_source_failure(tmp_path):
    """The consumer that writes model inputs: a configured, missing source stops the annotation; it never becomes defaults."""
    from genomic_variant_classifier.data.real_data_prep import AnnotationConfig, DataPrepPipeline

    pipeline = DataPrepPipeline(annotation_config=AnnotationConfig(dbnsfp_path=tmp_path / "absent.parquet",
                                                                   esm2_allow_network=False))
    cohort = pd.DataFrame({"chrom": ["1"], "pos": [100], "ref": ["A"], "alt": ["C"]})
    with pytest.raises(DbNSFPSourceError, match="not a readable file") as err:
        pipeline._annotate_scores(cohort)
    assert isinstance(err.value.__cause__, FileNotFoundError)


def test_the_source_error_is_still_a_value_error():
    """Callers that caught ValueError for a refused source keep doing so; the cause chain carries the detail."""
    assert issubclass(DbNSFPSourceError, ValueError)
