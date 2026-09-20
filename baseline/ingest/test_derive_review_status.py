"""Acceptance tests at the ingestion/derivation boundary."""
import argparse
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from derive_review_status import (
    ReconciliationFailure, canonical_review_status, derive, join_derived_status,
)

REPO = None


def row(vid, review, sig="Pathogenic", meta=...):
    m = {"review_status": review, "rs_id": 1} if meta is ... else meta
    return dict(variant_id=vid, clinical_sig=sig, ref="A", alt="G",
                gene_symbol="G1", metadata=m)


class DerivationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _derive(self, rows, name="out.parquet", **kw):
        src = self.tmp / "in.parquet"
        pd.DataFrame(rows).to_parquet(src, index=False)
        return derive(src, REPO, self.tmp / name, **kw)

    def test_unmatched_join_refuses_never_fabricates(self):
        vcf_map = {"1:100:A:G": "criteria provided, single submitter"}
        self.assertEqual(join_derived_status("1:100:A:G", vcf_map),
                         "criteria provided, single submitter")
        with self.assertRaises(ReconciliationFailure):
            join_derived_status("1:200:AT:A", vcf_map)

    def test_absent_metadata_refuses(self):
        with self.assertRaises(ReconciliationFailure):
            canonical_review_status(None)

    def test_metadata_without_review_key_refuses(self):
        with self.assertRaises(ReconciliationFailure):
            canonical_review_status({"rs_id": 123})

    def test_absent_metadata_fails_whole_derivation(self):
        with self.assertRaises(ReconciliationFailure):
            self._derive([row("a", "practice guideline"), row("b", None, meta=None)])

    def test_round_trip_preserves_status_and_provenance(self):
        rows = [row("a", "practice guideline"),
                row("b", "criteria provided, single submitter"),
                row("c", "-")]
        report = self._derive(rows)
        back = pd.read_parquet(self.tmp / "out.parquet")
        self.assertEqual(list(back["ReviewStatus"]),
                         ["practice guideline", "criteria provided, single submitter", "-"])
        for meta, status in zip(back["metadata"], back["ReviewStatus"]):
            self.assertEqual(meta["review_status"], status)
        prov = json.loads(Path(str(self.tmp / "out.parquet") + ".derivation.json").read_text())
        for key in ("policy_digest", "resolver_sha256", "source_cohort_sha256", "output_sha256"):
            self.assertRegex(prov[key], r"^[0-9a-f]{64}$")
        self.assertEqual(prov["canonical_source"], "metadata.review_status")

    def test_stale_existing_column_detected_not_trusted(self):
        rows = [row("a", "practice guideline"), row("b", "criteria provided, single submitter")]
        for r in rows:
            r["ReviewStatus"] = ""
        with self.assertRaises(ReconciliationFailure) as ctx:
            self._derive(rows)
        self.assertIn("disagrees with the canonical record", str(ctx.exception))

    def test_agreeing_existing_column_passes(self):
        rows = [row("a", "practice guideline"), row("b", "criteria provided, single submitter")]
        for r in rows:
            r["ReviewStatus"] = r["metadata"]["review_status"]
        report = self._derive(rows)
        self.assertTrue(report["existing_column_present"])
        self.assertEqual(report["existing_disagrees_rows"], 0)

    def test_deliberate_overwrite_allowed_and_recorded(self):
        rows = [row("a", "practice guideline")]
        rows[0]["ReviewStatus"] = ""
        report = self._derive(rows, overwrite_existing_column=True)
        self.assertEqual(report["existing_disagrees_rows"], 1)
        back = pd.read_parquet(self.tmp / "out.parquet")
        self.assertEqual(back.loc[0, "ReviewStatus"], "practice guideline")

    def test_unknown_vocabulary_fails_closed(self):
        with self.assertRaises(Exception) as ctx:
            self._derive([row("a", "some future status"), row("b", "practice guideline")])
        self.assertIn("some future status", str(ctx.exception))

    def test_recognised_missing_token_is_not_unknown(self):
        report = self._derive([row("a", "-"), row("b", "practice guideline")])
        self.assertEqual(report["n_rows"], 2)

    def test_refuses_to_overwrite_output(self):
        rows = [row("a", "practice guideline")]
        self._derive(rows)
        src = self.tmp / "in.parquet"
        with self.assertRaises(FileExistsError):
            derive(src, REPO, self.tmp / "out.parquet")

    def test_json_string_metadata_accepted(self):
        text = (chr(123) + chr(34) + "review_status" + chr(34) + ":" + chr(34)
                + "practice guideline" + chr(34) + chr(125))
        self.assertEqual(canonical_review_status(text), "practice guideline")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    args = p.parse_args()
    REPO = args.repo
    unittest.main(argv=["test_derive_review_status"], verbosity=2)
