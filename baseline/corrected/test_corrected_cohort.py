"""Refusal tests at the ingestion/derivation boundary."""
import argparse
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_corrected_cohort import (
    ReviewEvidenceError, allele_state, build, nested_review_status,
)

REPO = None


def row(vid, sig, review, ref="A", alt="G"):
    return dict(variant_id=vid, clinical_sig=sig, ref=ref, alt=alt,
                metadata={"review_status": review}, gene_symbol="G1",
                allele_freq=0.01, consequence="missense_variant")


class CorrectedCohortTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self, rows, **kw):
        src = self.tmp / "in.parquet"
        pd.DataFrame(rows).to_parquet(src, index=False)
        out = self.tmp / "out"
        return build(src, REPO, out, **kw)

    def test_accounting_closes(self):
        m = self._build([row("a", "Pathogenic", "practice guideline"),
                         row("b", "Uncertain significance", "practice guideline")])
        self.assertEqual(m["n_input_universe"], 2)
        self.assertEqual(m["n_included"] + m["n_excluded"], 2)
        self.assertEqual(m["n_included"], 1)

    def test_unknown_vocabulary_refused_when_eligible(self):
        with self.assertRaises(ReviewEvidenceError):
            self._build([row("a", "Pathogenic", "some future status")])

    def test_unknown_vocabulary_tolerated_when_label_ineligible(self):
        m = self._build([row("a", "Uncertain significance", "some future status"),
                         row("b", "Pathogenic", "practice guideline")])
        self.assertEqual(m["n_included"], 1)

    def test_absent_review_evidence_excludes_not_fabricates(self):
        m = self._build([row("a", "Pathogenic", "-")])
        self.assertEqual(m["n_included"], 0)
        self.assertIn("review_tier_above_threshold", str(m["exclusion_reason_counts"]))

    def test_malformed_allele_named_not_coerced(self):
        self.assertEqual(allele_state("A", "<DEL>"), "non_acgt_allele")
        self.assertEqual(allele_state("", "A"), "empty_allele")
        self.assertEqual(allele_state(None, "A"), "non_string_allele")
        self.assertEqual(allele_state("A", "A"), "ref_equals_alt")
        self.assertEqual(allele_state("AT", "A"), "resolved")

    def test_malformed_allele_excluded_when_filter_enabled(self):
        m = self._build([row("a", "Pathogenic", "practice guideline", ref="A", alt="<DEL>")],
                        apply_allele_filter=True)
        self.assertEqual(m["n_included"], 0)
        self.assertIn("allele_non_acgt_allele", str(m["exclusion_reason_counts"]))

    def test_malformed_allele_recorded_but_included_by_default(self):
        m = self._build([row("a", "Pathogenic", "practice guideline", ref="A", alt="<DEL>")])
        self.assertEqual(m["n_included"], 1)
        self.assertFalse(m["apply_allele_filter"])
        self.assertEqual(m["n_included_with_unresolved_allele"], 1)
        self.assertIn("non_acgt_allele", m["included_allele_state_counts"])

    def test_metadata_shape_refused(self):
        with self.assertRaises(ValueError):
            nested_review_status(["not", "an", "object"])

    def test_metadata_json_string_accepted(self):
        text = chr(123) + chr(34) + "review_status" + chr(34) + ":" + chr(34) + "practice guideline" + chr(34) + chr(125)
        self.assertEqual(nested_review_status(text), "practice guideline")

    def test_refuses_to_overwrite(self):
        rows = [row("a", "Pathogenic", "practice guideline")]
        self._build(rows)
        src = self.tmp / "in.parquet"
        with self.assertRaises(FileExistsError):
            build(src, REPO, self.tmp / "out")

    def test_duplicate_identity_refused(self):
        with self.assertRaises(ValueError):
            self._build([row("a", "Pathogenic", "practice guideline"),
                         row("a", "Benign", "practice guideline")])

    def test_conflicting_classification_excluded(self):
        m = self._build([row("a", "Conflicting classifications of pathogenicity", "practice guideline"),
                         row("b", "Pathogenic", "practice guideline")])
        self.assertEqual(m["n_included"], 1)

    def test_tier_threshold_respected(self):
        m3 = self._build([row("a", "Pathogenic", "no assertion criteria provided")], max_review_tier=3)
        self.assertEqual(m3["n_included"], 0)
        shutil.rmtree(self.tmp / "out")
        m4 = self._build([row("a", "Pathogenic", "no assertion criteria provided")], max_review_tier=4)
        self.assertEqual(m4["n_included"], 1)

    def test_exact_case_sensitivity(self):
        m = self._build([row("a", "pathogenic", "practice guideline")])
        self.assertEqual(m["n_included"], 0)

    def test_policy_digests_recorded(self):
        m = self._build([row("a", "Pathogenic", "practice guideline")])
        for key in ("label_policy_sha256", "review_policy_sha256", "resolver_sha256",
                    "source_snapshot_sha256", "decision_table_sha256", "corrected_cohort_sha256"):
            self.assertRegex(m[key], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    args = p.parse_args()
    REPO = args.repo
    unittest.main(argv=["test_corrected_cohort"], verbosity=2)

