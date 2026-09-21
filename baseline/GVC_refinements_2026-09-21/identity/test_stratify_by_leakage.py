import json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from stratify_by_leakage import auroc, influence

HERE = Path(__file__).resolve().parent


class StratifyTests(unittest.TestCase):
    def test_auroc_with_ties_matches_hand_value(self):
        # pairs (pos, neg): (0.8,0.3)=1 (0.8,0.8)=0.5 (0.4,0.3)=1 (0.4,0.8)=0 -> 2.5/4
        self.assertAlmostEqual(auroc(np.array([1, 1, 0, 0]), np.array([0.8, 0.4, 0.3, 0.8])), 0.625)
        self.assertIsNone(auroc(np.array([1, 1]), np.array([0.1, 0.2])))

    def run_fixture(self, min_components):
        d = Path(tempfile.mkdtemp())
        pd.DataFrame({"variant_id": [f"v{i}" for i in range(1, 7)], "source_id": [str(i) for i in range(1, 7)]}).to_parquet(d / "c.parquet", index=False)
        pd.DataFrame({"variant_id": [f"v{i}" for i in range(1, 7)],
                      "partition": ["train", "validation", "validation", "validation", "test", "validation"]}).to_parquet(d / "m.parquet", index=False)
        pd.DataFrame({"variation_id": ["1", "2", "3", "4", "5"],
                      "component_genes_only": ["c1", "c1", "c2", "c3", "c3"]}).to_parquet(d / "k.parquet", index=False)
        pd.DataFrame({"variant_id": ["v2", "v3", "v4", "v6"], "label": [1, 0, 1, 0],
                      "a": [0.8, 0.3, 0.6, 0.2], "b": [0.9, 0.2, 0.5, 0.1]}).to_parquet(d / "p.parquet", index=False)
        subprocess.run([sys.executable, str(HERE / "stratify_by_leakage.py"), "--predictions", str(d / "p.parquet"),
                        "--components", str(d / "k.parquet"), "--cohort", str(d / "c.parquet"), "--membership", str(d / "m.parquet"),
                        "--contrast", "b", "a", "--n-boot", "200", "--min-components", str(min_components),
                        "--output", str(d / "o.json")], check=True, capture_output=True)
        with open(d / "o.json", encoding="utf-8") as fh:
            return json.load(fh)

    def test_strata_use_full_membership_span(self):
        r = self.run_fixture(30)
        self.assertEqual(r["strata"]["leaked"]["rows"], 1)      # v2: its component holds a training row
        self.assertEqual(r["strata"]["unseen"]["rows"], 3)      # v3; v4 (validation+test only); v6 (no component)
        self.assertEqual(r["rows_without_component"], 1)

    def test_too_few_components_makes_no_claim(self):
        c = self.run_fixture(30)["strata"]["leaked"]["contrasts"]["b - a"]
        self.assertFalse(c["interval_reliable"]); self.assertIsNone(c["excludes_zero"])
        c = self.run_fixture(1)["strata"]["leaked"]["contrasts"]["b - a"]
        self.assertTrue(c["interval_reliable"]); self.assertIsNotNone(c["excludes_zero"])


class InfluenceTests(unittest.TestCase):
    def test_attribution_and_leave_out_by_hand(self):
        # A: 3 rows mean 0.10 (sum 0.30); B: 1 row 0.02; C: 6 rows mean -0.01 (sum -0.06). N = 10.
        g = pd.DataFrame({"component": ["A"] * 3 + ["B"] + ["C"] * 6,
                          "gene_symbol": ["TTN"] * 3 + ["BRCA1"] + ["X"] * 6})
        delta = np.array([0.1] * 3 + [0.02] + [-0.01] * 6)
        r = influence(g, delta, 10)
        self.assertEqual([t["component"] for t in r["top"]], ["A", "C", "B"])   # ranked by |contribution|
        self.assertAlmostEqual(r["top"][0]["contribution"], 0.03)
        self.assertAlmostEqual(sum(t["contribution"] for t in r["top"]), delta.mean())
        self.assertAlmostEqual(r["top"][0]["share_of_stratum_delta"], 0.30 / 0.26)
        self.assertAlmostEqual(r["leave_out_point_estimates"]["without_top_1"], (0.26 - 0.30) / 7)
        self.assertIsNone(r["leave_out_point_estimates"]["without_top_3"])   # nothing left to average
        self.assertEqual(r["top"][0]["genes"], ["TTN"])
        # |contributions| 0.03, 0.006, 0.002 -> absolute shares bounded in [0, 1] and summing to 1
        shares = [t["share_of_absolute_contribution"] for t in r["top"]]
        self.assertAlmostEqual(sum(shares), 1.0)
        self.assertAlmostEqual(shares[0], 0.03 / 0.038)

    def test_absolute_share_stays_bounded_when_total_is_near_zero(self):
        g = pd.DataFrame({"component": ["A", "B"], "gene_symbol": ["x", "y"]})
        r = influence(g, np.array([0.5, -0.4999]), 10)
        self.assertGreater(abs(r["top"][0]["share_of_stratum_delta"]), 100)   # the signed share explodes
        self.assertLessEqual(max(t["share_of_absolute_contribution"] for t in r["top"]), 1.0)


if __name__ == "__main__":
    unittest.main()
