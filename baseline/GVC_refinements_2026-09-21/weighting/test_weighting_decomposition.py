import unittest
import numpy as np
from weighting_decomposition import WeightingError, decompose, per_group_contributions


def build(sizes, effects, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    g, v = [], []
    for i, (n, e) in enumerate(zip(sizes, effects)):
        g += [f"G{i}"] * int(n)
        v += list(rng.normal(e, noise, int(n)))
    return np.array(v), np.array(g)


class WeightingTests(unittest.TestCase):
    def test_identity_exact_on_arbitrary_data(self):
        rng = np.random.default_rng(9)
        for _ in range(20):
            sizes = rng.integers(1, 500, rng.integers(3, 60))
            v, g = build(sizes, rng.normal(0, 0.03, sizes.size), seed=int(rng.integers(1e6)))
            r = decompose(v, g)
            self.assertTrue(np.isclose(r["gap"], r["G_times_cov"], rtol=1e-9, atol=1e-12))

    def test_equal_sizes_means_no_disagreement(self):
        v, g = build([100] * 50, np.linspace(-0.05, 0.05, 50))
        r = decompose(v, g)
        self.assertAlmostEqual(r["gap"], 0.0, places=12)
        # Correlation with a constant is undefined: must be None with a reason,
        # never a silent NaN.
        self.assertIsNone(r["spearman_size_vs_delta"])
        self.assertEqual(r["spearman_undefined_reason"], "all groups have equal size")

    def test_reversal_localised_to_large_groups(self):
        rng = np.random.default_rng(3)
        sizes = np.concatenate([rng.integers(2000, 20000, 15), rng.integers(1, 40, 1800)])
        v, g = build(sizes, np.where(sizes > 1000, 0.02, -0.015), seed=3)
        r = decompose(v, g)
        self.assertGreater(r["variant_weighted"], 0)
        self.assertLess(r["group_weighted"], 0)
        bins = r["by_size_bin"]
        self.assertLess(bins[0]["row_weighted_delta"], 0)
        self.assertGreater(bins[-1]["row_weighted_delta"], 0)

    def test_bins_partition_rows_exactly(self):
        rng = np.random.default_rng(4)
        v, g = build(rng.integers(1, 300, 200), rng.normal(0, 0.02, 200), seed=4)
        r = decompose(v, g)
        self.assertEqual(sum(b["rows"] for b in r["by_size_bin"]), r["n_rows"])
        self.assertEqual(sum(b["n_groups"] for b in r["by_size_bin"]), r["n_groups"])

    def test_refusals(self):
        v, g = build([10, 10], [0.0, 0.1])
        with self.assertRaises(WeightingError):
            decompose(v, g[:-1])
        with self.assertRaises(WeightingError):
            decompose(np.array([0.1, np.nan]), np.array(["a", "b"]))
        with self.assertRaises(WeightingError):
            decompose(np.array([0.1, 0.2]), np.array(["a", "a"]))
        with self.assertRaises(WeightingError):
            decompose(np.array([]), np.array([]))


class ReversalAndContributionTests(unittest.TestCase):
    def test_difference_without_reversal(self):
        # The reviewer's counterexample: effects 1 and 2, unequal sizes.
        v, g = build([10, 90], [1.0, 2.0], noise=0.0)
        r = decompose(v, g)
        self.assertTrue(r["means_differ"])
        self.assertFalse(r["sign_reversal"])

    def test_reversal_flagged(self):
        rng = np.random.default_rng(3)
        sizes = np.concatenate([rng.integers(2000, 20000, 15), rng.integers(1, 40, 1800)])
        v, g = build(sizes, np.where(sizes > 1000, 0.02, -0.015), seed=3)
        self.assertTrue(decompose(v, g)["sign_reversal"])

    def test_contributions_sum_to_gap(self):
        rng = np.random.default_rng(8)
        v, g = build(rng.integers(1, 400, 150), rng.normal(0, 0.03, 150), seed=8)
        per = per_group_contributions(v, g)
        r = decompose(v, g)
        self.assertTrue(np.isclose(per["contribution"].sum(), r["gap"], atol=1e-12))


if __name__ == "__main__":
    unittest.main()
