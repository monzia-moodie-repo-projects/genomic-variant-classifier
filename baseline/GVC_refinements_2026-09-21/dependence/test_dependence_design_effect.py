import unittest
import numpy as np
from dependence_design_effect import DependenceRatioError, dependence_design_effect


def clustered(rho, m=10, ng=1500, seed=2):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(ng), m).astype(str)
    v = np.repeat(rng.normal(0, np.sqrt(rho), ng), m) + rng.normal(0, np.sqrt(1 - rho), ng * m)
    return v, g


class DependenceTests(unittest.TestCase):
    def test_no_dependence_with_class_gap_returns_one(self):
        # The regime where the project's width ratio reported up to 2.55.
        rng = np.random.default_rng(1)
        n = 20000
        g = rng.integers(0, 2000, n).astype(str)
        y = rng.binomial(1, 0.19, n)
        v = rng.normal(0, 0.05, n) + 0.30 * y
        r = dependence_design_effect(np.mean, v, g, n_boot=2000)
        self.assertAlmostEqual(r["variance_design_effect"], 1.0, delta=0.12)

    def test_tracks_kish_closed_form(self):
        for rho in (0.05, 0.2, 0.5):
            v, g = clustered(rho)
            r = dependence_design_effect(np.mean, v, g, n_boot=3000)
            theory = 1 + (10 - 1) * rho
            self.assertAlmostEqual(r["variance_design_effect"] / theory, 1.0, delta=0.08,
                                   msg=f"rho={rho}")

    def test_monotone_in_dependence(self):
        des = [dependence_design_effect(np.mean, *clustered(r), n_boot=2000)["variance_design_effect"]
               for r in (0.0, 0.2, 0.5)]
        self.assertLess(des[0], des[1])
        self.assertLess(des[1], des[2])

    def test_deterministic(self):
        v, g = clustered(0.2)
        a = dependence_design_effect(np.mean, v, g, n_boot=500, seed=7)
        b = dependence_design_effect(np.mean, v, g, n_boot=500, seed=7)
        self.assertEqual(a["variance_design_effect"], b["variance_design_effect"])

    def test_refusals(self):
        v, g = clustered(0.2)
        with self.assertRaises(DependenceRatioError):
            dependence_design_effect(np.mean, v, g[:-1])
        with self.assertRaises(DependenceRatioError):
            dependence_design_effect(np.mean, np.array([1.0, np.nan]), np.array(["a", "b"]))
        with self.assertRaises(DependenceRatioError):
            dependence_design_effect(np.mean, v, g, n_boot=10)
        with self.assertRaises(DependenceRatioError):
            dependence_design_effect(np.mean, v, np.array(["same"] * v.size))
        with self.assertRaises(DependenceRatioError):
            dependence_design_effect(np.mean, np.ones(50), np.arange(50).astype(str))


if __name__ == "__main__":
    unittest.main()
