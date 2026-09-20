import unittest
from migration import SplitRegistry,confirmation_screen,cohort_cells,brier_factorial

class MigrationTests(unittest.TestCase):
    def setUp(self): self.registry=SplitRegistry("policy1","fixed-salt",(7,1,2),{"known":"test"})
    def test_preserves_known(self):
        r=self.registry.extend(["known","new"]);self.assertEqual(r.assignments["known"],"test")
    def test_order_independent(self):
        self.assertEqual(self.registry.extend(["a","b"]).sha256,self.registry.extend(["b","a"]).sha256)
    def test_extension_stable(self):
        self.assertEqual(self.registry.extend(["a"]).extend(["b"]).sha256,self.registry.extend(["a","b"]).sha256)
    def test_no_mutation(self):
        self.registry.extend(["new"]);self.assertNotIn("new",self.registry.assignments)
        with self.assertRaises(TypeError):self.registry.assignments["new"]="test"
    def test_reject_unknown_partition(self):
        with self.assertRaises(ValueError):SplitRegistry("p","s",(7,1,2),{"g":"holdout_typo"})
    def test_reject_empty_group(self):
        with self.assertRaises(ValueError):self.registry.extend([""])
    def test_test_feedback_blocks(self):
        r=confirmation_screen([{"variant_id":"v","group_id":"g"}],[{"variant_id":"v","group_id":"g","use":"test_feedback"}])
        self.assertFalse(r[0]["passes_recorded_exposure_screen"])
    def test_new_variant_exposed_gene(self):
        r=confirmation_screen([{"variant_id":"new","group_id":"g"}],[{"variant_id":"old","group_id":"g","use":"exploration"}])
        self.assertEqual(r[0]["blockers"],["previously_exposed_group"])
    def test_no_known_overlap(self):
        self.assertTrue(confirmation_screen([{"variant_id":"v","group_id":"g"}],[])[0]["passes_recorded_exposure_screen"])
    def test_membership(self):
        self.assertEqual(cohort_cells(["a","b"],["b","c"],["a","b","c","d"]),
                         {"common":["b"],"added":["c"],"removed":["a"],"excluded_both":["d"]})
    def test_membership_outside_universe(self):
        with self.assertRaises(ValueError):cohort_cells(["a"],["b"],["a"])
    def test_factorial(self):
        rows=[dict(variant_id="a",evaluation_cell="common",label=0,legacy_probability=.2,corrected_probability=.1),
              dict(variant_id="b",evaluation_cell="added",label=1,legacy_probability=.2,corrected_probability=.8)]
        r=brier_factorial(rows)
        self.assertAlmostEqual(r["delta_common"],-.03)
        self.assertAlmostEqual(r["delta_added"],-.60)
        self.assertAlmostEqual(r["interaction"],-.57)
    def test_factorial_invalid(self):
        with self.assertRaises(ValueError):brier_factorial([])
    def test_duplicate_members(self):
        with self.assertRaises(ValueError):cohort_cells(["a","a"],[],["a"])
if __name__=="__main__":unittest.main()
