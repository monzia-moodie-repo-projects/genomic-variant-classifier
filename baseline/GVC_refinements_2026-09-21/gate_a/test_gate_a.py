import unittest
from gate_a import *
class Tests(unittest.TestCase):
 def setUp(self):
  self.r=GeneResolver([Gene('NCBIGene:1',9606,'LOC1',('old',)),Gene('NCBIGene:2',9606,'ABC',('old',),'HGNC:2')])
 def test_loc_retained(self):self.assertEqual(self.r.resolve(symbol='LOC1')['gene_id'],'NCBIGene:1')
 def test_no_loc_guess(self):self.assertEqual(self.r.resolve(symbol='LOC999')['state'],'unresolved')
 def test_alias_review(self):self.assertEqual(self.r.resolve(symbol='old')['state'],'review_required')
 def test_source_conflict(self):
  with self.assertRaises(GateError):self.r.resolve(source_gene_id='NCBIGene:1',symbol='ABC')
 def test_direct_id(self):self.assertEqual(self.r.resolve(source_gene_id='NCBIGene:1')['state'],'resolved')
 def test_nonhuman_refused(self):
  with self.assertRaises(GateError):GeneResolver([Gene('NCBIGene:1',10090,'A')])
 def test_crossref_conflict(self):
  with self.assertRaises(GateError):GeneResolver([Gene('NCBIGene:1',9606,'A',(),'HGNC:1'),Gene('NCBIGene:2',9606,'B',(),'HGNC:1')])
 def test_so_id(self):self.assertEqual(normalize_consequence(so_id='SO:0001587',raw_term='nonsense',active_terms={'SO:0001587':'stop_gained'},obsolete_ids=set())['canonical_term'],'stop_gained')
 def test_label_only_not_zero(self):self.assertEqual(normalize_consequence(so_id=None,raw_term='nonsense',active_terms={},obsolete_ids=set())['state'],'needs_mapping')
 def test_obsolete_refused(self):
  with self.assertRaises(GateError):normalize_consequence(so_id='SO:0001587',raw_term='nonsense',active_terms={},obsolete_ids={'SO:0001587'})
 def test_covariance_not_reversal(self):
  r,_=weighting_attribution([1,2,2],['a','b','b']);self.assertGreater(r['gap'],0);self.assertFalse(r['sign_reversal'])
 def test_exact_attribution(self):
  r,t=weighting_attribution([-2,1,1,1],['a','b','b','b']);self.assertTrue(r['sign_reversal']);self.assertAlmostEqual(t.gap_contribution.sum(),r['gap'])
if __name__=='__main__':unittest.main()
