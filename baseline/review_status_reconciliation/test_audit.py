import argparse
import unittest
import pandas as pd
from audit_review import audit,load_policy,nested_status,representation
POLICY=None
class AuditTests(unittest.TestCase):
    def rows(self):
        return pd.DataFrame([
            dict(variant_id="a",clinical_sig="Pathogenic",ReviewStatus="",metadata={"review_status":"criteria provided, single submitter"},ref="AT",alt="A"),
            dict(variant_id="b",clinical_sig="Uncertain significance",ReviewStatus="",metadata={"review_status":"criteria provided, single submitter"},ref="A",alt="AT")])
    def test_review_count_is_not_final_count(self):
        s,_=audit(self.rows(),POLICY)
        self.assertEqual(s["n_review_only_changed"],2)
        self.assertEqual(s["n_final_eligibility_changed"],1)
    def test_production_case_sensitivity(self):
        df=self.rows();df.loc[0,"clinical_sig"]="pathogenic"
        s,_=audit(df,POLICY);self.assertEqual(s["n_nested_included"],0)
    def test_whitespace_strip(self):
        df=self.rows();df.loc[0,"clinical_sig"]=" Pathogenic "
        s,_=audit(df,POLICY);self.assertEqual(s["n_nested_included"],1)
    def test_unknown_eligible_refused(self):
        df=self.rows();df.at[0,"metadata"]={"review_status":"future status"}
        with self.assertRaises(ValueError):audit(df,POLICY)
    def test_unknown_ineligible_recorded(self):
        df=self.rows();df.at[1,"metadata"]={"review_status":"future status"}
        s,t=audit(df,POLICY);self.assertEqual(t.loc[1,"nested_path"],"UNKNOWN_VOCABULARY")
    def test_duplicate_ids(self):
        df=self.rows();df.loc[1,"variant_id"]="a"
        with self.assertRaises(ValueError):audit(df,POLICY)
    def test_missing_column(self):
        with self.assertRaises(ValueError):audit(self.rows().drop(columns="ReviewStatus"),POLICY)
    def test_unknown_metadata_shape(self):
        with self.assertRaises(ValueError):nested_status(["not","an","object"])
    def test_json(self):
        self.assertEqual(nested_status(chr(123)+chr(34)+"review_status"+chr(34)+":"+chr(34)+"practice guideline"+chr(34)+chr(125)),"practice guideline")
    def test_missing_marker(self):
        df=self.rows();df.at[0,"metadata"]={"review_status":"-"}
        s,_=audit(df,POLICY);self.assertEqual(s["n_final_eligibility_changed"],0)
    def test_frameshift_not_claimed(self):
        self.assertEqual(representation("AT","A"),"net_length_loss")
        self.assertEqual(representation(None,"A"),"unresolved")
        self.assertEqual(representation("A","<DEL>"),"unresolved")
    def test_order_invariance(self):
        a,_=audit(self.rows(),POLICY);b,_=audit(self.rows().iloc[::-1],POLICY)
        self.assertEqual(a,b)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True);args=p.parse_args()
    POLICY=load_policy(args.repo)
    unittest.main(argv=["test_audit"],verbosity=2)
