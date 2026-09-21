import json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path
from generate_gate_a_report import verdict

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"           # genuine outputs of the real scripts, shipped with the package


def args(src, out, override=None):
    names = {"mc-census": "mc.json", "geneinfo-census": "gi.json", "provenance": "prov.json",
             "authentication": "auth.json", "residuals": "res.json",
             "resolution": "resolution/resolution_summary.json", "components": "components/components_summary.json",
             "bundle": "BUNDLE_PROVENANCE.json", "strata-constraint": "strat_c.json", "strata-representation": "strat_r.json"}
    a = [sys.executable, str(HERE / "generate_gate_a_report.py")]
    for k, v in names.items():
        a += [f"--{k}", str((override or {}).get(k, src / v))]
    return a + ["--output", str(out)]


class ReportTests(unittest.TestCase):
    def test_verdict_mapping(self):
        self.assertEqual(verdict({"interval_reliable": True, "excludes_zero": True}), "detectable")
        self.assertEqual(verdict({"interval_reliable": True, "excludes_zero": False}), "not detectable")
        self.assertEqual(verdict({"interval_reliable": False, "excludes_zero": None}), "not assessed (too few components)")

    def test_generates_and_refuses_overwrite(self):
        out = Path(tempfile.mkdtemp()) / "r.md"
        subprocess.run(args(FIX, out), check=True, capture_output=True)
        text = out.read_text(encoding="utf-8")
        self.assertIn("omitted: delta not detectable", text)
        self.assertIn("registry `gene_symbol` strings of its rows (not resolved genes)", text)
        r = subprocess.run(args(FIX, out), capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0); self.assertIn("refusing to overwrite", r.stderr)

    def test_every_consistency_check_can_fail(self):
        cases = [
            ("res.json", lambda d: d.__setitem__("clinical_sig_disagreement_kinds", {}), "residuals classified 0"),
            ("gi.json", lambda d: d["provenance"].__setitem__("relation_rows", 3), "census wrote 3 relation rows"),
            ("components/components_summary.json",
             lambda d: d["genes_only"]["leakage_restricted"].__setitem__("validation_rows", 99), "evaluated validation rows"),
            ("components/components_summary.json",
             lambda d: d["genes_only"]["leakage_restricted"].__setitem__("validation_rows_sharing_component_with_train", 99),
             "leaked validation rows"),
            ("strat_r.json", lambda d: d.__setitem__("component_column", "component_all_resolved"), "different component columns"),
            ("strat_r.json", lambda d: d["strata"]["unseen"].__setitem__("rows", 99), "stratum unseen"),
        ]
        keys = {"res.json": "residuals", "gi.json": "geneinfo-census",
                "components/components_summary.json": "components", "strat_r.json": "strata-representation"}
        for fname, mutate, expect in cases:
            d = Path(tempfile.mkdtemp())
            j = json.loads((FIX / fname).read_text(encoding="utf-8")); mutate(j)
            bad = d / "bad.json"; bad.write_text(json.dumps(j), encoding="utf-8")
            out = d / "r.md"
            r = subprocess.run(args(FIX, out, {keys[fname]: bad}), capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0, f"check did not fire for {fname}")
            self.assertIn(expect, r.stderr, f"{fname}: {r.stderr}")
            self.assertFalse(out.exists())

    def test_consistent_fixtures_list_checks_and_provenance(self):
        out = Path(tempfile.mkdtemp()) / "r.md"
        subprocess.run(args(FIX, out), check=True, capture_output=True)
        text = out.read_text(encoding="utf-8")
        self.assertIn("## Cross-output consistency", text)
        self.assertIn("## Appendix: provenance of every figure", text)
        self.assertIn("| authentication | `clinical_sig` |", text)

    def test_missing_key_stops_and_writes_nothing(self):
        d = Path(tempfile.mkdtemp()); bad = d / "prov.json"
        j = json.loads((FIX / "prov.json").read_text(encoding="utf-8")); del j["cohort_in_vcf"]
        bad.write_text(json.dumps(j), encoding="utf-8")
        out = d / "r.md"
        r = subprocess.run(args(FIX, out, {"provenance": bad}), capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("provenance: missing key cohort_in_vcf", r.stderr)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
