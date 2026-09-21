import gzip, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import pandas as pd
from build_components import ContractError, NCBI, canonical_components

HERE = Path(__file__).resolve().parent


def fixture(d):
    gi = ["#tax_id\tGeneID\tSymbol\tLocusTag\tSynonyms\tdbXrefs\tchromosome\tmap_location\tdescription\ttype_of_gene"]
    for gid, sym, t in [("7273", "TTN", "protein-coding"), ("100506866", "TTN-AS1", "ncRNA"),
                        ("672", "BRCA1", "protein-coding"), ("4211", "MEIS1", "protein-coding"),
                        ("111258505", "HHC2:066588", "biological-region"), ("999", "GENEZ", "protein-coding")]:
        gi.append(f"9606\t{gid}\t{sym}\t-\t-\t-\t1\t-\tdesc\t{t}")
    with gzip.open(d / "gi.gz", "wt") as f:
        f.write("\n".join(gi) + "\n")
    rel = [("1", 7273), ("1", 100506866), ("2", 7273), ("3", 672), ("4", 4211), ("4", 111258505),
           ("5", 111258505), ("5", 999), ("6", 111258505), ("7", 555)]
    pd.DataFrame([dict(variation_id=v, source_symbol="s", source_gene_id=g, genes_in_record=1) for v, g in rel]
                 ).to_parquet(d / "rel.parquet", index=False)
    res = [(g, "current", g) for g in ["7273", "100506866", "672", "4211", "111258505", "999"]]
    res.append(("555", "discontinued_no_replacement", None))
    pd.DataFrame([dict(source_gene_id=g, source_symbol="s", state=s, resolved_gene_id=r) for g, s, r in res]
                 ).to_parquet(d / "res.parquet", index=False)
    pd.DataFrame({"variant_id": [f"var{i}" for i in range(1, 8)], "source_id": [str(i) for i in range(1, 8)]}
                 ).to_parquet(d / "cohort.parquet", index=False)
    parts = {1: "train", 2: "validation", 3: "validation", 4: "train", 5: "validation", 6: "test", 7: "test"}
    pd.DataFrame({"variant_id": [f"var{i}" for i in parts], "partition": list(parts.values())}
                 ).to_parquet(d / "mem.parquet", index=False)
    pd.DataFrame({"variant_id": ["var2", "var3", "var5"]}).to_parquet(d / "restrict.parquet", index=False)


class ComponentTests(unittest.TestCase):
    def test_ncbi_ids_link_composite_symbols(self):
        v, _ = canonical_components([("a", ["NCBIGene:7273", "NCBIGene:100506866"]), ("b", ["NCBIGene:7273"])], NCBI)
        self.assertEqual(v["a"], v["b"])

    def test_refusals(self):
        for bad in ([("v", "NCBIGene:1;NCBIGene:2")], [("v", [])], [("v", ["HGNC:5"])],
                    [("v", ["NCBIGene:1"]), ("v", ["NCBIGene:2"])]):
            with self.assertRaises(ContractError):
                canonical_components(bad, NCBI)

    def test_end_to_end_including_restricted_span(self):
        d = Path(tempfile.mkdtemp()); fixture(d)
        subprocess.run([sys.executable, str(HERE / "build_components.py"), "--relation", str(d / "rel.parquet"),
                        "--resolution", str(d / "res.parquet"), "--gene-info", str(d / "gi.gz"),
                        "--cohort", str(d / "cohort.parquet"), "--membership", str(d / "mem.parquet"),
                        "--restrict-to", str(d / "restrict.parquet"), "--output-dir", str(d / "out")],
                       check=True, capture_output=True)
        with open(d / "out" / "components_summary.json", encoding="utf-8") as fh:
            s = json.load(fh)
        self.assertEqual(s["all_resolved"]["components"], 3)
        self.assertEqual(s["genes_only"]["components"], 4)
        self.assertEqual(s["variants_left_without_gene_after_exclusion"], 1)
        for k, expect in (("all_resolved", 2), ("genes_only", 1)):
            self.assertEqual(s[k]["leakage_vs_old_registry"]["validation_rows_sharing_component_with_train"], expect)
            # Regression: the span must come from FULL membership, not the restricted rows.
            self.assertEqual(s[k]["leakage_restricted"]["validation_rows_sharing_component_with_train"], expect)


if __name__ == "__main__":
    unittest.main()
