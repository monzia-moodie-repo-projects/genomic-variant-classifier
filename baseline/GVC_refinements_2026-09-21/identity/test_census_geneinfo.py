import gzip, tempfile, unittest
from pathlib import Path
from census_geneinfo import census, parse_geneinfo

ROWS = ["1\t100\t11\tA\tG\t.\t.\tGENEINFO=TTN-AS1:100506866|TTN:7273",
        "1\t200\t12\tA\tG\t.\t.\tGENEINFO=BRCA1:672",
        "1\t300\t13\tA\tG\t.\t.\tCLNSIG=Benign",
        "1\t400\t14\tA\tG\t.\t.\tGENEINFO=BAD_NO_ID"]


class GeneInfoTests(unittest.TestCase):
    def setUp(self):
        self.vcf = Path(tempfile.mkdtemp()) / "f.vcf.gz"
        with gzip.open(self.vcf, "wt") as f:
            f.write("#h\n" + "\n".join(ROWS) + "\n")

    def test_every_gene_kept_with_its_id(self):
        self.assertEqual(parse_geneinfo("GENEINFO=TTN-AS1:100506866|TTN:7273"),
                         [("TTN-AS1", 100506866), ("TTN", 7273)])

    def test_states_partition(self):
        _, s = census(self.vcf)
        self.assertEqual((s["records_with_geneinfo"], s["records_without_geneinfo"],
                          s["records_with_malformed_geneinfo"]), (2, 1, 1))

    def test_relation_rows_one_per_association(self):
        rows, _ = census(self.vcf)
        self.assertEqual(len(rows), 3)

    def test_symbol_containing_colon_parsed_by_last_colon(self):
        self.assertEqual(parse_geneinfo("GENEINFO=MEIS1:4211|HHC2:066588:111258505"),
                         [("MEIS1", 4211), ("HHC2:066588", 111258505)])

    def test_malformed_refused_not_coerced(self):
        for bad in ("GENEINFO=X:0", "GENEINFO=X:abc", "GENEINFO=X:12|X:12", "GENEINFO=:12", "GENEINFO=X:012"):
            with self.assertRaises(ValueError):
                parse_geneinfo(bad)

    def test_duplicate_variation_id_refused(self):
        with gzip.open(self.vcf, "wt") as f:
            f.write("#h\n1\t1\t9\tA\tG\t.\t.\tGENEINFO=A:1\n1\t2\t9\tA\tG\t.\t.\tGENEINFO=B:2\n")
        with self.assertRaises(ValueError):
            census(self.vcf)


if __name__ == "__main__":
    unittest.main()
