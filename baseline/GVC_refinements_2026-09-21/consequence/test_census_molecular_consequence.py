import gzip, json, tempfile, unittest
from pathlib import Path
from census_molecular_consequence import census, parse_mc

ROWS = [
    "1\t100\t11\tA\tG\t.\t.\tMC=SO:0001587|nonsense",
    "1\t200\t12\tAT\tA\t.\t.\tMC=SO:0001627|intron_variant,SO:0001587|nonsense",
    "1\t300\t13\tC\tT\t.\t.\tMC=SO:0001619|non-coding_transcript_variant",
    "1\t400\t14\tG\tA\t.\t.\tCLNSIG=Benign",
    "1\t700\t17\tG\tC\t.\t.\tMC=garbage_no_pipe",
]


class CensusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.vcf = self.tmp / "f.vcf.gz"
        with gzip.open(self.vcf, "wt") as f:
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + "\n".join(ROWS) + "\n")

    def test_every_entry_kept_not_just_the_first(self):
        self.assertEqual(parse_mc("MC=SO:0001627|intron_variant,SO:0001587|nonsense"),
                         [("SO:0001627", "intron_variant"), ("SO:0001587", "nonsense")])

    def test_absent_and_malformed_are_distinct(self):
        r = census(self.vcf)
        self.assertEqual((r["records_with_mc"], r["records_without_mc"],
                          r["records_with_malformed_mc"]), (3, 1, 1))

    def test_accession_retained_with_label(self):
        r = census(self.vcf)
        pairs = {(p["so_id"], p["label"]) for p in r["pairs"]}
        self.assertIn(("SO:0001587", "nonsense"), pairs)

    def test_malformed_entry_refused(self):
        with self.assertRaises(ValueError):
            parse_mc("MC=SO:123|short_accession")

    def test_label_to_multiple_accessions_detected(self):
        with gzip.open(self.vcf, "wt") as f:
            f.write("#h\n1\t1\t1\tA\tG\t.\t.\tMC=SO:0001587|x\n1\t2\t2\tA\tG\t.\t.\tMC=SO:0001583|x\n")
        self.assertEqual(census(self.vcf)["labels_with_more_than_one_accession"],
                         {"x": ["SO:0001583", "SO:0001587"]})


if __name__ == "__main__":
    unittest.main()
