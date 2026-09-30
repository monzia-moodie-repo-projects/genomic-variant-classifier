"""C2 receipt output (ported from the owner's reviewed reference tests/test_receipt_io.py). Author: Monzia Moodie"""
from pathlib import Path
import base64
import tempfile
import unittest
from tests.unit.test_c2_protocol import fixture
from genomic_variant_classifier.source_monitor.c2_receipt_io import emit_job_output, write_receipt
from genomic_variant_classifier.source_monitor import c2_protocol as c


class OutputTests(unittest.TestCase):
    def test_failed_verification_still_writes_receipt(self):
        p = fixture()
        p["decision"]["verified"] = False
        p["decision"]["flags"]["claims_reconciled"] = False
        p["decision"]["flags"][c.FLAGS[-1]] = False
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "receipt.json"
            self.assertEqual(write_receipt(p, path), 2)
            self.assertEqual(path.read_bytes(), c.seal(p))

    def test_multiline_diagnostics_cannot_inject_outputs(self):
        p = fixture()
        p["diagnostics"] = ["hello\nother_output=forged\n::warning::not a command"]
        with tempfile.TemporaryDirectory() as d:
            receipt, output = Path(d) / "receipt.json", Path(d) / "output"
            self.assertEqual(write_receipt(p, receipt), 0)
            emit_job_output(receipt, output)
            lines = output.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(lines[0].startswith("receipt_b64="))
            self.assertEqual(base64.b64decode(lines[0].split("=", 1)[1]), c.seal(p))
