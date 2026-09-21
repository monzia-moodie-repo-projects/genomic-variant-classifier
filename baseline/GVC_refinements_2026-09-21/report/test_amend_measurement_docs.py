import subprocess, sys, tempfile, unittest
from pathlib import Path
from amend_measurement_docs import BLOCKS, GATE_A, MARK

HERE = Path(__file__).resolve().parent


def make_root(newline, with_gate_a=True):
    root = Path(tempfile.mkdtemp()); (root / "docs").mkdir()
    for rel, (title, _) in BLOCKS.items():
        body = newline.join([title, "", "original line one", "original line two", ""])
        (root / rel).write_bytes(body.encode("utf-8"))
    if with_gate_a:
        (root / GATE_A).write_text("x", encoding="utf-8")
    return root


def run(root):
    return subprocess.run([sys.executable, str(HERE / "amend_measurement_docs.py"), str(root)],
                          capture_output=True, text=True)


class AmendTests(unittest.TestCase):
    def test_original_bytes_preserved_in_both_newline_styles(self):
        for nl in ("\n", "\r\n"):
            root = make_root(nl)
            before = {rel: (root / rel).read_bytes() for rel in BLOCKS}
            self.assertEqual(run(root).returncode, 0)
            for rel, (title, _) in BLOCKS.items():
                after = (root / rel).read_bytes(); b = nl.encode()
                rest = before[rel].partition(b)[2]
                self.assertTrue(after.startswith((title + nl).encode("utf-8")))
                self.assertTrue(after.endswith(rest))
                self.assertIn(MARK.encode(), after)
                if nl == "\r\n":
                    self.assertEqual(after.count(b"\r\n"), after.count(b"\n"))   # no bare LF introduced

    def test_mixed_newlines_lf_body_with_trailing_crlf(self):
        # The case that failed on the real working copy: LF lines plus a trailing CRLF.
        root = make_root("\n")
        for rel in BLOCKS:
            (root / rel).write_bytes((root / rel).read_bytes().rstrip(b"\n") + b"\r\n")
        before = {rel: (root / rel).read_bytes() for rel in BLOCKS}
        r = run(root); self.assertEqual(r.returncode, 0, r.stderr)
        for rel, (title, _) in BLOCKS.items():
            after = (root / rel).read_bytes()
            rest = before[rel].partition(b"\n")[2]
            self.assertTrue(after.startswith((title + "\n").encode("utf-8")))
            self.assertTrue(after.endswith(rest))                       # every original byte after the title kept
            self.assertEqual(after.count(b"\r\n"), before[rel].count(b"\r\n"))   # inserted lines add no CRLF

    def test_refuses_without_gate_a_and_on_second_run(self):
        root = make_root("\n", with_gate_a=False)
        before = {rel: (root / rel).read_bytes() for rel in BLOCKS}
        r = run(root); self.assertNotEqual(r.returncode, 0); self.assertIn("must exist", r.stderr)
        self.assertEqual(before, {rel: (root / rel).read_bytes() for rel in BLOCKS})
        (root / GATE_A).write_text("x", encoding="utf-8")
        self.assertEqual(run(root).returncode, 0)
        r = run(root); self.assertNotEqual(r.returncode, 0); self.assertIn("already has a status block", r.stderr)

    def test_title_mismatch_writes_nothing(self):
        root = make_root("\n"); first = next(iter(BLOCKS))
        (root / first).write_text("# a different title\n", encoding="utf-8")
        before = {rel: (root / rel).read_bytes() for rel in BLOCKS}
        r = run(root); self.assertNotEqual(r.returncode, 0)
        self.assertEqual(before, {rel: (root / rel).read_bytes() for rel in BLOCKS})   # neither file touched


if __name__ == "__main__":
    unittest.main()
