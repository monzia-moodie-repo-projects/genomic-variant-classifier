#!/usr/bin/env python3
"""
scripts/make_roadmap_docx.py
============================
Generate docs/ROADMAP.docx IN-PLACE from docs/ROADMAP.md (the source of
truth). This permanently removes the recurring "ROADMAP.docx not in
Downloads" install failure: the .docx is rebuilt locally from the committed
Markdown every session, so the two never drift.

Usage (from repo root):
    python scripts/make_roadmap_docx.py
    # or explicit paths:
    python scripts/make_roadmap_docx.py docs/ROADMAP.md docs/ROADMAP.docx

Strategy: prefer pandoc (best fidelity) if it is on PATH; otherwise fall back
to a self-contained python-docx converter (`pip install python-docx`).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


def _via_pandoc(md: Path, docx: Path) -> bool:
    if shutil.which("pandoc") is None:
        return False
    try:
        subprocess.run(
            ["pandoc", str(md), "-f", "gfm", "-o", str(docx)],
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"pandoc failed ({exc}); falling back to python-docx\n")
        return False


def _via_python_docx(md: Path, docx: Path) -> None:
    try:
        from docx import Document
    except ImportError:
        sys.exit(
            "python-docx not installed and pandoc not found.\n"
            "  pip install python-docx        (or install pandoc and re-run)"
        )

    doc = Document()
    in_code = False
    code_buf: list[str] = []
    bold = re.compile(r"\*\*(.+?)\*\*")
    code_inline = re.compile(r"`([^`]+)`")

    def flush_code() -> None:
        nonlocal code_buf
        if code_buf:
            p = doc.add_paragraph()
            run = p.add_run("\n".join(code_buf))
            run.font.name = "Consolas"
            code_buf = []

    def clean(s: str) -> str:
        return code_inline.sub(r"\1", bold.sub(r"\1", s))

    for raw in md.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip("\n")
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
            in_code = not in_code
            continue
        if in_code:
            code_buf.append(line)
            continue
        s = line.strip()
        if not s:
            continue
        if s.startswith("### "):
            doc.add_heading(clean(s[4:].strip()), level=3)
        elif s.startswith("## "):
            doc.add_heading(clean(s[3:].strip()), level=2)
        elif s.startswith("# "):
            doc.add_heading(clean(s[2:].strip()), level=1)
        elif re.match(r"^[-*+]\s+", s):
            doc.add_paragraph(clean(s[2:].strip()), style="List Bullet")
        elif re.match(r"^\d+\.\s+", s):
            doc.add_paragraph(clean(re.sub(r"^\d+\.\s+", "", s)), style="List Number")
        else:
            doc.add_paragraph(clean(s))
    flush_code()
    doc.save(str(docx))


def main(argv: list[str]) -> None:
    md = Path(argv[1]) if len(argv) > 1 else Path("docs/ROADMAP.md")
    docx = Path(argv[2]) if len(argv) > 2 else Path("docs/ROADMAP.docx")

    if not md.exists():
        sys.exit(f"Source Markdown not found: {md} (run from the repo root).")

    docx.parent.mkdir(parents=True, exist_ok=True)

    if _via_pandoc(md, docx):
        print(f"Wrote {docx} via pandoc.")
    else:
        _via_python_docx(md, docx)
        print(f"Wrote {docx} via python-docx.")

    size = docx.stat().st_size
    print(f"  {docx}  ({size:,} bytes)")
    if size < 1024:
        sys.stderr.write("WARNING: output < 1 KB - verify ROADMAP.md had content.\n")


if __name__ == "__main__":
    main(sys.argv)
