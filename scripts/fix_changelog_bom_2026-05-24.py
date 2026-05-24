r"""
Fix the CHANGELOG.md BOM-induced ordering bug from commit 6e0e379.

Root cause:
  PowerShell's `Set-Content -Encoding UTF8` writes a BOM (\xEF\xBB\xBF) at
  byte 0. My create_run10b_docs.py read CHANGELOG with `encoding="utf-8"`,
  which preserves the BOM as `\ufeff` in the Python string. The regex
  `^(# CHANGELOG\s*?\n\n+)` then failed to match because position 0 was
  `\ufeff`, not `#`. The script fell back to prepending the new entry
  BEFORE the BOM+header, leaving `# CHANGELOG` floating in the middle of
  the file with an embedded BOM in front of it.

This fix:
  1. Reads CHANGELOG.md with `utf-8-sig` (strips any leading BOM)
  2. Removes any internal `\ufeff` characters
  3. Removes the misplaced `# CHANGELOG` line (wherever it is)
  4. Prepends canonical `# CHANGELOG\n\n` at the top
  5. Collapses runs of 3+ newlines to 2
  6. Writes without BOM
  7. Patches scripts/create_run10b_docs.py to use utf-8-sig (future-proofs)

Usage:
  python scripts\\fix_changelog_bom_2026-05-24.py
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(r"C:\Projects\genomic-variant-classifier")
CHANGELOG = REPO / "docs" / "CHANGELOG.md"
DOCS_SCRIPT = REPO / "scripts" / "create_run10b_docs.py"


def fix_changelog() -> int:
    if not CHANGELOG.exists():
        print(f"  FAIL: {CHANGELOG} not found")
        return 1

    # Read with utf-8-sig: strips any leading BOM automatically
    content = CHANGELOG.read_text(encoding="utf-8-sig")
    original_chars = len(content)
    original_size = CHANGELOG.stat().st_size

    # Count and remove any internal \ufeff (embedded BOM characters)
    internal_boms = content.count("\ufeff")
    content = content.replace("\ufeff", "")

    # Find all `# CHANGELOG` header occurrences
    header_pattern = re.compile(r'(?m)^# CHANGELOG\s*$')
    headers_before = list(header_pattern.finditer(content))

    print(f"  Pre-fix state:")
    print(f"    File size:                {original_size:,} bytes ({original_chars:,} chars)")
    print(f"    Internal BOMs found:      {internal_boms}")
    print(f"    '# CHANGELOG' occurrences: {len(headers_before)}")
    if headers_before:
        for h in headers_before:
            line_num = content[:h.start()].count("\n") + 1
            print(f"      - line {line_num} (offset {h.start():,})")

    # Detect if header is already correctly at the top
    if (headers_before
            and headers_before[0].start() == 0
            and len(headers_before) == 1
            and internal_boms == 0):
        print("  CHANGELOG already correctly structured; nothing to do")
        return 0

    # Remove ALL `# CHANGELOG` lines and any surrounding blank-line runs
    content = re.sub(r'\n*^# CHANGELOG\s*$\n*', '\n\n', content, flags=re.MULTILINE)

    # Strip leading whitespace (prepare for prepending canonical header)
    content = content.lstrip()

    # Prepend canonical header
    content = "# CHANGELOG\n\n" + content

    # Collapse runs of 3+ consecutive newlines to exactly 2
    content = re.sub(r'\n{3,}', '\n\n', content)

    # Ensure trailing newline
    if not content.endswith("\n"):
        content += "\n"

    # Write without BOM
    CHANGELOG.write_text(content, encoding="utf-8")
    new_size = CHANGELOG.stat().st_size

    print(f"\n  Post-fix state:")
    print(f"    File size:  {new_size:,} bytes ({len(content):,} chars)")
    print(f"    Size delta: {new_size - original_size:+,} bytes")

    # Verification: re-read and check structure
    final = CHANGELOG.read_text(encoding="utf-8-sig")
    final_lines = final.split("\n")
    final_headers = [(i, l) for i, l in enumerate(final_lines)
                     if l.strip() == "# CHANGELOG"]

    print(f"\n  Verification - first 8 lines:")
    for i, line in enumerate(final_lines[:8], 1):
        marker = "  <-- canonical header" if line.startswith("# CHANGELOG") else ""
        marker = "  <-- new entry"        if line.startswith("## 2026-05-24") else marker
        marker = "  <-- prior entry"      if line.startswith("## 2026-05-23") else marker
        truncated = line[:70] + ("..." if len(line) > 70 else "")
        print(f"    line {i:2}: {truncated}{marker}")

    # Final asserts
    print(f"\n  Final checks:")
    if "\ufeff" in final:
        print(f"    FAIL: BOM still present in file content")
        return 1
    print(f"    PASS: no BOM anywhere in file")

    if len(final_headers) != 1:
        print(f"    FAIL: expected exactly 1 '# CHANGELOG' line, found {len(final_headers)}")
        return 1
    print(f"    PASS: exactly 1 '# CHANGELOG' header")

    if final_headers[0][0] != 0:
        print(f"    FAIL: '# CHANGELOG' at line {final_headers[0][0] + 1}, expected line 1")
        return 1
    print(f"    PASS: '# CHANGELOG' at line 1")

    # Confirm new entry comes BEFORE prior entry
    new_idx = next((i for i, l in enumerate(final_lines)
                    if l.startswith("## 2026-05-24")), -1)
    old_idx = next((i for i, l in enumerate(final_lines)
                    if l.startswith("## 2026-05-23")), -1)
    if new_idx == -1 or old_idx == -1:
        print(f"    FAIL: entries not found (new={new_idx}, old={old_idx})")
        return 1
    if new_idx >= old_idx:
        print(f"    FAIL: 2026-05-24 entry (line {new_idx + 1}) is not before 2026-05-23 (line {old_idx + 1})")
        return 1
    print(f"    PASS: 2026-05-24 at line {new_idx + 1}, 2026-05-23 at line {old_idx + 1}")

    return 0


def patch_docs_script() -> int:
    if not DOCS_SCRIPT.exists():
        print(f"  {DOCS_SCRIPT} not found; skipping patch")
        return 0

    content = DOCS_SCRIPT.read_text(encoding="utf-8")

    # Idempotence check
    if 'encoding="utf-8-sig"' in content:
        print(f"  Already patched: {DOCS_SCRIPT.name}")
        return 0

    # Target the specific line that reads CHANGELOG
    old = '        current = CHANGELOG.read_text(encoding="utf-8")'
    new = ('        current = CHANGELOG.read_text(encoding="utf-8-sig")\n'
           '        current = current.replace("\\ufeff", "")'
           '  # belt-and-suspenders strip embedded BOMs')

    if old not in content:
        print(f"  WARN: target line not found in {DOCS_SCRIPT.name}; may have been edited")
        print(f"        expected: {old!r}")
        return 1

    # Apply patch
    patched = content.replace(old, new, 1)
    DOCS_SCRIPT.write_text(patched, encoding="utf-8")
    print(f"  Patched: {DOCS_SCRIPT.name}")
    print(f"    Old: {old.strip()}")
    print(f"    New: read_text(encoding='utf-8-sig') + .replace('\\ufeff', '')")
    return 0


def main() -> int:
    print("=" * 70)
    print("Fix Run 10b docs artifacts: CHANGELOG BOM/ordering bug")
    print("=" * 70)

    print("\n[1/2] Fixing docs/CHANGELOG.md...")
    rc1 = fix_changelog()

    print("\n[2/2] Patching scripts/create_run10b_docs.py...")
    rc2 = patch_docs_script()

    print("\n" + "=" * 70)
    if rc1 == 0 and rc2 == 0:
        print("BOTH FIXES SUCCEEDED")
        print()
        print("Next steps (run in PowerShell):")
        print("  git add docs\\CHANGELOG.md scripts\\create_run10b_docs.py")
        print("  git status     # expect exactly 2 modified files")
        print("  git diff --cached --stat")
        print('  git commit -m "fix(docs): CHANGELOG BOM ordering bug + harden create_run10b_docs.py with utf-8-sig"')
        print("  git push origin main")
        return 0
    else:
        print(f"FAIL: changelog={rc1}, script_patch={rc2}")
        print("Review the output above before staging anything.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
