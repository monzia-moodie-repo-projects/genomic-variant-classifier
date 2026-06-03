#!/usr/bin/env python3
"""Fix the Monthly Drift Monitor: (A) grant notify job issues:write, (B) capture the
drift exit code without -e aborting, (C) no-op cleanly (drift_level=none) when reference
splits are absent so it does not file false alerts, plus a default case. Idempotent,
backup-first, count-guarded, YAML-validated, and line-ending agnostic (handles CRLF/LF)."""
from __future__ import annotations
import shutil, sys
from pathlib import Path
try:
    import yaml
except Exception:
    yaml = None

MARKER = "avoids false alert"
EDITS = [
    ("    needs: [feature-drift, label-drift]\n"
     "    if: needs.feature-drift.outputs.drift_level != 'none'\n"
     "\n"
     "    steps:\n"
     "      - name: Create GitHub Issue for drift alert",
     "    needs: [feature-drift, label-drift]\n"
     "    if: needs.feature-drift.outputs.drift_level != 'none'\n"
     "    permissions:\n"
     "      contents: read\n"
     "      issues: write\n"
     "\n"
     "    steps:\n"
     "      - name: Create GitHub Issue for drift alert",
     1),
    ("          python scripts/run_drift_monitor.py \\\n",
     "          EXIT=0\n"
     "          if [ ! -d outputs/phase2_with_gnomad/splits ] || [ -z \"$(ls -A outputs/phase2_with_gnomad/splits 2>/dev/null)\" ]; then\n"
     "            echo \"No reference splits available -- skipping drift check (avoids false alert).\"\n"
     "            echo \"exit_code=0\" >> \"$GITHUB_OUTPUT\"\n"
     "            echo \"drift_level=none\" >> \"$GITHUB_OUTPUT\"\n"
     "            exit 0\n"
     "          fi\n"
     "          python scripts/run_drift_monitor.py \\\n",
     1),
    ("            --output-dir outputs/drift_reports/\n"
     "          EXIT=$?\n",
     "            --output-dir outputs/drift_reports/ || EXIT=$?\n",
     1),
    ('            3) echo "drift_level=severe"  >> "$GITHUB_OUTPUT" ;;\n'
     "          esac",
     '            3) echo "drift_level=severe"  >> "$GITHUB_OUTPUT" ;;\n'
     '            *) echo "drift_level=none"    >> "$GITHUB_OUTPUT" ;;\n'
     "          esac",
     1),
]

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")            # normalize for matching (anchors are LF)
    if MARKER in data:
        print(f"SKIP: {path} already patched (idempotent no-op)"); return 0
    for old, _new, n in EDITS:
        c = data.count(old)
        if c != n:
            print(f"ABORT: expected {n} of an anchor, got {c}; no change. Anchor head:\n{old[:80]!r}"); return 2
    out = data
    for old, new, _n in EDITS:
        out = out.replace(old, new, 1)
    if yaml is not None:
        try:
            yaml.safe_load(out)
        except Exception as e:
            print(f"ABORT: patched YAML invalid: {e}; no change"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out   # restore original line endings
    backup = path.with_suffix(path.suffix + ".driftfix.bak")
    shutil.copy2(path, backup)
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path}  (backup {backup}); applied {len(EDITS)} edits; line endings = {'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ".github/workflows/drift_monitor.yml"))
