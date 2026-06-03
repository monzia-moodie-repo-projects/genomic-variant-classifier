#!/usr/bin/env python3
"""Replace the TCP-only network guard in test_esm2_not_in_stub_mode with real
dependency probes (model embedding + UniProt fetch), so CI false-regressions
become xfail while genuine all-zeros-with-working-deps still fails. Idempotent,
backup-first, ast-validated, line-ending agnostic. Test-only; no production change."""
from __future__ import annotations
import ast, shutil, sys
from pathlib import Path

MARKER = "Probe the ACTUAL dependencies"
OLD = (
    "    if not _has_network():\n"
    "        pytest.xfail(\n"
    "            \"rest.uniprot.org unreachable -- network flake, not a regression. \"\n"
    "            \"ESM-2 connector needs UniProt to fetch canonical sequences.\"\n"
    "        )\n"
)
NEW = (
    "    # Probe the ACTUAL dependencies rather than a TCP-only reachability check:\n"
    "    # a 443 handshake does not prove the HF model weights downloaded or that\n"
    "    # UniProt returned a sequence -- the gap that made CI #250 a false regression.\n"
    "    conn = esm2_connector._get_conn()\n"
    "    if esm2_mod._embed_sequence(\n"
    "        \"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\", esm2_connector.model_name, conn\n"
    "    ) is None:\n"
    "        pytest.xfail(\n"
    "            \"ESM-2 model weights could not be loaded/run here (e.g. HuggingFace \"\n"
    "            \"download failed/timed out) -- environmental, not a regression.\"\n"
    "        )\n"
    "    if esm2_mod._fetch_uniprot_sequence(\"TP53\", esm2_connector.request_timeout) is None:\n"
    "        pytest.xfail(\n"
    "            \"UniProt sequence fetch failed or returned empty -- environmental, \"\n"
    "            \"not a regression. ESM-2 needs UniProt canonical sequences.\"\n"
    "        )\n"
)

def main(path_str: str) -> int:
    path = Path(path_str)
    raw = path.open(encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    data = raw.replace("\r\n", "\n")
    if MARKER in data:
        print(f"SKIP: {path} already probes real deps (idempotent)"); return 0
    if data.count(OLD) != 1:
        print(f"ABORT: expected 1 of the TCP-guard anchor, got {data.count(OLD)}; no change"); return 2
    out = data.replace(OLD, NEW, 1)
    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"ABORT: patched test invalid: {e}; no change"); return 3
    final = out.replace("\n", nl) if nl == "\r\n" else out
    backup = path.with_suffix(path.suffix + ".esm2probe.bak")
    shutil.copy2(path, backup)
    path.open("w", encoding="utf-8", newline="").write(final)
    print(f"patched {path} (backup {backup}); endings={'CRLF' if nl==chr(13)+chr(10) else 'LF'}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "tests/unit/test_esm2_activation.py"))
