#!/usr/bin/env python3
"""patch_train_esm2_wiring.py -- thread ESM-2 connector settings through
scripts/train.py so a regen can select the 650M model and an offline UniProt
index (the validated path), instead of silently using the 8M default with
live per-gene REST.

Three anchored edits:
  1. Add --esm2-model / --esm2-uniprot-index / --esm2-cache / --esm2-device.
  2. Pass them into AnnotationConfig(...).
  3. Record the active ESM-2 settings (and finngen/dbnsfp) in the metrics
     JSON annotation_sources block for honest run provenance.

Idempotent, backup-first, py_compile-gated, ASCII-only, newline-preserving.
Author: Monzia Moodie.
"""
from __future__ import annotations
import py_compile, shutil, sys
from pathlib import Path

TARGET = Path("scripts/train.py")
MARKER = "--esm2-model"

A1_OLD = (
    "    p.add_argument(\n"
    "        \"--skip-nn\", action=\"store_true\",\n"
    "        help=\"Skip neural network models (faster without TensorFlow/GPU)\",\n"
    "    )\n"
    "    return p.parse_args()\n"
)
A1_NEW = (
    "    p.add_argument(\n"
    "        \"--skip-nn\", action=\"store_true\",\n"
    "        help=\"Skip neural network models (faster without TensorFlow/GPU)\",\n"
    "    )\n"
    "    p.add_argument(\n"
    "        \"--esm2-model\",\n"
    "        default=\"esm2_t6_8M_UR50D\",\n"
    "        metavar=\"NAME\",\n"
    "        help=(\n"
    "            \"ESM-2 model for esm2_delta_norm / esm2_llr (HuggingFace facebook/<NAME>). \"\n"
    "            \"Default esm2_t6_8M_UR50D (fast, for smoke tests). Set \"\n"
    "            \"esm2_t33_650M_UR50D for production runs.\"\n"
    "        ),\n"
    "    )\n"
    "    p.add_argument(\n"
    "        \"--esm2-uniprot-index\",\n"
    "        default=None,\n"
    "        metavar=\"PATH\",\n"
    "        help=(\n"
    "            \"Local UniProt sequence index parquet for ESM-2 (e.g. \"\n"
    "            \"data/external/uniprot/uniprot_human_reviewed.parquet). When set, \"\n"
    "            \"ESM-2 resolves sequences offline with NO run-time UniProt REST. \"\n"
    "            \"Default None -> live REST per gene (slow; not for large regens).\"\n"
    "        ),\n"
    "    )\n"
    "    p.add_argument(\n"
    "        \"--esm2-cache\",\n"
    "        default=None,\n"
    "        metavar=\"PATH\",\n"
    "        help=\"SQLite cache path for ESM-2 sequences/embeddings. Default None -> connector default.\",\n"
    "    )\n"
    "    p.add_argument(\n"
    "        \"--esm2-device\",\n"
    "        default=None,\n"
    "        metavar=\"DEV\",\n"
    "        help=\"Device for ESM-2 ('cpu','cuda','auto'). Default None -> cuda if available else cpu.\",\n"
    "    )\n"
    "    return p.parse_args()\n"
)

A2_OLD = (
    "    annotation_config = AnnotationConfig(\n"
    "        alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,\n"
    "        lovd_path=Path(args.lovd_path) if args.lovd_path else None,\n"
    "        finngen_path=Path(args.finngen_path) if args.finngen_path else None,\n"
    "        dbnsfp_path=Path(args.dbnsfp_path) if args.dbnsfp_path else None,\n"
    "    )\n"
)
A2_NEW = (
    "    annotation_config = AnnotationConfig(\n"
    "        alphamissense_path=Path(args.alphamissense) if args.alphamissense else None,\n"
    "        lovd_path=Path(args.lovd_path) if args.lovd_path else None,\n"
    "        finngen_path=Path(args.finngen_path) if args.finngen_path else None,\n"
    "        dbnsfp_path=Path(args.dbnsfp_path) if args.dbnsfp_path else None,\n"
    "        esm2_model_name=args.esm2_model,\n"
    "        esm2_uniprot_index_path=Path(args.esm2_uniprot_index) if args.esm2_uniprot_index else None,\n"
    "        esm2_cache_path=Path(args.esm2_cache) if args.esm2_cache else None,\n"
    "        esm2_device=args.esm2_device,\n"
    "    )\n"
)

# Edit 3 anchors ONLY on the unambiguous single-space alphamissense JSON-key line
# (avoids the alignment-padded 'lovd' line); inserts the missing provenance keys.
A3_OLD = (
    "                    \"alphamissense\": str(args.alphamissense) if args.alphamissense else None,\n"
)
A3_NEW = (
    "                    \"alphamissense\": str(args.alphamissense) if args.alphamissense else None,\n"
    "                    \"finngen\": str(args.finngen_path) if args.finngen_path else None,\n"
    "                    \"dbnsfp\": str(args.dbnsfp_path) if args.dbnsfp_path else None,\n"
    "                    \"esm2_model\": args.esm2_model,\n"
    "                    \"esm2_uniprot_index\": str(args.esm2_uniprot_index) if args.esm2_uniprot_index else None,\n"
)

EDITS = [("argparse", A1_OLD, A1_NEW), ("config", A2_OLD, A2_NEW), ("provenance", A3_OLD, A3_NEW)]

def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found"); return 1
    raw = TARGET.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if MARKER in text:
        print("already applied (--esm2-model present); no change."); return 0
    for name, old, _ in EDITS:
        c = text.count(old)
        if c != 1:
            print(f"ABORT: anchor '{name}' found {c} times (expected 1); no change."); return 1
    for _, old, new in EDITS:
        text = text.replace(old, new, 1)
    bak = TARGET.with_suffix(TARGET.suffix + ".bak")
    shutil.copy2(TARGET, bak)
    TARGET.write_text(text.replace("\n", nl), encoding="utf-8", newline="")
    try:
        py_compile.compile(str(TARGET), doraise=True)
    except py_compile.PyCompileError as exc:
        shutil.copy2(bak, TARGET); print(f"ABORT: py_compile failed, restored:\n{exc}"); return 1
    print(f"OK: train.py ESM-2 wiring applied; backup at {bak}"); return 0

if __name__ == "__main__":
    sys.exit(main())
