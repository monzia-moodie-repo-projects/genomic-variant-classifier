#!/usr/bin/env python3
"""patch_readme_directive_pass.py -- README directive pass (Run 8 baseline relabel,
run-table trim, 1.70M residual fix, histopathology->roadmap, honest DB framing,
real agent count). Docker/FastAPI claims are intentionally LEFT INTACT (confirmed real).

Every marker is a string that exists ONLY AFTER its own edit (lesson from the
recent_runs_cohort guard, whose marker collided with the headline phrase and made the
edit silently skip). Per-edit resilient, backup-first, idempotent, EOL-aware, ASCII-only.
Author: Monzia Moodie.
"""
from __future__ import annotations
import datetime as _dt, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RM = REPO / "README.md"

# (old, new, marker, is_delete, label)
EDITS = [
    # --- Directive 1: relabel "publication snapshot" -> "Run 8 baseline" (+ real 1.70M fix) ---
    ("publication snapshot; recent runs use the full 1.70 M-variant matrix.",
     "Run 8 baseline; recent runs (Run 14/15) use the full ~1.49 M-variant cohort.",
     "Run 8 baseline; recent runs (Run 14/15)", False, "perf preamble: relabel + 1.70M->1.49M fix"),
    ("| Holdout AUROC (publication snapshot) | **0.9847** |",
     "| Holdout AUROC (Run 8 baseline) | **0.9847** |",
     "Holdout AUROC (Run 8 baseline)", False, "metrics row: relabel"),
    ("### Per-model performance (validation set, publication snapshot)",
     "### Per-model performance (validation set, Run 8 baseline)",
     "validation set, Run 8 baseline", False, "per-model header: relabel"),
    # --- Directive 2: trim run-history table to Run 14 + Run 15 ---
    ("| Run 6 | 2026-04 | GCP n2-highmem-32 (CPU) | 0.9862 | First full 78-feature run; ESM-2 silently inert |\n"
     "| Run 7 | 2026-04 | GCP n2-highmem-32 (CPU) | 0.9862 | gnomAD v4.1 constraint wired; GNN still CPU-only |\n"
     "| **Run 8** | **2026-04-16** | **Vast.ai RTX 4090** | **0.9863** (test 0.9833) | **AUPRC 0.9461, MCC 0.8482, Brier 0.0358; AlphaMissense ranked 7/78** |\n"
     "| Run 9 | 2026-05-09 | Vast.ai RTX 4090 | OOF 0.9916 (blend) | Best single LightGBM OOF 0.9911; locked test lost to `save()` PicklingError |\n"
     "| Run 10 | scheduled | Vast.ai RTX 4090 | -- | Phase-1.7 launch script + dual-layer preflight; targets locked test recovery |\n",
     "", "First full 78-feature run", True, "run-history: trim to Run 14 + Run 15"),
    # --- Directive 3a: intro sentence (histo->future, DBs->suite, agents->7 core; FastAPI kept) ---
    ("The system integrates genomic sequence data, population-stratified allele frequencies\n"
     "from eighteen biological databases, protein structural annotations, tissue-specific\n"
     "gene expression, variant co-classification evidence, and whole-slide histopathology\n"
     "imaging from The Cancer Genome Atlas into a unified stacking ensemble architecture,\n"
     "deployed as a production FastAPI REST service and continuously supervised by an\n"
     "autonomous agent layer of thirteen specialised monitoring agents communicating\n"
     "over a typed inter-agent message bus.",
     "The system integrates genomic sequence data, population-stratified allele frequencies,\n"
     "protein structural annotations, tissue-specific gene expression, and variant\n"
     "co-classification evidence from a suite of biological databases into a unified\n"
     "stacking-ensemble architecture, deployed as a production FastAPI REST service and\n"
     "continuously supervised by an autonomous agent layer of seven core monitoring agents\n"
     "-- plus a committed drift-detection suite -- over a typed inter-agent message bus.\n"
     "Whole-slide histopathology imaging (TCGA) is a future multi-modal phase tracked in\n"
     "`docs/ROADMAP.md`.",
     "seven core monitoring agents", False, "intro sentence: histo->future, DB suite, 7 core agents"),
    # --- Directive 3b: dedicated histopathology paragraph -> planned (your approved text) ---
    ("**Histopathology Branch** -- A ResNet-50 CNN fine-tuned on TCGA whole-slide image tiles\n"
     "(224x224 px at 20x magnification) across TCGA-BRCA, TCGA-LUAD, and TCGA-COAD cohorts,\n"
     "providing phenotypic validation that anchors molecular classifications in observable\n"
     "tissue-level consequences.",
     "**Histopathology Branch (planned -- future multi-modal expansion).** A ResNet-50 branch\n"
     "over TCGA whole-slide tiles is a roadmap ambition for the multi-modal program; it is not\n"
     "yet implemented. The current system is the tabular variant classifier described above.\n"
     "Image, RNA, and protein-structure modalities are tracked as future phases in `docs/ROADMAP.md`.",
     "Histopathology Branch (planned", False, "histo paragraph: -> planned/roadmap"),
    # --- Directive 3c: ASCII diagram histopathology block -> [PLANNED] (whitespace-safe) ---
    ("ResNet-50 Histopathology Branch",
     "ResNet-50 Histopathology Branch [PLANNED - see ROADMAP]",
     "ResNet-50 Histopathology Branch [PLANNED", False, "ASCII diagram: histo [PLANNED]"),
    # --- Directive 3d: 'Phenotypically grounded' bullet -> planned ---
    ("**Phenotypically grounded** -- The TCGA histopathology branch provides an empirical\n"
     "link between variant pathogenicity classification and observable tumor-tissue\n"
     "morphology, validated across breast, lung adenocarcinoma, and colorectal cancer\n"
     "cohorts.",
     "**Phenotypically grounded (planned).** A future TCGA histopathology branch will link\n"
     "variant pathogenicity classification to observable tumor-tissue morphology across breast,\n"
     "lung adenocarcinoma, and colorectal cancer cohorts -- a multi-modal capability on the\n"
     "roadmap (`docs/ROADMAP.md`), not yet implemented.",
     "Phenotypically grounded (planned)", False, "phenotypic bullet: -> planned"),
    # --- Directive 3e: architecture DB count (L39) -> honest suite framing ---
    ("graph. Input features span **80 dimensions** drawn from eighteen biological databases.",
     "graph. Input features span **80 dimensions** drawn from a suite of biological databases "
     "(further sources are being wired in the current data-expansion phase).",
     "drawn from a suite of biological databases (further sources", False, "architecture: DB count -> suite"),
    # --- Directive 3f: agents badge 13 -> 7 core ---
    ("[![Agents](https://img.shields.io/badge/Autonomous%20agents-13-blueviolet.svg)]()",
     "[![Agents](https://img.shields.io/badge/Core%20agents-7-blueviolet.svg)]()",
     "Core%20agents-7", False, "badge: 13 -> 7 core agents"),
    # --- Directive 3g: ASCII diagram agent count (whitespace-safe substring) ---
    ("13 monitoring agents",
     "7 core agents + drift suite",
     "7 core agents + drift suite", False, "ASCII diagram: 13 -> 7 core + drift suite"),
]

def run(raw):
    log=[]
    for old,new,marker,is_del,label in EDITS:
        if is_del:
            if old not in raw: log.append(("skip",label)); continue
        else:
            if marker in raw: log.append(("skip",label)); continue
        c = raw.count(old)
        if c!=1: log.append((f"MISS({c})",label)); continue
        raw = raw.replace(old,new,1); log.append(("ok",label))
    return raw, log

def main()->int:
    if not RM.exists(): print(f"ABORT: missing {RM}"); return 2
    raw0 = RM.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw0 else "\n"
    norm = (lambda s: s.replace("\n", nl)) if nl!="\n" else (lambda s: s)
    # normalize anchors to file EOL
    global EDITS
    EDITS = [(norm(o),norm(n),norm(m),d,l) for (o,n,m,d,l) in EDITS]
    ts=_dt.datetime.now().strftime("%Y%m%d_%H%M%S"); shutil.copy2(RM,f"{RM}.bak_{ts}")
    raw,log = run(raw0)
    ok=sum(1 for s,_ in log if s=="ok"); sk=sum(1 for s,_ in log if s=="skip"); ms=[l for s,l in log if s.startswith("MISS")]
    for s,l in log: print(f"  {s}: {l}")
    RM.write_bytes(raw.encode("utf-8"))
    print(f"\napplied={ok} skipped={sk} missed={len(ms)} (backup -> README.md.bak_{ts})")
    if ms:
        print("MISSED (left unchanged -- paste the exact current line and I will fix the anchor):")
        for m in ms: print(f"  - {m}")
        return 1
    print("DONE.")
    return 0

if __name__=="__main__": sys.exit(main())
