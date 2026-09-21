"""Insert dated status blocks into the two 2026-09-20 measurement documents.

Preserve-and-correct: the original text is left intact; a block is inserted directly after the
title line. Guarded: the title must match exactly, the block must not already be present, and the
file's own newline convention (LF or CRLF) is preserved. Writes nothing unless every check passes.
"""
import re, sys
from pathlib import Path

GATE_A = "docs/MEASUREMENT_2026-09-21_gate-a-identity-and-leakage.md"
CORR = "baseline/GVC_refinements_2026-09-21/CORRECTIONS.md"
MARK = "**Status 2026-09-21"

BLOCKS = {
"docs/MEASUREMENT_2026-09-20_constraint-does-not-survive-repair.md": (
"# MEASUREMENT 2026-09-20 \u2014 Constraint extension does not survive cohort repair",
f"""> {MARK} \u2014 superseded in part.** Read with `{GATE_A}` (Gate A) and `{CORR}`.
>
> - The headline conclusion does not hold as a general finding. Split by resolved gene-component
>   leakage (Gate A section 7), the LightGBM constraint effect is not detectable on components unseen
>   in training. Among components seen in training it is detectable and concentrated in three
>   components (TTN, PKD1/TSC2, SON). Logistic regression shows no detectable effect in any stratum.
> - "Gene disjointness ... overlap 0" measured literal registry strings. Resolved through source
>   GeneIDs, a substantial share of validation rows share a gene component with training (Gate A
>   section 6), so "every validation gene is unseen" is false.
> - The transfer diagnostic's "constraint absent" stratum was defined by `loeuf_is_missing` alone and
>   pooled subgroups of opposite sign.
> - The "design effect" values compare a whole-gene interval with a class-stratified row interval at
>   different replicate counts; they do not measure within-gene dependence, and the conclusion drawn
>   from them is withdrawn.
> - "Verdicts stable across seeds 0-4" reflects Monte Carlo error of one analysis, not replication.
> - The recommendation to exclude constraint from step 7 rested on harm to unseen genes, which is
>   not supported.
>
> Retained: the per-tier metrics and the all-rows paired deltas, which Gate A section 7 reproduces
> exactly. Its intervals differ slightly because Gate A resamples resolved components, not registry
> genes.
"""),
"docs/MEASUREMENT_2026-09-20_lr-lightgbm-gap-is-representational.md": (
"# MEASUREMENT 2026-09-20 \u2014 The LR/LightGBM gap is 98.5% representational",
f"""> {MARK} \u2014 core finding confirmed on unseen gene components; several statements corrected.**
> Read with `{GATE_A}` (Gate A) and `{CORR}`.
>
> - Confirmed: split by resolved gene-component leakage (Gate A section 8), `lr_representation`
>   improves on `lr_current` and remains detectably behind LightGBM in both the unseen and leaked
>   strata, and in the unseen stratum neither contrast is carried by a few components.
> - "Gene overlap 0" measured literal registry strings; see Gate A section 6.
> - The stated reason for excluding constraint, that it degrades LightGBM on unseen genes, is not
>   supported; see the companion document's status block.
> - `allele_freq` is null in every cohort row, so `af_raw` and the log10 allele-frequency term were
>   constants. The transformed arm's gain came from the length transforms and categorical severity;
>   "`af_raw` ... badly non-linear" is withdrawn.
> - Consequence severity, including its categorical encoding, was computed from a vocabulary mapping
>   that scored ClinVar `nonsense` and other unmapped terms as 0.
> - `lr_splines` used B-splines (`SplineTransformer`), not restricted cubic splines.
> - "Attributable to model capacity" is not a causal decomposition.
> - The design-effect values use the flawed width ratio described in the companion status block.
"""),
}


def main(root):
    root = Path(root); plan = []
    if not (root / GATE_A).is_file():
        sys.exit(f"ABORT, nothing written: {GATE_A} must exist before it is referenced")
    for rel, (title, block) in BLOCKS.items():
        p = root / rel; raw = p.read_bytes().decode("utf-8")
        # A working copy can MIX conventions (e.g. LF lines plus one trailing CRLF appended by
        # PowerShell). So the newline is taken from the TITLE LINE'S OWN terminator -- the lines
        # inserted sit beside it -- never inferred from whether CRLF appears anywhere in the file.
        m = re.match(r"([^\r\n]*)(\r\n|\n)", raw)
        if not m:
            sys.exit(f"ABORT, nothing written: no terminated first line in {rel}")
        first, nl, rest = m.group(1), m.group(2), raw[m.end():]
        crlf = raw.count("\r\n"); lf = raw.count("\n") - crlf
        print(f"{rel}: CRLF {crlf}, bare LF {lf}; title terminated by {'CRLF' if nl == chr(13) + chr(10) else 'LF'}")
        if first.lstrip("\ufeff") != title:
            sys.exit(f"ABORT, nothing written: title mismatch in {rel}: {first!r}")
        if MARK in raw:
            sys.exit(f"ABORT, nothing written: {rel} already has a status block")
        new = first + nl + nl + nl.join(block.rstrip("\n").split("\n")) + nl + rest
        plan.append((p, new))
    for p, new in plan:
        p.write_bytes(new.encode("utf-8"))
        print(f"amended {p.name}: {len(new.encode('utf-8')):,} bytes")


if __name__ == "__main__":
    main(sys.argv[1])
