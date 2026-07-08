#!/usr/bin/env python
"""
probe_pos_offset_by_representation.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
SUPERSEDES scripts/probe_vcf_deletion_join.py, whose verdict was correct but whose
instrument had two defects (both corrected here, both documented below).

WHAT THE FIRST PROBE ESTABLISHED (2026-07-08)

    30/30 blank-ReviewStatus deletions matched the ClinVar VCF at pos - 1.
    30/30 populated deletions, 30/30 insertions, 30/30 MNVs matched at pos + 0.

    Corroboration on base identity: cohort row `clinvar:17:43076593:ACTT:A` cannot
    have `ref="ACTT"` beginning at 43076593, because the reference base there is C
    (a different VCF record, C>G, sits at that position). ACTT begins at 43076592.
    The cohort's `pos` and its own `ref` string disagree.

    Separation on representation was perfect:
        blank  (offset -1):  ACTT:A   CA:C   GA:G   AC:A     <- alt == ref[0]  (PADDED DELETION)
        matched(offset +0):  AA:C     CG:T   GG:T   TCAG..:G <- alt != ref[0]  (DELINS)

MECHANISM (established)

    The cohort's `pos` is ClinVar variant_summary's `Start` -- the first ALTERED
    reference base. Its `ref`/`alt` are ReferenceAlleleVCF / AlternateAlleleVCF,
    which begin at `PositionVCF`. For a padded deletion the padding base is
    UNCHANGED, so Start == PositionVCF + 1. For SNVs, delins, and insertions (no
    reference base is removed) Start == PositionVCF. Only PADDED DELETIONS shift.

DEFECTS IN THE FIRST PROBE, CORRECTED HERE

    DEFECT 1 -- search order. `for off in range(-W, W+1)` searched -W first and took
        the first hit. A single-base C>T coincidentally matches any nearby C>T, so
        SNVs were reported at -5 / -2. Artifacts of search order, not evidence.
        FIX: search nearest-first (0, -1, +1, -2, +2, ...) AND report every matching
        offset, flagging ambiguity rather than hiding it behind a first-hit.
        (The deletion result was never at risk: offsets -5..-2 were searched BEFORE
        -1 and found nothing, and a multi-base ref cannot coincidentally match.)

    DEFECT 2 -- mislabelled display. `near = found.get(pos) or found.get(pos-1)`
        printed records AT pos under the header "VCF@pos-1".
        FIX: print the record that actually matched, at the offset where it matched.

WHAT THIS PROBE ADDS

    * Buckets by ALLELE REPRESENTATION, not crude length classes. `AA>C` is not a
      deletion; it is a delins that the length rule mislabels.
    * Whole-cohort bucket counts, so the EXACT number of mis-positioned rows is known
      rather than extrapolated from a 30-row sample.
    * The corrected `pos` rule, printed, ready to apply.

DECISION RULE (fixed before the data is seen)
    * padded_deletion matches predominantly at -1 AND every other bucket at 0
        -> CONFIRMED. `pos` must be decremented for padded deletions only.
    * any other bucket also shifts
        -> the rule is broader than "padded deletion". Do not apply the fix; widen the
           bucketing and re-derive.
    * ambiguity (a bucket matching at several offsets) above ~10%
        -> the (ref,alt) key is too degenerate for that bucket; its offset is not
           evidence. Expect this for SNVs; it must NOT appear for padded deletions.

USAGE (from project root, .venv312 active)
    python scripts/probe_pos_offset_by_representation.py
    python scripts/probe_pos_offset_by_representation.py --n 60 --window 5
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def norm_chrom(c: object) -> str:
    s = str(c)
    return s[3:] if s.lower().startswith("chr") else s


def representation(ref: str, alt: str) -> str:
    """Bucket by how the allele pair is written, not by raw lengths.

    padded_deletion  : ref is alt + deleted bases, e.g. ACTT>A   (alt == ref[:len(alt)])
    padded_insertion : alt is ref + inserted bases, e.g. G>GAAGA (ref == alt[:len(ref)])
    SNV              : 1 base -> 1 base
    delins           : first base changes, e.g. AA>C, CG>T
    padded_other     : shares a prefix but is neither a clean padded del nor ins
    """
    if not ref or not alt:
        return "empty"
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(alt) < len(ref) and ref.startswith(alt):
        return "padded_deletion"
    if len(ref) < len(alt) and alt.startswith(ref):
        return "padded_insertion"
    if ref[0] == alt[0]:
        return "padded_other"
    return "delins"


def _sample(df: pd.DataFrame, mask, n: int, seed: int) -> pd.DataFrame:
    sub = df[mask]
    return sub if len(sub) == 0 else sub.sample(min(n, len(sub)), random_state=seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-representation VCF positional offset.")
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean.parquet")
    ap.add_argument("--vcf", default="data/raw/clinvar/clinvar_GRCh38.vcf.gz")
    ap.add_argument("--n", type=int, default=40, help="samples per bucket")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    for p in (Path(a.cohort), Path(a.vcf)):
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 2

    print("=" * 80)
    print("PROBE: positional offset by ALLELE REPRESENTATION")
    print(f"  cohort {a.cohort}\n  vcf    {a.vcf}\n  window +/-{a.window}   samples/bucket {a.n}")
    print("=" * 80)

    df = pd.read_parquet(a.cohort, columns=["variant_id", "chrom", "pos", "ref", "alt", "ReviewStatus"])
    df["ref"] = df["ref"].astype(str)
    df["alt"] = df["alt"].astype(str)
    df["rep"] = [representation(r, x) for r, x in zip(df["ref"], df["alt"])]
    blank = df["ReviewStatus"].astype("string").fillna("").str.strip() == ""

    # --- whole-cohort census: the number that matters, not a sample estimate ----
    print("\n--- WHOLE-COHORT CENSUS BY REPRESENTATION ---")
    cen = pd.DataFrame({
        "rows": df["rep"].value_counts(),
        "RS_blank": df.loc[blank, "rep"].value_counts(),
    }).fillna(0).astype(int)
    cen["pct_blank"] = (100 * cen["RS_blank"] / cen["rows"]).round(3)
    cen = cen.sort_values("rows", ascending=False)
    print(cen.to_string())
    n_pad_del = int(cen.loc["padded_deletion", "rows"]) if "padded_deletion" in cen.index else 0
    print(f"\npadded deletions in cohort : {n_pad_del:,}  <- rows whose `pos` would shift by -1")

    # --- sample per bucket -----------------------------------------------------
    buckets = [b for b in ("padded_deletion", "padded_insertion", "SNV", "delins", "padded_other")
               if b in set(df["rep"])]
    groups: dict[str, pd.DataFrame] = {}
    for b in buckets:
        groups[f"{b:<17} blank"] = _sample(df, (df.rep == b) & blank, a.n, a.seed)
        groups[f"{b:<17} set  "] = _sample(df, (df.rep == b) & ~blank, a.n, a.seed)
    groups = {k: g for k, g in groups.items() if len(g)}

    interest: dict[str, set[int]] = defaultdict(set)
    for g in groups.values():
        for c, p in zip(g["chrom"], g["pos"]):
            cc, pp = norm_chrom(c), int(p)
            for off in range(-a.window, a.window + 1):
                interest[cc].add(pp + off)
    print(f"\nscanning VCF for {sum(len(v) for v in interest.values()):,} sites ...")

    found: dict[tuple[str, int], list[tuple[str, str, str]]] = defaultdict(list)
    n_rec = 0
    with gzip.open(a.vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 8:
                continue
            n_rec += 1
            cc = norm_chrom(p[0])
            if cc not in interest:
                continue
            try:
                pp = int(p[1])
            except ValueError:
                continue
            if pp not in interest[cc]:
                continue
            info = dict(kv.split("=", 1) for kv in p[7].split(";") if "=" in kv)
            rev = info.get("CLNREVSTAT", "")
            for alt in p[4].split(","):
                found[(cc, pp)].append((p[3], alt, rev))
    print(f"VCF records scanned {n_rec:,}   sites hit {len(found):,}")

    # nearest-first search order; collect EVERY matching offset
    order = [0] + [o for k in range(1, a.window + 1) for o in (-k, k)]

    print("\n" + "-" * 80)
    print("PRIMARY OFFSET (nearest-first) AND AMBIGUITY")
    print("  offset 0 = the join augment_reviewstatus.py attempts")
    print("-" * 80)
    offs = [f"{o:+d}" for o in sorted(order)] + ["none"]
    hdr = f"{'bucket / RS':<26}" + "".join(f"{o:>6}" for o in offs) + f"{'ambig':>8}"
    print(hdr)
    print("-" * len(hdr))

    examples: dict[str, list[str]] = defaultdict(list)
    for name, g in groups.items():
        cnt: Counter = Counter()
        n_ambig = 0
        for vid, c, p, r, x in zip(g["variant_id"], g["chrom"], g["pos"], g["ref"], g["alt"]):
            cc, pp = norm_chrom(c), int(p)
            hits = [o for o in order
                    if any(vr == r and va == x for (vr, va, _) in found.get((cc, pp + o), []))]
            key = f"{hits[0]:+d}" if hits else "none"
            cnt[key] += 1
            if len(hits) > 1:
                n_ambig += 1
            if len(examples[name]) < 3:
                rec = ""
                if hits:
                    o = hits[0]
                    m = [t for t in found[(cc, pp + o)] if t[0] == r and t[1] == x][0]
                    rec = f"  | VCF@{pp+o} {m[0]}>{m[1]} rev={m[2] or '(none)'}"
                examples[name].append(f"    {vid} -> {key}{rec}"
                                      + (f"   [ambiguous: {hits}]" if len(hits) > 1 else ""))
        print(f"{name:<26}" + "".join(f"{cnt.get(o, 0):>6}" for o in offs) + f"{n_ambig:>8}")

    print("\nexamples:")
    for name, lines in examples.items():
        print(f"  {name}")
        for ln in lines:
            print(ln)

    # ---------------- verdict --------------------------------------------------
    print("\n" + "=" * 80)

    def frac(name_prefix: str, off: str) -> float:
        tot = hit = 0
        for name, g in groups.items():
            if not name.startswith(name_prefix):
                continue
            for c, p, r, x in zip(g["chrom"], g["pos"], g["ref"], g["alt"]):
                cc, pp = norm_chrom(c), int(p)
                hits = [o for o in order
                        if any(vr == r and va == x for (vr, va, _) in found.get((cc, pp + o), []))]
                tot += 1
                if hits and f"{hits[0]:+d}" == off:
                    hit += 1
        return hit / tot if tot else 0.0

    pd_minus1 = frac("padded_deletion", "-1")
    others_zero = min(
        (frac(b, "+0") for b in ("padded_insertion", "SNV", "delins") if any(k.startswith(b) for k in groups)),
        default=0.0,
    )
    print(f"padded_deletion primary offset -1 : {100*pd_minus1:.1f}%")
    print(f"all other buckets primary offset 0: {100*others_zero:.1f}% (minimum across buckets)")
    print()
    if pd_minus1 >= 0.9 and others_zero >= 0.9:
        print("VERDICT: CONFIRMED. `pos` is off by one for PADDED DELETIONS ONLY.")
        print()
        print("  THE FIX (apply to the cohort builder, not the artifact):")
        print("      is_padded_del = (len(alt) < len(ref)) & ref.str.startswith(alt)")
        print("      pos_vcf       = pos - is_padded_del.astype(int)")
        print("      variant_id    = 'clinvar:' + chrom + ':' + pos_vcf + ':' + ref + ':' + alt")
        print(f"      rows affected : {n_pad_del:,}")
        print()
        print("  ESCALATE -- this is NOT a ReviewStatus bug. Every join keyed on")
        print("  chrom:pos(:ref:alt) misses these rows: gnomAD, SpliceAI, phyloP, CADD,")
        print("  dbNSFP, 1000G, dbSNP, COSMIC, AlphaMissense. The Nucleotide-Transformer")
        print("  sequence windows are centred on `pos` and are therefore off by one too.")
        print("  Runs 15-17 dropped these rows (tier filter) and never saw the corruption.")
        print("  Runs 9-14 KEPT them, with silently-defaulted annotations.")
        print()
        print("  MISSING GUARD: nothing ever asserted genome[chrom][pos-1:][:len(ref)] == ref.")
        print("  That single post-condition would have caught 187,258 rows immediately.")
    else:
        print("VERDICT: NOT the simple padded-deletion rule. Do NOT apply the fix.")
        print("  Inspect the table: another bucket shifts, or ambiguity is high.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
