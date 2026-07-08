#!/usr/bin/env python
"""
probe_vcf_deletion_join.py  (2026-07-08)  -- READ-ONLY. Writes nothing.
==========================================================================
THE QUESTION

    `scripts/augment_reviewstatus.py` joins the clean cohort to the ClinVar VCF on
    `chrom:pos:ref:alt`. Both sides construct that key identically -- no case,
    prefix, or normalisation asymmetry. Yet the join misses **98.834% of deletions**
    (187,258 of 189,468) while insertions (0.483%), MNVs (0.519%) and SNVs (5.771%)
    are largely unaffected. `.fillna("")` then converts every miss into review tier 5,
    and `--min-review-tier 3` discards them. See
    docs/incidents/INCIDENT_2026-07-08_deletion-reviewstatus-loss.md.

    The keys are built the same way, so the mismatch must be in the DATA, not the
    code. Where, exactly?

THE HYPOTHESIS THIS FALSIFIES OR CONFIRMS

    `source_id` on the first cohort rows reads 2, 3, 4 -- the canonical ClinVar
    VariationIDs of the first `AP5Z1` entries in `variant_summary.txt`. A VCF is
    position-sorted and would not begin at chr7. So the cohort was probably built
    from `variant_summary.txt`, which carries BOTH:

        Start        = first changed base
        PositionVCF  = padding base (what the VCF's POS column holds)

    These coincide for SNVs, insertions and MNVs. For a DELETION they differ by
    exactly one: `Start == PositionVCF + 1`.

    PREDICTION. If the cohort's `pos` came from `Start`, then for a blank-ReviewStatus
    deletion the VCF holds the same `ref`/`alt` at **pos - 1**. Insertions, SNVs and
    MNVs will match at **offset 0**.

WHY IT MATTERS FAR BEYOND ReviewStatus

    If `pos` is off by one for 189,468 deletions, then EVERY position-keyed
    annotation join -- gnomAD, SpliceAI, phyloP, CADD, dbNSFP, 1000G, dbSNP, COSMIC,
    AlphaMissense -- misses them as well. It would never have surfaced, because the
    review-tier filter discarded those rows before any feature audit ran.

    A systematic `-1` for deletions and `0` for everything else CONFIRMS it.
    A scatter of offsets, or no match at any offset, REFUTES it and points instead at
    a left-alignment / trimming difference, or at a VCF release mismatch.

DECISION RULE (fixed before the data is seen)
    * deletions match predominantly at offset -1, controls at offset 0
        -> CONFIRMED. The cohort's `pos` is off-by-one for deletions. Escalate: audit
           every position-keyed annotation for deletion coverage before any re-run.
    * deletions match predominantly at offset 0 (i.e. the join should have worked)
        -> the VCF lacks these records. Check CLNREVSTAT presence and VCF release.
    * no match at any offset within the window
        -> representation differs (left-alignment / trimming). Widen the window and
           compare `ref`/`alt` directly at the nearest record.

USAGE (from project root, .venv312 active)
    python scripts/probe_vcf_deletion_join.py
    python scripts/probe_vcf_deletion_join.py --n 50 --window 8
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


def variant_class(ref: str, alt: str) -> str:
    lr, la = len(ref), len(alt)
    if lr == 1 and la == 1:
        return "SNV"
    if lr > 1 and la == 1:
        return "deletion"
    if lr == 1 and la > 1:
        return "insertion"
    return "MNV/other"


def _sample(df: pd.DataFrame, mask, n: int, seed: int) -> pd.DataFrame:
    sub = df[mask]
    if len(sub) == 0:
        return sub
    return sub.sample(min(n, len(sub)), random_state=seed)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Locate the VCF offset at which cohort keys match.")
    ap.add_argument("--cohort", default="data/processed/clinvar_grch38_clean.parquet")
    ap.add_argument("--vcf", default="data/raw/clinvar/clinvar_GRCh38.vcf.gz")
    ap.add_argument("--n", type=int, default=30, help="samples per group")
    ap.add_argument("--window", type=int, default=5, help="+/- positional offsets to search")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    vcf = Path(a.vcf)
    if not vcf.exists():
        print(f"ERROR: VCF not found at {vcf}", file=sys.stderr)
        print("Pass --vcf <path>. `augment_reviewstatus.py` defaults to this same path.",
              file=sys.stderr)
        return 2
    cohort = Path(a.cohort)
    if not cohort.exists():
        print(f"ERROR: cohort not found at {cohort}", file=sys.stderr)
        return 2

    print("=" * 78)
    print("PROBE: at which VCF offset does each cohort key match?")
    print(f"  cohort : {cohort}")
    print(f"  vcf    : {vcf}")
    print(f"  window : +/-{a.window}   samples/group: {a.n}")
    print("=" * 78)

    df = pd.read_parquet(cohort, columns=["variant_id", "chrom", "pos", "ref", "alt", "ReviewStatus"])
    df["ref"] = df["ref"].astype(str)
    df["alt"] = df["alt"].astype(str)
    df["cls"] = [variant_class(r, x) for r, x in zip(df["ref"], df["alt"])]
    blank = df["ReviewStatus"].astype("string").fillna("").str.strip() == ""
    print(f"cohort rows {len(df):,}   blank ReviewStatus {int(blank.sum()):,}")

    groups: dict[str, pd.DataFrame] = {
        "deletion  RS=blank": _sample(df, (df.cls == "deletion") & blank, a.n, a.seed),
        "deletion  RS=set  ": _sample(df, (df.cls == "deletion") & ~blank, a.n, a.seed),
        "insertion RS=set  ": _sample(df, (df.cls == "insertion") & ~blank, a.n, a.seed),
        "SNV       RS=set  ": _sample(df, (df.cls == "SNV") & ~blank, a.n, a.seed),
        "SNV       RS=blank": _sample(df, (df.cls == "SNV") & blank, a.n, a.seed),
        "MNV       RS=set  ": _sample(df, (df.cls == "MNV/other") & ~blank, a.n, a.seed),
    }
    for k, g in groups.items():
        print(f"  sampled {len(g):>3} of group  {k}")

    # positions of interest: chrom -> set(pos), covering the whole window
    interest: dict[str, set[int]] = defaultdict(set)
    for g in groups.values():
        for c, p in zip(g["chrom"], g["pos"]):
            cc, pp = norm_chrom(c), int(p)
            for off in range(-a.window, a.window + 1):
                interest[cc].add(pp + off)
    n_interest = sum(len(v) for v in interest.values())
    print(f"\nscanning VCF for {n_interest:,} (chrom,pos) sites ...")

    # one streaming pass
    found: dict[tuple[str, int], list[tuple[str, str, str]]] = defaultdict(list)
    n_rec = n_rev = n_norev = 0
    with gzip.open(vcf, "rt") as fh:
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
    # CLNREVSTAT census over the whole file would need a second pass; report on hits only
    for recs in found.values():
        for _, _, rev in recs:
            if rev:
                n_rev += 1
            else:
                n_norev += 1
    print(f"VCF records scanned {n_rec:,}   sites hit {len(found):,}")
    print(f"  alt-records at those sites: with CLNREVSTAT {n_rev:,}, without {n_norev:,}")

    # classify each sample by the offset at which (ref,alt) matches
    print("\n" + "-" * 78)
    print("OFFSET AT WHICH THE COHORT KEY MATCHES A VCF RECORD")
    print("  offset 0  = the join augment_reviewstatus.py actually attempts")
    print("  offset -1 = cohort pos is one BASE RIGHT of the VCF's POS (the Start/PositionVCF gap)")
    print("-" * 78)

    table: dict[str, Counter] = {}
    details: dict[str, list[str]] = defaultdict(list)
    for name, g in groups.items():
        cnt: Counter = Counter()
        for vid, c, p, r, x in zip(g["variant_id"], g["chrom"], g["pos"], g["ref"], g["alt"]):
            cc, pp = norm_chrom(c), int(p)
            hit = None
            for off in range(-a.window, a.window + 1):
                for (vref, valt, _rev) in found.get((cc, pp + off), []):
                    if vref == r and valt == x:
                        hit = off
                        break
                if hit is not None:
                    break
            key = f"{hit:+d}" if hit is not None else "none"
            cnt[key] += 1
            if len(details[name]) < 4:
                near = found.get((cc, pp), []) or found.get((cc, pp - 1), [])
                details[name].append(
                    f"    {vid}  ->  offset {key}"
                    + (f"   | VCF@pos-1: {near[:1]}" if hit == -1 and near else "")
                )
        table[name] = cnt

    offs = [f"{o:+d}" for o in range(-a.window, a.window + 1)] + ["none"]
    hdr = f"{'group':<20}" + "".join(f"{o:>7}" for o in offs)
    print(hdr)
    print("-" * len(hdr))
    for name, cnt in table.items():
        print(f"{name:<20}" + "".join(f"{cnt.get(o, 0):>7}" for o in offs))

    print("\nexamples:")
    for name, lines in details.items():
        print(f"  {name}")
        for ln in lines:
            print(ln)

    # verdict
    print("\n" + "=" * 78)
    dblank = table.get("deletion  RS=blank", Counter())
    ctrl = Counter()
    for k in ("insertion RS=set  ", "SNV       RS=set  ", "MNV       RS=set  "):
        ctrl.update(table.get(k, Counter()))
    del_minus1 = dblank.get("-1", 0)
    del_total = sum(dblank.values()) or 1
    ctrl_zero = ctrl.get("+0", 0)
    ctrl_total = sum(ctrl.values()) or 1

    print(f"blank deletions matching at -1 : {del_minus1}/{del_total} ({100*del_minus1/del_total:.1f}%)")
    print(f"controls matching at  0        : {ctrl_zero}/{ctrl_total} ({100*ctrl_zero/ctrl_total:.1f}%)")
    print()
    if del_minus1 / del_total >= 0.8 and ctrl_zero / ctrl_total >= 0.8:
        print("VERDICT: CONFIRMED -- the cohort's `pos` is off by one for DELETIONS.")
        print("  `pos` came from variant_summary's `Start`; the VCF uses `PositionVCF`.")
        print()
        print("  ESCALATE. Every position-keyed annotation join (gnomAD, SpliceAI, phyloP,")
        print("  CADD, dbNSFP, 1000G, dbSNP, COSMIC, AlphaMissense) uses chrom:pos(:ref:alt)")
        print("  and therefore ALSO misses these 189,468 deletions. This was invisible")
        print("  because the review-tier filter discarded them before any feature audit.")
        print("  Audit deletion coverage per annotation BEFORE any cohort-v2 re-run.")
    elif dblank.get("+0", 0) / del_total >= 0.5:
        print("VERDICT: the join SHOULD have matched at offset 0. The VCF lacks CLNREVSTAT")
        print("  for these records, or the VCF release differs from the cohort's source.")
    elif dblank.get("none", 0) / del_total >= 0.5:
        print("VERDICT: no match at any offset in the window. Representation differs")
        print("  (left-alignment / trimming), or the VCF release is wrong. Widen --window")
        print("  and inspect the nearest records printed above.")
    else:
        print("VERDICT: mixed. Do not generalise -- inspect the examples above, raise --n.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
