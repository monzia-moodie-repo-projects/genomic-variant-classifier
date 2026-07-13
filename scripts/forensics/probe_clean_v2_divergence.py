#!/usr/bin/env python
"""probe_clean_v2_divergence.py (2026-07-09)
187,245 clean.parquet rows are absent from v2 by full 5-tuple key, yet counts partition
(v2 == clean + structural). Distinguish a DIFFERENT ClinVar SNAPSHOT (H1) from KEY-FORMAT
DRIFT (H2) by comparing clean vs v2 on progressively looser keys:
  A. source_id set overlap                (snapshot: many source_ids differ; format: ~identical)
  B. (chrom,pos,ref,alt) without variant_id (isolates variant_id-string drift)
  C. (source_id,chrom,pos)                 (isolates allele/variant_id format, same variant)
  D. (source_id, ref, alt)                 (isolates pos/chrom format)
Also inspects a few rows that mismatch on the 5-tuple but MATCH on source_id, printing the
v2 and clean versions side by side to show EXACTLY which field differs. Pure evidence.
"""
import sys, argparse
print("=== probe_clean_v2_divergence START ===", flush=True)
try:
    import pandas as pd
    print("python:", sys.version.split()[0], "pandas:", pd.__version__, flush=True)
except Exception as e:
    print("FATAL pandas:", e, flush=True); sys.exit(11)

def clean_id(s):
    s=str(s).strip(); return s[:-2] if s.endswith(".0") else s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--v2", required=True)
    a = ap.parse_args()
    cl = pd.read_parquet(a.clean); v2 = pd.read_parquet(a.v2)
    for df in (cl, v2):
        df["source_id"] = df["source_id"].map(clean_id)
        df["chrom"] = df["chrom"].astype(str); df["pos"] = df["pos"].astype(int)
        df["ref"] = df["ref"].astype(str); df["alt"] = df["alt"].astype(str)
        df["variant_id"] = df["variant_id"].astype(str)
    print(f"clean {len(cl):,}  v2 {len(v2):,}", flush=True)

    def ov(setc, setv, label):
        only_c = len(setc - setv); only_v = len(setv - setc); inter = len(setc & setv)
        print(f"  [{label:32s}] shared={inter:>10,}  clean_only={only_c:>8,}  v2_only={only_v:>8,}", flush=True)
        return only_c, only_v

    print("\n--- overlap on progressively looser keys ---", flush=True)
    A = ov(set(cl["source_id"]), set(v2["source_id"]), "source_id")
    B = ov(set(zip(cl["chrom"],cl["pos"],cl["ref"],cl["alt"])),
           set(zip(v2["chrom"],v2["pos"],v2["ref"],v2["alt"])), "chrom,pos,ref,alt (no varid)")
    C = ov(set(zip(cl["source_id"],cl["chrom"],cl["pos"])),
           set(zip(v2["source_id"],v2["chrom"],v2["pos"])), "source_id,chrom,pos")
    D = ov(set(zip(cl["source_id"],cl["ref"],cl["alt"])),
           set(zip(v2["source_id"],v2["ref"],v2["alt"])), "source_id,ref,alt")
    E = ov(set(zip(cl["variant_id"],cl["chrom"],cl["pos"],cl["ref"],cl["alt"])),
           set(zip(v2["variant_id"],v2["chrom"],v2["pos"],v2["ref"],v2["alt"])), "full 5-tuple")

    print("\n--- rows matching on source_id but NOT on 5-tuple: what field differs? ---", flush=True)
    # find source_ids present in both, then compare their rows
    common_sids = set(cl["source_id"]) & set(v2["source_id"])
    cl_i = cl[cl["source_id"].isin(common_sids)].drop_duplicates("source_id").set_index("source_id")
    v2_i = v2[v2["source_id"].isin(common_sids)].drop_duplicates("source_id").set_index("source_id")
    shown = 0
    for sid in list(common_sids):
        c = cl_i.loc[sid]; v = v2_i.loc[sid]
        diffs = []
        for f in ("variant_id","chrom","pos","ref","alt"):
            if str(c[f]) != str(v[f]): diffs.append(f)
        if diffs:
            print(f"  sid={sid} differs on {diffs}:", flush=True)
            print(f"    clean: vid={c['variant_id']} {c['chrom']}:{c['pos']} {c['ref']}>{c['alt']}", flush=True)
            print(f"    v2   : vid={v['variant_id']} {v['chrom']}:{v['pos']} {v['ref']}>{v['alt']}", flush=True)
            shown += 1
            if shown >= 15: break
    if shown == 0:
        print("  (no source_id-matched rows differ on any field -> mismatch is pure set diff = snapshot)", flush=True)

    print("\nINTERPRETATION:", flush=True)
    print("  If source_id overlap is near-total (clean_only, v2_only both tiny) but 5-tuple", flush=True)
    print("    differs by ~187k -> H2 KEY-FORMAT DRIFT (same variants, different key strings).", flush=True)
    print("    The side-by-side rows above name the drifting field.", flush=True)
    print("  If source_id overlap ALSO shows ~187k clean_only/v2_only -> H1 DIFFERENT SNAPSHOT", flush=True)
    print("    (clean and v2 are different ClinVar vintages; not a safe drop-in).", flush=True)
    print("=== probe_clean_v2_divergence DONE ===", flush=True)

if __name__ == "__main__":
    main()
