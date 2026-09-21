"""Census of ClinVar GENEINFO gene associations in a pinned VCF, joined by VariationID.

scripts/patch_clinvar_alleles.py defined GENEINFO=([^:;]+), which captures only the
FIRST symbol and stops before its GeneID. ClinVarConnector._to_canonical then drops the
variant_summary GeneID column. This recovers every (symbol, GeneID) association the
source records, so the Gate A resolver can start from source GeneIDs rather than
display strings. Recovery from the same bytes; not reannotation.

Every entry must match `symbol:positive-integer`; anything else is counted as
malformed and reported, never coerced. Read-only; writes one JSON and one parquet.
"""
import argparse, collections, gzip, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

GI_RE = re.compile(r"(?:^|;)GENEINFO=([^;]+)")
# The GeneID is the integer after the LAST colon; a symbol may itself contain colons.
# NCBI gene_info confirms e.g. GeneID 111258505 has symbol "HHC2:066588" (a biological
# region). This is the unique parse under the grammar symbol:positive-integer.
ENTRY_RE = re.compile(r"(.+):([1-9][0-9]*)")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_geneinfo(info):
    m = GI_RE.search(info)
    if not m:
        return None
    out = []
    for entry in m.group(1).split("|"):
        mm = ENTRY_RE.fullmatch(entry)
        if not mm:
            raise ValueError(f"malformed GENEINFO entry: {entry!r}")
        if "|" in mm.group(1):
            raise ValueError(f"malformed GENEINFO entry: {entry!r}")
        out.append((mm.group(1), int(mm.group(2))))
    if len({g for _, g in out}) != len(out):
        raise ValueError("duplicate GeneID within one record")
    return out


def census(vcf):
    genes_per_record = collections.Counter()
    symbols_per_id = collections.defaultdict(set)
    ids_per_symbol = collections.defaultdict(set)
    malformed = collections.Counter()
    colon_symbols = collections.Counter()
    rows, records, with_gi = [], 0, 0
    seen_ids = set()
    opener = gzip.open if str(vcf).endswith(".gz") else open
    with opener(vcf, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t", 8)
            if len(c) < 8:
                raise ValueError("VCF line with fewer than 8 columns")
            records += 1
            vid = c[2]
            if vid in seen_ids:
                raise ValueError(f"duplicate VariationID in VCF: {vid}")
            seen_ids.add(vid)
            try:
                gi = parse_geneinfo(c[7])
            except ValueError as e:
                malformed[str(e)[:90]] += 1
                continue
            if gi is None:
                continue
            with_gi += 1
            genes_per_record[len(gi)] += 1
            for sym, gid in gi:
                if ":" in sym:
                    colon_symbols[(sym, gid)] += 1
                symbols_per_id[gid].add(sym)
                ids_per_symbol[sym].add(gid)
                rows.append((vid, sym, gid, len(gi)))
    n_mal = sum(malformed.values())
    if with_gi + n_mal + (records - with_gi - n_mal) != records:
        raise AssertionError("record states do not partition the file")
    return rows, {
        "records": records, "records_with_geneinfo": with_gi,
        "records_without_geneinfo": records - with_gi - n_mal,
        "records_with_malformed_geneinfo": n_mal,
        "malformed_examples": dict(malformed.most_common(10)),
        "genes_per_record": {str(k): v for k, v in sorted(genes_per_record.items())},
        "symbols_containing_colon": [{"symbol": k[0], "gene_id": k[1], "entries": v}
                                     for k, v in colon_symbols.most_common()],
        "distinct_gene_ids": len(symbols_per_id),
        "distinct_symbols": len(ids_per_symbol),
        "gene_ids_with_more_than_one_symbol": {str(k): sorted(v) for k, v in symbols_per_id.items() if len(v) > 1},
        "symbols_with_more_than_one_gene_id": {k: sorted(v) for k, v in ids_per_symbol.items() if len(v) > 1},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vcf", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-parquet", required=True)
    a = p.parse_args()
    for o in (a.output_json, a.output_parquet):
        if Path(o).exists():
            sys.exit(f"ABORT: {o} exists; refusing to overwrite")
    import pandas as pd
    rows, s = census(a.vcf)
    rel = pd.DataFrame(rows, columns=["variation_id", "source_symbol", "source_gene_id", "genes_in_record"])
    rel.to_parquet(a.output_parquet, index=False)
    s["provenance"] = {"vcf": str(Path(a.vcf).resolve()), "vcf_sha256": sha256_file(a.vcf),
                       "relation_parquet": str(Path(a.output_parquet).resolve()),
                       "relation_parquet_sha256": sha256_file(a.output_parquet),
                       "relation_rows": len(rel), "run_utc": datetime.now(timezone.utc).isoformat()}
    Path(a.output_json).write_text(json.dumps(s, indent=2), encoding="utf-8")
    print(f"records {s['records']:,} | with GENEINFO {s['records_with_geneinfo']:,} "
          f"| without {s['records_without_geneinfo']:,} | malformed {s['records_with_malformed_geneinfo']:,}")
    print("genes per record:", s["genes_per_record"])
    print(f"distinct GeneIDs {s['distinct_gene_ids']:,} | distinct symbols {s['distinct_symbols']:,}")
    print(f"GeneIDs with >1 symbol: {len(s['gene_ids_with_more_than_one_symbol']):,} "
          f"| symbols with >1 GeneID: {len(s['symbols_with_more_than_one_gene_id']):,}")
    for k, v in list(s["symbols_with_more_than_one_gene_id"].items())[:10]:
        print(f"   symbol {k!r} -> GeneIDs {v}")
    for c in s["symbols_containing_colon"]:
        print(f"   symbol containing ':' -> {c}")
    if s["malformed_examples"]:
        print("malformed examples:", s["malformed_examples"])
    print(f"relation rows written: {len(rel):,}")


if __name__ == "__main__":
    main()
