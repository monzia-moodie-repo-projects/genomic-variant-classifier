import pandas as pd

LOVD = r"C:\Projects\genomic-variant-classifier\data\external\lovd\lovd_all_variants.parquet"
CLINVAR = r"C:\Projects\genomic-variant-classifier\models\v1\clinvar_enriched.parquet"

lovd = pd.read_parquet(LOVD)
clin = pd.read_parquet(CLINVAR)

print(f"LOVD rows:    {len(lovd):,}")
print(f"ClinVar rows: {len(clin):,}")
print()

# Right-side keys (LOVD) — replicates connector's _load
parts = lovd["variant_id"].str.split(":", expand=True)
lovd["_chrom"] = parts[0].astype(str).str.lstrip("chr")
lovd["_pos"]   = parts[1].astype(str)
lovd["_ref"]   = parts[2].astype(str)
lovd["_alt"]   = parts[3].astype(str)

# Left-side keys (ClinVar) — replicates connector's annotate_dataframe
clin = clin.copy()
clin["_chrom"] = clin["chrom"].astype(str).str.lstrip("chr")
clin["_pos"]   = clin["pos"].astype(str)
clin["_ref"]   = clin["ref"].astype(str)
clin["_alt"]   = clin["alt"].astype(str)

print("LOVD key dtypes:")
print(lovd[["_chrom","_pos","_ref","_alt"]].dtypes.to_string())
print()
print("ClinVar key dtypes:")
print(clin[["_chrom","_pos","_ref","_alt"]].dtypes.to_string())
print()

print("LOVD first 3 keys:")
print(lovd[["_chrom","_pos","_ref","_alt"]].head(3).to_string(index=False))
print()
print("ClinVar first 3 keys:")
print(clin[["_chrom","_pos","_ref","_alt"]].head(3).to_string(index=False))
print()

lovd_chroms = set(lovd["_chrom"].unique())
clin_chroms = set(clin["_chrom"].unique())
print(f"LOVD chroms:           {sorted(lovd_chroms)}")
print(f"ClinVar chroms (count): {len(clin_chroms)}")
print(f"Chroms in common:       {sorted(lovd_chroms & clin_chroms)}")
print()

LOVD_GENES = sorted(lovd["gene_symbol"].unique())
print(f"LOVD genes: {LOVD_GENES}")
clin_in_lovd_genes = clin[clin["gene_symbol"].isin(LOVD_GENES)]
print(f"ClinVar rows in LOVD genes: {len(clin_in_lovd_genes):,}")
print()

merged = clin_in_lovd_genes.merge(
    lovd[["_chrom","_pos","_ref","_alt","classification_raw"]],
    on=["_chrom","_pos","_ref","_alt"],
    how="inner",
)
print(f"Inner-join matches: {len(merged):,}")
print()

if len(merged) == 0:
    print("=== Zero matches. BRCA1 side-by-side: ===")
    lovd_brca1 = lovd[lovd["gene_symbol"] == "BRCA1"].head(3)
    clin_brca1 = clin[clin["gene_symbol"] == "BRCA1"].head(3)
    print("LOVD BRCA1 (first 3):")
    print(lovd_brca1[["_chrom","_pos","_ref","_alt","classification_raw"]].to_string(index=False))
    print()
    print("ClinVar BRCA1 (first 3):")
    print(clin_brca1[["_chrom","_pos","_ref","_alt"]].to_string(index=False))
