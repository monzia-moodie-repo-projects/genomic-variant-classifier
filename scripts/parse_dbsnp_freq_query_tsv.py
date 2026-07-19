from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NCBI_TO_CHROM = {
    "NC_000001.11": "1",
    "NC_000002.12": "2",
    "NC_000003.12": "3",
    "NC_000004.12": "4",
    "NC_000005.10": "5",
    "NC_000006.12": "6",
    "NC_000007.14": "7",
    "NC_000008.11": "8",
    "NC_000009.12": "9",
    "NC_000010.11": "10",
    "NC_000011.10": "11",
    "NC_000012.12": "12",
    "NC_000013.11": "13",
    "NC_000014.9": "14",
    "NC_000015.10": "15",
    "NC_000016.10": "16",
    "NC_000017.11": "17",
    "NC_000018.10": "18",
    "NC_000019.10": "19",
    "NC_000020.11": "20",
    "NC_000021.9": "21",
    "NC_000022.11": "22",
    "NC_000023.11": "X",
    "NC_000024.10": "Y",
    "NC_012920.1": "MT",
}


def parse_freq_by_alt(alts: str, freq: str) -> dict[str, float]:
    alt_list = str(alts).split(",")
    best: dict[str, float] = {}

    if not freq or freq == ".":
        return {alt: 0.0 for alt in alt_list}

    for source_block in str(freq).split("|"):
        if ":" not in source_block:
            continue
        _, values = source_block.split(":", 1)
        vals = values.split(",")

        for i, alt in enumerate(alt_list, start=1):
            if i >= len(vals):
                continue
            v = vals[i]
            if v in {"", "."}:
                continue
            try:
                af = float(v)
            except ValueError:
                continue
            best[alt] = max(best.get(alt, 0.0), af)

    for alt in alt_list:
        best.setdefault(alt, 0.0)

    return best


def parse_query_tsv(path: Path) -> pd.DataFrame:
    rows = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 5:
                raise ValueError(f"{path}:{line_no}: expected 5 tab columns, got {len(parts)}")

            chrom_raw, pos, ref, alts, freq = parts
            chrom = NCBI_TO_CHROM.get(chrom_raw, chrom_raw)

            for alt, af in parse_freq_by_alt(alts, freq).items():
                rows.append({
                    "variant_id": f"{chrom}:{pos}:{ref}:{alt}",
                    "allele_freq": float(af),
                    "chrom": chrom,
                    "pos": int(pos),
                    "ref": ref,
                    "alt": alt,
                    "dbsnp_source_chrom": chrom_raw,
                })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "variant_id", "allele_freq", "chrom", "pos", "ref", "alt", "dbsnp_source_chrom"
        ])

    out["allele_freq"] = out["allele_freq"].clip(lower=0.0, upper=1.0)
    out = (
        out.sort_values(["variant_id", "allele_freq"], ascending=[True, False])
        .drop_duplicates("variant_id", keep="first")
        .reset_index(drop=True)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)

    if not inp.exists():
        raise SystemExit(f"missing input TSV: {inp}")

    df = parse_query_tsv(inp)

    assert {"variant_id", "allele_freq"}.issubset(df.columns)
    assert df["variant_id"].is_unique
    assert df["allele_freq"].between(0, 1).all()

    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(outp, index=False)

    print(f"wrote {outp}")
    print(f"rows={len(df)}")
    print(df.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
