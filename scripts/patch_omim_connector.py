#!/usr/bin/env python3
r"""patch_omim_connector.py

Make OMIMConnector derive BOTH disease counts AND the AD flag from genemap2.txt
(the file that actually carries gene->phenotype relationships). Fixes the long-
standing omim_n_diseases ~88 bug (it read mim2gene.txt, whose own header says it
is NOT a gene-phenotype table) and adds omim_n_diseases_molecular (the (3)
confirmed-molecular-basis disease count).

Three anchored edits to src/genomic_variant_classifier/data/omim.py:

  EDIT 1 — replace _parse_genemap2_autosomal_dominant(...) with _parse_genemap2(...)
           returning gene_symbol, omim_n_diseases, omim_n_diseases_molecular,
           omim_is_autosomal_dominant. SAME row-parse as the proven-37%-AD logic
           (len(parts)==len(header) kept verbatim) so AD coverage is unchanged;
           the two counts ride on the identical successfully-parsed rows.

  EDIT 2 — _get_gene_table: make genemap2 the PRIMARY source. Relax the
           mim2gene-None guard so genemap2 alone works; drop the broken
           _parse_mim2gene count path; return the 4-col genemap2 frame.

  EDIT 3 — annotate_dataframe: add fillna(0).astype(int) for the new
           omim_n_diseases_molecular column (mirror the existing two).

Anchors verified against live reads 10a/13a/13b/16a. ANCHOR-BASED, IDEMPOTENT.
CRLF-safe: backup written with newline="" (preserve), patched file newline="\n".
"""
from __future__ import annotations
import argparse
import py_compile
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/omim.py")
MARKER = "def _parse_genemap2(self"   # idempotency sentinel

# ---- EDIT 1: replace the whole _parse_genemap2_autosomal_dominant method ----
OLD_PARSE = '''    def _parse_genemap2_autosomal_dominant(self, path: Path) -> pd.DataFrame:
        """Parse genemap2.txt into gene-level autosomal-dominant flags."""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            logger.error("OMIMConnector: failed to read genemap2 %s: %s", path, exc)
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("# Chromosome") and "Approved Gene Symbol" in line and "Phenotypes" in line:
                header_idx = i
                break

        if header_idx is None:
            logger.warning("OMIMConnector: could not find genemap2 header in %s.", path)
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        header = lines[header_idx].lstrip("# ").split("\\t")
        rows = []

        for line in lines[header_idx + 1:]:
            if not line or line.startswith("#"):
                continue
            parts = line.split("\\t")
            if len(parts) == len(header):
                rows.append(parts)

        if not rows:
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        raw = pd.DataFrame(rows, columns=header)
        required = {"Approved Gene Symbol", "Phenotypes"}
        if not required.issubset(raw.columns):
            logger.warning("OMIMConnector: genemap2 missing required columns in %s.", path)
            return pd.DataFrame(columns=["gene_symbol", "omim_is_autosomal_dominant"])

        x = raw[["Approved Gene Symbol", "Phenotypes"]].copy()
        x = x.rename(columns={"Approved Gene Symbol": "gene_symbol"})
        x["gene_symbol"] = x["gene_symbol"].astype(str).str.strip()
        x["Phenotypes"] = x["Phenotypes"].astype(str)
        x = x[x["gene_symbol"].str.len() > 0].copy()

        x["omim_is_autosomal_dominant"] = (
            x["Phenotypes"]
            .str.contains("Autosomal dominant", case=False, na=False)
            .astype(int)
        )

        return (
            x.groupby("gene_symbol", as_index=False)["omim_is_autosomal_dominant"]
            .max()
        )'''

NEW_PARSE = '''    @staticmethod
    def _count_phenotypes(phenotypes: str) -> "tuple[int, int, int]":
        """Return (n_diseases_all, n_diseases_molecular, is_autosomal_dominant)
        for one gene's genemap2 Phenotypes string.

        - n_diseases_all:       count of ;-separated entries that are real diseases
                                (EXCLUDES [non-disease] bracketed entries: biomarkers/QTLs).
                                INCLUDES plain, {susceptibility}, and ?provisional entries.
        - n_diseases_molecular: count of entries containing the (3) mapping key
                                = molecular basis of the disorder is known (confirmed gene).
        - is_autosomal_dominant: 1 if any entry mentions "Autosomal dominant".

        Counting entries that CONTAIN "(3)" is robust to the 2/8953 entries that
        embed a stray "(N)" inside disease text (verified against live genemap2).
        """
        import re as _re
        s = str(phenotypes).strip()
        if not s:
            return 0, 0, 0
        n_all = 0
        n_mol = 0
        is_ad = 0
        for entry in s.split(";"):
            e = entry.strip()
            if not e:
                continue
            if e.startswith("["):          # [non-disease] — exclude from disease counts entirely
                continue
            n_all += 1
            if _re.search(r"\\(3\\)", e):
                n_mol += 1
            if "autosomal dominant" in e.lower():
                is_ad = 1
        return n_all, n_mol, is_ad

    def _parse_genemap2(self, path: Path) -> pd.DataFrame:
        """Parse genemap2.txt into gene-level OMIM features.

        Returns gene_symbol, omim_n_diseases, omim_n_diseases_molecular,
        omim_is_autosomal_dominant (one row per gene; aggregated across the gene's
        genemap2 rows). genemap2.txt is the file that actually carries
        gene->phenotype relationships (mim2gene.txt explicitly is NOT).
        """
        empty_cols = ["gene_symbol", "omim_n_diseases",
                      "omim_n_diseases_molecular", "omim_is_autosomal_dominant"]
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            logger.error("OMIMConnector: failed to read genemap2 %s: %s", path, exc)
            return pd.DataFrame(columns=empty_cols)

        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("# Chromosome") and "Approved Gene Symbol" in line and "Phenotypes" in line:
                header_idx = i
                break

        if header_idx is None:
            logger.warning("OMIMConnector: could not find genemap2 header in %s.", path)
            return pd.DataFrame(columns=empty_cols)

        header = lines[header_idx].lstrip("# ").split("\\t")
        rows = []

        for line in lines[header_idx + 1:]:
            if not line or line.startswith("#"):
                continue
            parts = line.split("\\t")
            if len(parts) == len(header):
                rows.append(parts)

        if not rows:
            return pd.DataFrame(columns=empty_cols)

        raw = pd.DataFrame(rows, columns=header)
        required = {"Approved Gene Symbol", "Phenotypes"}
        if not required.issubset(raw.columns):
            logger.warning("OMIMConnector: genemap2 missing required columns in %s.", path)
            return pd.DataFrame(columns=empty_cols)

        x = raw[["Approved Gene Symbol", "Phenotypes"]].copy()
        x = x.rename(columns={"Approved Gene Symbol": "gene_symbol"})
        x["gene_symbol"] = x["gene_symbol"].astype(str).str.strip()
        x["Phenotypes"] = x["Phenotypes"].astype(str)
        x = x[x["gene_symbol"].str.len() > 0].copy()

        counts = x["Phenotypes"].map(self._count_phenotypes)
        x["omim_n_diseases"]            = counts.map(lambda t: t[0]).astype(int)
        x["omim_n_diseases_molecular"]  = counts.map(lambda t: t[1]).astype(int)
        x["omim_is_autosomal_dominant"] = counts.map(lambda t: t[2]).astype(int)

        # One gene may appear on multiple genemap2 rows: take max per gene so a
        # gene's disease count / AD flag reflect its richest annotation.
        agg = (
            x.groupby("gene_symbol", as_index=False)[
                ["omim_n_diseases", "omim_n_diseases_molecular", "omim_is_autosomal_dominant"]
            ].max()
        )
        logger.info(
            "OMIMConnector: parsed genemap2 -> %d genes; %d with >=1 disease, "
            "%d with >=1 molecular (3) disease, %d autosomal-dominant.",
            len(agg),
            int((agg["omim_n_diseases"] > 0).sum()),
            int((agg["omim_n_diseases_molecular"] > 0).sum()),
            int((agg["omim_is_autosomal_dominant"] > 0).sum()),
        )
        return agg'''

# ---- EDIT 2: _get_gene_table body (genemap2-primary, relaxed guard) ----
OLD_GGT = '''    def _get_gene_table(self) -> pd.DataFrame:
        """Return a gene-level summary DataFrame, or empty if unavailable."""
        if self.mim2gene_path is None:
            logger.warning(
                "OMIMConnector: mim2gene_path not set — returning default values "
                "(omim_n_diseases=0, omim_is_autosomal_dominant=0).  "
                "Download mim2gene.txt from https://omim.org/downloads."
            )
            return pd.DataFrame(columns=["gene_symbol", "omim_n_diseases", "omim_is_autosomal_dominant"])

        cache_key = f"gene_table:mim2gene={self.mim2gene_path}:genemap2={self.genemap2_path}"
        cached = None
        if self.genemap2_path is None:
            cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("OMIMConnector: loaded gene table from cache (%d genes).", len(cached))
            return cached

        if not self.mim2gene_path.exists():
            logger.warning(
                "OMIMConnector: mim2gene.txt not found at '%s' — returning default values.",
                self.mim2gene_path,
            )
            return pd.DataFrame(columns=["gene_symbol", "omim_n_diseases", "omim_is_autosomal_dominant"])

        gene_table = self._parse_mim2gene(self.mim2gene_path)
        if self.genemap2_path is not None and self.genemap2_path.exists():
            ad_table = self._parse_genemap2_autosomal_dominant(self.genemap2_path)
            if not ad_table.empty:
                gene_table = gene_table.drop(columns=["omim_is_autosomal_dominant"], errors="ignore")
                gene_table = gene_table.merge(ad_table, on="gene_symbol", how="outer")
                gene_table["omim_n_diseases"] = (
                    gene_table["omim_n_diseases"].fillna(DEFAULT_N_DISEASES).astype(int)
                )
                gene_table["omim_is_autosomal_dominant"] = (
                    gene_table["omim_is_autosomal_dominant"].fillna(DEFAULT_IS_AD).astype(int)
                )
        if not gene_table.empty:
            if self.genemap2_path is None:
                self._save_cache(cache_key, gene_table)
                logger.info("OMIMConnector: parsed and cached %d genes.", len(gene_table))
            else:
                logger.info("OMIMConnector: parsed %d genes with genemap2 cache bypass.", len(gene_table))
        return gene_table'''

NEW_GGT = '''    def _get_gene_table(self) -> pd.DataFrame:
        """Return a gene-level summary DataFrame, or empty if unavailable.

        genemap2.txt is the PRIMARY (and sufficient) source: it carries the
        gene->phenotype relationships from which omim_n_diseases,
        omim_n_diseases_molecular and omim_is_autosomal_dominant are all derived.
        mim2gene.txt is an ID cross-reference whose own header states it is NOT a
        gene-phenotype table, so it is no longer used for the disease count.
        """
        empty_cols = ["gene_symbol", "omim_n_diseases",
                      "omim_n_diseases_molecular", "omim_is_autosomal_dominant"]

        if self.genemap2_path is None or not self.genemap2_path.exists():
            logger.warning(
                "OMIMConnector: genemap2.txt not available (path=%s) — returning default "
                "values (omim_n_diseases=0, omim_n_diseases_molecular=0, "
                "omim_is_autosomal_dominant=0).  Download genemap2.txt from "
                "https://omim.org/downloads.",
                self.genemap2_path,
            )
            return pd.DataFrame(columns=empty_cols)

        cache_key = f"gene_table:genemap2={self.genemap2_path}"
        cached = self._load_cache(cache_key)
        if cached is not None and not cached.empty:
            logger.info("OMIMConnector: loaded gene table from cache (%d genes).", len(cached))
            return cached

        gene_table = self._parse_genemap2(self.genemap2_path)
        if not gene_table.empty:
            self._save_cache(cache_key, gene_table)
            logger.info("OMIMConnector: parsed and cached %d genes from genemap2.", len(gene_table))
        return gene_table'''

# ---- EDIT 3: annotate_dataframe — add molecular fillna+astype ----
OLD_ANN = '''        result["omim_n_diseases"] = (
            result["omim_n_diseases"].fillna(DEFAULT_N_DISEASES).astype(int)
        )
        result["omim_is_autosomal_dominant"] = (
            result["omim_is_autosomal_dominant"].fillna(DEFAULT_IS_AD).astype(int)
        )'''

NEW_ANN = '''        result["omim_n_diseases"] = (
            result["omim_n_diseases"].fillna(DEFAULT_N_DISEASES).astype(int)
        )
        result["omim_n_diseases_molecular"] = (
            result["omim_n_diseases_molecular"].fillna(DEFAULT_N_DISEASES).astype(int)
        )
        result["omim_is_autosomal_dominant"] = (
            result["omim_is_autosomal_dominant"].fillna(DEFAULT_IS_AD).astype(int)
        )'''

# ---- EDIT 4: annotate_dataframe no-gene_symbol guard branch ----
OLD_GUARD = '''            result["omim_n_diseases"]           = pd.Series(dtype=int)
            result["omim_is_autosomal_dominant"] = pd.Series(dtype=int)'''
NEW_GUARD = '''            result["omim_n_diseases"]           = pd.Series(dtype=int)
            result["omim_n_diseases_molecular"] = pd.Series(dtype=int)
            result["omim_is_autosomal_dominant"] = pd.Series(dtype=int)'''

# ---- EDIT 5: annotate_dataframe gene_table.empty branch ----
OLD_EMPTY = '''            result["omim_n_diseases"]           = DEFAULT_N_DISEASES
            result["omim_is_autosomal_dominant"] = DEFAULT_IS_AD'''
NEW_EMPTY = '''            result["omim_n_diseases"]           = DEFAULT_N_DISEASES
            result["omim_n_diseases_molecular"] = DEFAULT_N_DISEASES
            result["omim_is_autosomal_dominant"] = DEFAULT_IS_AD'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ns = ap.parse_args()
    if not TARGET.exists():
        print(f"FAIL: {TARGET} not found."); return 2
    src = TARGET.read_text(encoding="utf-8")

    if MARKER in src:
        print("OK (idempotent): _parse_genemap2 already present."); return 0

    anchors = {"EDIT1 parse method": OLD_PARSE,
               "EDIT2 _get_gene_table": OLD_GGT,
               "EDIT3 annotate merge": OLD_ANN,
               "EDIT4 no-gene guard": OLD_GUARD,
               "EDIT5 empty branch": OLD_EMPTY}
    ok = True
    for name, anc in anchors.items():
        c = src.count(anc)
        if c != 1:
            print(f"FAIL: anchor '{name}' occurs {c}x (need 1)."); ok = False
    if not ok:
        return 3
    if ns.check:
        print("CHECK: all 3 connector anchors found exactly once."); print("RESULT: PASS (check)"); return 0

    patched = (src.replace(OLD_PARSE, NEW_PARSE, 1)
                  .replace(OLD_GGT, NEW_GGT, 1)
                  .replace(OLD_ANN, NEW_ANN, 1)
                  .replace(OLD_GUARD, NEW_GUARD, 1)
                  .replace(OLD_EMPTY, NEW_EMPTY, 1))

    backup = TARGET.with_suffix(".py.pre_genemap2_counts.bak")
    if not backup.exists():
        backup.write_text(src, encoding="utf-8", newline="")
        print(f"OK: backup -> {backup}")
    TARGET.write_text(patched, encoding="utf-8", newline="\n")

    after = TARGET.read_text(encoding="utf-8")
    checks = [
        ("_parse_genemap2 defined", "def _parse_genemap2(self" in after),
        ("_count_phenotypes helper", "_count_phenotypes" in after),
        ("old AD-only parser gone", "_parse_genemap2_autosomal_dominant" not in after),
        ("genemap2-primary guard", "genemap2.txt is the PRIMARY" in after),
        ("annotate molecular col", 'result["omim_n_diseases_molecular"]' in after),
        ("molecular in parse return", '"omim_n_diseases_molecular"' in after),
        ("guard branch molecular", after.count('result["omim_n_diseases_molecular"] = pd.Series(dtype=int)') == 1),
        ("empty branch molecular", 'result["omim_n_diseases_molecular"] = DEFAULT_N_DISEASES' in after),
        ("all 3 branches set molecular", after.count("omim_n_diseases_molecular") >= 5),
    ]
    allok = True
    for label, present in checks:
        print(f"  {'OK' if present else 'MISSING'}  {label}"); allok &= present
    try:
        py_compile.compile(str(TARGET), doraise=True); print("  OK  omim.py compiles")
    except py_compile.PyCompileError as exc:
        print(f"  FAIL  compile: {exc}"); allok = False
    print("RESULT:", "PASS" if allok else "FAIL")
    return 0 if allok else 5


if __name__ == "__main__":
    raise SystemExit(main())
