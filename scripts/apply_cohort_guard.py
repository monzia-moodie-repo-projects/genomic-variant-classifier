#!/usr/bin/env python3
"""Insert a fail-loud cohort guard into real_data_prep.py (B1 hardening).

Self-validating + idempotent + CRLF-safe. Inserts:
  * a @staticmethod _assert_clean_cohort(...) before def _load_and_label
  * a call self._assert_clean_cohort(df, clinvar_path) after the cohort load
Aborts (no write) if either anchor is absent or non-unique, or if already applied.
"""
from __future__ import annotations
import sys
from pathlib import Path

TARGET = Path("src/genomic_variant_classifier/data/real_data_prep.py")
ANCHOR_DEF = "    def _load_and_label(self, clinvar_path: str) -> pd.DataFrame:"
ANCHOR_LOG = '        logger.info("Loaded %d rows from %s.", len(df), clinvar_path)'
MARKER = "_assert_clean_cohort"

STATICMETHOD = '''    @staticmethod
    def _assert_clean_cohort(df: pd.DataFrame, source: str) -> None:
        """Fail loud on null/empty alleles or duplicate variant_id.

        See docs/incidents/INCIDENT_2026-05-31_null-key-leak.md. The clean cohort
        guarantees these properties; this guard prevents silent reintroduction of
        the leak by a future ClinVar re-pull (astype(str) below would otherwise
        collapse null alleles onto shared join keys).
        """
        _bad_tokens = ["", "nan", "none", "na", ".", "null", "-"]
        bad = (
            df["ref"].isna()
            | df["alt"].isna()
            | df["ref"].astype(str).str.strip().str.lower().isin(_bad_tokens)
            | df["alt"].astype(str).str.strip().str.lower().isin(_bad_tokens)
        )
        n_bad = int(bad.sum())
        if n_bad:
            raise ValueError(
                f"{n_bad} rows have null/empty ref or alt in {source}; "
                "run scripts/clean_cohort.py --apply and use clinvar_grch38_clean.parquet."
            )
        if bool(df["variant_id"].duplicated().any()):
            raise ValueError(
                f"duplicate variant_id in {source}; run scripts/clean_cohort.py --apply."
            )
'''
CALL = "        self._assert_clean_cohort(df, clinvar_path)"


def main() -> int:
    if not TARGET.exists():
        print(f"ABORT: {TARGET} not found (run from project root).")
        return 1
    raw = TARGET.read_bytes()
    text = raw.decode("utf-8")
    eol = "\r\n" if "\r\n" in text else "\n"

    if MARKER in text:
        print("ALREADY APPLIED: _assert_clean_cohort present; no change.")
        return 0

    # Work on a normalized-LF copy for anchoring, re-apply EOL at the end.
    norm = text.replace("\r\n", "\n")
    if norm.count(ANCHOR_DEF) != 1:
        print(f"ABORT: def anchor found {norm.count(ANCHOR_DEF)} times (need exactly 1).")
        return 1
    if norm.count(ANCHOR_LOG) != 1:
        print(f"ABORT: log anchor found {norm.count(ANCHOR_LOG)} times (need exactly 1).")
        return 1

    norm = norm.replace(ANCHOR_DEF, STATICMETHOD + "\n" + ANCHOR_DEF, 1)
    norm = norm.replace(ANCHOR_LOG, ANCHOR_LOG + "\n" + CALL, 1)

    out = norm.replace("\n", eol) if eol == "\r\n" else norm
    TARGET.write_bytes(out.encode("utf-8"))
    _eol_label = "CRLF" if eol == '\\r\\n' else "LF"
    print(f"APPLIED: staticmethod + call inserted (eol={_eol_label}).")
    print("Verify:  grep -n _assert_clean_cohort " + str(TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
