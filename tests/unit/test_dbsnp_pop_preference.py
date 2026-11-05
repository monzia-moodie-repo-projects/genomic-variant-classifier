"""scripts/build_dbsnp_parquet.py: the dbSNP FREQ population preference, coverage-first, with ancestry-specific studies last.

REWRITTEN 2026-11-05. Until then this file (added 2026-06-26, 499ccc6) was a script: its checks ran as module-level `assert`
statements while pytest COLLECTED it, and it defined no test -- so pytest reported nothing for it, a run of this file alone said
"no tests ran" (exit 5, measured 2026-11-04), and its print() calls wrote into collection's output. The same checks, with the same
inputs and expected values -- the preference order, and seven frequency cases (the old "Test 6" checked two alternate alleles of
one line) -- are tests now.

Author: Monzia Moodie
"""
from __future__ import annotations

import pytest

import build_dbsnp_parquet as builder          # scripts/ is on sys.path (tests/conftest.py)


def test_the_preference_is_the_coverage_first_order_with_the_build_157_labels():
    assert builder._POP_PREFERENCE == ["dbGaP_PopFreq", "TOPMED", "GnomAD_genomes", "GnomAD_exomes", "1000Genomes_30X",
                                       "1000Genomes"]


@pytest.mark.parametrize(("freq", "alt", "expected", "why"), [
    ("GnomAD_genomes:0.9,0.1|dbGaP_PopFreq:0.8,0.2|TOPMED:0.85,0.15", 1, 0.2, "dbGaP wins when dbGaP and gnomAD both report"),
    ("GnomAD_genomes:0.97,0.03", 1, 0.03, "the gnomAD label matches (the 2026-06-26 fix)"),
    ("1000Genomes:0.5,0.5|1000Genomes_30X:0.6,0.4", 1, 0.4, "1000Genomes_30X is preferred over plain 1000Genomes"),
    ("KOREAN:0.99,0.01", 1, 0.01, "an ancestry-specific study is used when it is the only one"),
    ("dbGaP_PopFreq:1,.|TOPMED:0.9,0.1|KOREAN:0.5,0.5", 1, 0.1, "a '.' falls through to the next broad study, not to KOREAN"),
    ("KOREAN:0.9891,0.0109,.|SGDP_PRJ:0,1,.|dbGaP_PopFreq:1,.,0", 1, 0.0109, "the build-157 file's line 1, alternate allele 1"),
    ("KOREAN:0.9891,0.0109,.|SGDP_PRJ:0,1,.|dbGaP_PopFreq:1,.,0", 2, 0.0, "the build-157 file's line 1, alternate allele 2"),
])
def test_the_alternate_allele_frequency_comes_from_the_most_preferred_study_that_reports_it(freq, alt, expected, why):
    assert builder.alt_freq_for(builder.parse_freq(freq), alt) == expected, why
