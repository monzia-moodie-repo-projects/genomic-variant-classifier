import importlib.util, pathlib
_BUILDER = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_dbsnp_parquet.py"
spec = importlib.util.spec_from_file_location("b", _BUILDER)
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

# Confirm the preference list is exactly the coverage-first order with build-157 labels
assert b._POP_PREFERENCE == ["dbGaP_PopFreq","TOPMED","GnomAD_genomes","GnomAD_exomes",
                             "1000Genomes_30X","1000Genomes"], b._POP_PREFERENCE
print("preference order OK:", b._POP_PREFERENCE)

# Test 1: when dbGaP AND gnomAD both report, dbGaP wins (coverage-first)
freq = "GnomAD_genomes:0.9,0.1|dbGaP_PopFreq:0.8,0.2|TOPMED:0.85,0.15"
studies = b.parse_freq(freq)
f = b.alt_freq_for(studies, 1)
print(f"  dbGaP+gnomAD+TOPMED all present -> alt1={f} (expect 0.2 from dbGaP)")
assert f == 0.2, f

# Test 2: gnomAD label now MATCHES (the bug fix) -- when only GnomAD_genomes present
freq2 = "GnomAD_genomes:0.97,0.03"
f2 = b.alt_freq_for(b.parse_freq(freq2), 1)
print(f"  only GnomAD_genomes -> alt1={f2} (expect 0.03; old code would've missed the label)")
assert f2 == 0.03, f2

# Test 3: 1000Genomes_30X now preferred over plain 1000Genomes
freq3 = "1000Genomes:0.5,0.5|1000Genomes_30X:0.6,0.4"
f3 = b.alt_freq_for(b.parse_freq(freq3), 1)
print(f"  1000G + 1000G_30X -> alt1={f3} (expect 0.4 from _30X, higher priority)")
assert f3 == 0.4, f3

# Test 4: ancestry-specific pop used ONLY as last resort
freq4 = "KOREAN:0.99,0.01"   # no broad pop present
f4 = b.alt_freq_for(b.parse_freq(freq4), 1)
print(f"  only KOREAN (last resort) -> alt1={f4} (expect 0.01)")
assert f4 == 0.01, f4

# Test 5: broad pop has '.', falls through to next broad pop, NOT to ancestry pop first
freq5 = "dbGaP_PopFreq:1,.|TOPMED:0.9,0.1|KOREAN:0.5,0.5"
f5 = b.alt_freq_for(b.parse_freq(freq5), 1)
print(f"  dbGaP alt='.' -> falls to TOPMED -> alt1={f5} (expect 0.1, NOT KOREAN 0.5)")
assert f5 == 0.1, f5

# Test 6: the REAL line 1 from the VCF, re-evaluated under new preference
# FREQ=KOREAN:0.9891,0.0109,.|SGDP_PRJ:0,1,.|dbGaP_PopFreq:1,.,0
# alt1 (A): dbGaP alt1 is '.', no other broad pop -> falls to KOREAN 0.0109
# alt2 (C): dbGaP alt2 is 0 -> 0.0
freqR = "KOREAN:0.9891,0.0109,.|SGDP_PRJ:0,1,.|dbGaP_PopFreq:1,.,0"
sR = b.parse_freq(freqR)
fa1 = b.alt_freq_for(sR, 1); fa2 = b.alt_freq_for(sR, 2)
print(f"  REAL line1: alt1(A)={fa1} (expect 0.0109 KOREAN-fallback), alt2(C)={fa2} (expect 0.0 dbGaP)")
assert fa1 == 0.0109 and fa2 == 0.0, (fa1, fa2)

print("\nALL ORDERING TESTS PASSED")
