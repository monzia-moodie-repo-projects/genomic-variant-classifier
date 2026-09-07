# CORRECTION 2026-09-04 part 4 -- a coordinate that does not separate, and a phenomenon enumerated from half of itself

**Author: Monzia Moodie**
**Measured at commit:** 5bc42e4
**Applies to two records, both pinned and both unchanged when this was written:**

```
docs/incidents/INCIDENT_2026-08-29_a-source-nobody-declared.md
    7e31d24e8170b7f57699de974f89c7283be5c372f89782c1ab4cb33d1e9a4070
docs/measurements/MEASUREMENT_2026-09-04_artifact-key-one-of-three.md
    e34169d501c34cffe16c9d7b043019a83dc8722dccddb84ff19ee309b25a3f5b
```

Corrections belong BESIDE records, never inside them. Neither file is edited.

---

## 1. `COORDINATE-CONTEXT-DOES-NOT-SEPARATE-ARTIFACT-KEYS-1`

`INCIDENT_2026-08-29`, line 23, states verbatim:

```
| `ClinVar/vcf` | 2 | `CoordinateContext` -- they differ by ASSEMBLY |
```

The table's column is "already modelled by". **It is not modelled.** MEASURED at
`5bc42e4` by construction:

```
two ClinVar VCFs, SAME assembly, no product
  REFUSED: artifact key(s) ['clinvar/vcf'] appear more than once

two ClinVar VCFs, DIFFERENT assemblies, no product
  REFUSED: artifact key(s) ['clinvar/vcf'] appear more than once
```

The full refusal, 257 characters, identical in both cases:

> artifact key(s) `['clinvar/vcf']` appear more than once. One analysis reads
> ONE artifact of each KIND per authority -- several kinds from one authority
> are several dependencies, which is measured practice: one module consumes
> three distinct ClinVar artifacts.

**The mechanism, enumerated rather than argued.**

```
SourceArtifactKey fields                    ['source', 'artifact_kind', 'product']
canonical_key of both                       ('clinvar', 'vcf', '')
GRCh38 identity.key == GRCh37 identity.key  True
the two IDENTITIES differ                   True
```

`CoordinateContext` lives on `SourceArtifactIdentity`. The KEY does not carry
it. So two identities that genuinely differ share ONE canonical key, and the
assembly distinguishes IDENTITY without permitting CO-MEMBERSHIP in one
manifest.

**The two guards are independent, proven by their messages.** With distinct
products supplied, the same cross-assembly pair reaches a DIFFERENT refusal:

> the manifest mixes genome assemblies `['GRCh37', 'GRCh38']`. Coordinates from
> different assemblies are not comparable, and a join across them would be
> silently wrong. Build-independent evidence is unaffected and may accompany
> any assembly.

The cross-assembly case never reaches the assembly check; it is refused on key
duplication first. A record calling the pair "already modelled by
`CoordinateContext`" is describing the wrong axis.

### Why the earlier claim was plausible

`CoordinateContext` DOES distinguish the two artifacts as identities, and the
2026-08-29 census was classifying collisions by what tells the members apart.
That question and "can they coexist in one evidence manifest" are different,
and the table's column heading asks the second while its entry answers the
first.

---

## 2. `PHENOMENON-A-ENUMERATED-FROM-ONE-INSTANCE-OF-TWO-1`

`INCIDENT_2026-08-28`, line 62, names TWO instances of phenomenon A:

> **A. Several published products.** GENCODE's three FASTAs; **ClinVar's
> `.vcf.gz` and `_GRCh38.vcf.gz`.** A product coordinate is genuinely missing.

`MEASUREMENT_2026-09-04_artifact-key-one-of-three.md` cites only the first. Its
line 37 reads `A. several PUBLISHED PRODUCTS   GENCODE's three FASTAs`, and its
section 2 reproduces the GENCODE case alone at lines 54 to 61.

**The conclusion survives; the enumeration did not.** MEASURED at `5bc42e4`,
the ClinVar instance behaves exactly as the GENCODE instance does:

```
two ClinVar VCFs, no product          REFUSED on the key
the same two, each with its product   ACCEPTED
                                        clinvar/vcf/clinvar
                                        clinvar/vcf/clinvar_GRCh38
```

So phenomenon A is resolved for BOTH of its instances by the `product` field,
and the measurement record's finding-level claim stands. What was wrong is that
it asserted a claim about A while having reproduced half of A.

### Why this matters more than a missing example

The record's section 5 states that `ARTIFACT-KEY-INSUFFICIENT-1` survives on
phenomena B and C. That conclusion depends on A being FULLY resolved. Had the
ClinVar instance behaved differently, the record would have understated what
remains open while appearing to have measured it.

A count of instances is a claim; an enumeration is a check.
`COUNTED-LINES-NOT-ITEMS-1` in its other direction.

---

## 3. What this does NOT change

`ARTIFACT-KEY-INSUFFICIENT-1` remains OPEN, on B and C unchanged:

```
B  partitioned members        AMBIGUOUS -- three EVE partitions as products and
                              the whole score set as one artifact are BOTH accepted
C  project-derived misattributed  INEXPRESSIBLE -- no field distinguishes a
                              publisher's bytes from this project's
```

`PARTITION-AXIS-UNDECIDED-1` and `PUBLISHER-VERSUS-PROJECT-AXIS-ABSENT-1` stand
as registered.

`INCIDENT_2026-08-28`'s ruling stands: **Phase 1C must not persist a source
manifest yet.** A manifest persisted today could still record
`clinvar/primary_release/clean_DRIVE` as a ClinVar publication.

`DRIFT-SOURCE-KERNEL-HAS-NO-PRODUCTION-CALLER-1` remains open and blocked on
that ruling.

---

## 4. What is NOT claimed

That `INCIDENT_2026-08-29`'s other three collision rows were re-measured. Only
the `ClinVar/vcf` row was. `ClinVar/primary_release`, `EVE/csv` and
`GENCODE/sequence_fasta` carry their own classifications and this correction
establishes nothing about the first two beyond what the 2026-09-04 measurement
already recorded.

That the key SHOULD carry the coordinate context. It does not, and whether it
should is a design question this correction does not answer -- the same
question `PARTITION-AXIS-UNDECIDED-1` and
`PUBLISHER-VERSUS-PROJECT-AXIS-ABSENT-1` leave open.

That `product` is the right model for phenomenon A. It REPRESENTS both
instances; whether the schema should instead declare separate sources, the form
`mim2gene` and `omim` already take among the 33 declarations, is unanswered.

---

## 5. How both were found

The first: I stated that `CoordinateContext` "models identity, not
co-membership" as a prediction BEFORE running the cross-assembly case, then ran
it. The prediction was right and the record it contradicts is eight days old.

The second: reading `INCIDENT_2026-08-28` section 1 line by line to determine
which finding owns which phenomenon. That read answered the attribution
question -- all three phenomena are inside section 1, so
`ARTIFACT-KEY-INSUFFICIENT-1` owns all three and the measurement record's
attribution is correct -- and incidentally exposed that line 62 names an
instance I had never reproduced.

The attribution check was the one I set out to do. The enumeration defect was
found by reading the line rather than the section heading, which is the
discipline the whole session has been about.
