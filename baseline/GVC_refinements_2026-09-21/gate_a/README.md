# Gate A identity and consequence revision

This is a tested reference core, not a source downloader or a production adapter.
Run: python -m unittest discover -s . -p "test_*.py" -v
Twelve tests passed in the authoring environment.

Decisions:
- For the new retrospective repair study, freeze current human NCBI Gene resources,
  HGNC complete/withdrawn records and a dated Sequence Ontology release. Record full
  hashes, actual acquisition time and source effective release metadata.
- Historical prediction-time claims need a separately authenticated as-of bundle.
  A current download is not an old release merely because a record has an old date.
- Prefer original ClinVar GeneIDs; use NCBIGene IDs as primary keys in this
  ClinVar-derived adapter. HGNC IDs are authenticated cross-references. Retain
  NCBI-only human records. LOC digits must never be guessed as authoritative IDs.
- The GeneResolver expects active human records from validated adapters. Use a
  separate explicit gene-history adapter for discontinued/merged IDs, preserving
  the original ID and path. Do not silently resolve split/deleted records.
- Current exact symbols can resolve only uniquely; alias-only evidence is returned
  for review. Neither aliases nor regulatory edges are identity-equivalence edges.
- Prefer SO accessions on original source records. normalize_consequence validates
  against a pinned active ontology and preserves raw display labels. A label-only
  row remains needs_mapping, not zero. Obsolete terms require explicit migration.
- Preserve allele, transcript accession/version, assembly, source release/record,
  consequence accession and relation type in upstream adapter tables. The small
  functions here validate identity and vocabulary, not all this provenance.
- Construct partition components only from declared variant-to-gene associations;
  regulatory targets and disease evidence belong in different relation tables.
- weighting_attribution uses population covariance (ddof=0). Nonzero covariance
  means differing weighted means, not necessarily opposing signs or causality.

Included tests cover NCBI-only LOC retention, no numeric inference, alias ambiguity,
ID/symbol conflicts, direct IDs, human-species checks, cross-reference collisions,
SO accession use, label-only states, obsolete terms and exact weighting algebra.
They do not authenticate downloaded public resources or validate a full production
execution. No real-cohort refits were performed by this package.

Read the response accompanying this package for migration criteria, interpretation
of the reported size-bin contributions and qualifications on design-effect claims.
