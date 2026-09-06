# Architecture decision records

**Author: Monzia Moodie**

This directory is the canonical location for accepted architecture decision
records. ADR-0003 reserved it and rejected two alternatives: `docs/adr/`, which
flattens a normative document class into a top-level special case, and
`docs/strategy/<date>/`, where an accepted record would be filed among the dated
proposals and evidence of the session that produced it, rather than outliving
them.

A canonical location for a normative document class is not a runtime
preference. Installers offer no override. Tests needing an alternate location
parameterise the lower-level function, never the production interface.

---

## The contract

Every accepted record is enforced by `tests/unit/test_adr_contract.py`, which
checks what a record must be rather than what it must say. Content is the
author's work; no test can validate reasoning.

**Filename.** `ADR-NNNN-lowercase-hyphenated-slug.md`. The pattern is what makes
a record findable by identifier instead of by remembering its title.

**Identifier.** Four digits, unique, contiguous from `0001`. A gap means a
record was deleted rather than superseded. A record is superseded by a later
record that says so; never by removal.

**Header.** Two shapes, deliberately distinct. The byline places the name inside
the bold, as the project's authorship rule requires; metadata places the key
inside and the value outside.

```
# ADR-NNNN -- Title

**Author: Monzia Moodie**
**Status:** accepted
**Date:** YYYY-MM-DD
**Authority:** normative
**Domains:** execution, project_state
**Measured at commit:** <short sha>
```

`Status`, `Date`, `Authority`, `Domains` and `Measured at commit` are required
and must appear within the first twelve lines. A field further down is body
text, not a declaration.

**Status** is one of `draft`, `accepted`, `superseded`, `rejected`. A superseded
record must name what superseded it.

**Domains** come from the vocabulary ADR-0001 defines. A record that governs no
single domain but defines the lattice itself declares `meta`. If a genuinely new
domain is needed, add it to the vocabulary and to ADR-0001 in the same commit --
not to one of them.

**Measured at commit** records the repository state the record's evidence was
gathered from. A decision made against a tree nobody can name is a decision
nobody can re-examine.

---

## Amending a record

A record's substance is not edited after acceptance. It is superseded by a later
record, and the superseding relationship is declared in both.

A missing or malformed **metadata field** is a different matter: an index entry
is not a historical claim. It may be repaired in place, and the repair must
declare itself with an `**Amended:**` field naming the finding, the reason, and
an explicit statement that no ruling, consequence or reasoning is altered.

`ADR-0001` carries exactly such an amendment, dated 2026-08-22.

---

## Accepted records

This list is bound to the directory by
`test_adr_contract.py::test_the_index_lists_exactly_the_records_present`. It
enumerates the records, which makes it a second copy of the record list -- the
same shape that once let `README.md` state a feature count in nine places with
four different values. A list nobody checks goes stale on a schedule, so this
one is checked: a record present but unlisted, or listed but absent, fails the
suite and names both.

| Record | Status | Domains | Subject |
|---|---|---|---|
| [ADR-0001-authority-and-contract-governance.md](ADR-0001-authority-and-contract-governance.md) | accepted | meta | Authority is typed by domain, not ranked globally. One semantic concept, one typed owner; derived presentation is not the source of truth; direct evidence outranks arithmetic reconstruction. |
| [ADR-0002-runtime-path-ownership.md](ADR-0002-runtime-path-ownership.md) | accepted | execution | Where runtime state lives, and what the repository certification surface is. The transaction journal resolves outside the repository, so hygiene needs no ignore exception. |
| [ADR-0003-project-knowledge-and-historical-record.md](ADR-0003-project-knowledge-and-historical-record.md) | accepted | execution, data_schema, project_state, historical_repository_record | Which artifact owns which question, and what each must never own. INVARIANT-HANDOFF-1, SuiteTransition by node identity, typed freshness, archival proof by blob object identifier. |
| [ADR-0004-repository-records-evidence-preservation-and-authority-succession.md](ADR-0004-repository-records-evidence-preservation-and-authority-succession.md) | accepted | meta, data_schema, project_state, historical_repository_record | Machine records are not documentation. The `records/` plane, ArtifactRole determining placement, disclosure and preservation as separate axes, verbatim import as a distinct validation policy, AUTHORITY-SUCCESSION-1, and the law that structure may not classify semantics. |
| [ADR-0005-repository-measurement-corpus-and-claim-semantics.md](ADR-0005-repository-measurement-corpus-and-claim-semantics.md) | accepted | data_schema | The Observation role ADR-0001 declared acquires a typed owner. A repository inspection declares its corpus, enumeration semantics, analysis completeness and evidentiary scope; measurement evidence is never state authority. |
