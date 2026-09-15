"""Qualified source monitoring: does a check run, and can its evidence be qualified?

Author: Monzia Moodie

WHY THIS IS NOT UNDER monitoring/
=================================
MEASURED 2026-09-14 at 5cb1a330, two facts decided the placement.

FIRST: src/genomic_variant_classifier/monitoring/ has NO __init__.py. It is an
implicit namespace package, while drift/ inside it is a regular package.
Creating monitoring/__init__.py would CONVERT it, changing import semantics for
sixteen existing modules -- a behavioural change disguised as a new file.

SECOND: monitoring/registry.py already declares Category, Check, Verdict,
Source, REGISTRY, and all_sources / by_key / probeable / by_verdict /
critical_assets, over sources including _ALPHAMISSENSE_GCS, _CLINVAR_FTP,
_CLINVAR_SUMMARY and _LOVD_API -- a source registry with a Verdict enum and
probe URLs of the same kind this subsystem watches.

This subsystem declares Health, Reason, ContractFinding and a required-target
policy. Two registries and two verdict vocabularies in one package, with no
relationship between them, is the structure that produced an eight-target
agent carrying a seven-target docstring and a promised gnomAD target that was
never dispatched.

    monitoring/registry.py  answers  "is this source stale?"
    source_monitor/         answers  "did the check run, and can its
                                      evidence be qualified?"

Different questions. A future reader must not have to guess which registry
governs what.
"""
