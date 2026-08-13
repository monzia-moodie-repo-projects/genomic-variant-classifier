"""Dependency governance: one vocabulary, several analyzers.

    model.py                canonical identities and scope vocabulary
    requirements_parse.py   fail-closed requirements/lock parsing
    dependency_census.py    AST-based Python import census

Separate analyzers are fine; separate AUTHORITIES are not. Every identity,
scope and artifact-role concept lives in `model`, because two analyzers each
inventing a normalisation rule is how parallel vocabularies begin -- and that
had already started:

    requirements_parse     req.name.lower()
    dependency_census      p.lower().replace("-", "_")

Measured 2026-08-13: a naive `.lower()` disagrees with
`packaging.utils.canonicalize_name` on six of ten sampled distribution names,
and the hyphen-to-underscore rule disagrees with real package metadata on four
of seven installed distributions -- including pyBigWig, whose module keeps its
capitals while Python imports are case-sensitive.

Author: Monzia Moodie
"""

from __future__ import annotations