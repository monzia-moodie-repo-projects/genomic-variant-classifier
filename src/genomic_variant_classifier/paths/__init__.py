"""Where this project keeps things -- one authority, four roots.

    project_root    where the source lives. Identity, not destination.
    artifact_root   where generated output goes.
    state_root      where mutable operational state persists, beside the
                    checkout it belongs to.
    cache_root      machine-scoped, OUTSIDE any repository. Where an
                    in-flight installer transaction records its preimages,
                    so an interrupted run survives a working-tree reset.

This package exists because the repository accumulated FIVE independent
conventions for the same question -- `root: str = "."`, a hard-coded
PROJECT_ROOT, ADAPTATION_PROJECT_ROOT, Path(__file__).parent, and a bare
relative path -- and every new component invented a sixth.

A path derives from the authority that owns what the path contains. That is
why these are four domains rather than one root with four subdirectories:
repository identity, artifact destination, checkout state and machine cache
answer different questions and diverge under a real deployment.

    OUTPUT-ROOT-CONFLATION-1 (f89ce6b) moved the report directories off
    project_root, because where output goes is a deployment decision rather
    than a fact about where source lives.

    INSTALLER-TRANSACTION-1 step 2 (05f1a72) added cache_root, because a
    rollback journal must outlive the checkout it repairs.

THIS DOCSTRING UNDERCOUNTED THE DOMAINS UNTIL 2026-08-19. cache_root landed at
05f1a72 and the enumeration here was not re-derived -- the same defect as a
count stated once and carried forward. Corrected in place rather than
superseded: this describes CURRENT STRUCTURE, not what was believed at a past
moment. Records get corrections beside them; live descriptions get fixed.

Author: Monzia Moodie
"""

from __future__ import annotations