"""Where this project keeps things -- one authority, three roots.

    project_root    where the source lives. Identity, not destination.
    artifact_root   where generated output goes.
    state_root      where mutable operational state persists.

This package exists because the repository accumulated FIVE independent
conventions for the same question -- `root: str = "."`, a hard-coded
PROJECT_ROOT, ADAPTATION_PROJECT_ROOT, Path(__file__).parent, and a bare
relative path -- and every new component invented a sixth.

Author: Monzia Moodie
"""

from __future__ import annotations