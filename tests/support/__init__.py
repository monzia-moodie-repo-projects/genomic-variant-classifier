"""Test-support packages.

`tests/` is a package (`tests/__init__.py` is tracked), and `tests/conftest.py`
inserts the repository root onto `sys.path`, so `from tests.support...` resolves
during collection. `tests/fixtures/`, `tests/integration/` and `tests/unit/`
each carry an `__init__.py`; this file keeps `tests/support/` consistent with
that convention rather than relying on namespace-package resolution.

Nothing here is collected: pytest matches `test_*.py`, and neither this file
nor `identity_laws.py` does.

Author: Monzia Moodie
"""
