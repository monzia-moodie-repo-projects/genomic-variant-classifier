"""
tests/unit/test_data_dir_not_shadowed.py
========================================
Guard tests for the recurring `data/`-directory shadow collision.

When a stray file -- or a dangling symlink/junction -- named `data` shadows
the `data/` directory in the repo root, every connector and pipeline that
constructs a FetchConfig / DataPrepConfig used to fail with a cryptic
``FileExistsError: [WinError 183]`` deep inside pathlib.mkdir, producing ~79
opaque failures instead of one clear signal. (See
docs/incidents/INCIDENT_2026-06-08_data-dir-shadow.md.)

These tests:
  1. Convert that whole failure class into ONE fast, unambiguous diagnostic.
  2. Lock the design property that constructing a FetchConfig is
     side-effect-free (creates no directories), which is what makes the suite
     robust to CWD / filesystem state.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# tests/unit/<this file>  ->  parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_repo_data_path_is_dir_or_absent():
    """`data` in the repo root must be a real directory (or not exist yet).

    A non-directory `data` (stray file / dangling symlink / junction) shadows
    the data/ tree and breaks every connector and pipeline construction.
    """
    data = REPO_ROOT / "data"
    if data.exists():
        assert data.is_dir(), (
            f"{data} exists but is NOT a directory. A stray file or dangling "
            f"symlink/junction is shadowing the data/ directory. This causes "
            f"cryptic WinError 183 across all connector/pipeline tests. "
            f"Recover with: (1) identify it "
            f"(`Get-Item data | Format-List Name,Attributes,LinkType,Target,Length`), "
            f"(2) move it aside, (3) `git checkout HEAD -- data` to restore "
            f"tracked contents, (4) recreate untracked data subdirs."
        )


def test_fetchconfig_construction_is_side_effect_free(tmp_path, monkeypatch):
    """Constructing FetchConfig() must not create directories on disk.

    Regression lock for patch_fetchconfig_lazy_mkdir.py: the cache directory is
    created lazily at first write (ensure_dir in _save_cache), never at
    construction. Skips cleanly if the connectors module cannot be imported.
    """
    pytest.importorskip("genomic_variant_classifier.data.database_connectors")
    from genomic_variant_classifier.data.database_connectors import FetchConfig

    monkeypatch.chdir(tmp_path)  # isolate CWD so we observe construction I/O
    cfg = FetchConfig()  # default cache_dir == Path("data/raw/cache") (relative)

    assert not (tmp_path / "data").exists(), (
        "FetchConfig() created data/ as a construction side effect. The cache "
        "dir must be created lazily at first write (ensure_dir/_save_cache), "
        "not in __post_init__. Re-run patch_fetchconfig_lazy_mkdir.py."
    )
    # cache_dir is still a usable Path pointing at the intended location
    assert str(cfg.cache_dir).replace("\\", "/").endswith("data/raw/cache")


def test_ensure_dir_helper_gives_clear_error_on_file_shadow(tmp_path):
    """ensure_dir() must raise a clear NotADirectoryError, never WinError 183,
    when a path component is shadowed by a file."""
    ed = pytest.importorskip(
        "genomic_variant_classifier.data.database_connectors"
    )
    ensure_dir = getattr(ed, "ensure_dir", None)
    if ensure_dir is None:
        pytest.skip("ensure_dir not present yet (apply patch_fetchconfig_lazy_mkdir.py)")

    shadow = tmp_path / "data"
    shadow.write_text("not a directory", encoding="utf-8")  # file shadows 'data'
    with pytest.raises(NotADirectoryError) as exc:
        ensure_dir(tmp_path / "data" / "raw" / "cache")
    assert "not a directory" in str(exc.value).lower()
