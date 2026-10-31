# CORRECTION 2026-10-12 -- the suite still wrote into the repository after 2026-09-23

**Author: Monzia Moodie**
**Measured at tree:** e72a9092 (main after the run9a-prep record's merge)

**Applies to two tracked documents, each pinned at all sixty-four characters as they stand in that tree:**

```
docs/CHANGELOG.md
    489514e678f4861f8af1a741bfb3a02200b745d89c6faa277c4aeeaba6d94f41
docs/ROADMAP.md
    fcd3fb1aa26436d87c12578c8c5c0ad1fd1ed76701d1b2a8f7686775847ffc11
```

Corrections belong BESIDE records, never inside them. The CHANGELOG entry named below is not edited. The ROADMAP is the living
statement of present state, so its 2026-09-23 entry now carries a one-line pointer to this record.

## 1. What was written

- `docs/CHANGELOG.md`, entry "2026-09-23 (repair) -- test-isolation prerequisite: the suite stops writing into the repository": its
  title, and "Content-digest inventory of the whole tree before and after the full suite: unchanged."
- `docs/ROADMAP.md`, entry "2026-09-23 -- test-isolation prerequisite": "The suite no longer writes into the repository."

## 2. What was measured

- **On the owner's machine, 2026-10-11.** Both validation clones that had run the full suite held `models/best_gat.pt` and compiled
  files under `src/genomic_variant_classifier/__pycache__` and `src/genomic_variant_classifier/evaluation/__pycache__`. The owner's
  clean-up block H3 refused to delete the clones because of them, naming the first five (upload of 2026-10-11 20:16 local, lines
  563-564).
- **In the development sandbox, 2026-10-12.** The full suite was run at 8a65002d, which has the same tree. The environment was the
  owner's package versions, read from his validation's `environment.log`; collection gave 8,507, as on his machine. The run
  reproduced the conditions of the installer's full-suite stage:
  - `PYTHONPATH` pointed at the checkout's `src`;
  - `PYTHONDONTWRITEBYTECODE=1` was set;
  - the run had its own temporary root and `--basetemp`.
- **What that run left in the clone:**
  - `models/best_gat.pt`;
  - 14 `.pyc` files in those two directories, among them the five H3 named;
  - `renv/library/R-4.3/x86_64-pc-linux-gnu`.
- **What it left in its temporary root**, outside pytest's own folders: 173 `tmp*` folders, three modules PyTorch Geometric
  generated (`torch_geometric.nn.conv.*.py`), and PyTorch's `torchinductor_root` cache.

## 3. Why the 2026-09-23 verification did not see it

Its own entry says so: "the remaining failures are identical on both trees (PyTorch absent in the review environment)". Without
PyTorch the graph-neural-network tests cannot run. Those are the tests that wrote `models/best_gat.pt`, through `GNNTrainer`'s
default checkpoint path, which is relative to the current directory. The compiled files come from two child interpreters in
`tests/unit/test_bootstrap_reconciliation.py`. Each is started with a fresh environment, so `PYTHONDONTWRITEBYTECODE` does not
reach it; an inventory taken without that variable set would not single these files out.

## 4. What this changes

- The 2026-10-12 change stops both writes and adds a working-tree guard. The guard fails any test session that leaves the tree
  changed, so a claim like the 2026-09-23 one is now checked on every run rather than once.
- The `renv/library` directory is not changed by that unit. It is created when `tests/unit/test_inference_backend_trace.py` runs
  `Rscript` in the checkout at collection time: the checkout's `.Rprofile` activates renv. `renv/` is declared not this repository's
  content (`repository_hygiene.backup_artifacts.NOT_THIS_REPOSITORY`), so the guard does not enter it. This is recorded as open:
  `TEST-R-PROJECT-PROFILE-1`.
