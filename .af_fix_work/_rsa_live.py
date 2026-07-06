import importlib.util, tempfile, time, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("bap", str(Path("scripts")/"build_alphafold_parquet.py"))
bap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bap)
acc = "P01023"  # A2M, 1474 residues -- the 23s worst case pre-fix
with tempfile.TemporaryDirectory() as d:
    dd = Path(d)
    cif = bap._download_cif(acc, dd)
    if cif is None:
        print("LIVE_FAIL: could not fetch A2M"); sys.exit(1)
    sites = bap._fetch_active_sites(acc, dd)
    t = time.time(); rows = bap._extract_one(acc, cif, sites); dt = time.time() - t
    print(f"A2M extract_one {dt:.1f}s -> {len(rows)} rows")
    ok = (len(rows) == 1474) and (dt < 18.0)
    print("LIVE_OK" if ok else f"LIVE_FAIL: rows={len(rows)} (want 1474), dt={dt:.1f}s (want <18s)")
    sys.exit(0 if ok else 1)