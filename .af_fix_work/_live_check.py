import importlib.util, tempfile, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location(
    "bap", str(Path("scripts") / "build_alphafold_parquet.py"))
bap = importlib.util.module_from_spec(spec); spec.loader.exec_module(bap)
with tempfile.TemporaryDirectory() as d:
    p = bap._download_cif("P04637", Path(d))
    if p is None:
        print("LIVE_FAIL: _download_cif returned None"); sys.exit(1)
    txt = p.read_text(encoding="utf-8")
    ok = p.name.endswith("model_v6.cif") and txt.lstrip().startswith("data_") and "_atom_site" in txt
    print("LIVE_NAME", p.name, "BYTES", len(txt))
    print("LIVE_OK" if ok else "LIVE_FAIL: unexpected name/content")
    sys.exit(0 if ok else 1)