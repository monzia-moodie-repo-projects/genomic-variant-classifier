import subprocess, sys, tempfile, json, time
from pathlib import Path
import pandas as pd
uni = "data/external/uniprot/uniprot_human_reviewed.parquet"
d = Path(tempfile.mkdtemp())
def run(workers, tag):
    out = d / f"{tag}.parquet"; cache = d / f"cache_{tag}"
    cmd = [sys.executable, "scripts/build_alphafold_parquet.py",
           "--uniprot-index", uni, "--out", str(out),
           "--cache-dir", str(cache), "--max-genes", "50", "--workers", str(workers)]
    t0=time.time(); r = subprocess.run(cmd, capture_output=True, text=True); dt=time.time()-t0
    if r.returncode != 0:
        print(f"[workers={workers}] exit {r.returncode}\n{r.stderr[-800:]}"); sys.exit(1)
    cov = out.parent / "alphafold_coverage.json"
    covj = json.loads(cov.read_text()).get("genes", {}) if cov.exists() else {}
    return out, covj, dt
o1, c1, t1 = run(1, "serial")
o8, c8, t8 = run(8, "par8")
df1 = pd.read_parquet(o1).sort_values(["uniprot_accession","residue_pos"]).reset_index(drop=True)
df8 = pd.read_parquet(o8).sort_values(["uniprot_accession","residue_pos"]).reset_index(drop=True)
rows_id = df1.equals(df8); cov_id = c1 == c8
print(f"serial: {len(df1)} rows in {t1:.0f}s | parallel(8): {len(df8)} rows in {t8:.0f}s | speedup={t1/max(t8,0.1):.1f}x")
print(f"rows_identical={rows_id} coverage_identical={cov_id}")
print("PAR_IDENTITY_OK" if (rows_id and cov_id) else "PAR_IDENTITY_FAIL")
sys.exit(0 if (rows_id and cov_id) else 1)
