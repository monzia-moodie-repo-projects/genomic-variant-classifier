import sys, json
from pathlib import Path
import pandas as pd
t=Path(sys.argv[1])
a=pd.read_parquet(t/"s.parquet"); b=pd.read_parquet(t/"p.parquet")
rows=a.equals(b)
try:
    ca=json.loads((t/"cov_s.json").read_text()).get("genes",{})
    cb=json.loads((t/"cov_p.json").read_text()).get("genes",{})
    cov=(ca==cb)
except Exception:
    cov=False
print(f"serial rows={len(a)} parallel rows={len(b)} rows_identical={rows} coverage_identical={cov}")
print("PAR_IDENTITY_OK" if (rows and cov) else "PAR_IDENTITY_FAIL")
sys.exit(0 if (rows and cov) else 1)
