#!/usr/bin/env python3
"""patch_readme_consistency_ghcr.py -- close the two stragglers the enumeration grep
found (committed half-corrected in b4618cb) + soften the unverified GHCR/CI claim.

E1 (L115) and E2 (L135) are mandatory: they make the README consistent with the agent
count (7 core + drift suite) and DB framing already on main. E3 softens the
"image published to GHCR / full CI" claim to what is actually verifiable (Dockerfile +
FastAPI are REAL and kept; registry-publish + full CI are roadmap). If you DO push to
GHCR / run a build CI, drop E3 before committing -- review it in the diff.

Markers are unique-to-result (the recent_runs_cohort guard collided with the headline
phrase and silently skipped; never again). Per-edit resilient, backup-first, idempotent,
EOL-aware, ASCII-only. Author: Monzia Moodie.
"""
from __future__ import annotations
import datetime as _dt, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RM = REPO / "README.md"

EDITS = [
    # E1 -- 'Scientifically current' DB count -> consistent with intro/L39 "suite"
    ("Integrates 18 biological databases spanning",
     "Integrates a broad suite of biological databases spanning",
     "a broad suite of biological databases spanning", "L115 Scientifically-current: 18 -> suite"),
    # E2 -- 'Autonomously maintained' agent count -> consistent with badge/intro (growth-safe)
    ("A 13-agent monitoring layer (",
     "A monitoring layer of seven core agents plus a committed drift-detection suite (",
     "seven core agents plus a committed drift-detection suite", "L135 Autonomously-maintained: 13-agent -> 7 core + suite"),
    # E3 -- 'Production deployed' GHCR/CI -> verifiable (Docker + FastAPI kept; publish/CI -> roadmap)
    ("**Production deployed** -- FastAPI service on port 8000, multi-stage Dockerfile\n"
     "(builder / api / trainer targets), image published to GHCR\n"
     "(`ghcr.io/monzia-moodie/genomic-variant-api`), CI/CD via GitHub Actions with\n"
     "lockfile checks, full pytest sweep, Docker smoke tests, and monthly scheduled\n"
     "drift monitoring.",
     "**Containerised** -- FastAPI service on port 8000 with a multi-stage Dockerfile\n"
     "(builder / api / trainer targets) that builds the `genomic-variant-api` image locally,\n"
     "plus a scheduled GitHub Actions drift-monitoring workflow. Publishing the image to a\n"
     "container registry such as GHCR and a full build/test CI pipeline are roadmap items.",
     "builds the `genomic-variant-api` image locally", "L129 Production-deployed: GHCR/CI -> verifiable"),
]

def run(raw):
    log=[]
    for old,new,marker,label in EDITS:
        if marker in raw: log.append(("skip",label)); continue
        c = raw.count(old)
        if c!=1: log.append((f"MISS({c})",label)); continue
        raw = raw.replace(old,new,1); log.append(("ok",label))
    return raw, log

def main()->int:
    if not RM.exists(): print(f"ABORT: missing {RM}"); return 2
    raw0 = RM.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw0 else "\n"
    norm = (lambda s: s.replace("\n", nl)) if nl!="\n" else (lambda s: s)
    global EDITS
    EDITS = [(norm(o),norm(n),norm(m),l) for (o,n,m,l) in EDITS]
    ts=_dt.datetime.now().strftime("%Y%m%d_%H%M%S"); shutil.copy2(RM,f"{RM}.bak_{ts}")
    raw,log = run(raw0)
    ok=sum(1 for s,_ in log if s=="ok"); sk=sum(1 for s,_ in log if s=="skip"); ms=[l for s,l in log if s.startswith("MISS")]
    for s,l in log: print(f"  {s}: {l}")
    RM.write_bytes(raw.encode("utf-8"))
    print(f"\napplied={ok} skipped={sk} missed={len(ms)} (backup -> README.md.bak_{ts})")
    if ms:
        print("MISSED (paste the exact current line; anchor needs fixing):")
        for m in ms: print(f"  - {m}")
        return 1
    print("DONE.")
    return 0

if __name__=="__main__": sys.exit(main())
