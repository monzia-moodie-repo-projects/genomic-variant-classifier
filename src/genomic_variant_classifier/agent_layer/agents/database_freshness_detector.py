"""
database_freshness_detector.py -- Monzia Moodie

Registry-driven upstream + local freshness detection. Pure logic (no BaseAgent / no SharedState) so it is
unit-testable with a mocked probe. Iterates monitoring.registry: for each PROBEABLE source it checks the
upstream for change vs a prior fingerprint; for EVERY source it checks local-asset presence/size and flags
known cruft (.OOMbak / .STALE_ / .pre_reviewstatus.bak / 78col.bak). Network calls degrade gracefully and
NEVER raise -- a failed probe yields status 'unreachable', not an exception (matches the agent-layer
convention of graceful external-I/O degradation).
"""
from __future__ import annotations

import ftplib
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from genomic_variant_classifier.monitoring import registry as R

# upstream statuses
UNCHANGED = "unchanged"
CHANGED = "changed"
UNREACHABLE = "unreachable"
MANUAL_SKIP = "manual_skip"
# local statuses
PRESENT = "present"
MISSING = "missing"
CRUFT = "cruft"

_CRUFT_MARKERS = (".OOMbak", ".STALE_", ".pre_reviewstatus.bak", "78col.bak")


@dataclass
class UpstreamResult:
    key: str
    status: str
    previous: str | None
    current: str | None
    detail: str


@dataclass
class LocalResult:
    key: str
    path: str | None
    status: str
    size_bytes: int | None
    detail: str


def _default_probe(source: R.Source) -> tuple[str | None, str]:
    """Return (fingerprint, detail) for a source's upstream. Raises on network failure (caller catches)."""
    method, url = source.check, source.upstream_url
    if method is R.Check.FTP_LISTING:
        rest = url[len("ftp://"):] if url.startswith("ftp://") else url
        host, _, path = rest.partition("/")
        with ftplib.FTP(host, timeout=20) as ftp:
            ftp.login()
            names = ftp.nlst("/" + path)
        vcfs = sorted(n for n in names if n.endswith(".vcf.gz"))
        return (vcfs[-1] if vcfs else None, f"{len(vcfs)} vcf(s)")
    if method is R.Check.HTTP_ETAG:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=20) as resp:
            fp = resp.headers.get("ETag") or resp.headers.get("Last-Modified") or str(resp.status)
        return (fp, "etag/last-modified")
    if method is R.Check.HTTP_HASH:
        with urllib.request.urlopen(url, timeout=20) as resp:
            body = resp.read()
        return (hashlib.md5(body).hexdigest(), f"{len(body)} bytes")
    if method is R.Check.GITHUB_RELEASE:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read())
        return (data.get("tag_name"), "github tag")
    return (None, "no automated probe")


def check_upstream(source: R.Source, last_seen: str | None, *, probe=_default_probe) -> UpstreamResult:
    if source.check is R.Check.MANUAL or not source.upstream_url:
        return UpstreamResult(source.key, MANUAL_SKIP, last_seen, None, "manual source -- no automated probe")
    try:
        current, detail = probe(source)
    except Exception as exc:  # graceful: a probe failure is 'unreachable', never raised
        return UpstreamResult(source.key, UNREACHABLE, last_seen, None, f"{type(exc).__name__}: {exc}")
    if current is None:
        return UpstreamResult(source.key, UNREACHABLE, last_seen, None, detail)
    if last_seen is None:
        return UpstreamResult(source.key, CHANGED, None, current, f"first observation ({detail})")
    if str(current) != str(last_seen):
        return UpstreamResult(source.key, CHANGED, last_seen, current, f"changed ({detail})")
    return UpstreamResult(source.key, UNCHANGED, last_seen, current, detail)


def check_local(source: R.Source, *, root: str = ".") -> LocalResult:
    if not source.local_path:
        status = MISSING if source.verdict is R.Verdict.ACTIVE else PRESENT
        return LocalResult(source.key, None, status, None, "no local_path declared")
    p = Path(root) / source.local_path
    if not p.exists():
        return LocalResult(source.key, source.local_path, MISSING, None, "absent on disk")
    if p.is_file():
        size = p.stat().st_size
        parent = p.parent
    else:
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        parent = p
    cruft = [f.name for f in parent.glob("*") if any(m in f.name for m in _CRUFT_MARKERS)]
    if cruft:
        return LocalResult(source.key, source.local_path, CRUFT, size, f"present; cruft alongside: {sorted(cruft)[:3]}")
    return LocalResult(source.key, source.local_path, PRESENT, size, "present")


def scan(state_freshness: dict, *, root: str = ".", probe=_default_probe) -> dict:
    """Full registry scan. state_freshness = prior {key: {'last_seen': ...}}. Returns a report dict."""
    upstream, local, changes = [], [], []
    for s in R.all_sources():
        prev = (state_freshness.get(s.key) or {}).get("last_seen")
        ur = check_upstream(s, prev, probe=probe)
        upstream.append(ur)
        if ur.status == CHANGED:
            changes.append(ur)
        local.append(check_local(s, root=root))
    return {"upstream": upstream, "local": local, "changes": changes}
