"""Acquire and freeze the identity bundle for RETROSPECTIVE HARMONISATION.

These resources are regenerated continually (NCBI daily; HGNC Tuesdays and Fridays),
so a download is a dated SNAPSHOT, not a named release. Per resource this records the
source URL, retrieval time, HTTP status, Last-Modified, ETag, declared Content-Length,
bytes actually received and SHA-256. A transfer whose received bytes differ from the
declared length is refused, so truncation cannot pass as a complete download.
Refuses to overwrite. Writes into a new directory named by retrieval TIMESTAMP (UTC),
so a failed run never blocks an immediate retry.
"""
import argparse, hashlib, json, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

RESOURCES = {
    "Homo_sapiens.gene_info.gz":
        "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
    "gene_history.gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_history.gz",
    "hgnc_complete_set.txt":
        "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
}


def fetch(url, dest):
    h = hashlib.sha256(); got = 0
    req = urllib.request.Request(url, headers={"User-Agent": "gvc-identity-bundle/1"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as out:
            status = r.status
            hdr = {k: r.headers.get(k) for k in ("Last-Modified", "ETag", "Content-Length", "Content-Type")}
            for block in iter(lambda: r.read(1 << 20), b""):
                out.write(block); h.update(block); got += len(block)
    except BaseException:
        # Any failure after the file was opened leaves a partial file that could later
        # be mistaken for a complete download. Remove it before re-raising.
        if dest.exists():
            dest.unlink()
        raise
    declared = int(hdr["Content-Length"]) if hdr.get("Content-Length") else None
    if declared is not None and declared != got:
        dest.unlink()
        raise IOError(f"truncated transfer: declared {declared} bytes, received {got}")
    return {"http_status": status, "headers": hdr, "bytes_received": got,
            "declared_length_matched": declared == got if declared is not None else None,
            "sha256": h.hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="parent directory, e.g. data/external/identity")
    ap.add_argument("--only", nargs="*", default=None, help="subset of resource names")
    a = ap.parse_args()
    when = datetime.now(timezone.utc)
    # Timestamped, not dated: a failed run must not block a retry for the rest of the day.
    out = Path(a.root) / when.strftime("%Y-%m-%dT%H%M%SZ")
    if out.exists():
        sys.exit(f"ABORT: {out} exists; refusing to overwrite a frozen bundle")
    names = a.only or list(RESOURCES)
    unknown = set(names) - set(RESOURCES)
    if unknown:
        sys.exit(f"ABORT: unknown resources {sorted(unknown)}")
    out.mkdir(parents=True)
    record = {"purpose": "retrospective harmonisation identity bundle",
              "temporal_mode": "retrospective_harmonization",
              "note": "snapshot of continually regenerated resources; retrieval time is not a release date",
              "resources": {}, "failures": {}}
    for name in names:
        url = RESOURCES[name]
        t = datetime.now(timezone.utc).isoformat()
        try:
            meta = fetch(url, out / name)
            record["resources"][name] = {"source_url": url, "retrieved_at_utc": t, **meta}
            print(f"OK    {name:28} {meta['bytes_received']:>13,} bytes  sha256 {meta['sha256'][:16]}  "
                  f"Last-Modified {meta['headers'].get('Last-Modified')}")
        except Exception as e:
            record["failures"][name] = {"source_url": url, "attempted_at_utc": t, "error": repr(e)[:300]}
            print(f"FAIL  {name:28} {e!r}"[:200])
    (out / "BUNDLE_PROVENANCE.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nWrote {out}  ({len(record['resources'])} ok, {len(record['failures'])} failed)")
    if record["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
