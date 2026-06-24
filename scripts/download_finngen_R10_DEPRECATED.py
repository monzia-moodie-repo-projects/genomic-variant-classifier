import requests, pathlib, sys

url  = "https://storage.googleapis.com/finngen-public-data-r10/annotations/R10_annotated_variants_v2.gz"
dest = pathlib.Path(r"C:\Projects\genomic-variant-classifier\data\external\finngen\R10_annotated_variants_v2.gz")
dest.parent.mkdir(parents=True, exist_ok=True)

# Resume support: check existing file size
existing = dest.stat().st_size if dest.exists() else 0
headers = {"Range": f"bytes={existing}-"} if existing else {}

print(f"Starting from byte {existing:,} ...")
with requests.get(url, headers=headers, stream=True, timeout=60) as r:
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0)) + existing
    downloaded = existing
    mode = "ab" if existing else "wb"
    with open(dest, mode) as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total else 0
            gb = downloaded / 1e9
            print(f"\r{gb:.2f} GB / {total/1e9:.2f} GB  ({pct:.1f}%)", end="", flush=True)
print(f"\nDone. Saved to {dest}")
