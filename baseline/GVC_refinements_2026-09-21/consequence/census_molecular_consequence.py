"""Census of ClinVar molecular-consequence annotations in a pinned VCF.

scripts/patch_clinvar_alleles.py kept only the LABEL of only the FIRST MC entry:
    first = m.group(1).split(",")[0]; return first.split("|")[1]
discarding every Sequence Ontology accession and every consequence after the
first. This recovers what was discarded, from the same source bytes, so it is
recovery -- not reannotation. Read-only; writes one JSON file.

The ClinVar VCF covers a SUBSET of ClinVar records. A record absent here is not
evidence that it has no consequence.
"""
import argparse, collections, gzip, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

MC_RE = re.compile(r"(?:^|;)MC=([^;]+)")
SO_RE = re.compile(r"SO:\d{7}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def parse_mc(info):
    """Return list of (so_id, label) or None if MC absent. Refuses malformed entries."""
    m = MC_RE.search(info)
    if not m:
        return None
    out = []
    for entry in m.group(1).split(","):
        parts = entry.split("|")
        if len(parts) != 2 or not SO_RE.fullmatch(parts[0]) or not parts[1]:
            raise ValueError(f"malformed MC entry: {entry!r}")
        out.append((parts[0], parts[1]))
    return out


def census(vcf, so_terms_path=None):
    so_terms = json.load(open(so_terms_path)) if so_terms_path else None
    pair = collections.Counter()
    label_to_so = collections.defaultdict(set)
    n_entries = collections.Counter()
    records = with_mc = multi_distinct_label = first_differs = 0
    malformed = collections.Counter()
    opener = gzip.open if str(vcf).endswith(".gz") else open
    with opener(vcf, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t", 8)
            if len(cols) < 8:
                raise ValueError("VCF line with fewer than 8 columns")
            records += 1
            try:
                mc = parse_mc(cols[7])
            except ValueError as e:
                malformed[str(e)[:80]] += 1
                continue
            if mc is None:
                continue
            with_mc += 1
            n_entries[len(mc)] += 1
            for so_id, label in mc:
                pair[(so_id, label)] += 1
                label_to_so[label].add(so_id)
            labels = {l for _, l in mc}
            if len(labels) > 1:
                multi_distinct_label += 1
    if with_mc + sum(malformed.values()) + (records - with_mc - sum(malformed.values())) != records:
        raise AssertionError("record states do not partition the file")
    ambiguous = {l: sorted(s) for l, s in label_to_so.items() if len(s) > 1}
    validation = None
    if so_terms is not None:
        ids = {s for s, _ in pair}
        validation = {
            "unknown_accessions": sorted(i for i in ids if i not in so_terms),
            "obsolete_accessions": sorted(i for i in ids if i in so_terms and so_terms[i]["obsolete"]),
            "label_differs_from_so_name": sorted(
                {f"{s}|{l}|so_name={so_terms[s]['name']}" for s, l in pair
                 if s in so_terms and so_terms[s]["name"] != l}),
        }
    return {
        "records": records, "records_with_mc": with_mc,
        # Absent and malformed are DIFFERENT states: a record with an unparseable
        # MC field has an annotation we failed to read, not no annotation.
        "records_without_mc": records - with_mc - sum(malformed.values()),
        "records_with_malformed_mc": sum(malformed.values()),
        "malformed_examples": dict(malformed.most_common(10)),
        "mc_entries_per_record": {str(k): v for k, v in sorted(n_entries.items())},
        "records_with_more_than_one_distinct_label": multi_distinct_label,
        "pairs": [{"so_id": s, "label": l, "entries": c} for (s, l), c in pair.most_common()],
        "labels_with_more_than_one_accession": ambiguous,
        "so_validation": validation,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vcf", required=True)
    p.add_argument("--so-terms", default=None, help="so_terms.json from the pinned ontology")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"ABORT: {out} exists; refusing to overwrite")
    result = census(a.vcf, a.so_terms)
    result["provenance"] = {"vcf": str(Path(a.vcf).resolve()), "vcf_sha256": sha256_file(a.vcf),
                            "so_terms_sha256": sha256_file(a.so_terms) if a.so_terms else None,
                            "run_utc": datetime.now(timezone.utc).isoformat()}
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"records {result['records']:,} | with MC {result['records_with_mc']:,} "
          f"| malformed {result['records_with_malformed_mc']:,}")
    print("entries per record:", result["mc_entries_per_record"])
    print(f"records with >1 distinct label: {result['records_with_more_than_one_distinct_label']:,}")
    print("labels mapped to more than one accession:", result["labels_with_more_than_one_accession"] or "none")
    if result["so_validation"]:
        v = result["so_validation"]
        print("unknown accessions:", v["unknown_accessions"] or "none")
        print("obsolete accessions:", v["obsolete_accessions"] or "none")
        print(f"(accession, label) pairs where the label is not the SO name: {len(v['label_differs_from_so_name'])}")
        for row in v["label_differs_from_so_name"]:
            print("  ", row)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
