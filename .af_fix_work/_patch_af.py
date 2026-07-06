from __future__ import annotations
import io, sys
from pathlib import Path

BUILDER = Path(sys.argv[1])
TESTS = Path(sys.argv[2])
WORK = Path(sys.argv[3])
NEW_CONST = (WORK / "_new_constant.txt").read_text(encoding="utf-8").rstrip("\n")
NEW_FUNCS = (WORK / "_new_funcs.txt").read_text(encoding="utf-8").rstrip("\n")
NEW_TESTS = (WORK / "_new_tests.txt").read_text(encoding="utf-8")


def read(p):
    with io.open(p, "r", encoding="utf-8", newline="\n") as fh:
        return fh.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)


def guarded_replace(src, old, new, label):
    n = src.count(old)
    if n != 1:
        raise SystemExit("ABORT [%s]: expected exactly 1 match, found %d. No changes written." % (label, n))
    return src.replace(old, new)


src = read(BUILDER)
if 'if __name__ == "__main__":' not in src:
    raise SystemExit('ABORT: no `if __name__ == "__main__":` guard in builder; importlib test would execute main().')

R1_OLD = 'ALPHAFOLD_CIF_URL = "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.cif"'
R2_OLD = (
    'def _download_cif(accession: str, cache_dir: Path) -> Optional[Path]:\n'
    '    """Download+cache an AlphaFold CIF; reuse if present. Returns path or None."""\n'
    '    cache_file = cache_dir / f"AF-{accession}-F1-model_v4.cif"\n'
    '    if cache_file.exists() and cache_file.stat().st_size > 0:\n'
    '        return cache_file\n'
    '    url = ALPHAFOLD_CIF_URL.format(accession=accession)\n'
    '    try:\n'
    '        resp = requests.get(url, timeout=_REQUEST_TIMEOUT)\n'
    '    except Exception as exc:\n'
    '        logger.debug("download failed for %s: %s", accession, exc)\n'
    '        return None\n'
    '    if not resp.ok or not resp.text:\n'
    '        return None\n'
    '    cache_file.write_text(resp.text, encoding="utf-8")\n'
    '    time.sleep(_POLITE_DELAY_S)\n'
    '    return cache_file'
)

src = guarded_replace(src, R1_OLD, NEW_CONST, "R1 constant->API")
src = guarded_replace(src, R2_OLD, NEW_FUNCS, "R2 _download_cif rewrite")
if "model_v4.cif" in src:
    raise SystemExit("ABORT: 'model_v4.cif' still present in builder after patch.")
write(BUILDER, src)
print("builder patched OK")

tsrc = read(TESTS)
if "test_download_cif_resolves_api_and_writes_server_version" in tsrc:
    print("tests already contain fetch-path block; skipping append")
else:
    if not tsrc.endswith("\n"):
        tsrc += "\n"
    write(TESTS, tsrc + NEW_TESTS)
    print("tests appended OK")