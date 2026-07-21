#!/usr/bin/env python3
"""audit_disk_census.py -- correct whole-volume census. READ ONLY.

RENAMED 2026-07-21 from audit_disk_reclaim_v3_2026-07-20.py. A tool that lives
in the repository must not carry a version number and a date in its filename:
the project's own data-layout standard forbids version suffixes on directory
names for exactly this reason -- the version belongs in history, not in the
name. The old name also implied "reclaim", which it never did; it deletes
nothing.

WHY VERSION 3 EXISTS -- TWO DEFECTS IN VERSION 2, BOTH MINE, BOTH SERIOUS
=========================================================================

DEFECT 3 -- Windows directory JUNCTIONS were followed. (Critical.)
-------------------------------------------------------------------
Version 2's walker guarded against cycles with `DirEntry.is_symlink()` and its
docstring claimed it "never follows links or reparse points", naming Google
Drive File Stream junctions specifically. That claim was FALSE.

On Windows a symbolic link carries reparse tag IO_REPARSE_TAG_SYMLINK, and
`is_symlink()` detects it. A DIRECTORY JUNCTION carries a different tag,
IO_REPARSE_TAG_MOUNT_POINT, and `is_symlink()` returns False for it. Windows
ships several legacy compatibility junctions that are therefore invisible to
that check:

    C:\\Documents and Settings              -> C:\\Users
    C:\\Users\\<u>\\Local Settings             -> C:\\Users\\<u>\\AppData\\Local
    C:\\Users\\<u>\\AppData\\Local\\Application Data -> C:\\Users\\<u>\\AppData\\Local   (SELF)
    C:\\ProgramData\\Application Data         -> C:\\ProgramData                    (SELF)

The last two point at their own parent. Version 2 recursed into them forever.

Measured consequences on 2026-07-20, on a 935.59 GiB volume:

    C:\\Documents and Settings   reported 5557.03 GiB   =  5.94x the whole volume
    C:\\ProgramData              reported 2039.15 GiB   =  2.18x the whole volume
    census total                reported 5593.90 GiB   =  5.98x the whole volume
    reconciliation difference   reported -502.2 %      =  negative; impossible
    docker_data.vhdx (73.49 GiB) listed 30 times, each one "Application Data"
      level deeper than the last -- 2204.70 GiB of double-counting from ONE file
    31,083,584 files walked, elapsed exactly 1800.0 s -- the walk never finished

THE FIX. Three independent guards, because one is not enough:
  1. Reparse-point detection proper: DirEntry.is_junction() (Python 3.12+,
     which this project runs), falling back to the FILE_ATTRIBUTE_REPARSE_POINT
     bit in st_file_attributes, plus is_symlink(). Any of the three excludes.
  2. A visited set keyed on (st_dev, st_ino). A directory already accounted for
     is never descended twice, whatever path led to it.
  3. A hard depth ceiling. If guards 1 and 2 are ever defeated by a filesystem
     this code has not met, the walk still terminates.

DEFECT 5 -- the data breakdown reused the census walker and reported only
loose files. (Found 2026-07-21.)
--------------------------------------------------------------------------
`main()` passed the SAME Walker to `data_breakdown()` that had already walked
the whole volume, including C:\\Projects\\genomic-variant-classifier. That walk
recursed into data/ and added every nested directory to `_seen_dirs`.
`size_of()` skips any child directory already in that set, and does NOT check
the root it is handed -- so each subtotal counted ONLY the files sitting loose
in that directory, silently omitting every subdirectory.

Measured 2026-07-21 against an independent measurement of the same tree taken
the same hour:

    directory            true      in subdirs   loose only   v3 reported
    data/external       75.18 GiB   75.18 GiB     0.00 GiB     0.003 GiB
    data/processed       3.50 GiB    0.54 GiB     2.96 GiB     2.950 GiB
    data/raw            19.80 GiB   19.80 GiB     0.00 GiB     0.000 GiB
    data/_drift_check    0.26 GiB    0.00 GiB     0.26 GiB     0.264 GiB

    reported subtotal 3.21 GiB against a true 98.75 GiB -- short by 95.54 GiB.

The de-duplication is itself correct, and necessary: in a CENSUS, where the sum
over many roots must equal one volume, an overlapping root must not be counted
twice. The defect was applying census semantics to an INDEPENDENT measurement,
where the only question is how large one directory is, regardless of what was
walked before.

Worse than wrong: the section printed "Compare the figure above against 161.38
GiB" directly beneath its own bad number, inviting the reader to conclude that
158 GiB had been reclaimed. Nothing had.

size_of() now takes an explicit `independent` flag and the two semantics are
NAMED rather than implied. Cycle safety holds in both modes; only the SCOPE of
the visited sets differs.

DEFECT 4 -- after the time budget expired, zeros were reported as measurements.
--------------------------------------------------------------------------------
Version 2's `_tick` returned False once past the deadline, so `size_of` broke
out immediately and returned (0, 0). Every section that ran AFTER the budget was
exhausted printed confident zeros. In the 2026-07-20 run the repository data
directory printed fifteen subdirectories at "0.0 MiB, 0 files" -- the same
directory version 1 had measured at 161.38 GiB across 15,260 files. Duplicate
detection likewise reported "none found". Those were not measurements. They
were the absence of measurement, formatted to look identical to one.

THE FIX. `size_of` returns an explicit `complete` flag. Nothing that is not
complete is ever printed as a number; it prints INCOMPLETE and is excluded from
every total. A measurement and a non-measurement must not look alike.

ALSO NEW IN VERSION 3
---------------------
  - Hard-link de-duplication by (st_dev, st_ino). WinSxS is a hard-link farm;
    counting apparent size overstates real occupancy substantially.
  - Default mode is TARGETED and fast (roughly one to three minutes). The full
    volume walk is opt-in via --full, because on this machine the targeted
    checks alone already identify more reclaimable space than is needed.

STILL READ-ONLY. Measures, reconciles, proposes. Deletes nothing.

USAGE (PowerShell 5.1, from the Downloads folder; no copying required)
----------------------------------------------------------------------
    python "C:\\Users\\monzi\\Downloads\\audit_disk_reclaim_v3_2026-07-20.py" `
        --json "C:\\Users\\monzi\\Downloads\\disk_census_v3_2026-07-20.json"

    add --full for the whole-volume walk (now terminates; 5-20 minutes)

Author: written for Monzia Moodie, 2026-07-20.
"""
from __future__ import annotations

import argparse
import json
import os
import stat as statmod
import shutil
import sys
import time
from pathlib import Path

GiB = 1024 ** 3
MiB = 1024 ** 2

REPO = r"C:\Projects\genomic-variant-classifier"
USERPROFILE = os.environ.get("USERPROFILE", r"C:\Users\monzi")

HEADROOM_FRACTION = 0.05
HEADROOM_MIN = 20 * GiB
JEPA_CACHE_GIB = 14.7
MAX_DEPTH = 64

FILE_ATTRIBUTE_REPARSE_POINT = getattr(statmod, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def human(n) -> str:
    if n is None:
        return "   INCOMPLETE"
    n = float(n)
    if n >= GiB:
        return f"{n / GiB:9.2f} GiB"
    return f"{n / MiB:9.1f} MiB"


def is_reparse_point(entry) -> bool:
    """True for symbolic links, directory junctions, and any other reparse point.

    Three checks because each alone is insufficient on Windows:
      - is_junction() exists only in Python 3.12 and later
      - is_symlink() misses IO_REPARSE_TAG_MOUNT_POINT (the junction tag)
      - the FILE_ATTRIBUTE_REPARSE_POINT bit catches tags neither method names
    """
    try:
        if entry.is_symlink():
            return True
    except OSError:
        return True  # cannot tell -> refuse to descend
    is_junction = getattr(entry, "is_junction", None)
    if is_junction is not None:
        try:
            if is_junction():
                return True
        except OSError:
            return True
    try:
        st = entry.stat(follow_symlinks=False)
        if getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT:
            return True
    except OSError:
        return True
    return False


class Walker:
    """Cycle-safe, hard-link-aware recursive size accumulator."""

    def __init__(self, deadline: float | None = None):
        self.errors: list[str] = []
        self.n_files = 0
        self.deadline = deadline
        self.timed_out = False
        self.reparse_skipped: list[str] = []
        self._seen_dirs: set[tuple] = set()
        self._seen_files: set[tuple] = set()
        self.hardlink_savings = 0
        self._next_report = 250_000

    def _past_deadline(self) -> bool:
        if self.deadline is not None and time.monotonic() > self.deadline:
            self.timed_out = True
            return True
        return False

    def size_of(self, root: Path, collect_files: list | None = None,
                *, independent: bool = False):
        """Return (size_bytes, file_count, complete).

        `complete` is False if the deadline expired mid-walk. A caller must never
        print an incomplete total as if it were a measurement -- that was defect
        4 in version 2.

        TWO SEMANTICS, NAMED RATHER THAN IMPLIED (defect 5, 2026-07-21):

        independent=False (default) -- CENSUS. Participates in the walker's
            shared visited sets. Correct when the sum over many roots must equal
            one volume: an overlapping root, or a legacy junction such as
            C:\\Documents and Settings, must not be counted twice. A directory
            already accounted for anywhere is skipped.

        independent=True -- STANDALONE MEASUREMENT. Visited sets are scoped to
            THIS CALL. Correct when the question is how large one directory is,
            regardless of what was walked before. Cycle safety is unchanged --
            a self-referencing junction still terminates -- but a subtree that
            an earlier call consumed is measured again, because it genuinely
            belongs to this root.

        Applying census semantics to a standalone measurement is what made the
        data breakdown report 3.21 GiB for a 98.75 GiB tree: every nested
        directory had already been visited, so only loose files were counted.
        `hardlink_savings` is deliberately NOT mutated in independent mode,
        since that statistic describes the census.
        """
        total = 0
        count = 0
        complete = True
        seen_dirs = set() if independent else self._seen_dirs
        seen_files = set() if independent else self._seen_files
        stack = [(root, 0)]
        while stack:
            if self._past_deadline():
                complete = False
                break
            d, depth = stack.pop()
            if depth > MAX_DEPTH:
                self.errors.append(f"{d}: depth ceiling {MAX_DEPTH} reached (not descended)")
                continue
            try:
                with os.scandir(d) as it:
                    for e in it:
                        try:
                            if is_reparse_point(e):
                                self.reparse_skipped.append(e.path)
                                continue
                            if e.is_dir(follow_symlinks=False):
                                try:
                                    key = (e.stat(follow_symlinks=False).st_dev, e.inode())
                                except OSError:
                                    key = None
                                if key is not None and key != (0, 0):
                                    if key in seen_dirs:
                                        continue
                                    seen_dirs.add(key)
                                stack.append((Path(e.path), depth + 1))
                            elif e.is_file(follow_symlinks=False):
                                st = e.stat(follow_symlinks=False)
                                sz = st.st_size
                                if getattr(st, "st_nlink", 1) > 1:
                                    fkey = (st.st_dev, e.inode())
                                    if fkey in seen_files:
                                        if not independent:
                                            self.hardlink_savings += sz
                                        continue
                                    seen_files.add(fkey)
                                total += sz
                                count += 1
                                self.n_files += 1
                                if self.n_files >= self._next_report:
                                    print(f"    ... {self.n_files:,d} files", flush=True)
                                    self._next_report += 250_000
                                if collect_files is not None and sz >= 512 * MiB:
                                    collect_files.append((e.path, sz))
                        except OSError as inner:
                            self.errors.append(f"{e.path}: {inner}")
            except OSError as outer:
                self.errors.append(f"{d}: {outer}")
        return total, count, complete


def locked_file_size(path: Path):
    try:
        return path.stat().st_size, "os.stat"
    except OSError as e:
        return None, f"UNREADABLE ({type(e).__name__})"


def report(label: str, size, count, complete, extra: str = "") -> None:
    if not complete:
        print(f"  {'INCOMPLETE':>13s}  {label}")
        print("                 time budget expired mid-walk; NOT a measurement")
        return
    line = f"  {human(size)}  "
    if count is not None:
        line += f"{count:>9,d} files  "
    print(line + label)
    if extra:
        print(f"                 {extra}")


def targeted(volume: Path, w: Walker) -> list[dict]:
    up = Path(USERPROFILE)
    rows: list[dict] = []
    print("-" * 78)
    print("TARGETED CHECKS -- known large space consumers")
    print("-" * 78)

    for p, what, how in [
        (volume / "pagefile.sys", "virtual memory pagefile",
         "cap it: System > Advanced system settings > Performance > Virtual memory"),
        (volume / "hiberfil.sys", "hibernation image, roughly RAM-sized",
         "reclaim ALL of it: powercfg /hibernate off   (loses Fast Startup)"),
        (volume / "swapfile.sys", "modern-app swapfile", "leave alone"),
    ]:
        sz, src = locked_file_size(p)
        if sz is None:
            print(f"  {'UNREADABLE':>13s}  {p}  ({src})")
        elif sz > 0:
            report(str(p), sz, None, True, what)
            print(f"                 {how}")
            rows.append({"path": str(p), "size": sz, "what": what, "how": how})

    for p, what, how in [
        (volume / "System Volume Information", "restore points and shadow copies",
         "vssadmin list shadowstorage ; vssadmin resize shadowstorage"),
        (volume / "$Recycle.Bin", "deleted files not yet purged", "empty the Recycle Bin"),
        (volume / "Windows.old", "previous Windows installation",
         "Settings > System > Storage > Temporary files"),
        (volume / "$GetCurrent", "Windows upgrade staging leftovers", "safe to delete"),
        (volume / "Windows/WinSxS", "component store (hard-link farm)",
         "Dism /Online /Cleanup-Image /StartComponentCleanup /ResetBase"),
        (volume / "Windows/SoftwareDistribution/Download", "Windows Update cache",
         "stop the wuauserv service, then clear"),
        (volume / "Config.Msi", "installer rollback staging", "usually safe once no install is running"),
        (volume / "cabal", "Haskell cabal package store", "delete if Haskell is not in use"),
        (up / "AppData/Local/Docker", "Docker Desktop data", "see virtual disks below"),
        (up / "AppData/Local/Packages", "Store apps and Windows Subsystem for Linux state",
         "see virtual disks below"),
        (up / "AppData/Local/Programs", "per-user installed programs", "inspect before touching"),
        (up / "AppData/Local/pip/Cache", "pip download and build cache", "python -m pip cache purge"),
        (up / ".cache/huggingface", "Hugging Face model cache", "delete unused model snapshots"),
        (up / "AppData/Local/Temp", "user temporary files", "safe to clear when idle"),
        (up / "OneDrive", "OneDrive local mirror", "enable Files On-Demand to free local copies"),
        (up / "anaconda3", "Anaconda installation", "redundant if .venv312 is the working environment"),
        (Path(REPO), "the project repository", "see data directory breakdown below"),
    ]:
        if not p.exists():
            continue
        sz, cnt, ok = w.size_of(p)
        if ok and sz < 100 * MiB:
            continue
        report(str(p), sz, cnt, ok, what)
        if ok:
            print(f"                 {how}")
            rows.append({"path": str(p), "size": sz, "n_files": cnt,
                         "what": what, "how": how})

    print("\n  virtual disk images (these NEVER shrink on their own):")
    seen = set()
    found = False
    for base in [up / "AppData/Local/Docker", up / "AppData/Local/Packages",
                 Path(r"C:\ProgramData\DockerDesktop")]:
        if not base.exists():
            continue
        for pat in ("**/*.vhdx", "**/*.vhd", "**/*.qcow2"):
            try:
                for f in base.glob(pat):
                    try:
                        st = f.stat()
                    except OSError:
                        continue
                    key = (st.st_dev, st.st_ino)
                    if key in seen or st.st_size < 512 * MiB:
                        continue
                    seen.add(key)
                    found = True
                    print(f"    {human(st.st_size)}  {f}")
                    rows.append({"path": str(f), "size": st.st_size,
                                 "what": "virtual disk image",
                                 "how": "wsl --shutdown, then Optimize-VHD -Mode Full "
                                        "(or diskpart: select vdisk / compact vdisk)"})
            except OSError as e:
                w.errors.append(f"{base}: {e}")
    if not found:
        print("    none found at or above 512 MiB")
    print()
    return rows


def data_breakdown(w: Walker, top: int) -> list[dict]:
    data = Path(REPO) / "data"
    if not data.is_dir():
        return []
    print("-" * 78)
    print(f"REPOSITORY DATA DIRECTORY -- {data}")
    print("-" * 78)
    rows = []
    try:
        with os.scandir(data) as it:
            for e in it:
                if is_reparse_point(e):
                    print(f"  {'JUNCTION':>13s}  {e.path}  (not followed)")
                    continue
                if not e.is_dir(follow_symlinks=False):
                    continue
                # independent=True: this is a standalone measurement of one
                # directory, NOT part of the volume census. Without it the
                # whole-volume walk has already consumed every subdirectory
                # and only loose files are counted -- defect 5.
                sz, cnt, ok = w.size_of(Path(e.path), independent=True)
                rows.append({"path": e.path, "size": sz if ok else None,
                             "n_files": cnt, "complete": ok})
    except OSError as err:
        w.errors.append(f"{data}: {err}")
    rows.sort(key=lambda r: -(r["size"] or 0))
    for r in rows[:top]:
        report(r["path"], r["size"], r["n_files"], r["complete"])
    known = sum(r["size"] for r in rows if r["size"])
    print(f"\n  measured subtotal: {human(known)}")
    print("  History of this section, because it has been wrong three times:")
    print("    v1 2026-07-20 : 161.38 GiB / 15,260 files.")
    print("    v2 2026-07-20 : every subdirectory 0.0 MiB -- the time budget had")
    print("                    expired; a non-measurement formatted as a measurement.")
    print("    v3 2026-07-21 :   3.21 GiB -- the walker was shared with the volume")
    print("                    census, so only loose files were counted. True: 98.75.")
    print("  This run uses independent=True per subdirectory. Treat any figure that")
    print("  disagrees sharply with a direct measurement as suspect, not as news.")
    print("\n  Canonical store is Google Drive: genvarcla:genomic-variant-classifier/data/")
    print("  Verify BEFORE removing anything local, per subdirectory:")
    print('    rclone check "C:\\Projects\\genomic-variant-classifier\\data\\<sub>" '
          '"genvarcla:genomic-variant-classifier/data/<sub>" --size-only --one-way')
    print()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--volume", default="C:\\")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-seconds", type=float, default=1800.0)
    ap.add_argument("--full", action="store_true",
                    help="also walk the whole volume top-level (5-20 minutes)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    volume = Path(args.volume)
    started = time.monotonic()
    w = Walker(deadline=started + args.max_seconds)

    print("=" * 78)
    print("DISK CENSUS -- junction-safe, cycle-safe. READ ONLY. DELETES NOTHING.")
    print("=" * 78)
    try:
        u = shutil.disk_usage(str(volume))
    except OSError as e:
        print(f"  cannot read volume {volume}: {e}")
        return 2
    total, used, free = u.total, u.used, u.free
    print(f"  {volume}")
    print(f"    total {total:>18,d} bytes  {total/GiB:9.2f} GiB")
    print(f"    used  {used:>18,d} bytes  {used/GiB:9.2f} GiB")
    print(f"    free  {free:>18,d} bytes  {free/GiB:9.2f} GiB   {100*free/total:.3f} %")
    if 100 * free / total < 5.0:
        print("    *** BELOW 5 % FREE -- pagefile growth, git operations and pytest")
        print("        temporary-file fixtures are all at risk. ***")
    print()

    rows = targeted(volume, w)
    big_files: list = []
    if args.full:
        print("-" * 78)
        print(f"WHOLE-VOLUME CENSUS -- top level of {volume}")
        print("-" * 78)
        try:
            with os.scandir(volume) as it:
                children = list(it)
        except OSError as e:
            children = []
            print(f"  cannot read {volume}: {e}")
        vol_rows = []
        for e in children:
            try:
                if is_reparse_point(e):
                    print(f"  {'JUNCTION':>13s}  {e.path}  (not followed)")
                    continue
                if e.is_dir(follow_symlinks=False):
                    sz, cnt, ok = w.size_of(Path(e.path), collect_files=big_files)
                elif e.is_file(follow_symlinks=False):
                    st = e.stat(follow_symlinks=False)
                    sz, cnt, ok = st.st_size, 1, True
                    if sz >= 512 * MiB:
                        big_files.append((e.path, sz))
                else:
                    continue
                vol_rows.append({"path": e.path, "size": sz if ok else None,
                                 "n_files": cnt, "complete": ok})
            except OSError as inner:
                w.errors.append(f"{e.path}: {inner}")
        vol_rows.sort(key=lambda r: -(r["size"] or 0))
        for r in vol_rows[:args.top]:
            report(r["path"], r["size"], r["n_files"], r["complete"])
        rows.extend([r for r in vol_rows if r["size"]])
        scanned = sum(r["size"] for r in vol_rows if r["size"])
        print()
        print("  RECONCILIATION")
        print(f"    volume reports used : {human(used)}")
        print(f"    census totalled     : {human(scanned)}")
        print(f"    difference          : {human(used - scanned)}")
        if scanned > used:
            print("    *** census EXCEEDS volume usage -- a cycle guard has failed."
                  " Do not trust this run. ***")
        print()

    if big_files:
        print("-" * 78)
        print("LARGEST INDIVIDUAL FILES (>= 512 MiB, de-duplicated)")
        print("-" * 78)
        for p, s in sorted(big_files, key=lambda kv: -kv[1])[:args.top]:
            print(f"  {human(s)}  {p}")
        print()

    data_rows = data_breakdown(w, args.top)

    print("=" * 78)
    print("WALK INTEGRITY")
    print("=" * 78)
    print(f"  files counted                  : {w.n_files:,d}")
    print(f"  reparse points skipped         : {len(w.reparse_skipped)}")
    for p in w.reparse_skipped[:12]:
        print(f"      {p}")
    if len(w.reparse_skipped) > 12:
        print(f"      ... and {len(w.reparse_skipped)-12} more")
    print(f"  directories de-duplicated      : {len(w._seen_dirs):,d} unique")
    print(f"  bytes saved by hard-link dedup : {human(w.hardlink_savings)}")
    print(f"  unreadable paths               : {len(w.errors)}")
    print(f"  timed out                      : {w.timed_out}")
    print(f"  elapsed                        : {time.monotonic()-started:.1f} s")
    if w.timed_out:
        print("  *** BUDGET EXPIRED. Any row above marked INCOMPLETE is NOT a")
        print("      measurement. Re-run with a larger --max-seconds. ***")
    print()

    headroom = max(HEADROOM_FRACTION * total, HEADROOM_MIN)
    need = JEPA_CACHE_GIB * GiB + headroom
    print("=" * 78)
    print("VERDICT -- headroom-aware")
    print("=" * 78)
    print(f"  JEPA embedding cache          : {JEPA_CACHE_GIB:.1f} GiB")
    print(f"  required operational headroom : {headroom/GiB:.2f} GiB  (max of 5 %, 20 GiB)")
    print(f"  free space required           : {need/GiB:.2f} GiB")
    print(f"  free space now                : {free/GiB:.2f} GiB")
    if free >= need:
        print("  VERDICT: sufficient.")
    else:
        print(f"  VERDICT: INSUFFICIENT -- short by {(need-free)/GiB:.2f} GiB")
    print()
    print("  NOTHING WAS DELETED.")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "volume": {"path": str(volume), "total": total, "used": used, "free": free},
            "targeted": rows,
            "data_breakdown": data_rows,
            "largest_files": [{"path": p, "size": s} for p, s in
                              sorted(big_files, key=lambda kv: -kv[1])],
            "integrity": {"files": w.n_files,
                          "reparse_skipped": w.reparse_skipped[:500],
                          "hardlink_savings": w.hardlink_savings,
                          "errors": len(w.errors), "timed_out": w.timed_out},
            "verdict": {"required": need, "free": free, "sufficient": free >= need},
        }, indent=2), encoding="utf-8", newline="\n")
        print(f"  census written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
