# CORRECTION 2026-09-04 -- a repository I called empty holds two hundred megabytes

**Author: Monzia Moodie**
**Applies to:** `e970fcd` (`docs/CHANGELOG.md`, the `2026-09-04 part 3` entry)
**Status:** the corrected claim is recorded here; `e970fcd` is not amended.

---

## 0. What went wrong

The `2026-09-04 part 3` changelog entry carries, in its list of findings
carried forward and still open:

> `HOME-DIRECTORY-IS-AN-EMPTY-GIT-REPOSITORY-1`

The name asserts that the git repository at `C:\Users\monzi\.git` is empty. It
is not, and I never checked.

I measured four properties -- refs, commits, tracked files, remotes -- saw
zero in each, and named the finding from them. `git fsck` was never run, and
the object store was never queried. **Four zeros and a conclusion.**

`FINDING-NAMED-FROM-FOUR-PROPERTIES-OF-SEVEN-1`.

---

## 1. WITHDRAWN: `HOME-DIRECTORY-IS-AN-EMPTY-GIT-REPOSITORY-1`

MEASURED 2026-09-04 by the precondition gate of
`Resolve_StrayHomeRepository_2026-09-04.ps1`, which REFUSED to rename the
directory:

```
commits 0    refs 0    remotes 0    tracked files 0    stashes 0
fsck unreachable objects              849
files inside .git                     869
bytes inside .git              72,791,946
```

The eighth precondition -- zero unreachable objects -- was the only one of the
eight that failed, and it is the only one the original finding had not
already claimed. Had I written that gate from my own picture of the
directory, it would have renamed a directory holding content I had never
looked at.

**The refusal is the instrument working.** Both the measuring run and the
`-Apply` run stopped before touching anything.

---

## 2. A SECOND error, made after the first was known

Having found the object store populated, I then reported its content as
**72.8 megabytes**. That is the COMPRESSED size on disk. MEASURED by
`git cat-file --batch-check` over all 849 objects:

```
uncompressed content       200,027,701 B
compressed on disk          72,791,946 B
ratio                             2.75x
understated by             127,235,755 B
```

Both numbers are correct. I attached the wrong meaning to one of them, which
is the same failure as the first: **reporting a quantity without checking what
it measures.**

`COMPRESSED-SIZE-REPORTED-AS-CONTENT-SIZE-1`.

---

## 3. What the evidence actually shows

849 objects: 848 blobs and one tree. The size distribution accounts for every
one, `89 + 705 + 43 + 7 + 5 = 849`, and the largest single blob is 91,380,224
bytes, 87.1 mebibytes.

### The names are unrecoverable, and that is PROVEN rather than assumed

The single tree is `4b825dc642cb6eb9a060e54bf8d69288fbee4904`. That identity
is computable without any repository at all, because a git object identity is
`sha1("<type> <length>\0<content>")` and an empty tree has zero-length
content:

```
sha1(b"tree 0\x00") = 4b825dc642cb6eb9a060e54bf8d69288fbee4904
```

It is git's canonical EMPTY TREE, and `ls-tree` confirmed zero entries. With
no commit, no non-empty tree and no index, **nothing anywhere records what
those 848 blobs were called.** A blob carries content and no name.

### What the on-disk state supports

```
HEAD -> refs/heads/master        a branch was named
refs 0                           that branch never existed
index ABSENT                     the staging area is gone
848 blobs present                yet content WAS staged: git add wrote these
no [remote] section              never connected anywhere
description unedited             git init default
created 2024-02-13, written 2026-08-08
```

`git init` ran in the home directory, `git add` staged a large set of files,
nothing was ever committed, and the index was later removed. The blobs are
orphaned staging residue.

---

## 4. REPLACEMENT: `HOME-DIRECTORY-REPOSITORY-HAS-NO-HISTORY-AND-ORPHANED-OBJECTS-1`

A git repository exists at `C:\Users\monzi\.git` with no commits, no refs, no
remotes and no index, and an object store holding 849 unreachable objects
totalling 200,027,701 bytes of content whose original path names are
unrecoverable.

It still captures any bare git command issued from beneath the home directory.
That is the operational hazard the original finding correctly identified, and
it is unchanged by this correction: a bare `git push` from `Downloads` reports
"no configured push destination" rather than reporting that it is in the wrong
repository, which is exactly how a push silently failed earlier on 2026-09-04.

**NOT RESOLVED.** The rename gate refused and the gate is not being relaxed. A
gate loosened because it fired is not a gate.

---

## 5. What is not claimed

That the blobs are recoverable in any useful sense. They are content without
names; identifying any of them would require reading it, which no probe here
does.

That deleting the repository is safe or unsafe. Renaming preserves every byte
and removes the hazard; deleting reclaims 72,791,946 bytes of disk and
destroys content nothing can identify. That is a decision about personal files
and it is not one this record makes.

Who created the repository, or why. Only what the objects show.

That any other finding in the `2026-09-04 part 3` entry is affected. The
Phase 3B.0 measurements -- 1,720 tracked paths, 11,372 decision sites, 2,252
origins, seventeen cyclic components each of size one -- were each verified
against their census artifacts and are unchanged by this.

---

## 6. Why this is a correction and not an amendment

`docs/CHANGELOG.md` is pinned by digest `759061636db1c75b` in the attestation
for `e970fcd`, and the file is append-only by convention. Amending the entry
would break that binding and violate the convention in the same act.

The convention is unchanged: corrections sit beside records, never inside
them.

---

## 7. Status

One finding withdrawn, one replacement registered, two new findings about how
the original was reached. The operational hazard is unresolved and the
decision on it is not mine.

No file in the repository is changed by this record beyond its own creation
and the changelog entry accompanying it.
