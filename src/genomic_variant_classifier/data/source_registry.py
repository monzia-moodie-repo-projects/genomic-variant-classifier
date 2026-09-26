"""A typed reader over the source declarations in the data manifest.

Created 2026-08-29 after `AUTHORITY-SEARCH-SCOPED-TO-ONE-LANGUAGE-1`.

WHY THIS EXISTS
---------------
`configs/data_manifest.yaml` describes itself on its third line as the
"Canonical registry of every data source under data/", declares 32 sources, and
is read by five scripts under `scripts/maintenance/`. MEASURED 2026-08-29: all
five walk RAW DICTIONARIES.

    sources = man.get("sources", {})
    bad = [s for s, m in sources.items()
           if m.get("sync") and m.get("tier") == "controlled"]
    loc = meta.get("location", "external")

`StoragePolicy` in `scripts/maintenance/preflight_data_guard.py` is the only
typed reader, and it reads the `storage:` block alone. This is its sibling for
the `sources:` block, built to the same pattern.

WHAT RAW DICTIONARY ACCESS PERMITS
----------------------------------
A MISSPELLED KEY IS SILENTLY A DEFAULT. `meta.get("tier")` returns None for
`teir`, and `None != "controlled"`, so the compliance gate in
`setup_data_tree.py` would admit a controlled source. The gate that hard-aborts
when a controlled source is marked for synchronisation depends on a key name
matching exactly, with nothing checking that it does.

ONE DEFAULT LIVES IN FOUR PLACES. `meta.get("location", "external")` appears
independently in `setup_data_tree.py`, `audit_data_tree.py` and
`consolidate_aliases.py`. `StoragePolicy` avoids this by declaring defaults
once in `DEFAULT_POLICY` and pinning them with a test.

NOTHING VALIDATES. `StoragePolicy.__post_init__` refuses a policy whose
severity bands are unreachable. Nothing refuses a source declaring
`tier: contrlled`, an empty `class`, or an alias equal to its own canonical
name.

WHY THIS RAISES WHERE `StoragePolicy` FALLS BACK
------------------------------------------------
`StoragePolicy.load` warns and uses documented defaults when the manifest
cannot be read, because "refusing every run because a configuration file moved
would be a worse failure than the one being guarded against".

That reasoning does not transfer. There is no defensible default for a source
registry: one cannot invent 32 declarations, and a fallback registry would
silently answer questions about evidence the project does not have. So this
RAISES, and the caller decides.

STRICT TYPES AND UNIQUE KEYS (2026-09-25)
-----------------------------------------
MEASURED on main 38987f54 (owner review 2026-09-25, reproduced here): the loader
refused unknown KEYS but coerced VALUES, and let YAML keep the last duplicate:

    sync: "false"            -> sync=True             bool("false") is True
    aliases: abc             -> ("a", "b", "c")       a string is iterable
    version: "4.1" twice     -> the last one, silently

This reader is the policy authority for release approvals, so it now refuses
all three: duplicate keys at any depth, and any value whose type is not the
declared one. Nothing is converted. The real manifest had none of these
(measured 2026-09-25), so no declaration changed.

RELEASE APPROVALS (2026-09-26)
------------------------------
A separate top-level `release_approvals:` section, keyed by monitoring target,
SELECTS the active approval: exactly `{record, sha256}` -- a never-edited
record under docs/approvals/ and that record's full SHA-256. The record holds
the facts; `release_approval.load_approval` exposes `approved_release` and the
scope only AFTER verifying the record's bytes. Kept apart from `sources:`,
which describes the data the project USES: approval is not adoption. The
section exists only in manifest schema version 2.

WHAT THIS UNIT DOES NOT DO
--------------------------
It does not rewire the maintenance scripts. Those are deliberately standalone --
`preflight_data_guard.py` imports nothing from this repository so it can run
from any directory -- and changing that is a separate decision with its own
risk. This reader exists first; anything that adopts it comes after.

Acronyms: YAML = YAML Ain't Markup Language; DUA = Data Use Agreement.

Author: Monzia Moodie
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

#: The default the four scripts each spell separately.
DEFAULT_LOCATION = "external"

#: Where the declarations live, relative to the repository root.
DEFAULT_MANIFEST = "configs/data_manifest.yaml"


#: Manifest schema versions this reader understands. Version 2 (2026-09-26) adds the
#: `release_approvals:` selector section; a version-1 manifest may not carry one.
SUPPORTED_MANIFEST_VERSIONS = (1, 2)
_POINTER_FIELDS = frozenset({"record", "sha256"})


class SourceTier(str, Enum):
    """Access terms. FOUR values, not the three the manifest header lists.

    `docs/standards/DATA_LAYOUT_STANDARD.md` line 106 declares the fourth:
    `review` marks sources whose access tier or leakage-independence must be
    confirmed before they are synced or used. The manifest's own header comment
    lists three and is stale; the standard is the authority.
    """

    PUBLIC = "public"
    ACADEMIC = "academic"
    CONTROLLED = "controlled"
    REVIEW = "review"


class SourceClass(str, Enum):
    """Durability, which drives backup policy.

    From the standard, section 4. This axis has no counterpart in any type the
    drift package declares, and it is what decides whether losing an artifact
    costs a re-download or costs the artifact.
    """

    IRREPLACEABLE = "irreplaceable"
    REGENERABLE_EXPENSIVE = "regenerable_expensive"
    REGENERABLE_CHEAP = "regenerable_cheap"
    PUBLIC_REDOWNLOADABLE = "public_redownloadable"


class SourceLocation(str, Enum):
    """Which subtree of `data/` holds it."""

    EXTERNAL = "external"
    RAW = "raw"
    PROCESSED = "processed"


class SourceRegistryError(ValueError):
    """A declaration that cannot be acted on."""


@dataclass(frozen=True)
class SourceDeclaration:
    """One source, as the manifest declares it.

    `acquire` and `regenerate` together separate PUBLISHED from DERIVED, which
    is a convention already in use: measured 2026-08-29, 29 sources carry a
    non-empty `acquire` and 3 carry an empty `acquire` with a non-empty
    `regenerate`, under a heading that names them BUILT ARTIFACTS.
    """

    name: str
    location: SourceLocation
    tier: SourceTier
    cls: SourceClass
    aliases: Tuple[str, ...]
    version: str
    acquire: str
    regenerate: str
    sync: bool
    notes: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise SourceRegistryError("a source must have a name")
        if self.name != self.name.lower():
            raise SourceRegistryError(
                "source {!r} is not lower_snake_case. The standard, section 3: "
                "one directory per logical source, named in lower_snake_case."
                .format(self.name))
        for field, want in (("location", SourceLocation), ("tier", SourceTier),
                            ("cls", SourceClass)):
            value = getattr(self, field)
            if not isinstance(value, want):
                raise SourceRegistryError(
                    "{}.{} is {!r}; a raw string here is how a MISSPELLED key "
                    "becomes a silent default".format(self.name, field, value))
        if not isinstance(self.aliases, tuple):
            raise SourceRegistryError(
                "{}.aliases is {}; it must be a TUPLE so a declaration cannot "
                "be mutated after it is read"
                .format(self.name, type(self.aliases).__name__))
        for a in self.aliases:
            if not isinstance(a, str) or not a:
                raise SourceRegistryError(
                    "{}: alias {!r} is not a non-empty string"
                    .format(self.name, a))
            if a == self.name:
                raise SourceRegistryError(
                    "{}: declares itself as its own alias. An alias is a "
                    "NON-CANONICAL name the auditor folds away; naming the "
                    "canonical form would make the auditor migrate a directory "
                    "into itself.".format(self.name))
        if len(set(self.aliases)) != len(self.aliases):
            raise SourceRegistryError(
                "{}: duplicate aliases {}".format(self.name,
                                                  sorted(self.aliases)))
        if self.sync and self.tier is SourceTier.CONTROLLED:
            raise SourceRegistryError(
                "{} is tier CONTROLLED and marked sync=true. The standard, "
                "section 5: controlled sources are backed up encrypted or "
                "offline ONLY -- never to a personal cloud, which would breach "
                "the licence or Data Use Agreement.".format(self.name))
        if not isinstance(self.sync, bool):
            raise SourceRegistryError("{}.sync must be a bool".format(self.name))

    @property
    def is_published(self) -> bool:
        """Does a publisher supply the bytes? Non-empty `acquire`."""
        return bool(self.acquire.strip())

    @property
    def is_derived(self) -> bool:
        """Does this project build them? Empty `acquire`, non-empty `regenerate`."""
        return not self.acquire.strip() and bool(self.regenerate.strip())

    @property
    def must_back_up(self) -> bool:
        """Standard section 4: irreplaceable or expensive to rebuild."""
        return self.cls in (SourceClass.IRREPLACEABLE,
                            SourceClass.REGENERABLE_EXPENSIVE)

    def directory(self, data_dir: str = "data") -> str:
        return "{}/{}/{}".format(data_dir, self.location.value, self.name)


@dataclass(frozen=True)
class ApprovalPointer:
    """The manifest's SELECTION of an approval record: path and full SHA-256 only.

    The approval's facts live in the record; see `release_approval.load_approval`.
    """

    target: str
    record: str
    sha256: str

    def __post_init__(self) -> None:
        from genomic_variant_classifier.data.release_approval import _RECORD_PATH, _SHA256

        for field in ("target", "record", "sha256"):
            if type(getattr(self, field)) is not str:
                raise SourceRegistryError(
                    "release_approvals.{}.{} must be a quoted string, not {}".format(
                        self.target, field, type(getattr(self, field)).__name__))
        if not self.target or self.target != self.target.strip():
            raise SourceRegistryError("a release approval needs a non-blank target")
        if _RECORD_PATH.fullmatch(self.record) is None:
            raise SourceRegistryError(
                "release_approvals.{}.record {!r} must be a direct docs/approvals/*.json path"
                .format(self.target, self.record))
        if _SHA256.fullmatch(self.sha256) is None:
            raise SourceRegistryError(
                "release_approvals.{}.sha256 must be all 64 lowercase hex digits".format(self.target))


@dataclass(frozen=True)
class SourceRegistry:
    """Every declared source, and where the declarations came from.

    `manifest_source` records the path read, exactly as `StoragePolicy.source`
    does. A reader that cannot say where its values came from cannot be
    audited.
    """

    declarations: Tuple[SourceDeclaration, ...]
    manifest_source: str
    release_approvals: Tuple[ApprovalPointer, ...] = ()

    def __post_init__(self) -> None:
        targets = [a.target for a in self.release_approvals]
        if len(set(targets)) != len(targets):
            raise SourceRegistryError("duplicate release-approval target(s)")
        if targets != sorted(targets):
            raise SourceRegistryError("release approvals are not in canonical order")
        if not self.declarations:
            raise SourceRegistryError(
                "the registry is empty. An empty registry would answer every "
                "membership question with 'no' and look like a working reader.")
        names = [d.name for d in self.declarations]
        if len(set(names)) != len(names):
            raise SourceRegistryError(
                "duplicate source name(s) {}".format(
                    sorted({n for n in names if names.count(n) > 1})))
        seen: Dict[str, str] = {}
        for d in self.declarations:
            for a in d.aliases:
                if a in names:
                    raise SourceRegistryError(
                        "{!r} is an alias of {!r} AND a canonical source. The "
                        "auditor would fold a real source into another one."
                        .format(a, d.name))
                if a in seen:
                    raise SourceRegistryError(
                        "alias {!r} claimed by both {!r} and {!r}"
                        .format(a, seen[a], d.name))
                seen[a] = d.name
        if list(self.declarations) != sorted(self.declarations,
                                             key=lambda d: d.name):
            raise SourceRegistryError(
                "declarations are not in canonical order; `load` sorts them, "
                "so two registries read from one file would compare unequal")

    @classmethod
    def load(cls, manifest: str | Path = DEFAULT_MANIFEST) -> "SourceRegistry":
        """Read and TYPE every declaration. RAISES; there is no default.

        `StoragePolicy.load` falls back to documented defaults because refusing
        every run over a moved configuration file would be worse than the
        problem. That does not transfer: one cannot invent 32 source
        declarations, and a fallback registry would silently answer questions
        about evidence this project does not have.
        """
        p = Path(manifest)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as exc:
            raise SourceRegistryError(
                "cannot read {}: {}. There is no defensible default for a "
                "source registry.".format(p, exc)) from exc
        return cls.from_text(text, str(p))

    @classmethod
    def from_text(cls, text: str, source: str) -> "SourceRegistry":
        """Parse manifest TEXT -- e.g. a Git blob at a pinned commit (the admission check
        reads Git objects, never the working tree). `source` is recorded for audit."""
        p = source
        raw = _load_unique_yaml(text) or {}
        if not isinstance(raw, dict):
            raise SourceRegistryError("{} is not a mapping at the top level".format(p))
        version = raw.get("version")
        if type(version) is not int or version not in SUPPORTED_MANIFEST_VERSIONS:
            raise SourceRegistryError(
                "{}: manifest version {!r} is not one of {}".format(p, version, SUPPORTED_MANIFEST_VERSIONS))
        if "release_approvals" in raw and version < 2:
            raise SourceRegistryError(
                "{}: release_approvals requires manifest version 2 (found {})".format(p, version))
        block = raw.get("sources")
        if not isinstance(block, dict) or not block:
            raise SourceRegistryError(
                "{} has no non-empty 'sources' section".format(p))

        out = []
        for name, meta in block.items():
            if not isinstance(meta, dict):
                raise SourceRegistryError(
                    "source {!r} is not a mapping".format(name))
            unknown = set(meta) - {"location", "tier", "class", "aliases",
                                   "version", "acquire", "regenerate", "sync",
                                   "notes"}
            if unknown:
                raise SourceRegistryError(
                    "source {!r} declares unknown key(s) {}. A misspelled key "
                    "would otherwise read as a default and never be noticed."
                    .format(name, sorted(unknown)))
            out.append(SourceDeclaration(
                name=str(name),
                location=_enum(SourceLocation, meta.get("location",
                                                        DEFAULT_LOCATION),
                               name, "location"),
                tier=_enum(SourceTier, meta.get("tier"), name, "tier"),
                cls=_enum(SourceClass, meta.get("class"), name, "class"),
                aliases=_typed_aliases(meta, name),
                version=_typed(meta, "version", str, "", name),
                acquire=_typed(meta, "acquire", str, "", name),
                regenerate=_typed(meta, "regenerate", str, "", name),
                sync=_typed(meta, "sync", bool, False, name),
                notes=_typed(meta, "notes", str, "", name),
            ))
        return cls(declarations=tuple(sorted(out, key=lambda d: d.name)),
                   manifest_source=str(p),
                   release_approvals=_release_approvals(raw.get("release_approvals")))

    def approval_pointer(self, target: str) -> ApprovalPointer:
        for a in self.release_approvals:
            if a.target == target:
                return a
        raise SourceRegistryError(
            "no release approval declared for {!r}; declared: {}".format(
                target, [a.target for a in self.release_approvals]))

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(d.name for d in self.declarations)

    @property
    def aliases(self) -> FrozenSet[str]:
        return frozenset(a for d in self.declarations for a in d.aliases)

    def declaration(self, name: str) -> SourceDeclaration:
        for d in self.declarations:
            if d.name == name:
                return d
        raise SourceRegistryError(
            "{!r} is not a declared source. Known: {}".format(
                name, list(self.names)))

    def canonical_for(self, name: str) -> str:
        """Resolve a canonical name or a declared alias. REFUSES anything else.

        The standard, section 3: "Aliases are forbidden: a source has exactly
        one canonical name. The manifest records known aliases so the auditor
        can flag and guide migration." So an alias resolves HERE, and the
        auditor is what removes it from disk.
        """
        if name in self.names:
            return name
        for d in self.declarations:
            if name in d.aliases:
                return d.name
        raise SourceRegistryError(
            "{!r} is neither a declared source nor a declared alias. "
            "Registering it explicitly is the only way to admit it; guessing "
            "would mint a scientifically duplicate authority.".format(name))

    def by_tier(self, tier: SourceTier) -> Tuple[SourceDeclaration, ...]:
        return tuple(d for d in self.declarations if d.tier is tier)

    @property
    def controlled(self) -> Tuple[SourceDeclaration, ...]:
        return self.by_tier(SourceTier.CONTROLLED)

    @property
    def syncable(self) -> Tuple[SourceDeclaration, ...]:
        """Standard section 5: sync=true, tier not controlled, and durable."""
        return tuple(d for d in self.declarations
                     if d.sync and d.tier is not SourceTier.CONTROLLED
                     and d.must_back_up)

    def describe(self) -> str:
        return ("{} source(s) from {}\n"
                "  published {} | derived {} | must back up {}\n"
                "  controlled {} | review {} | aliases {}").format(
            len(self.declarations), self.manifest_source,
            sum(1 for d in self.declarations if d.is_published),
            sum(1 for d in self.declarations if d.is_derived),
            sum(1 for d in self.declarations if d.must_back_up),
            len(self.controlled), len(self.by_tier(SourceTier.REVIEW)),
            len(self.aliases))


def _enum(kind, value, source: str, field: str):
    """Convert, and name the source and field when it fails.

    A bare `ValueError: 'contrlled' is not a valid SourceTier` does not say
    WHICH declaration is wrong, and 32 declarations is too many to search by
    hand.
    """
    if value is None:
        raise SourceRegistryError(
            "source {!r} declares no {}. It is required: omitting it would "
            "otherwise read as a default and never be noticed."
            .format(source, field))
    try:
        return kind(value)
    except ValueError as exc:
        raise SourceRegistryError(
            "source {!r} declares {} {!r}; expected one of {}".format(
                source, field, value, [m.value for m in kind])) from exc


def _load_unique_yaml(text: str):
    """Safe YAML with DUPLICATE KEYS REFUSED at every depth.

    `yaml.safe_load` keeps the last duplicate silently -- MEASURED 2026-09-25: two
    `version:` keys read as the second.
    """
    import yaml

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _mapping(loader, node, deep=False):
        loader.flatten_mapping(node)
        out = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in out:
                raise SourceRegistryError(
                    "duplicate YAML key {!r} at line {}; YAML would silently keep the last "
                    "one".format(key, key_node.start_mark.line + 1))
            out[key] = loader.construct_object(value_node, deep=deep)
        return out

    _UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)
    return yaml.load(text, Loader=_UniqueKeyLoader)


def _typed(meta, key: str, kind, default, source: str):
    """The declared type or a refusal. NOTHING IS CONVERTED (bool("false") is True)."""
    if key not in meta:
        return default
    value = meta[key]
    if type(value) is not kind:
        raise SourceRegistryError(
            "source {!r}: {} is {!r} ({}), not {}. Values are never converted: "
            "bool('false') is True and str(4.1) invents text.".format(
                source, key, value, type(value).__name__, kind.__name__))
    return value


def _typed_aliases(meta, source: str):
    if "aliases" not in meta or meta["aliases"] is None:
        return ()
    value = meta["aliases"]
    if type(value) is not list:
        raise SourceRegistryError(
            "source {!r}: aliases is {!r} ({}), not a list. A string would be split into "
            "characters.".format(source, value, type(value).__name__))
    for a in value:
        if type(a) is not str:
            raise SourceRegistryError(
                "source {!r}: alias {!r} is {}, not a string".format(source, a, type(a).__name__))
    return tuple(value)


def _release_approvals(block) -> Tuple[ApprovalPointer, ...]:
    if block is None:
        return ()
    if not isinstance(block, dict):
        raise SourceRegistryError("release_approvals must be a mapping of target to {record, sha256}")
    out = []
    for target, fields in block.items():
        if type(target) is not str:
            raise SourceRegistryError("release-approval target {!r} is not a string".format(target))
        if not isinstance(fields, dict) or set(fields) != _POINTER_FIELDS:
            got = sorted(fields) if isinstance(fields, dict) else type(fields).__name__
            raise SourceRegistryError(
                "release_approvals.{} must contain exactly {}; got {}".format(
                    target, sorted(_POINTER_FIELDS), got))
        out.append(ApprovalPointer(target=target, **fields))
    return tuple(sorted(out, key=lambda a: a.target))
