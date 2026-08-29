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
from typing import Dict, FrozenSet, Optional, Tuple

#: The default the four scripts each spell separately.
DEFAULT_LOCATION = "external"

#: Where the declarations live, relative to the repository root.
DEFAULT_MANIFEST = "configs/data_manifest.yaml"


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
class SourceRegistry:
    """Every declared source, and where the declarations came from.

    `manifest_source` records the path read, exactly as `StoragePolicy.source`
    does. A reader that cannot say where its values came from cannot be
    audited.
    """

    declarations: Tuple[SourceDeclaration, ...]
    manifest_source: str

    def __post_init__(self) -> None:
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
        import yaml

        p = Path(manifest)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise SourceRegistryError(
                "cannot read {}: {}. There is no defensible default for a "
                "source registry.".format(p, exc)) from exc
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
                aliases=tuple(str(a) for a in (meta.get("aliases") or [])),
                version=str(meta.get("version", "")),
                acquire=str(meta.get("acquire", "")),
                regenerate=str(meta.get("regenerate", "")),
                sync=bool(meta.get("sync", False)),
                notes=str(meta.get("notes", "")),
            ))
        return cls(declarations=tuple(sorted(out, key=lambda d: d.name)),
                   manifest_source=str(p))

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
