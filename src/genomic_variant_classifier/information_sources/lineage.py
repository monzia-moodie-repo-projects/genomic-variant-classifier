"""Stage 1D-1b.2: the structural integrity of information lineage (ADMISSION-CYCLIC-LINEAGE-1).

Owner ruling 7f4b69d0 (2026-11-09), section 3. The 1D-1b.1 admission was the GREATEST fixed point of "admitted iff the
disposition is admitted and every dependency is admitted". A cycle satisfies that rule with no grounded origin: A -> B -> A,
both admit_strict, are both admitted, and so is a self-loop A -> A. A scientific dependency must terminate in an identified
source or an authorised computational input, so two questions are answered separately, and the first one first:

1. Is the dependency graph structurally valid? Every dependency is declared, there is no cycle (``require_acyclic`` names
   EVERY cyclic component at once, self-loops included -- not only the first cycle a depth-first search meets), and every
   leaf names a registered logical source (``ungrounded_leaves``; the census ``datasets`` registry is the one authority).
2. Which names are admissible under a regime? ``least_fixed_point`` evaluates bottom-up in ``dependency_order``: a name is
   admitted only after all its dependencies were admitted, so a cycle could never admit itself even if the structural check
   were bypassed (defence in depth: two independent mechanisms).

On an acyclic graph the least and greatest fixed points coincide, so every admission the census has measured so far is
unchanged (tested on the real census). An iterative computation that legitimately uses feedback (a recurrent or
message-passing model) is represented as a versioned operation with declared inputs, not as a cycle among identities.

Everything is iterative (no recursion limit) and deterministic (sorted traversal). Standard library only.
"""
from __future__ import annotations

import heapq
import logging
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


def _check_declared(deps: Mapping[str, Sequence[str]]) -> None:
    unknown = sorted({d for ds in deps.values() for d in ds} - set(deps))
    if unknown:
        raise ValueError(f"undeclared dependencies: {unknown}")
    for name, ds in deps.items():
        if len(set(ds)) != len(tuple(ds)):
            raise ValueError(f"{name}: a dependency is listed twice")


def cyclic_components(deps: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Every strongly connected component that is a cycle (more than one name, or a self-loop), each sorted, sorted.

    Tarjan's algorithm with an explicit stack, so a dependency chain of any depth is handled."""
    _check_declared(deps)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: list[list[str]] = []
    counter = 0
    for root in sorted(deps):
        if root in index:
            continue
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(sorted(deps[root])))]
        while work:
            v, children = work[-1]
            child = next(children, None)
            if child is not None:
                if child not in index:
                    index[child] = low[child] = counter
                    counter += 1
                    stack.append(child)
                    on_stack.add(child)
                    work.append((child, iter(sorted(deps[child]))))
                elif child in on_stack:
                    low[v] = min(low[v], index[child])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[v])
            if low[v] == index[v]:
                component: list[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                if len(component) > 1 or v in deps[v]:
                    out.append(sorted(component))
    return sorted(out)


def require_acyclic(deps: Mapping[str, Sequence[str]]) -> None:
    """Refuse a lineage graph with any cycle, naming every cyclic component."""
    cycles = cyclic_components(deps)
    if cycles:
        raise ValueError("cyclic information lineage (ADMISSION-CYCLIC-LINEAGE-1): "
                         + "; ".join(" <-> ".join(c) for c in cycles))


def dependency_order(deps: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """A deterministic topological order (dependencies first; ties broken by name). Refuses a cyclic graph."""
    require_acyclic(deps)
    remaining = {n: set(ds) for n, ds in deps.items()}
    users: dict[str, list[str]] = {n: [] for n in deps}
    for n, ds in deps.items():
        for d in ds:
            users[d].append(n)
    ready = [n for n, ds in remaining.items() if not ds]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        n = heapq.heappop(ready)
        order.append(n)
        for u in users[n]:
            remaining[u].discard(n)
            if not remaining[u]:
                heapq.heappush(ready, u)
    if len(order) != len(deps):          # unreachable after require_acyclic; kept so a defect cannot pass silently
        raise AssertionError("topological order did not cover the graph")
    return tuple(order)


def least_fixed_point(disposition_admits: Mapping[str, bool], deps: Mapping[str, Sequence[str]]) -> dict[str, bool]:
    """Admitted iff the disposition admits the name AND every dependency was admitted FIRST (bottom-up).

    Does not itself require acyclicity: a name on or downstream of a cycle is never reached bottom-up and is therefore
    not admitted, whatever its disposition. Callers that admit run ``require_acyclic`` first; this is the second guard."""
    if set(disposition_admits) != set(deps):
        raise ValueError("dispositions and dependencies must name the same identities")
    _check_declared(deps)
    remaining = {n: set(ds) for n, ds in deps.items()}
    users: dict[str, list[str]] = {n: [] for n in deps}
    for n, ds in deps.items():
        for d in ds:
            users[d].append(n)
    admitted: dict[str, bool] = {}
    ready = [n for n, ds in remaining.items() if not ds]
    heapq.heapify(ready)
    while ready:
        n = heapq.heappop(ready)
        admitted[n] = bool(disposition_admits[n]) and all(admitted[d] for d in deps[n])
        for u in users[n]:
            remaining[u].discard(n)
            if not remaining[u]:
                heapq.heappush(ready, u)
    for n in deps:                      # never resolved: on, or downstream of, a cycle
        admitted.setdefault(n, False)
    return admitted


def ancestors(name: str, deps: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Every identity ``name`` depends on, directly or transitively, sorted (the transitive lineage)."""
    if name not in deps:
        raise ValueError(f"unknown identity {name!r}")
    seen: set[str] = set()
    work = list(deps[name])
    while work:
        d = work.pop()
        if d in seen:
            continue
        seen.add(d)
        work.extend(deps[d])
    return tuple(sorted(seen))


def ungrounded_leaves(source_ids: Mapping[str, Sequence[str]], deps: Mapping[str, Sequence[str]],
                      registered_datasets: Iterable[str]) -> list[str]:
    """Leaves (no dependency) that name no REGISTERED logical source ('dataset:<id>' with <id> in the registry)."""
    registry = {f"dataset:{d}" for d in registered_datasets}
    return sorted(n for n, ds in deps.items()
                  if not ds and not any(s in registry for s in source_ids.get(n, ())))
