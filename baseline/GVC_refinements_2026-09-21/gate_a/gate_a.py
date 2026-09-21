from dataclasses import dataclass
import re
import numpy as np
import pandas as pd

class GateError(ValueError):
    pass

@dataclass(frozen=True)
class Gene:
    gene_id: str
    tax_id: int
    symbol: str
    aliases: tuple = ()
    hgnc_id: str | None = None

class GeneResolver:
    """Inputs must come from validated, pinned human Gene/HGNC adapters.
    Discontinued IDs must be reconciled explicitly before construction.
    Aliases produce review candidates, never automatic equivalence edges.
    """
    def __init__(self, records):
        self.by_id, self.symbols, self.aliases = {}, {}, {}
        hgnc_owners = {}
        for r in records:
            if not re.fullmatch(r'NCBIGene:[1-9][0-9]*', r.gene_id):
                raise GateError('Invalid namespaced GeneID')
            if r.tax_id != 9606 or not r.symbol or r.symbol != r.symbol.strip():
                raise GateError('Require human records and nonempty exact symbols')
            if r.gene_id in self.by_id:
                raise GateError('Duplicate GeneID')
            if r.hgnc_id is not None:
                if not re.fullmatch(r'HGNC:[1-9][0-9]*', r.hgnc_id):
                    raise GateError('Invalid HGNC cross-reference')
                if r.hgnc_id in hgnc_owners:
                    raise GateError('HGNC cross-reference is not one-to-one')
                hgnc_owners[r.hgnc_id] = r.gene_id
            self.by_id[r.gene_id] = r
            self.symbols.setdefault(r.symbol, set()).add(r.gene_id)
            for a in r.aliases:
                self.aliases.setdefault(a, set()).add(r.gene_id)
        if not self.by_id:
            raise GateError('Empty identity map')

    def resolve(self, *, source_gene_id=None, symbol=None):
        exact = self.symbols.get(symbol, set())
        if source_gene_id is not None:
            if source_gene_id not in self.by_id:
                raise GateError('Source GeneID absent from pinned identity map')
            if exact and source_gene_id not in exact:
                raise GateError('Source ID conflicts with exact symbol evidence')
            return {'state': 'resolved', 'gene_id': source_gene_id,
                    'basis': 'source_gene_id',
                    'symbol_unconfirmed': symbol is not None and not exact}
        if len(exact) == 1:
            return {'state': 'resolved', 'gene_id': next(iter(exact)),
                    'basis': 'unique_current_symbol', 'symbol_unconfirmed': False}
        candidates = exact or self.aliases.get(symbol, set())
        return {'state': 'ambiguous' if len(exact) > 1 else
                         'review_required' if candidates else 'unresolved',
                'gene_id': None, 'candidates': sorted(candidates)}


def normalize_consequence(*, so_id, raw_term, active_terms, obsolete_ids):
    """Use a source-record SO accession; preserve the display term separately.
    active_terms is SO accession -> canonical label from a pinned ontology.
    Does not infer an accession from an arbitrary label or replace obsolete IDs.
    """
    if so_id is None:
        return {'state': 'missing' if raw_term in (None, '') else 'needs_mapping',
                'so_id': None, 'raw_term': raw_term}
    if not isinstance(so_id, str) or not re.fullmatch(r'SO:[0-9]{7}', so_id):
        raise GateError('Malformed SO accession')
    if so_id in obsolete_ids:
        raise GateError('Obsolete SO accession requires a versioned migration')
    if so_id not in active_terms:
        raise GateError('SO accession absent from pinned ontology')
    return {'state': 'observed', 'so_id': so_id,
            'canonical_term': active_terms[so_id], 'raw_term': raw_term}


def weighting_attribution(delta, groups):
    d = np.asarray(delta, dtype=float)
    g = np.asarray(groups, dtype=object)
    if d.ndim != 1 or g.ndim != 1 or len(d) != len(g) or not len(d):
        raise GateError('Require nonempty aligned 1D arrays')
    if not np.isfinite(d).all() or any(not isinstance(x, str) or
        not x or x != x.strip() for x in g):
        raise GateError('Invalid values or group identities')
    table = pd.DataFrame({'group': g, 'd': d}).groupby('group', sort=True).d.agg(
        n='size', effect='mean')
    G, N = len(table), len(d)
    p = table.n.to_numpy(float) / N
    a = table.effect.to_numpy(float)
    variant, group = float(p @ a), float(a.mean())
    table['gap_contribution'] = (p - 1/G) * a
    covariance = float(np.mean((p - p.mean()) * (a - a.mean())))
    if not np.isclose(variant-group, G*covariance, rtol=1e-12, atol=1e-14):
        raise GateError('Decomposition failed')
    return {'variant_weighted': variant, 'group_weighted': group,
            'gap': variant-group, 'G_times_population_covariance': G*covariance,
            'sign_reversal': bool(variant*group < 0)}, table
