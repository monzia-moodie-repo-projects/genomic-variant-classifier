import pandas as pd, numpy as np
import sys; sys.path.insert(0, 'scripts')
from build_cohort_from_source import (collapse_duplicate_variants,
    _dedup_severity as _severity, _DEDUP_SEVERITY as _SEVERITY)

COLS = ["variant_id","chrom","pos","ref","alt","pathogenicity","clinical_sig","source_id","metadata"]
def row(vid, path, sig, sid, chrom="1", pos=100, ref="A", alt="T", md=None):
    return {"variant_id":vid,"chrom":chrom,"pos":pos,"ref":ref,"alt":alt,
            "pathogenicity":path,"clinical_sig":sig,"source_id":sid,"metadata":md}
def frame(rows): return pd.DataFrame(rows)[COLS]

def test_severity_total_order():
    ranks=[_severity(k) for k in ["pathogenic","likely_pathogenic","uncertain","likely_benign","benign"]]
    assert ranks==[5,4,3,2,1]
    assert _severity("Conflicting")==0 and _severity(None)==0
    assert len(set(_SEVERITY.values()))==5  # strict distinct ranks

def test_identical_twin_collapse():
    df=frame([row("v1","uncertain","Uncertain significance",202444),
              row("v1","uncertain","Uncertain significance",1329154)])
    out,audit=collapse_duplicate_variants(df)
    assert len(out)==1
    md=out.iloc[0]["metadata"]
    assert md["classification_conflict"] is False
    assert md["conflict_span"]==0
    assert md["collapse_all_variation_ids"]==[202444,1329154]
    assert md["collapsed_from_n"]==2
    assert len(audit)==1 and audit[0]["classification_conflict"] is False

def test_conflict_collapse_keeps_most_severe():
    df=frame([row("v2","likely_pathogenic","Likely pathogenic",1325045),
              row("v2","pathogenic","Pathogenic",2924090)])
    out,audit=collapse_duplicate_variants(df)
    assert len(out)==1
    keep=out.iloc[0]
    assert keep["pathogenicity"]=="pathogenic"          # most severe wins
    assert keep["source_id"]==2924090                    # the pathogenic row
    md=keep["metadata"]
    assert md["classification_conflict"] is True
    assert md["conflict_span"]==1                         # 5-4
    assert set(md["collapse_all_pathogenicities"])=={"pathogenic","likely_pathogenic"}
    assert audit[0]["dropped_source_ids"]==[1325045]

def test_no_duplicate_noop():
    df=frame([row("a","pathogenic","Pathogenic",1),
              row("b","benign","Benign",2),
              row("c","uncertain","Uncertain significance",3)])
    out,audit=collapse_duplicate_variants(df)
    assert audit==[]
    # identical content (order-insensitive), metadata untouched
    pd.testing.assert_frame_equal(out.sort_values("variant_id").reset_index(drop=True),
                                  df.sort_values("variant_id").reset_index(drop=True))

def test_determinism_under_shuffle():
    base=[row("v2","likely_pathogenic","Likely pathogenic",1325045),
          row("v2","pathogenic","Pathogenic",2924090),
          row("v1","uncertain","U",202444), row("v1","uncertain","U",1329154),
          row("x","benign","Benign",9)]
    import random
    outs=[]
    for seed in range(5):
        r=base[:]; random.Random(seed).shuffle(r)
        out,_=collapse_duplicate_variants(frame(r))
        outs.append(out.sort_values("variant_id").reset_index(drop=True))
    for o in outs[1:]:
        pd.testing.assert_frame_equal(outs[0], o)

def test_three_way_collision():
    df=frame([row("v3","pathogenic","Pathogenic",30),
              row("v3","likely_pathogenic","Likely pathogenic",10),
              row("v3","uncertain","Uncertain significance",20)])
    out,audit=collapse_duplicate_variants(df)
    assert len(out)==1
    keep=out.iloc[0]
    assert keep["pathogenicity"]=="pathogenic" and keep["source_id"]==30
    md=keep["metadata"]
    assert md["collapsed_from_n"]==3
    assert md["conflict_span"]==2                          # 5-3
    assert md["collapse_all_variation_ids"]==[10,20,30]
    assert md["classification_conflict"] is True

def test_tiebreak_same_severity_lowest_source_id():
    df=frame([row("v4","pathogenic","Pathogenic",500),
              row("v4","pathogenic","Pathogenic",100)])
    out,audit=collapse_duplicate_variants(df)
    keep=out.iloc[0]
    assert keep["source_id"]==100                          # lowest id
    md=keep["metadata"]
    assert md["classification_conflict"] is False          # same label
    assert md["conflict_span"]==0
    assert md["collapse_all_variation_ids"]==[100,500]

def test_inertness_preserves_other_rows_and_count():
    # mix: 1 dup group (2 rows) + 3 unique -> 1 + 3 = 4 rows out
    df=frame([row("d","pathogenic","Pathogenic",2),row("d","pathogenic","Pathogenic",1),
              row("u1","benign","Benign",7),row("u2","uncertain","U",8),row("u3","likely_benign","LB",9)])
    out,audit=collapse_duplicate_variants(df)
    assert len(out)==4 and len(audit)==1
    # the 3 unique rows are untouched (metadata still None)
    for vid in ("u1","u2","u3"):
        assert out[out["variant_id"]==vid].iloc[0]["metadata"] is None

def test_metadata_original_preserved_and_not_mutated():
    orig_md={"rs_id":123}
    df=frame([row("v5","pathogenic","Pathogenic",2,md=dict(orig_md)),
              row("v5","likely_pathogenic","LP",1,md={"rs_id":456})])
    out,_=collapse_duplicate_variants(df)
    keep=out.iloc[0]
    assert keep["pathogenicity"]=="pathogenic"
    assert keep["metadata"]["rs_id"]==123                  # survivor's own metadata kept
    assert keep["metadata"]["classification_conflict"] is True
    # original dict object not mutated
    assert orig_md=={"rs_id":123}
