"""Read-only counterfactual review-source audit. Does not choose source authority.
Uses this checkout's canonical review resolver and literal production label sets.
Dependencies: pandas; a Parquet engine (e.g. pyarrow) for CLI input only.
"""
import argparse
import ast
from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_policy(repo):
    root = Path(repo) / "src/genomic_variant_classifier/data"
    resolver_path = root / "review_status.py"
    name = "_review_policy_" + sha256(resolver_path)[:16]
    spec = importlib.util.spec_from_file_location(name, resolver_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    prep = root / "real_data_prep.py"
    constants = {}
    for node in ast.parse(prep.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"PATHOGENIC_TERMS","BENIGN_TERMS"}:
                    if target.id in constants:
                        raise ValueError("Duplicate label constant definition")
                    constants[target.id] = ast.literal_eval(node.value)
    if set(constants) != {"PATHOGENIC_TERMS","BENIGN_TERMS"}:
        raise ValueError("Cannot extract literal production label sets; inspect implementation")
    pos, neg = constants["PATHOGENIC_TERMS"], constants["BENIGN_TERMS"]
    if not isinstance(pos, set) or not isinstance(neg, set) or pos & neg:
        raise ValueError("Invalid production label sets")
    return module, pos, neg, {
        "resolver_sha256": sha256(resolver_path), "data_prep_sha256": sha256(prep),
        "pathogenic_terms": sorted(pos), "benign_terms": sorted(neg)}


def nested_status(value):
    if value is None or (not isinstance(value, (Mapping,str)) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object or missing")
    return value.get("review_status")


def representation(ref, alt):
    if not isinstance(ref,str) or not isinstance(alt,str) or not ref or not alt:
        return "unresolved"
    if set(ref+alt)-set("ACGT") or ref == alt:
        return "unresolved"
    if len(ref)==len(alt)==1:
        return "SNV"
    if len(alt)>len(ref): return "net_length_gain"
    if len(ref)>len(alt): return "net_length_loss"
    return "equal_length_replacement"


def audit(df, policy, *, max_tier=3, exclude_conflicting=True):
    resolver, pos, neg, identity = policy
    required={"variant_id","clinical_sig","ReviewStatus","metadata","ref","alt"}
    if not df.columns.is_unique or not required <= set(df.columns):
        raise ValueError("Missing required or duplicate columns")
    if type(max_tier) is not int or not 1 <= max_tier <= 5:
        raise ValueError("max_tier must be an integer from 1 to 5")
    df=df.reset_index(drop=True)
    ids=df["variant_id"]
    if not ids.map(lambda x:isinstance(x,str) and bool(x.strip()) and x==x.strip()).all() or ids.duplicated().any():
        raise ValueError("Require unique, nonmissing normalized row identities")
    sig=df["clinical_sig"]
    if not sig.map(lambda x:isinstance(x,str) or pd.isna(x)).all():
        raise ValueError("clinical_sig must be text or missing")
    sig=sig.fillna("").str.strip()
    binary=sig.isin(pos|neg)
    labels=sig.map({**{x:1 for x in pos},**{x:0 for x in neg}}).astype("Int64")
    conflict=sig.str.contains("onflict",regex=False,na=False)
    other_eligible=binary & (~conflict if exclude_conflicting else True)
    nested=df["metadata"].map(nested_status)
    table=pd.DataFrame({"variant_id":ids,"clinical_sig":sig,"label":labels,
                        "representation":[representation(r,a) for r,a in zip(df.ref,df.alt)]})
    for name, raw in [("top",df["ReviewStatus"]),("nested",nested)]:
        keys=raw.map(resolver.normalise)
        outcomes={}
        for key in keys.unique():
            try:
                result=resolver.resolve(key)
                outcomes[key]=(result.tier,result.path.value)
            except resolver.UnmatchedReviewStatusError:
                outcomes[key]=(None,"UNKNOWN_VOCABULARY")
        table[name+"_status"]=keys
        table[name+"_tier"]=keys.map(lambda k:outcomes[k][0]).astype("Int64")
        table[name+"_path"]=keys.map(lambda k:outcomes[k][1])
        unknown=table[name+"_tier"].isna()
        if (unknown & other_eligible).any():
            values=keys[unknown & other_eligible].value_counts().to_dict()
            raise ValueError(f"{name}: unknown vocabulary among otherwise eligible rows: {values}")
        table[name+"_review_pass"]=table[name+"_tier"].le(max_tier)
        table[name+"_included"]=(table[name+"_review_pass"] & other_eligible).fillna(False).astype(bool)
    a,b=table.top_included,table.nested_included
    table["transition"]="excluded_both"
    table.loc[a & b,"transition"]="included_both"
    table.loc[~a & b,"transition"]="added_under_nested"
    table.loc[a & ~b,"transition"]="removed_under_nested"
    review_known=table.top_review_pass.notna() & table.nested_review_pass.notna()
    review_changed=(table.top_review_pass != table.nested_review_pass).fillna(False)
    summary={
        "scope":"Counterfactual local-source comparison; source authority and historical scope unresolved",
        "policy":{**identity,"max_tier":max_tier,"exclude_conflicting":exclude_conflicting},
        "n_raw":len(table),"n_binary_labels":int(binary.sum()),
        "n_review_comparable":int(review_known.sum()),
        "n_review_only_changed":int(review_changed.sum()),
        "n_final_eligibility_changed":int((a!=b).sum()),
        "n_top_included":int(a.sum()),"n_nested_included":int(b.sum()),
        "transitions":table.transition.value_counts().to_dict(),
        "by_representation":table.groupby(["representation","transition"]).size().rename("n").reset_index().to_dict("records"),
        "changed_clinical_significance":table.loc[a!=b,"clinical_sig"].value_counts().to_dict(),
        "review_transition_counts":table.groupby(["top_status","nested_status"],dropna=False).size().rename("n").reset_index().to_dict("records"),
    }
    return summary,table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",required=True)
    p.add_argument("--cohort",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--max-tier",type=int,default=3)
    args=p.parse_args()
    source=Path(args.cohort)
    before=sha256(source)
    columns=["variant_id","clinical_sig","ReviewStatus","metadata","ref","alt"]
    df=pd.read_parquet(source,columns=columns)
    report,detail=audit(df,load_policy(args.repo),max_tier=args.max_tier)
    if sha256(source)!=before:
        raise ValueError("Source changed during audit")
    report["cohort_sha256"]=before
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    changed=detail[detail.top_status.ne(detail.nested_status)]
    changed.to_csv(out/"review_source_disagreements.csv",index=False)
    report["detail_sha256"]=sha256(out/"review_source_disagreements.csv")
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    for k in ["n_raw","n_binary_labels","n_review_only_changed","n_final_eligibility_changed","n_top_included","n_nested_included"]:
        print(f"{k}: {report[k]}")

if __name__=="__main__": main()
