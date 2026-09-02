from __future__ import annotations

import json, sqlite3
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agreement import audit_slices, gwet_ac1, persona_flip_rate
from .data import generate_judgments
from .evaluate import global_baseline, risk_metrics
from .model import FEATURES, make_risk_model, temporal_split
from .semantic import semantic_clusters

ROOT=Path(__file__).resolve().parents[2]


def run() -> dict:
    for d in ["data","artifacts","reports"]:(ROOT/d).mkdir(exist_ok=True)
    data=generate_judgments(); data["semantic_cluster"]=semantic_clusters(data)
    train,test=temporal_split(data)
    model=make_risk_model(); model.fit(train[FEATURES],train.disagreement)
    p=model.predict_proba(test[FEATURES])[:,1]
    baseline=global_baseline(test.disagreement,float(train.disagreement.mean()))
    improved=risk_metrics(test.disagreement,p)
    slices=audit_slices(test)
    clusters=test.groupby("semantic_cluster").agg(n=("event_id","size"),disagreement_rate=("disagreement","mean"),mean_boundary_distance=("boundary_distance","mean")).reset_index()
    summary={"split":{"train_rows":len(train),"test_rows":len(test),"train_end_batch":int(train.batch.max()),"test_start_batch":int(test.batch.min())},"global_agreement":{"gwet_ac1":gwet_ac1(test.reference_label,test.judge_label),"disagreement_rate":float(test.disagreement.mean())},"baseline_global_risk":baseline,"cluster_aware_risk_model":improved,"relative_error_capture_gain":float(improved["error_capture_at_25pct_review"]/baseline["error_capture_at_25pct_review"]-1),"persona_sensitivity":{"mean_context_flip_rate":persona_flip_rate(test)},"intervention":{"label_disagreement":float(test[test.intervention=="label"].disagreement.mean()),"independent_first_disagreement":float(test[test.intervention=="independent_first"].disagreement.mean())},"hotspots":{"count":int((clusters.disagreement_rate>.20).sum()),"highest_cluster_rate":float(clusters.disagreement_rate.max())}}
    (ROOT/"reports"/"metrics.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    data.to_csv(ROOT/"data"/"synthetic_judgment_events.csv",index=False)
    scored=test[["event_id","batch","context","model","persona","intervention","semantic_cluster","reference_label","judge_label","disagreement"]].assign(disagreement_risk=p,review_required=p>=np.quantile(p,.75))
    scored.to_csv(ROOT/"artifacts"/"holdout_triage.csv",index=False); slices.to_csv(ROOT/"artifacts"/"slice_audit.csv",index=False); clusters.to_csv(ROOT/"artifacts"/"cluster_atlas.csv",index=False)
    joblib.dump(model,ROOT/"artifacts"/"model.joblib")
    with sqlite3.connect(ROOT/"artifacts"/"atlas.db") as con:
        scored.to_sql("fact_triage",con,if_exists="replace",index=False); slices.to_sql("slice_audit",con,if_exists="replace",index=False); clusters.to_sql("cluster_atlas",con,if_exists="replace",index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    slices.pivot(index="context",columns="model",values="disagreement_rate").plot(kind="bar",ax=axes[0],color=["#2563eb","#f59e0b","#dc2626"]);axes[0].set(title="Structured disagreement",ylabel="Rate",xlabel="Context");axes[0].legend(fontsize=7)
    clusters.plot(x="semantic_cluster",y="disagreement_rate",kind="bar",ax=axes[1],legend=False,color="#7c3aed");axes[1].axhline(.2,ls="--",color="#dc2626");axes[1].set(title="Semantic hotspots",ylabel="Disagreement",xlabel="Cluster")
    curves=pd.DataFrame({"review_fraction":np.linspace(.05,1,20)});order=np.argsort(-p);curves["captured_error"]=[test.disagreement.to_numpy()[order[:max(1,int(len(test)*f))]].sum()/test.disagreement.sum() for f in curves.review_fraction];axes[2].plot(curves.review_fraction,curves.captured_error,color="#059669",lw=2);axes[2].plot([0,1],[0,1],"--",color="#6b7280");axes[2].set(title="Selective verification",xlabel="Reviewed fraction",ylabel="Errors captured")
    for ax in axes:ax.grid(alpha=.2)
    plt.tight_layout();plt.savefig(ROOT/"reports"/"reliability_atlas.png",dpi=170);plt.close()
    return summary


if __name__=="__main__":print(json.dumps(run(),indent=2))
