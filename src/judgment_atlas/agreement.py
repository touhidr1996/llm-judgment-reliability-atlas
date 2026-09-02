from __future__ import annotations

import numpy as np
import pandas as pd


def gwet_ac1(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0: return float("nan")
    observed = float(np.mean(a == b))
    p = (np.mean(a == 1) + np.mean(b == 1)) / 2
    expected = 2 * p * (1-p)
    return float((observed-expected)/(1-expected)) if expected < 1 else 0.0


def audit_slices(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for keys, g in df.groupby(["context","model"], sort=True):
        rows.append({"context":keys[0],"model":keys[1],"n":len(g),"disagreement_rate":g.disagreement.mean(),"gwet_ac1":gwet_ac1(g.reference_label,g.judge_label),"false_positive_rate":((g.reference_label==0)&(g.judge_label==1)).mean(),"false_negative_rate":((g.reference_label==1)&(g.judge_label==0)).mean()})
    return pd.DataFrame(rows)


def persona_flip_rate(df: pd.DataFrame) -> float:
    rates=df.groupby(["context","persona"]).disagreement.mean().unstack()
    return float((rates.max(axis=1)-rates.min(axis=1)).mean())
