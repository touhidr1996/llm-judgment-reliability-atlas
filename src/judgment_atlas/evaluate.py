from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def risk_metrics(y, p, budget: float=.25) -> dict[str,float]:
    y,p=np.asarray(y),np.asarray(p)
    k=max(1,int(len(y)*budget)); reviewed=np.argsort(-p)[:k]
    captured=float(y[reviewed].sum()/max(y.sum(),1))
    residual=float((y.sum()-y[reviewed].sum())/(len(y)-k))
    return {"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"brier":float(brier_score_loss(y,p)),"error_capture_at_25pct_review":captured,"residual_error_rate":residual}


def global_baseline(y, train_rate: float, budget: float=.25) -> dict[str,float]:
    p=np.full(len(y),train_rate)
    return risk_metrics(y,p,budget)
