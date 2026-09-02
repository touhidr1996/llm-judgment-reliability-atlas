from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES=["context","model","persona","intervention","lexical_overlap","ambiguity","boundary_distance"]
CATEGORICAL=["context","model","persona","intervention"]


def temporal_split(df: pd.DataFrame, test_batches: int = 4):
    cutoff=df.batch.max()-test_batches+1
    train,test=df[df.batch<cutoff].copy(),df[df.batch>=cutoff].copy()
    assert train.batch.max()<test.batch.min()
    return train,test


def make_risk_model() -> Pipeline:
    prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),CATEGORICAL)],remainder="passthrough")
    clf=HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=18,l2_regularization=1.5,random_state=42)
    return Pipeline([("features",prep),("risk",clf)])
