from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def semantic_clusters(df: pd.DataFrame, n_clusters: int=8) -> pd.Series:
    text=(df["query"]+" [PAIR] "+df["document"]).tolist()
    pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=3)),("svd",TruncatedSVD(n_components=12,random_state=42)),("cluster",KMeans(n_clusters=n_clusters,n_init=20,random_state=42))])
    return pd.Series(pipe.fit_predict(text),index=df.index,name="semantic_cluster")
