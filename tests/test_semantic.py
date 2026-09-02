from judgment_atlas.data import generate_judgments
from judgment_atlas.semantic import semantic_clusters


def test_semantic_clusters_cover_all_rows():
    data=generate_judgments(200)
    labels=semantic_clusters(data,4)
    assert len(labels)==len(data) and labels.nunique()==4
