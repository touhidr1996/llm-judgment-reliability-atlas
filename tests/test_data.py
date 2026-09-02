import pandas as pd
from judgment_atlas.data import generate_judgments
from judgment_atlas.model import temporal_split


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_judgments(100,7),generate_judgments(100,7))


def test_temporal_split_prevents_leakage():
    train,test=temporal_split(generate_judgments(2000),2)
    assert train.batch.max()<test.batch.min()
