import numpy as np
from judgment_atlas.agreement import gwet_ac1
from judgment_atlas.evaluate import risk_metrics


def test_gwet_ac1_is_one_for_identical_labels():
    y=np.array([0,0,1,1,1]);assert abs(gwet_ac1(y,y)-1)<1e-12


def test_risk_metrics_bounds():
    result=risk_metrics([0,1,0,1],[.1,.9,.2,.8],.5)
    assert all(0<=v<=1 for v in result.values())


def test_high_risk_ranking_captures_all_errors():
    result=risk_metrics([0,1,0,1],[.1,.9,.2,.8],.5)
    assert result["error_capture_at_25pct_review"]==1
