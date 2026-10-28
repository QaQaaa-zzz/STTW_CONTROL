import numpy as np
import pytest
from sttw_control.geometric_path import Path
from sttw_control.goal_endpoint_replanning import pad_original_path,periodic_plan


def test_padding_keeps_every_original_coordinate_and_goal():
    p=Path(np.array([0.,1.,2.]),np.array([[0.,0.],[1.,0.],[2.,0.]]),np.zeros(3),np.zeros(3),0.,1.5)
    out=pad_original_path(p,count=6,spacing_m=.2)
    np.testing.assert_array_equal(out.xy[:3],p.xy)
    np.testing.assert_array_equal(out.s[:3],p.s)
    assert float(out.goal)==1.5 and float(out.turn_end)==0.
    np.testing.assert_allclose(out.xy[-1],[2.6,0.],atol=1e-6)


def test_periodic_planning_is_causal_and_retains_original_destination():
    path,meta=periodic_plan([0,0,.2],[4,2],task_tick=400,arrived=False,contact=False,fault=False)
    assert len(path.s)==4001 and meta['task_tick']==400
    np.testing.assert_allclose([np.interp(float(path.goal),path.s,path.xy[:,i]) for i in (0,1)],[4,2],atol=3e-6)
    assert meta['radius_m']==1.8 and not meta['safety_certificate']


@pytest.mark.parametrize('tick',[0,199,201,2000])
def test_no_early_off_cadence_or_post_horizon_plan(tick):
    assert periodic_plan([0,0,0],[4,2],task_tick=tick,arrived=False,contact=False,fault=False) is None


@pytest.mark.parametrize('key',['arrived','contact','fault'])
def test_first_arrival_or_failure_is_never_reset(key):
    flags=dict(arrived=False,contact=False,fault=False);flags[key]=True
    assert periodic_plan([0,0,0],[4,2],task_tick=400,**flags) is None


def test_invalid_geometry_does_not_become_a_fabricated_plan():
    with pytest.raises(ValueError):periodic_plan([0,0,np.nan],[4,2],task_tick=400,arrived=False,contact=False,fault=False)
