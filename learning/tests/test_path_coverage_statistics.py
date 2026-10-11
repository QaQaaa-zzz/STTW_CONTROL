import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as j
import numpy as np
import pytest

from sttw_control import path_command_statistics as stats


def fixture():
    cfg=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Feedback_V1.json').read_text())
    route=SimpleNamespace(path=SimpleNamespace(turn_end=3.),family=2,angle=.5,
                          speed=2.5,radius=2.1,sample_group=2)
    logs=dict(active_tick=j.array([1,1,1,0],bool),chi=j.array([0.,.6,.2,1.]),
        path_curvature=j.array([.2,-.2,0.,1.]),path_progress=j.array([1.,2.,4.,5.]),
        actual_forward_speed=j.array([2.6,2.4,2.5,np.nan]),v_user=j.full(4,2.5),
        path_cross_track=j.array([.1,-.2,.3,np.nan]),
        path_heading_error=j.array([.02,-.03,.04,np.nan]),
        peak_roll=j.array([.31,.4,.2,np.nan]),lower_error=j.array([[.1,.01],[-.2,-.02],[.3,.03],[np.nan,np.nan]]))
    for key in ('target_offset','filtered_offset','offsets','governed','nominal'):
        logs[key]=j.array([[.1,.02],[-.2,-.03],[.3,.04],[np.nan,np.nan]])
    for name in ('raw_components','effective_components','component_capped'):
        logs[name]={k:j.array([1.,2.,3.,np.nan]) for k in stats.COST_KEYS}
    return logs,route,cfg


def test_geometric_demand_and_local_signs_are_independent():
    logs,route,cfg=fixture()
    result=stats.summarize_statistics(stats.step_statistics(logs,route,cfg),cfg)
    w=result['windows']
    assert w['ordinary']['samples']==1
    assert w['bend']==w['demand_positive']
    assert w['geometric_turn']['samples']==2
    assert w['local_left']['samples']==w['local_right']['samples']==1
    assert w['local_right']['lateral_signed_mean_m']==pytest.approx(-.2)
    assert w['geometric_turn']['working_roll_exceedance_duration_s']==pytest.approx(.01)
    assert w['geometric_turn']['peak_roll_rad']==pytest.approx(.4)
    assert w['local_right']['commands']['offsets']['signed_mean']==pytest.approx([-.2,-.03])
    assert w['local_right']['lower_actual_minus_governed']['signed_mean']==pytest.approx([-.2,-.02])
    assert w['local_right']['costs']['speed']['raw_mean']==pytest.approx(2.)


def test_peak_uses_max_across_device_and_host_aggregation():
    logs,route,cfg=fixture()
    a=stats.step_statistics(logs,route,cfg)
    b=stats.step_statistics({**logs,'peak_roll':logs['peak_roll']*.5},route,cfg)
    merged=jax.jit(stats.merge_statistics)(a,b)
    stacked=jax.tree.map(lambda x,y:j.stack([x,y]),a,b)
    reduced=jax.jit(stats.reduce_statistics)(stacked)
    for key in a:
        np.testing.assert_allclose(merged[key],reduced[key])
    np.testing.assert_allclose(merged['window_peak_roll'],a['window_peak_roll'])
    host=stats.reduce_statistics(jax.device_get(stacked))
    assert isinstance(host['window_peak_roll'],np.ndarray)
    assert float(merged['count'])==6


def test_reset_counts_distinct_from_active_sample_weighting_and_legacy_summary():
    logs,route,cfg=fixture()
    s=stats.step_statistics(logs,route,cfg)
    assert float(s['reset_count'])==0
    s=stats.merge_statistics(s,stats.reset_statistics(route))
    s=stats.merge_statistics(s,stats.reset_statistics(route,False))
    result=stats.summarize_statistics(s,cfg)
    assert result['route_distribution']['reset_count']==1
    assert result['route_distribution']['reset_family_fraction']==[0.,0.,1.]
    assert result['route_distribution']['sample_group_fraction']==[0.,0.,1.]
    # Saved v1 trees have only the original four windows and no enriched leaves.
    old={k:v for k,v in s.items() if not k.startswith(('window_','reset_','route_sample_group'))}
    old.update(window_count=s['window_count'][:4],window_error_sumsq=s['window_error_sumsq'][:4])
    legacy=stats.summarize_statistics(old,cfg)
    assert legacy['schema']=='path_training_statistics_v1'
    assert legacy['windows']['bend']['samples']==2
    assert 'geometric_turn' not in legacy['windows']


def test_inactive_nan_padding_empty_windows_and_active_nan_rejection():
    logs,route,cfg=fixture()
    logs['active_tick']=j.zeros(4,bool)
    s=stats.step_statistics(logs,route,cfg)
    result=stats.summarize_statistics(s,cfg)
    assert result['active_lower_samples']==0
    assert all(w['peak_roll_rad'] is None for w in result['windows'].values())
    assert all(w['sample_fraction'] is None for w in result['windows'].values())
    logs['active_tick']=j.ones(4,bool)
    with pytest.raises(ValueError,match='nonfinite active'):
        stats.summarize_statistics(stats.step_statistics(logs,route,cfg),cfg)
