import copy
import jax
import jax.numpy as j
import numpy as np
import pytest
from test_path_coverage_statistics import fixture
from sttw_control import path_command_statistics as stats


def conditional(alpha):
    logs,route,cfg=fixture()
    logs.update(upper_alpha=j.asarray([alpha]*3+[np.nan]),
                scored_tick_reward=j.array([-1.,-2.,-10.,np.nan]),
                physical_failure=j.array([0,0,1,0],bool),
                domain_exit=j.zeros(4,bool))
    return logs,route,cfg


def test_alpha_bins_partition_endpoints_interior_and_keep_terminal_reward():
    combined={}
    for alpha in (0,.1,.25,.5,.75,1):
        logs,route,cfg=conditional(alpha)
        combined=stats.merge_statistics(combined,stats.step_statistics(logs,route,cfg))
    result=stats.summarize_statistics(combined,cfg)
    groups=result['alpha_groups']
    assert len(groups)==6
    assert sum(g['samples'] for g in groups.values())==result['active_lower_samples']==18
    for g in groups.values():
        assert g['samples']==3
        assert g['reward_mean']==pytest.approx(-13/3)
        assert g['physical_failures']==1
        assert g['domain_failures']==0
        assert g['strong_demand_fraction']==pytest.approx(1/3)
        assert g['peak_roll_rad']==pytest.approx(.4)
        assert g['lateral_rmse_m']==pytest.approx(np.sqrt(.14/3))
    assert groups['exact_0']['alpha_mean']==0
    assert groups['exact_1']['alpha_mean']==1


def test_alpha_peak_is_max_and_inactive_nan_does_not_contaminate():
    logs,route,cfg=conditional(.5)
    first=stats.step_statistics(logs,route,cfg)
    logs['peak_roll']*=.5
    second=stats.step_statistics(logs,route,cfg)
    tree=jax.tree.map(lambda x,y:j.stack((x,y)),first,second)
    combined=jax.jit(stats.reduce_statistics)(tree)
    np.testing.assert_allclose(combined['alpha_peak_roll'],first['alpha_peak_roll'])
    summary=stats.summarize_statistics(combined,cfg)
    assert summary['alpha_groups']['exact_1']['reward_mean'] is None
    assert summary['alpha_groups']['interior_05_075']['samples']==6


@pytest.mark.parametrize('alpha',[-.1,1.1,np.nan])
def test_invalid_active_alpha_is_rejected_not_silently_dropped(alpha):
    logs,route,cfg=conditional(alpha)
    with pytest.raises(ValueError,match='alpha'):
        stats.summarize_statistics(stats.step_statistics(logs,route,cfg),cfg)
