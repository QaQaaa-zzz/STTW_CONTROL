"""Do not fabricate task curves on hold or multiply zero-baseline episodes."""
import numpy as np
import pytest


def test_reward_blocks_preserve_partial_decisions_and_terminal_charge_without_hold_costs():
    from sttw_control.survival_endpoint_figures import reward_blocks
    p=dict(active_tick=np.array([1,1,1,0,1,1,1,1],bool),
        task_reward_valid=np.array([1,1,1,0,0,0,0,0],bool),
        time=np.array([0,.005,.01,0,.015,.02,.025,.03]),
        tick_reward=np.array([-1.,-2.,-3.,0,np.nan,np.nan,np.nan,np.nan]),
        scored_tick_reward=np.array([-1.,-2.,-13.,0,np.nan,np.nan,np.nan,np.nan]),
        effective_costs=np.array([[1,0],[2,0],[3,0],[0,0],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan]]))
    got=reward_blocks(p,scale=1.)
    np.testing.assert_array_equal(got['time'],[0.])
    np.testing.assert_array_equal(got['scored_reward'],[-16.])
    np.testing.assert_array_equal(got['terminal_charge'],[-10.])
    np.testing.assert_array_equal(got['signed_components'],[[-6.,0.]])


def test_plot_panels_reuse_zero_alias_without_counting_new_physics():
    from sttw_control.survival_endpoint_figures import comparison_panels
    rows=[dict(case_id='zero',panel='physical_deterministic',context_index=0,alpha=None,replicate=0,role='zero_middle')]
    for panel,rep in [('physical_deterministic',0),('physical_stochastic',3)]:
        for a in [0.,1.]:
            for role in ['parent','task_only','physical_credit','parent_speed_floor']:
                rows.append(dict(case_id=f'{panel}/{a}/{role}',panel=panel,context_index=0,alpha=a,replicate=rep,role=role))
    panels=comparison_panels(rows)
    assert len(panels)==4
    assert all(p['case_ids']['zero_middle']=='zero' for p in panels)
    assert len({c for p in panels for c in p['case_ids'].values()})==17
    with pytest.raises(ValueError,match='missing'):comparison_panels(rows[:-1])
