import numpy as np

from review_lineage_cost_mechanisms import common_masks


def test_common_prefix_uses_earliest_actual_arrival_and_empty_demand_is_not_evidence():
    cfg={'residual_interface':{'speed_reference_bounds_m_s':[1.,3.5],'steer_reference_bound_rad':.65}}
    def trace():
        return dict(physical_tick=np.arange(8),active_tick=np.ones(8,bool),
            reference=np.tile([1.2,.1],(8,1)),nominal_reference=np.tile([1.2,.1],(8,1)),
            actor_action_valid=np.ones(8,bool),chi=np.zeros(8))
    a,b=trace(),trace()
    masks,state=common_masks(a,b,[.02,None],cfg)
    assert masks['through_first_goal_entry'].tolist()==[True]*4+[False]*4
    assert masks['after_first_goal_entry'].sum()==4
    assert masks['both_chi_active'].sum()==0
    assert masks['prefix_both_legal_no_current_bound'].sum()==4
    assert masks['prefix_both_legal_in_old_reference_box'].sum()==0
    assert state['outside_old_box_ticks']==[8,8]
    masks,_=common_masks(a,b,[None,None],cfg)
    assert masks['through_first_goal_entry'].all()
    assert not masks['after_first_goal_entry'].any()
