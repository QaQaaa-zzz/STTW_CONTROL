import numpy as np
import pytest
from sttw_control.approach_planner import ApproachPlanner


def model():
    return dict(schema='approach_endpoint_v1',identity={'skill':'frozen'},indices=[0],
                mean=[0.],scale=[1.],candidates=[{'speed_m_s':2.,'alpha':0.},{'speed_m_s':2.,'alpha':1.}],
                endpoint_weights=[[-1.],[1.],[0.],[0.]],gate_weights=[8.,0.,0.,0.,0.],
                capability_weights=[0.,4.],entry_scale=[1.],entry_mean=[0.],
                support_states=[[0.]],maximum_support_rms=3.,keep_probability_tolerance=.02,scope='fixture')


def test_skill_replacement_changes_action_without_changing_ground_prediction():
    m=model();old=ApproachPlanner(m,identity=m['identity']).choose([0.],'exit',{'speed_m_s':2.,'alpha':0.})
    m['capability_weights']=[0.,-4.]
    new=ApproachPlanner(m,identity=m['identity']).choose([0.],'exit',{'speed_m_s':2.,'alpha':0.})
    assert old['candidate']['alpha']==1. and new['candidate']['alpha']==0.
    assert [r['predicted_entry'] for r in old['estimates']]==[r['predicted_entry'] for r in new['estimates']]


def test_unsupported_state_retains_actual_reference_and_identity_is_checked():
    m=model();current={'speed_m_s':1.9,'alpha':.5}
    result=ApproachPlanner(m,identity=m['identity']).choose([10.],'exit',current)
    assert result['status']=='unknown' and result['candidate']==current
    with pytest.raises(ValueError,match='identity'):
        ApproachPlanner(m,identity={'skill':'different'})
