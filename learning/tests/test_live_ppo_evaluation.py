from sttw_control.live_ppo_evaluation import evaluation_inventory
from sttw_control.live_ppo_schedule import make_evaluation_schedule


def test_evaluation_adapter_preserves_full_pairing_and_actual_reference_permissions():
    templates=[dict(context=n,route=n,speed=v,world_direction=[0.,1.,0.]) for n,v in [('left',1.8),('right',1.5)]]
    declared=make_evaluation_schedule(templates,seed=9)
    rows=evaluation_inventory(declared,'left',reference_floor=1.)
    assert len(rows)==100 and len({c['case_id'] for c in rows})==100
    for old,new in zip([c for c in declared if c['context']=='left'],rows):
        assert new['alpha']==new['state_alpha']==old['alpha']
        assert new['requested_speed_m_s']==old['speed']
        assert new['profile']==old['profile'] and new['force_N']==old['force_N']
        assert new['pulse_onset_task_tick']==old['pulse_onset_task_tick']
        assert new['source_batch_slot']==0 and new['stage']=='reentry'
        assert new['reference_speed_floor_m_s']==1. and not new['zero_mode']
    for alpha in [0.,.25,.5,.75,1.]:
        assert sum(c['alpha']==alpha and c['force_N']==0 for c in rows)==4
