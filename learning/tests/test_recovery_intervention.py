import copy
from itertools import product
import numpy as np
import pytest
from sttw_control.recovery_intervention import intervention_layout,intervention_schedule,masked_target
from sttw_control.tolerance_comparison import chunk_schedule


def fixture():
    d=dict(config={'timing':{'upper_dt_s':.02,'lower_dt_s':.005},
                   'training_future_phase_C':{'episode_s':10}},
           onsets_s=[2.],duration_s=.1,
           policies={k:dict(actor_sha256='a',checkpoint_sha256='b') for k in ('control','speed_off','steer_off')},
           recovery_intervention=dict(start_s=2.1,end_s=4.1,control='control',
              target_channel_masks={'control':[1,1],'speed_off':[0,1],'steer_off':[1,0]}))
    rows=[dict(policy=p,route=r,speed=2.35,alpha=a,force_N=f)
          for p,r,a,f in product(d['policies'],('left','right'),(0,.125),(-20,0,20))]
    return d,rows


def test_complete_matching_context_and_control_source():
    d,rows=fixture();layout=intervention_layout(d,rows)
    for row,index in zip(rows,layout['source_indices']):
        source=rows[index]
        assert source['policy']=='control'
        assert all(source[k]==row[k] for k in ('route','speed','alpha','force_N'))
    schedule=intervention_schedule(2.,10.,layout)
    assert [i for start,n,_ in schedule for i in range(start,start+n)]==list(range(500))
    assert {100,105,205}.issubset({x[0] for x in schedule})
    assert [s for s,_,branch in schedule if branch]==[100]


def test_restore_target_exactly_at_end_without_changing_other_channel():
    z=np.array([.7,-.3]);mask=np.array([0,1])
    for tick,expected in ((419,z),(420,[0,-.3]),(819,[0,-.3]),(820,z)):
        np.testing.assert_allclose(masked_target(z,tick,mask,420,820),expected)


def test_reject_confounded_or_misaligned_interventions():
    d,rows=fixture()
    for change in ('actor','time','pulse'):
        bad=copy.deepcopy(d)
        if change=='actor':bad['policies']['speed_off']['actor_sha256']='other'
        if change=='time':bad['recovery_intervention']['start_s']=2.101
        if change=='pulse':bad['recovery_intervention']['start_s']=2.08
        with pytest.raises(ValueError):intervention_layout(bad,rows)


def test_ordinary_evaluation_schedule_is_unchanged():
    d,rows=fixture();d.pop('recovery_intervention')
    assert intervention_layout(d,rows) is None
    assert intervention_schedule(2.,10.,None)==chunk_schedule(2.,duration=10.)
