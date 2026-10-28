import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.tolerance_coordination import choose_alpha
from sttw_control.native_goal_run import tolerance_inventory


def decide(targets,q,**kw):
    import jax.numpy as j
    return choose_alpha(j.array(targets),1.,j.array([2.,0.]),2.,j.array(q),kw.pop('permission',True),kw.pop('latched',False),kw.pop('prev',.5),user_alpha=kw.pop('user',1.),task_aware_fallback=kw.pop('enabled',True),**kw)


def test_selects_only_task_feasible_even_when_all_risk_reject():
    targets=[[1.,0.],[2.,0.],[2.2,0.]]
    old=decide(targets,[.4,.7,.75],enabled=False)
    new=decide(targets,[.4,.7,.75])
    assert old['alpha']==.5 and new['alpha']==0.
    assert old['degraded'] and new['degraded'] and new['latched']
    assert new['task_aware_selected']


def test_highest_q_then_proxy_cost_then_index_and_no_feasible_fallback():
    targets=[[1.,0.],[1.2,0.],[1.4,0.]]
    assert decide(targets,[.5,.7,.6])['alpha']==.5
    # Equal quality and equal cost for symmetric candidates0/1 with userhalf.
    tie=decide([[1.,0.],[2.,0.],[1.,0.]],[.7,.1,.7],user=.5,prev=.5)
    assert tie['alpha']==0.
    assert decide([[1.,0.],[1.2,0.],[1.4,0.]],[.7,.7,.1],user=.5)['alpha']==.5
    none=decide([[2.,0.],[2.1,0.],[2.2,0.]],[.9,.7,.6])
    assert none['alpha']==.5 and none['degraded'] and not none['task_aware_selected']


def test_permission_restore_and_joint_eligible_behavior_unchanged():
    targets=[[1.,0.],[1.2,0.],[1.4,0.]]
    for q in ([.2,.3,.9],[.9,.3,.2]):
        old=decide(targets,q,enabled=False);new=decide(targets,q)
        assert old['alpha']==new['alpha'] and old['latched']==new['latched']
        assert not new['degraded'] and not new['task_aware_selected']
    blocked=decide(targets,[.4,.6,.7],permission=False,latched=True,prev=.5)
    assert blocked['alpha']==.5 and blocked['latched'] and not blocked['task_aware_selected']


def test_best_effort_user_choice_does_not_clear_restore_latch_or_risk_status():
    d=decide([[1.,0.],[1.2,0.],[1.4,0.]],[.1,.2,.82],latched=True,user=1.)
    assert d['alpha']==1. and d['degraded'] and d['latched']


def test_exact72_same_physics_budget_and_user_coverage():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_task_aware_fallback72.json').read_text());rows=tolerance_inventory(p)
    assert len(rows)==72 and len({x['case_id'] for x in rows})==72
    assert p['comparison_total_records']==360 and p['cached_physical_cases']==288
    assert {x['method'] for x in rows}=={'task_aware'}
    assert all(sum(x['alpha']==a for x in rows)==24 for a in (0.,.5,1.))
    p['task_aware_fallback']=False
    with pytest.raises(ValueError):tolerance_inventory(p)



def test_task_aware_decision_jit_vmap_has_no_physics_and_keeps_degraded():
    import jax
    import jax.numpy as j
    fn=jax.jit(jax.vmap(lambda q:choose_alpha(j.array([[1.,0.],[2.,0.],[2.2,0.]]),1.,j.array([2.,0.]),2.,q,True,False,.5,task_aware_fallback=True)))
    out=fn(j.array([[.4,.7,.75],[.5,.6,.7]]))
    np.testing.assert_array_equal(out['alpha'],[0.,0.])
    assert np.all(out['degraded']) and np.all(out['latched'])
