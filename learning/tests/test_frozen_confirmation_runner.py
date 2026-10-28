import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.native_goal_run import tolerance_inventory,select_confirmation_arm
from sttw_control.tolerance_coordination import choose_alpha,speedcap_latent


def protocol():return json.loads((Path(__file__).parents[1]/'configs/h75_frozen_confirmation200.json').read_text())


def test_exact200_new_lanes_no_cache_and_all10_arm_reset_groups():
    p=protocol();rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==200
    assert p['maximum_lower_ticks']==200*10/.005==400000
    assert p['cached_physical_cases']==0 and 'comparison_source' not in p
    groups={}
    for r in rows:groups.setdefault((r['bank_index'],r['route'],r['event']),[]).append(r)
    assert len(groups)==20 and all(len(r)==10 for r in groups.values())
    assert {r['bank_index'] for r in rows}=={1,7}
    for user in (0.,.5,1.):
        for method in ('task_aware','guard','fixed'):assert sum(r['alpha']==user and r['method']==method for r in rows)==20
    assert sum(r['method']=='speedcap' for r in rows)==20
    assert all(r['alpha']==1. for r in rows if r['method']=='speedcap')
    assert sum(r['force_N']==0 for r in rows)==40


@pytest.mark.parametrize('key,value',[('expected_cases',72),('maximum_lower_ticks',144000),('bank_indices',[4,6]),('onsets_s',[2.2]),('optimizer_updates',1)])
def test_frozen_confirmation_protocol_cannot_change_budget_or_initials(key,value):
    p=protocol();p[key]=value
    with pytest.raises(ValueError):tolerance_inventory(p)


def test_dispatch_frozen_taskaware_guard_fixed_and_permission():
    import jax
    import jax.numpy as j
    methods=j.array([0,1,2,0,1,2,0,1,2,3]);users=j.array([0.,0.,0.,.5,.5,.5,1.,1.,1.,1.])
    def one(method,user):
        d=choose_alpha(j.array([[1.,0.],[2.,0.],[2.2,0.]]),1.,j.array([2.,0.]),2.,j.array([.3,.4,.5]),True,False,user,user_alpha=user,use_q=method!=1,task_aware_fallback=method==0)
        a,l=select_confirmation_arm(method,d,True,user,False,user)
        return a,l,d['task_aware_fallback']
    alpha,latch,flag=jax.jit(jax.vmap(one))(methods,users)
    np.testing.assert_array_equal(alpha,[0.,0.,0.,0.,0.,.5,0.,0.,1.,1.])
    np.testing.assert_array_equal(flag,[True,False,False,True,False,False,True,False,False,False])
    assert np.all(np.asarray(latch)[[0,3,6]])
    for m in (0,1,2,3):
        d=dict(alpha=j.asarray(.5),latched=j.asarray(True))
        a,l=select_confirmation_arm(j.asarray(m),d,False,.5,True,1.)
        assert a==.5 and l
