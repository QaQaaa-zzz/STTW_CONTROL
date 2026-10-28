import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.native_goal_run import tolerance_inventory
from sttw_control.tolerance_coordination import choose_alpha


def test_user_inventory_exact120_not216_physics():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_user_preference120.json').read_text())
    rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==120
    assert sum(r['alpha']==0 for r in rows)==72
    assert sum(r['alpha']==.5 for r in rows)==48
    assert all(r['alpha']==r['user_alpha']==r['query_alpha'] for r in rows)
    assert {r['method'] for r in rows}=={'last','guard','fixed'}
    p['arms'].reverse()
    with pytest.raises(ValueError):tolerance_inventory(p)


@pytest.mark.parametrize('user',[0.,.5,1.])
def test_discrete_user_priority_other_two_and_permission(user):
    import jax.numpy as j
    targets=j.array([[2.4,0.],[2.4,0.],[2.4,0.]])
    q=j.ones(3)
    args=(targets,j.array(.3),j.array([2.8,0.]),j.array(2.8))
    out=choose_alpha(*args,q,True,False,.5,user_alpha=user)
    assert out['alpha']==user and not out['latched']
    blocked=choose_alpha(*args,q,False,True,.5,user_alpha=user)
    assert blocked['alpha']==.5 and blocked['latched']
    q=q.at[int(user*2)].set(.1)
    fallback=choose_alpha(*args,q,True,False,user,user_alpha=user)
    assert fallback['alpha']!=user
    none=choose_alpha(*args,j.zeros(3),True,False,user,user_alpha=user)
    assert none['alpha']==.5 and none['degraded']


def test_default_user1_old_formula_regression():
    import jax.numpy as j
    rng=np.random.default_rng(42)
    for _ in range(30):
        targets=rng.uniform([1.5,-.35],[3.5,.35],(3,2)).astype('f');nominal=np.array([2.8,.1],'f');q=rng.uniform(0,1,3).astype('f');k=float(rng.uniform(0,.8));latched=bool(rng.integers(2));prev=float(rng.choice([0.,.5,1.]))
        task=targets[:,0]**2*k<=3.;quality=q>=.8;restore=task[2] and q[2]>=(.85 if latched else .8);eligible=task[:2]&quality[:2]
        alphas=np.array([0.,.5,1.]);cost=(targets[:,0]-2.8)**2+.1*((targets[:,1]-.1)/.2)**2+.05*(alphas-1)**2+.02*(alphas-prev)**2
        expected=1. if restore else alphas[np.argmin(np.where(eligible,cost[:2],np.inf))] if eligible.any() else .5
        out=choose_alpha(j.array(targets),k,j.array(nominal),2.8,j.array(q),True,latched,prev)
        assert float(out['alpha'])==expected
        np.testing.assert_allclose(out['proxy_cost'],cost,rtol=1e-6,atol=1e-6)
