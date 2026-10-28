import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.native_goal_run import tolerance_inventory
from sttw_control.tolerance_coordination import candidate_probabilities


def test_prior72_all_users_and_same24_conditions_only():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_user_prior_online72.json').read_text());rows=tolerance_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==72
    assert p['maximum_lower_ticks']==144000 and p['comparison_total_records']==288
    for user in (0.,.5,1.):assert sum(r['alpha']==user for r in rows)==24
    assert all(r['alpha']==r['user_alpha']==r['query_alpha'] for r in rows)
    assert {r['method'] for r in rows}=={'prior_last'}
    assert set(p['arm_to_export_index'].values())=={0}
    p['maximum_lower_ticks']=240000
    with pytest.raises(ValueError):tolerance_inventory(p)


@pytest.mark.parametrize('arms',[1,2])
def test_single_or_historical_two_ensembles_keep_equal_three_seed_mean(arms):
    import jax
    import jax.numpy as j
    params={}
    for layer,indim,outdim in [(0,41,16),(1,16,16),(2,16,1)]:
        params[f'w{layer}']=j.zeros((arms,3,indim,outdim));params[f'b{layer}']=j.zeros((arms,3,outdim))
    params['b2']=j.array(np.broadcast_to(np.array([-1.,0.,2.])[None,:,None],(arms,3,1)))
    interface=dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.2)
    p=jax.jit(lambda obs:candidate_probabilities(params,obs,j.zeros((3,2)),0.,interface))(j.zeros(351))
    assert p.shape==(arms,3,4)
    np.testing.assert_allclose(p,np.mean(1/(1+np.exp(-np.array([-1.,0.,2.])))),atol=1e-7)
