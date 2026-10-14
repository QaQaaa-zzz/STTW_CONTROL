import copy
import numpy as np
import pytest
from test_path_safety_reward import safety_config,sample,trace
from sttw_control.path_command_reporting import reward_series,reconstruct_costs
from sttw_control.path_command_reward import failure_reward


def v2():
    c=safety_config();c['path_reward'].update(reward_version='path_safety_preference_v2',primary_speed_multiplier=2.)
    return c


@pytest.mark.parametrize('alpha',[0.,.25,.5,1.])
def test_only_primary_speed_branch_is_multiplied_before_same_cap(alpha):
    old=safety_config();new=v2();args=dict(alpha=alpha,ev=.25,ey=.25,roll_angle=.29)
    before=sample(old,**args);after=sample(new,**args)
    assert set(after['raw'])==set(before['raw'])
    # H((.25-.05)/.1)=3, weighted20*chi*alpha.
    assert float(after['raw']['primary']-before['raw']['primary'])==pytest.approx(60*alpha)
    for k in before['raw']:
        if k!='primary':assert float(after['raw'][k])==float(before['raw'][k])
    assert float(after['effective']['primary'])==min(float(after['raw']['primary']),new['path_reward']['component_caps']['primary'])
    assert sum(new['path_reward']['component_caps'].values())==5080.2
    assert float(failure_reward(37,new))==float(failure_reward(37,old))


def test_new_numpy_and_jax_reward_agree_and_alpha_zero_is_exact_regression():
    data=trace();old=safety_config();new=v2()
    for alpha in [0,.25,.5,.75,1]:
        before=reconstruct_costs(data,alpha,old);after=reconstruct_costs(data,alpha,new)
        assert np.all(after['raw']['primary']>=before['raw']['primary'])
        if alpha==0:
            for group in ['raw','effective','capped']:
                for k in before[group]:np.testing.assert_array_equal(after[group][k],before[group][k])
        r,_=reward_series(data,alpha,new)
        np.testing.assert_allclose(r,-new['path_reward']['scale']*.005*sum(after['effective'].values()))


@pytest.mark.parametrize('multiplier',[None,0.,-1.,np.nan,np.inf])
def test_v2_invalid_multiplier_rejected_every_entrypoint(multiplier):
    c=v2();c['path_reward']['primary_speed_multiplier']=multiplier
    for op in [lambda:sample(c,roll_angle=.29),lambda:failure_reward(10,c),lambda:reward_series(trace(),.5,c)]:
        with pytest.raises(ValueError,match='multiplier'):op()
