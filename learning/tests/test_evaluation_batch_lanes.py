import numpy as np
from sttw_control.isaac_evaluation_batch import batch_settings,split_case


def test_mixed_width_batch_preserves_alpha_zero_force_and_lane_identity():
    cases=[]
    for i,n in enumerate([8,11]):
        cases.append(dict(alphas=[.25]*n,zero_middle=[False]*(n-1)+[True],user_speed_m_s=2.3+i*.3,terrain_path='same',output=str(i),evaluation_pulse=dict(start_s=1.5,duration_s=.1,world_force_N=[0,8*(-1)**i,0])))
    settings=batch_settings({},cases)
    assert settings['num_envs']==19
    assert cases[1]['lane_start']==8
    assert settings['zero_middle']==cases[0]['zero_middle']+cases[1]['zero_middle']
    assert settings['evaluation_pulse']['world_force_N'][7][1]==8
    assert settings['evaluation_pulse']['world_force_N'][8][1]==-8
    arrays=dict(reward=np.arange(38).reshape(2,19),lower_trace=np.arange(2*4*19*11).reshape(2,4,19,11))
    contract=dict(settings=settings,origin=list(range(19)),terrain=list(range(19)))
    data,result=split_case(arrays,contract,dict(cases[1],batch_output='batch'),1)
    np.testing.assert_array_equal(data['reward'],arrays['reward'][:,8:])
    np.testing.assert_array_equal(data['lower_trace'],arrays['lower_trace'][:,:,8:])
    assert result['settings']['zero_middle']==cases[1]['zero_middle']
    assert result['settings']['evaluation_pulse']==cases[1]['evaluation_pulse']
    assert result['origin']==list(range(8,19))


def test_legacy_six_lane_batch_is_unchanged():
    cases=[dict(user_speed_m_s=2.3,terrain_path='a'),dict(user_speed_m_s=2.6,terrain_path='b')]
    settings=batch_settings({},cases)
    assert settings['num_envs']==12
    assert settings['lane_alphas']==[0,.25,.5,.75,1,0]*2
    assert settings['lane_terrain_ids']==[0]*6+[1]*6
