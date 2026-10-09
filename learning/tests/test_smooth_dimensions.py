

def test_resolved_dimensions_and_budget(tmp_path):
    from sttw_control.smooth_command_training import SmoothCampaign
    c=SmoothCampaign(tmp_path/'run',fresh=True,preference_v51=True,num_envs=256,rollout_steps=128)
    for alpha in (0,1):
        spec=c.resolve(alpha);p=spec['ppo']
        assert (c.n,c.steps)==(p['num_envs'],p['rollout_policy_steps'])==(256,128)
        assert p['minibatches']*p['minibatch_size']==256*128
        assert spec['budget']['default_policy_transitions']==2*100*256*128
        assert spec['budget']['default_control_transitions_upper']==2*100*256*128*4
        assert p['validation_updates']==[25,100]


def test_smoke_dimensions_are_resolved(tmp_path):
    from sttw_control.smooth_command_training import SmoothCampaign
    c=SmoothCampaign(tmp_path/'run',fresh=True,preference_v51=True,smoke=True)
    assert (c.n,c.steps)==(8,16)
    assert c.spec['budget']['default_policy_transitions']==2*100*8*16
