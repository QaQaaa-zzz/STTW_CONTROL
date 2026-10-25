import copy
from collections import Counter
import numpy as np
import pytest
from sttw_control.live_ppo_schedule import make_schedule, make_evaluation_schedule, fresh_learners


TEMPLATES=[dict(context=r,speed=v,world_direction=[0.,1.,0.],route=r)
           for r,v in [('left',1.8),('right',1.5)]]


def test_paired_schedule_balances_every_update_and_hides_event_in_case_metadata():
    a=make_schedule(TEMPLATES,seed=421,updates=3)
    assert a==make_schedule(TEMPLATES,seed=421,updates=3)
    assert a!=make_schedule(TEMPLATES,seed=422,updates=3)
    assert len(a)==3 and all(len(x)==120 for x in a)
    for cases in a:
        counts=Counter((c['arm'],c['context'],c['alpha'],c['force_N']) for c in cases)
        assert len(counts)==60 and set(counts.values())=={2}
        for i in range(0,120,2):
            x,y=cases[i:i+2]
            assert x['arm']=='plain' and y['arm']=='support'
            assert x['profile']==0 and y['profile']==7
            assert {k:v for k,v in x.items() if k not in ['profile','arm']}=={
                k:v for k,v in y.items() if k not in ['profile','arm']}
            assert 320<=x['pulse_onset_task_tick']<=680
            assert x['rng_fold_in']<60
    assert a[0]!=a[1]
    assert TEMPLATES[0]==dict(context='left',speed=1.8,world_direction=[0.,1.,0.],route='left')


def test_ambiguous_templates_and_invalid_counts_are_rejected():
    with pytest.raises(ValueError):make_schedule(TEMPLATES[:1],seed=1,updates=1)
    with pytest.raises(ValueError):make_schedule(TEMPLATES,seed=1,updates=0)


def test_fixed_evaluation_has_four_paired_roles_without_repeated_zero_force():
    cases=make_evaluation_schedule(TEMPLATES,seed=81)
    assert len(cases)==200
    for role in ['parent_plain','parent_support','trained_plain','trained_support']:
        rows=[c for c in cases if c['actor_role']==role]
        assert len(rows)==50 and sum(c['force_N']==0 for c in rows)==10
    groups={}
    for c in cases:
        groups.setdefault((c['context'],c['alpha'],c['force_N'],c['replicate']),[]).append(c)
    assert len(groups)==50
    for rows in groups.values():
        assert len(rows)==4 and len({c['pulse_onset_task_tick'] for c in rows})==1


def test_weights_only_initialization_drops_old_adam_but_copies_all_parameters():
    import torch
    from sttw_control.path_command_policy import make_torch_path_network
    from sttw_control.survival_online_learner import OnlineSurvivalLearner
    cfg={'network':{'architecture':'alpha_modulated_v1'}}
    torch.manual_seed(15)
    actor=make_torch_path_network(351,2,cfg);critic=make_torch_path_network(352,1,cfg)
    policy={'actor.'+k:v for k,v in actor.state_dict().items()}
    policy.update({'critic.'+k:v for k,v in critic.state_dict().items()})
    policy['log_std']=torch.tensor([-1.,-2.])
    old=OnlineSurvivalLearner(cfg,policy,beta=0.,seed=16)
    with torch.no_grad():old.physical_critic[0].bias.add_(.7)
    source=copy.deepcopy(old.checkpoint())
    source['next_collection']=64;source['actor_steps']=248
    fresh=fresh_learners(cfg,source,seed=17)
    assert set(fresh)=={'plain','support'}
    for obj in fresh.values():
        assert obj.settings['beta']==0 and obj.settings['fixed_denominator']==30000
        assert obj.next_collection==0 and obj.optimizer_steps==[0,0,0]
        assert all(not opt.state for opt in obj.optimizers)
        for module in ['actor','task_critic','physical_critic']:
            assert all(torch.equal(v,source[module][k]) for k,v in getattr(obj,module).state_dict().items())
        assert torch.equal(obj.log_std,source['log_std'])
    assert fresh['plain'].fingerprint()==fresh['support'].fingerprint()
    assert source['next_collection']==64
