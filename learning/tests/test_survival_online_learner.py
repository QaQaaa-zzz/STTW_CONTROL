"""Actual-network on-policy lifecycle, Critic warmup and boundary resume."""
import copy
import torch
import pytest
from sttw_control.path_command_policy import make_torch_path_network
from sttw_control.survival_online_learner import OnlineSurvivalLearner

CFG={'network':{'architecture':'alpha_modulated_v1'}}


def learner():
    torch.manual_seed(41)
    actor=make_torch_path_network(351,2,CFG);critic=make_torch_path_network(352,1,CFG)
    policy={'actor.'+k:v for k,v in actor.state_dict().items()}
    policy.update({'critic.'+k:v for k,v in critic.state_dict().items()})
    policy['log_std']=torch.tensor([-1.,-2.])
    return OnlineSurvivalLearner(CFG,policy,beta=1.,seed=42,epochs=2,fixed_denominator=4)


def batch(obj):
    frozen=obj.freeze_collection()
    obs=torch.sin(torch.arange(702,dtype=torch.float32).reshape(2,351))*.1
    view=torch.cat([obs,torch.tensor([[1.],[.9]])],dim=1)
    with torch.no_grad():
        mean=obj.actor(obs);action=mean+torch.tensor([[.03,-.02],[-.01,.02]])
        logp=torch.distributions.Normal(mean,obj.log_std.exp()).log_prob(action).sum(-1)
        value=obj.task_critic(view).squeeze(-1);physical=obj.physical_critic(view).squeeze(-1)
    return dict(collection_id=frozen['collection_id'],obs=obs,critic=view,action=action,old_logp=logp,
        old_mean=mean,old_task=value,old_physical=physical,task_advantage=torch.tensor([1.,-.5]),
        task_returns=value+torch.tensor([1.,-.5]),physical_targets=torch.tensor([1.,0.]))


def state(obj):return [p.detach().clone() for p in obj.actor.parameters()]+[obj.log_std.detach().clone()]


def test_critic_warmup_changes_both_critics_but_not_actor_or_std():
    obj=learner();before=state(obj);old_task=copy.deepcopy(obj.task_critic.state_dict());b=batch(obj)
    result=obj.update(b,warmup=True)
    assert all(torch.equal(a,b) for a,b in zip(before,state(obj)))
    assert any(not torch.equal(v,old_task[k]) for k,v in obj.task_critic.state_dict().items())
    assert result['actor_steps']==0 and result['critic_steps']==2
    assert obj.next_collection==1


def test_consumed_batch_and_mutated_collecting_parameters_are_rejected():
    obj=learner();b=batch(obj);obj.update(b,warmup=False)
    with pytest.raises(ValueError,match='collection'):obj.update(b,warmup=False)
    b=batch(obj)
    with torch.no_grad():next(obj.physical_critic.parameters()).add_(.1)
    with pytest.raises(ValueError,match='changed'):obj.update(b,warmup=False)


def test_boundary_restore_preserves_adam_for_identical_next_update():
    obj=learner();obj.update(batch(obj),warmup=False)
    saved=copy.deepcopy(obj.checkpoint());restored=OnlineSurvivalLearner.restore(CFG,saved)
    b=batch(obj);b2=batch(restored)
    assert b['collection_id']==b2['collection_id']
    obj.update(b,warmup=False);restored.update(b2,warmup=False)
    assert all(torch.equal(a,b) for a,b in zip(state(obj),state(restored)))
    for module in ['task_critic','physical_critic']:
        assert all(torch.equal(a,b) for a,b in zip(getattr(obj,module).parameters(),getattr(restored,module).parameters()))


def test_checkpoint_does_not_pretend_to_resume_an_unfinished_collection():
    obj=learner();obj.freeze_collection()
    with pytest.raises(ValueError,match='boundary'):obj.checkpoint()


def test_small_recorded_value_roundoff_is_logged_without_discarding_the_batch():
    obj=learner();b=batch(obj);b['old_task']=b['old_task']+.0002
    result=obj.update(b,warmup=True)
    assert result['value_diagnostics']['task']['max_abs_error']>.0001
    assert result['actor_steps']==0 and result['critic_steps']==2


@pytest.mark.parametrize('mode,expected', [('legal_mean',-.3),('fixed_slots',-.15)])
def test_entropy_gradient_respects_declared_sample_or_slot_normalization(mode,expected):
    source=learner().checkpoint()
    policy={'actor.'+k:v for k,v in source['actor'].items()}
    policy.update({'critic.'+k:v for k,v in source['task_critic'].items()})
    policy['log_std']=source['log_std']
    obj=OnlineSurvivalLearner(CFG,policy,beta=0.,seed=43,epochs=1,fixed_denominator=4,
        entropy=.3,grad_clip=1e6,entropy_normalization=mode)
    b=batch(obj);b['task_advantage'].zero_()
    obj.update(b,warmup=False)
    torch.testing.assert_close(obj.log_std.grad,torch.full_like(obj.log_std,expected))
    restored=OnlineSurvivalLearner.restore(CFG,obj.checkpoint())
    assert restored.settings['entropy_normalization']==mode
