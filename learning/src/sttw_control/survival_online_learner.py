"""Independent repeated on-policy learner with explicit physical-tail lifecycle.

This supplies known PPO/MC training machinery, not a new gradient theorem or
risk certificate. Collection and physical provenance remain the coordinator's
responsibility; pending/unknown episodes must never be supplied as failures.
"""
import copy
import hashlib
import json
import numpy as np
import torch
from .path_command_policy import make_torch_path_network,export_torch_actor,MODULATION_SHAPES
from .preference_best import config_hash
from .survival_learning import dual_ppo_loss


class OnlineSurvivalLearner:
    def __init__(self,config,parent_policy,*,beta,seed,epochs=4,fixed_denominator=24000,
                 actor_lr=1e-4,critic_lr=3e-4,clip=.2,entropy=.001,grad_clip=1.):
        self.config=copy.deepcopy(config)
        self.settings=dict(beta=beta,seed=seed,epochs=epochs,fixed_denominator=fixed_denominator,
            actor_lr=actor_lr,critic_lr=critic_lr,clip=clip,entropy=entropy,grad_clip=grad_clip)
        torch.manual_seed(seed)
        self.actor=make_torch_path_network(351,2,config)
        self.task_critic=make_torch_path_network(352,1,config)
        self.actor.load_state_dict({k[6:]:v for k,v in parent_policy.items() if k.startswith('actor.')},strict=True)
        self.task_critic.load_state_dict({k[7:]:v for k,v in parent_policy.items() if k.startswith('critic.')},strict=True)
        self.log_std=torch.nn.Parameter(parent_policy['log_std'].detach().clone())
        self.physical_critic=torch.nn.Sequential(torch.nn.Linear(352,1),torch.nn.Sigmoid())
        torch.nn.init.zeros_(self.physical_critic[0].weight);torch.nn.init.zeros_(self.physical_critic[0].bias)
        self.actor_parameters=list(self.actor.parameters())+[self.log_std]
        self.groups=[self.actor_parameters,list(self.task_critic.parameters()),list(self.physical_critic.parameters())]
        self.optimizers=[torch.optim.Adam(group,lr=lr) for group,lr in zip(self.groups,[actor_lr,critic_lr,critic_lr])]
        self.next_collection=0;self.actor_steps=0;self.critic_steps=0;self.optimizer_steps=[0,0,0];self._pending=None

    def fingerprint(self,*,sampling_only=False):
        h=hashlib.sha256(config_hash(self.config).encode())
        modules=[('actor',self.actor)]
        if not sampling_only:modules += [('task',self.task_critic),('physical',self.physical_critic)]
        for prefix,module in modules:
            for name,value in sorted(module.state_dict().items()):
                a=value.detach().cpu().numpy();h.update((prefix+name+str(a.dtype)+str(a.shape)).encode());h.update(a.tobytes())
        h.update(self.log_std.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def freeze_collection(self):
        if self._pending is not None:raise ValueError('previous collection is unresolved/unconsumed')
        state=self.fingerprint()
        identity=hashlib.sha256(json.dumps([state,self.next_collection,self.settings['beta'],self.settings['seed']]).encode()).hexdigest()
        self._pending=dict(collection_id=identity,state_sha256=state,sampling_sha256=self.fingerprint(sampling_only=True),
                           collection_index=self.next_collection,log_std=self.log_std.detach().tolist())
        return copy.deepcopy(self._pending)

    def sampling_parameters(self):
        """Dynamic JAX parameters for Actor and both collection-time value heads."""
        import jax.numpy as j
        params={}
        for i,k in enumerate((0,2,4,6)):
            layer=self.task_critic[k]
            params[f'Dense_{i}']=dict(kernel=j.asarray(layer.weight.detach().numpy().T),bias=j.asarray(layer.bias.detach().numpy()))
        for k in MODULATION_SHAPES:
            if hasattr(self.task_critic,k):params[k]=j.asarray(getattr(self.task_critic,k).detach().numpy())
        return dict(actor=export_torch_actor(self.actor,self.config),task_critic={'params':params},
            physical=dict(kernel=j.asarray(self.physical_critic[0].weight.detach().numpy().T),
                          bias=j.asarray(self.physical_critic[0].bias.detach().numpy())))

    def update(self,batch,*,warmup):
        frozen=self._pending
        if frozen is None or batch.get('collection_id')!=frozen['collection_id']:
            raise ValueError('collection identity is stale, missing or already consumed')
        if self.fingerprint()!=frozen['state_sha256']:
            raise ValueError('Actor or value parameters changed during collection')
        fields=['obs','critic','action','old_logp','old_mean','old_task','old_physical',
                'task_advantage','task_returns','physical_targets']
        data={k:batch[k].detach() for k in fields}
        if not all(torch.isfinite(x).all() for x in data.values()):raise ValueError('nonfinite selected collection data')
        n=len(data['obs'])
        for case in batch.get('case_ranges',[]):
            if not 0<=case['start']<case['end']<=n:raise ValueError('invalid case diagnostic range')
        if (data['obs'].shape!=(n,351) or data['critic'].shape!=(n,352) or data['action'].shape!=(n,2)
                or data['old_mean'].shape!=(n,2) or any(data[k].shape!=(n,) for k in fields[3:] if k!='old_mean')):
            raise ValueError('aligned legal observation/action/value vectors required')
        value_diagnostics={}
        with torch.no_grad():
            mean=self.actor(data['obs']);logp=torch.distributions.Normal(mean,self.log_std.exp()).log_prob(data['action']).sum(-1)
            if not torch.allclose(logp,data['old_logp'],rtol=0,atol=1e-3):raise ValueError('collection action density mismatch')
            if not torch.allclose(mean,data['old_mean'],rtol=1e-5,atol=1e-4):raise ValueError('collection mean mismatch')
            for name,actual,old in [('task',self.task_critic(data['critic']).squeeze(-1),data['old_task']),
                                    ('physical',self.physical_critic(data['critic']).squeeze(-1),data['old_physical'])]:
                if not torch.isfinite(actual).all():raise ValueError('nonfinite collection value recheck')
                difference=(actual-old).abs()
                value_diagnostics[name]=dict(max_abs_error=float(difference.max()),rms_error=float(difference.square().mean().sqrt()))
            # The exact frozen parameter fingerprint is the provenance guard.
            # GPU-recorded old values remain the frozen GAE/MC baselines. Small
            # CPU/GPU output roundoff is a diagnostic, not a batch-drop gate.
        center=float(data['task_advantage'].mean());scale=float(data['task_advantage'].std(unbiased=False).clamp_min(1e-8))
        advantage=(data['task_advantage']-center)/scale
        history=[]
        for epoch in range(self.settings['epochs']):
            for optimizer in self.optimizers:optimizer.zero_grad(set_to_none=True)
            distribution=torch.distributions.Normal(self.actor(data['obs']),self.log_std.exp())
            losses=dual_ppo_loss(new_log_prob=distribution.log_prob(data['action']).sum(-1),
                old_log_prob=data['old_logp'],task_advantage=advantage,
                task_values=self.task_critic(data['critic']).squeeze(-1),task_returns=data['task_returns'],
                physical_values=self.physical_critic(data['critic']).squeeze(-1),physical_targets=data['physical_targets'],
                old_physical_values=data['old_physical'],beta=self.settings['beta'],
                fixed_denominator=self.settings['fixed_denominator'],clip=self.settings['clip'])
            total=(losses['task_value_loss']+losses['physical_value_loss']) if warmup else (
                losses['total_loss']-self.settings['entropy']*distribution.entropy().sum(-1).mean())
            if not torch.isfinite(total):raise ValueError('nonfinite learner loss')
            total.backward();norms=[]
            for index,(group,optimizer) in enumerate(zip(self.groups,self.optimizers)):
                if warmup and index==0:
                    if any(p.grad is not None for p in group):raise ValueError('warmup unexpectedly reached Actor')
                    norms.append(0.);continue
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in group):raise ValueError('nonfinite/missing gradient')
                norms.append(float(torch.nn.utils.clip_grad_norm_(group,self.settings['grad_clip'],error_if_nonfinite=True)));optimizer.step()
                self.optimizer_steps[index]+=1
                self.actor_steps=self.optimizer_steps[0];self.critic_steps=self.optimizer_steps[1]
            if not warmup:
                with torch.no_grad():self.log_std.clamp_(np.log(.05),np.log(.5))
            if not all(torch.isfinite(p).all() for group in self.groups for p in group):raise ValueError('nonfinite updated parameters')
            history.append(dict(epoch=epoch+1,**{k:float(v.detach()) for k,v in losses.items()},
                actor_gradient_norm=norms[0],task_value_gradient_norm=norms[1],physical_value_gradient_norm=norms[2]))
        with torch.no_grad():
            source=torch.distributions.Normal(data['old_mean'],torch.tensor(frozen['log_std']).exp())
            target=torch.distributions.Normal(self.actor(data['obs']),self.log_std.exp())
            kl=torch.distributions.kl_divergence(source,target).sum(-1)
            ratio=(target.log_prob(data['action']).sum(-1)-data['old_logp']).exp()
        self.next_collection+=1;self._pending=None
        case_shift=[]
        for case in batch.get('case_ranges',[]):
            start,end=case['start'],case['end']
            r=ratio[start:end]
            case_shift.append(dict(case_index=case['case_index'],exact_gaussian_KL_mean=float(kl[start:end].mean()),
                ratio_clip_fraction=float(((r<1-self.settings['clip'])|(r>1+self.settings['clip'])).float().mean())))
        return dict(collection_id=frozen['collection_id'],collection_index=frozen['collection_index'],warmup=bool(warmup),
            actor_steps=self.actor_steps,critic_steps=self.critic_steps,samples=n,
            task_advantage_center=center,task_advantage_scale=scale,
            exact_gaussian_KL_mean=float(kl.mean()),likelihood_ratio_clip_fraction=float(((ratio<1-self.settings['clip'])|(ratio>1+self.settings['clip'])).float().mean()),
            history=history,case_shift=case_shift,value_diagnostics=value_diagnostics)

    def checkpoint(self):
        if self._pending is not None:raise ValueError('exact checkpoint requires a completed collection boundary')
        return copy.deepcopy(dict(schema='sttw_dual_survival_boundary_v1',config_sha256=config_hash(self.config),
            settings=self.settings,actor=self.actor.state_dict(),task_critic=self.task_critic.state_dict(),
            physical_critic=self.physical_critic.state_dict(),log_std=self.log_std.detach(),
            optimizers=[o.state_dict() for o in self.optimizers],next_collection=self.next_collection,
            actor_steps=self.actor_steps,critic_steps=self.critic_steps,exact_boundary_resume=True,
            optimizer_steps=self.optimizer_steps,
            torch_rng_state=torch.get_rng_state()))

    @classmethod
    def restore(cls,config,state):
        if state['schema']!='sttw_dual_survival_boundary_v1' or config_hash(config)!=state['config_sha256']:
            raise ValueError('checkpoint schema/config mismatch')
        policy={'actor.'+k:v for k,v in state['actor'].items()}
        policy.update({'critic.'+k:v for k,v in state['task_critic'].items()});policy['log_std']=state['log_std']
        obj=cls(config,policy,**state['settings']);obj.physical_critic.load_state_dict(state['physical_critic'],strict=True)
        for optimizer,saved in zip(obj.optimizers,state['optimizers']):optimizer.load_state_dict(copy.deepcopy(saved))
        obj.next_collection=state['next_collection'];obj.actor_steps=state['actor_steps'];obj.critic_steps=state['critic_steps']
        obj.optimizer_steps=list(state['optimizer_steps'])
        torch.set_rng_state(state['torch_rng_state'])
        return obj
