"""Short feedback-reference borrowing under the unchanged causal ESO gate.

Teacher alpha changes its Actor query only; the user's task state and scoring
alpha never change. No force, pulse onset or future outcome enters the gate.
"""
import jax
import jax.numpy as j
from .causal_reference_baseline import ALPHA,causal_reference_action


def relay_action(base_actor,teacher_actor,params,observation,offset,tick,gate,config,disabled):
    safe=j.where(disabled,j.zeros_like(observation),observation)
    base=base_actor.apply(params['base'],safe)
    _,candidate,receipt=causal_reference_action(base,safe,offset,tick,gate,config)
    teacher=teacher_actor.apply(params['teacher'],safe.at[ALPHA].set(params['teacher_alpha']))
    active=receipt['override_active']&~disabled
    selected=j.where(active,teacher,base)
    selected=j.where(disabled,j.zeros(2),selected)
    next_gate=jax.tree.map(lambda a,b:j.where(disabled,b,a),candidate,gate)
    return selected,next_gate,dict(receipt,
        detected=receipt['detected']&~disabled,override_active=active,
        engineering_fault=~disabled&(receipt['engineering_fault']|~j.all(j.isfinite(selected))),
        base_latent=base,teacher_latent=teacher)


class IssuedLatent:
    """Adapt an already selected legal latent to the shared single plant step."""
    def apply(self,params,_observation):
        return params['latent']


def make_expert_relay_step(env,base_actor,teacher_actor,*,horizon_s):
    from .anticipatory_reference import make_reference_step
    advance=make_reference_step(env,IssuedLatent(),horizon_s=horizon_s)

    def step(carry,params,path,speed,pulse,_profile,lower_bound,zero_mode):
        s=carry.state
        disabled=s.domain_exit|zero_mode|s.physical.failed|s.fault|(s.tick>=round(horizon_s/env.cc.dt))
        observation,_,obs_fault=env.observation(s,path,speed,horizon_s)
        action,gate,receipt=relay_action(base_actor,teacher_actor,params,observation,
            s.offsets,s.tick,carry.gate,env.config,disabled)
        fault=~disabled&(obs_fault|receipt['engineering_fault'])
        start=carry.replace(state=s.replace(fault=s.fault|fault),gate=gate)
        # Profile0 passes the preselected latent without any additional gate;
        # original filtering/slew/limits and all physical stepping stay shared.
        end,data=advance(start,dict(latent=action),path,speed,pulse,j.int32(0),lower_bound,zero_mode)
        active=data['active_tick'];used=active&receipt['override_active']
        data.update(original_actor_latent=j.where((active&~disabled)[:,None],j.tile(receipt['base_latent'],(4,1)),j.nan),
            teacher_latent=j.where(used[:,None],j.tile(receipt['teacher_latent'],(4,1)),j.nan),
            teacher_alpha_context=j.where(used,params['teacher_alpha'],j.nan),
            expert_override=used,gate_detected=active&receipt['detected'],
            gate_override=used,gate_signal=j.where(active&~disabled,receipt['signal'],j.nan))
        return end,data
    return step
