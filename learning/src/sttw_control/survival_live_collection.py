"""Optional stochastic wrapper around the verified live reference kernel.

Policy-inactive intervals still produce actual task rewards. This neither
changes historical held-tail collection nor authorizes querying invalid data.
"""
import math
import jax
import jax.numpy as j
from flax import struct
from .reference_reentry import actor_permission
from .survival_collection import evaluate_collection_networks,sample_gaussian_action


@struct.dataclass
class LiveSurvivalCollectionState:
    state: object
    key: object
    gate: object
    clean_blocks: object


class _SampledAction:
    def apply(self,latent,_protected_observation):
        return latent


def make_live_survival_collection_step(env,actor,*,horizon_s,task_critic):
    """Raw Gaussian samples are projected by the existing profile0/7 kernel.

    The recorded density is BEFORE deterministic tanh/support/filter mapping.
    Networks receive zero dummy views outside eligibility, their values are
    discarded there, and no such row becomes a PPO action. The RNG advances
    on real legal decisions only; all history and physical state carry through
    fallback and the existing16-clean-block reentry gate.
    """
    from .anticipatory_reference import make_reference_step
    dt=float(env.cc.dt);limit=round(horizon_s/dt)
    if (task_critic is None or not math.isfinite(horizon_s) or horizon_s<=0
            or limit%4 or not math.isclose(limit*dt,horizon_s,abs_tol=1e-8)):
        raise ValueError('task critic and positive four-lower-tick-aligned horizon required')
    advance=make_reference_step(env,_SampledAction(),horizon_s=horizon_s,allow_reentry=True)
    lower_bound=j.float32(env.config['residual_interface']['speed_reference_bounds_m_s'][0])

    def step(carry,params,log_std,path,speed,pulse,profile):
        s=carry.state;stopped=s.physical.failed|s.fault|(s.tick>=limit)
        eligible=actor_permission(s.domain_exit,s.follower.invalid,carry.clean_blocks,j.bool_(False),stopped)
        obs,critic_obs,obs_fault=env.observation(s,path,speed,horizon_s)
        protected_obs=j.where(eligible,obs,j.zeros_like(obs))
        protected_critic=j.where(eligible,critic_obs,j.zeros_like(critic_obs))
        mean,task_value,physical_value=evaluate_collection_networks(actor,task_critic,params,protected_obs,protected_critic)
        raw,logp,next_key=sample_gaussian_action(mean,log_std,carry.key)
        finite=j.all(j.isfinite(mean))&j.all(j.isfinite(raw))&j.isfinite(logp)&j.isfinite(task_value)&j.isfinite(physical_value)
        proposal_fault=eligible&(obs_fault|~finite|~((profile==0)|(profile==7)))
        prepared=carry.replace(state=s.replace(fault=s.fault|proposal_fault))
        end,logs=advance(prepared,raw,path,speed,pulse,profile,lower_bound,j.bool_(False))
        valid=j.any(logs['actor_action_valid'])&~proposal_fault
        key=jax.tree.map(lambda new,old:j.where(valid,new,old),next_key,carry.key)
        end=end.replace(key=key)
        decision=dict(valid=valid,observation=j.where(valid,obs,j.full_like(obs,j.nan)),
            critic_observation=j.where(valid,critic_obs,j.full_like(critic_obs,j.nan)),
            old_task_value=j.where(valid,task_value,j.nan),physical_value=j.where(valid,physical_value,j.nan),
            raw_action=j.where(valid,raw,j.full_like(raw,j.nan)),mean=j.where(valid,mean,j.full_like(mean,j.nan)),
            log_prob=j.where(valid,logp,j.nan),task_tick=j.where(valid,s.tick,j.int32(-1)),
            rng_key=carry.key,proposal_fault=proposal_fault)
        logs=dict(logs,upper_active=logs['active_tick']&logs['actor_action_valid'],
            fallback_active=logs['active_tick']&~logs['actor_action_valid'])
        return end,(decision,logs)
    return step
