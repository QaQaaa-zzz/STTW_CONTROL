"""Progress-gated legal reference candidates; no future disturbance features."""
import copy
import jax
import jax.numpy as j
from flax import struct
from .path_command_policy import map_latent
from .reference_prefix import inverse_offset_target


def progress_envelope(progress,prefix_end,turn_end):
    """Smooth rise over the last1.5m approach, release1.5m after the bend."""
    smooth=lambda u:(u:=j.clip(u,0.,1.))**2*(3.-2.*u)
    return smooth((progress-(prefix_end-1.5))/1.5)*smooth((turn_end+1.5-progress)/1.5)


def candidate_latent(original,nominal,progress,turn_end,profile,config):
    """Profiles0/base,1/speed,2/steer,3/joint; other profiles unchanged.

    Return finite latents using the original asymmetric correction interface.
    Untouched channels remain exactly the original latent, including at gate0.
    """
    gate=progress_envelope(progress,config['path']['straight_prefix_m'],turn_end)
    base=map_latent(original,config)
    desired=j.stack([j.float32(.24),-.15*nominal[1]])
    mask=j.stack([(profile==1)|(profile==3),(profile==2)|(profile==3)])
    target=j.where(mask,base+gate*(desired-base),base)
    latent,realized,_=inverse_offset_target(target,config)
    selected=j.where(mask&(gate>0),latent,original)
    return selected,gate,j.where(mask&(gate>0),realized,base)


def floor_config(config,lower_bound):
    """Per-lane reference lower bound; retain original filter/clip semantics."""
    interface=dict(config['residual_interface'])
    interface['speed_reference_bounds_m_s']=[lower_bound,interface['speed_reference_bounds_m_s'][1]]
    return dict(config,residual_interface=interface)


@struct.dataclass
class ReferenceEvaluationState:
    state: object
    gate: object


def make_reference_step(env,actor,*,horizon_s,allow_reentry=False):
    """One common four-lower-tick plant kernel for every candidate lane.

    At each50Hz boundary select an Actor/candidate or domain-exit zero target,
    hold it for the four real5ms ticks, and refresh actual history. The fixed
    original path never changes. Pulse is used only by the physical integrator.
    """
    from .live_path_continuation import make_live_reference_step
    from .causal_reference_baseline import causal_reference_action
    from .path_command_reward import configured_cost_keys
    cost_keys=tuple(configured_cost_keys(env.config))

    def step(carry,params,path,speed,pulse,profile,lower_bound,zero_mode):
        local=copy.copy(env);local.config=floor_config(env.config,lower_bound)
        s=carry.state
        stopped=s.physical.failed|s.fault|(s.tick>=round(horizon_s/local.cc.dt))
        disabled=s.domain_exit|zero_mode|stopped
        if allow_reentry:
            from .reference_reentry import actor_permission
            disabled=~actor_permission(s.domain_exit,s.follower.invalid,carry.clean_blocks,zero_mode,stopped)
        obs,_,obs_fault=local.observation(s,path,speed,horizon_s)
        # SIMD may evaluate a dummy network row; no domain-exited observation
        # is fed to it and no Actor action is published outside its domain.
        safe_obs=j.where(disabled,j.zeros_like(obs),obs)
        original=actor.apply(params,safe_obs)
        original=j.where(disabled,j.zeros(2),original)
        proposed,envelope,target=candidate_latent(original,s.physical.raw,
            s.projection.progress,path.turn_end,profile,local.config)
        causal_action,gate_candidate,receipt=causal_reference_action(original,safe_obs,
            s.offsets,s.tick,carry.gate,local.config,all_preferences=(profile==5))
        use_eso=((profile==4)|(profile==5))&~disabled
        action=j.where(use_eso,causal_action,proposed)
        action=j.where(disabled,j.zeros(2),action)
        gate=jax.tree.map(lambda a,b:j.where(use_eso,a,b),gate_candidate,carry.gate)
        fault=(~disabled&obs_fault)|(use_eso&receipt['engineering_fault'])
        advance=make_live_reference_step(local,horizon_s=horizon_s)
        end,logs=advance(s.replace(fault=s.fault|fault),action,path,speed,pulse)
        active=logs['active_tick'];ticks=logs['physical_tick']
        names=['active_tick','actual_xy','actual_forward_speed','wheel_speed_proxy',
            'phi','phi_dot','actual_delta','peak_roll','contact_failure','physical_failure',
            'nonfinite','lower_fault','policy_fault','time','path_cross_track','path_heading_error',
            'path_progress','goal_section_signed_distance','applied_force_impulse',
            'tick_reward','scored_tick_reward','physical_tick','chi']
        data={k:logs[k] for k in names}
        data.update(actual_yaw=logs['yaw_unwrapped'],reference=logs['governed'],
            nominal_reference=logs['nominal'],filtered_offset=logs['filtered_offset'],
            correction_offset=logs['offsets'],target_offset=logs['target_offset'],
            domain_invalid=logs['domain_exit'],
            ever_domain_exit=active&(s.domain_exit|(j.cumsum(logs['domain_exit'])>0)),
            zero_fallback=active&s.domain_exit,actor_action_valid=active&~disabled,
            task_reward_valid=active,requested_speed=j.repeat(speed,4),
            issued_latent=j.where(active[:,None],j.tile(action,(4,1)),j.nan),
            original_actor_latent=j.where((active&~disabled)[:,None],j.tile(original,(4,1)),j.nan),
            progress_envelope=j.where(active,envelope,0.),
            gate_detected=active&use_eso&receipt['detected'],
            gate_override=active&use_eso&receipt['override_active'],
            gate_signal=j.where(active&use_eso,receipt['signal'],j.nan),
            raw_costs=j.stack([logs['raw_components'][k] for k in cost_keys],axis=-1),
            effective_costs=j.stack([logs['effective_components'][k] for k in cost_keys],axis=-1),
            horizon_reached=active&(ticks==round(horizon_s/local.cc.dt)-1))
        if allow_reentry:
            from .reference_reentry import next_clean_blocks
            failures=logs['nonfinite']|logs['lower_fault']|logs['policy_fault']|logs['contact_failure']
            clean=next_clean_blocks(carry.clean_blocks,active,logs['domain_exit'],failures)
            data.update(zero_fallback=active&s.domain_exit&disabled,
                reentry_actor_active=active&s.domain_exit&~disabled,
                reentry_clean_blocks=j.repeat(carry.clean_blocks,4))
            return carry.replace(state=end,gate=gate,clean_blocks=clean),data
        return carry.replace(state=end,gate=gate),data
    return step
