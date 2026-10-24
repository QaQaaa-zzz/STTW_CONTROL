"""Fixed200Hz nominal-reference prefilter through the original bounded interface.

This is a conventional smoothing comparator, not a learned controller or safety
filter. It never observes a future force or bypasses correction_tick().
"""
import jax.numpy as j


def sampled_latent(nominal_steer_rate,config):
    c=config['residual_interface'];dt=config['timing']['lower_dt_s']
    decay=j.exp(-dt/c['filter_tau_s'][1])
    requested=-decay/(1-decay)*dt*nominal_steer_rate
    ratio=requested/c['steer_scale_rad']
    bounded=j.clip(ratio,-1+1e-6,1-1e-6)
    return j.stack([j.asarray(0.),j.arctanh(bounded)]),requested,ratio!=bounded


def install(env):
    original=env.tick
    def tick(state,zero_action,path,v_user,force_pulse=None,horizon=None):
        # Pure reference-stream preprocessing at its existing200Hz rate.
        # This is not a learned50Hz policy or a new physical-state feedback.
        latent,requested,clipped=sampled_latent(state.nominal_rates[1],env.config)
        state=state.replace(fault=state.fault|j.any(zero_action!=0))
        end,logs=original(state,latent,path,v_user,force_pulse,horizon)
        logs=dict(logs)
        logs['baseline_sampled_nominal_steer_rate']=state.nominal_rates[1]
        logs['baseline_requested_steer_target']=requested
        logs['baseline_inverse_map_clipped']=clipped
        return end,logs
    env.tick=tick
    return env


def load_environment(config,source,terminal_contract):
    from sttw_control.contact_only_environment import load_environment_contact_only
    env,initial=load_environment_contact_only(config,source,terminal_contract)
    return install(env),initial
