"""V3 objective rescoring of a fixed live physical trace, not a new rollout."""
import math
import numpy as np
from .path_command_reward import configured_cost_keys


def rescore_trace_alpha(data,*,alpha,config):
    """Keep executed geometry/inputs and all alpha-independent costs fixed.

    Cap V3 endpoint channels before mixing. Only actual contact replaces the
    final upper block; cumulative geometric exit is telemetry in this contract.
    Reweighting an Actor trace does not predict what a different Actor would do.
    """
    if not math.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('finite alpha within[0,1] required')
    c=config['path_reward']
    if c.get('reward_version')!='path_persistent_preference_v3':raise ValueError('explicit V3 reward required')
    keys=configured_cost_keys(config);a=np.asarray(data['active_tick'],bool);n=len(a)
    eff=np.asarray(data['effective_costs'],float).copy()
    if eff.shape!=(n,len(keys)):raise ValueError('cost columns must match configured contract')
    chi=np.asarray(data['chi'],float);ev=np.asarray(data['actual_forward_speed'],float)-np.asarray(data['requested_speed'],float)
    ey=np.asarray(data['path_cross_track'],float);eh=np.asarray(data['path_heading_error'],float)
    if not all(np.isfinite(x[a]).all() for x in [eff,chi,ev,ey,eh]):raise ValueError('nonfinite active data cannot be rescored')
    huber=lambda x:np.where(np.abs(x)<=1,x*x,2*np.abs(x)-1)
    speed0=(10-9*chi)*huber(ev/c['speed_scale_m_s']);speed1=10*huber(ev/c['speed_scale_m_s'])
    path=10*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad']))
    primary0=c['primary_weight']*chi*huber(np.maximum(np.abs(ey)-c['primary_path_band_m'],0)/c['primary_path_scale_m'])
    primary1=c['primary_weight']*chi*huber(np.maximum(np.abs(ev)-c['primary_speed_band_m_s'],0)/c['primary_speed_scale_m_s'])
    cap=c['component_caps'];eff[:,keys.index('speed')]=(1-alpha)*np.minimum(speed0,cap['speed'])+alpha*np.minimum(speed1,cap['speed'])
    eff[:,keys.index('path')]=(1-(1-c['path_secondary_ratio'])*alpha)*np.minimum(path,cap['path'])
    eff[:,keys.index('primary')]=(1-alpha)*np.minimum(primary0,cap['primary'])+alpha*np.minimum(primary1,cap['primary'])
    eff[~a]=0.;running=-c['scale']*config['timing']['lower_dt_s']*eff.sum(axis=-1);scored=running.copy()
    hits=np.flatnonzero(a&np.asarray(data['contact_failure'],bool))
    if len(hits):
        last=int(np.flatnonzero(a)[-1])
        if hits[0]!=last:raise ValueError('contact must stop the physical episode')
        block=(last//4)*4;upper=config['timing']['upper_dt_s'];remaining=round(config['training_future_phase_C']['episode_s']/upper)-last//4
        gamma=config['training_future_phase_C']['gamma']
        terminal=-c['failure_extra']-c['scale']*upper*c['failure_cost_rate_bound']*(1-gamma**remaining)/(1-gamma)
        scored[block:last+1]=0.;scored[last]=terminal
    return dict(running_reward=running,scored_reward=scored,effective_costs=eff,
                running_cost=float(-running.sum()),scored_cost=float(-scored.sum()),
                semantics='Fixed executed trajectory under a different user objective; not new dynamics or an independent observation.')
