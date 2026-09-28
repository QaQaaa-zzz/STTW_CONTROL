"""Finite candidate, slack lexicographic preference and shared heading recovery."""
import time
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jp
from .governor_state import Candidate
from .teleop_commands import validate_alpha

TRACK,CONFLICT,RECOVER,EMERGENCY=range(4)

def domain(raw,mode,gain):
    low=max(1.5,raw[0]-(.10 if mode==RECOVER else 1.))
    ds=(max(-.35,raw[1]-.10*gain),min(.35,raw[1]+.10*gain)) if mode==RECOVER else (-.35,.35)
    return np.array([low,ds[0]]),np.array([raw[0],ds[1]])

def coarse_grid(raw,last,current,delta_rec,mode,gain):
    raw=np.asarray(raw);lo,hi=domain(raw,mode,gain)
    vv,dd=np.meshgrid(np.linspace(lo[0],hi[0],9),np.linspace(lo[1],hi[1],15),indexing='ij')
    special=np.array([raw,last,current,raw,[lo[0],raw[1]],[raw[0],delta_rec],[lo[0],delta_rec],[raw[0],0],[lo[0],0]])
    # Fixed index 0 is bypass, index 1..8 specials, then grid. Stable across alpha.
    goals=np.concatenate([special,np.stack([vv.ravel(),dd.ravel()],axis=1)])
    goals=np.clip(goals,lo,hi);goals[0]=raw
    bypass=np.zeros(144,bool);bypass[0]=True
    return Candidate(goals,bypass,np.ones(144,bool))

def finite_mask(r):
    mask=np.asarray(r['finite']).copy()
    for key in ('steer_rmse','speed_rmse','heading_cost','smoothness','violation'):
        mask &= np.isfinite(r[key])
    return mask

def select(r,alpha,mode):
    alpha=validate_alpha(alpha);mask=finite_mask(r)
    if not mask.any():return None
    if mode==EMERGENCY or not np.any(mask&r['feasible']):
        indices=np.flatnonzero(mask)
        return int(min(indices,key=lambda i:(bool(r['physical_failure'][i]),r['violation'][i],r['smoothness'][i],i)))
    mask &= r['feasible']
    order=[('speed_rmse',.02),('heading_cost',.001)] if mode==RECOVER else (
        [('steer_rmse',.005),('speed_rmse',.02)] if alpha==0 else [('speed_rmse',.02),('steer_rmse',.005)])
    for name,slack in order:
        mask &= r[name]<=np.min(r[name][mask])+slack
    i=np.flatnonzero(mask)
    return int(i[np.argmin(r['smoothness'][i])])

def refine_grid(coarse,r,raw,mode,gain):
    valid=finite_mask(r);feasible=valid&r['feasible']
    if feasible.any():
        ids=np.flatnonzero(feasible)
        keys=['steer_rmse','speed_rmse','heading_cost' if mode==RECOVER else 'smoothness']
        centers=[int(ids[np.argmin(r[k][ids])]) for k in keys]
    else:
        ids=np.flatnonzero(valid)
        centers=sorted(ids,key=lambda i:(bool(r['physical_failure'][i]),r['violation'][i],r['smoothness'][i],i))[:3]
    centers=list(dict.fromkeys(centers))
    lo,hi=domain(raw,mode,gain);spacing=(hi-lo)/np.array([8.,14.])
    offsets=np.array([(a,b) for a in [-.5,-.25,0,.25,.5] for b in [-.5,-.25,0,.25,.5]])*spacing
    goals=np.tile(raw,(80,1));valid=np.zeros(80,bool)
    for k,c in enumerate(centers):
        goals[k*25:(k+1)*25]=np.clip(coarse.goal[c]+offsets,lo,hi);valid[k*25:(k+1)*25]=True
    return Candidate(goals,np.zeros(80,bool),valid)

def mode_update(raw_feasible,raw_reserve,reserve_ticks,previous_mode,error,exit_held,disable_recovery=False):
    reserve_ticks=reserve_ticks+1 if raw_reserve else 0
    if not raw_feasible:return CONFLICT,0
    if reserve_ticks<4 or disable_recovery:return TRACK,reserve_ticks
    if not exit_held and (abs(error)>.08 or previous_mode==RECOVER):return RECOVER,reserve_ticks
    return TRACK,reserve_ticks

@dataclass(frozen=True)
class Decision:
    goal: object
    normalized_residual: object
    mode: int
    bypass: bool
    raw_feasible: bool
    raw_reserve: bool
    selected_feasible: bool
    fallback_used: bool
    recovery_blocked: bool
    cost_summary: dict
    candidate_counts: dict
    predictor_ticks: int
    latency_ms: float
    cold_compile: bool
    governor: object

class PreferenceGovernor:
    def __init__(self,predictor,disable_recovery=False):
        self.predictor=predictor;self.disable_recovery=disable_recovery
        self.last_diagnostics=None

    def decide(self,s,alpha):
        validate_alpha(alpha);start=time.perf_counter();raw=np.asarray(s.raw)
        gs=s.governor
        # raw bypass prediction explicitly includes immediate virtual ref reset.
        rc=Candidate(raw[None],np.array([True]),np.array([True]))
        rr,cold=self.predictor.predict(s,rc)
        raw_ok=bool(rr['feasible'][0]);reserve=bool(rr['reserve'][0])
        error=float(s.reference_pose[2]-s.yaw_unwrapped)
        mode,ticks=mode_update(raw_ok,reserve,int(gs.reserve_ticks),int(gs.mode),error,
            int(gs.recovery_hold_ticks)>=100,self.disable_recovery)
        gain=min(1.,float(gs.recovery_gain)+.05/.30) if mode==RECOVER else 0.
        gov=gs.replace(reserve_ticks=jp.int32(ticks),recovery_gain=jp.asarray(gain,dtype=gs.recovery_gain.dtype),mode=jp.int32(mode))
        allc=rc;r=rr;index=0;used=240;blocked=False;coarse_count=0;fine_count=0
        if mode!=TRACK:
            _,_,speed=self.predictor.env.observe(s.data)
            extra=np.clip(.8*np.clip(error,-np.pi/2,np.pi/2),-.4,.4)
            omega=raw[0]*np.cos(self.predictor.env.cc.caster)*np.tan(raw[1])/self.predictor.env.cc.wheelbase
            drec=np.arctan(self.predictor.env.cc.wheelbase*(omega+extra)/(np.cos(self.predictor.env.cc.caster)*max(float(speed),1.5)))
            drec=np.clip(raw[1]+np.clip(drec-raw[1],-.10*gain,.10*gain),-.35,.35)
            coarse=coarse_grid(raw,np.asarray(gs.last_goal),np.asarray(gs.current_reference),drec,mode,gain)
            cr,ccold=self.predictor.predict(s,coarse)
            fine=refine_grid(coarse,cr,raw,mode,gain);fr,fcold=self.predictor.predict(s,fine)
            cold=cold or ccold or fcold;used+=224*240;coarse_count=144;fine_count=80
            r={k:np.concatenate([cr[k],fr[k]],axis=0) for k in cr}
            allc=Candidate(np.concatenate([coarse.goal,fine.goal]),np.concatenate([coarse.bypass,fine.bypass]),np.concatenate([coarse.valid,fine.valid]))
            index=select(r,alpha,mode)
            if index is None:raise FloatingPointError('numerical_failure: every candidate nonfinite')
            if not np.any(r['feasible']):mode=EMERGENCY;gain=0.;gov=gov.replace(mode=jp.int32(mode),recovery_gain=jp.asarray(0.,dtype=gs.recovery_gain.dtype))
            if mode==RECOVER:
                improves=r['feasible']&(r['heading_cost']<=rr['heading_cost'][0]-1e-4)
                if not np.any(improves):index=0;blocked=True
        goal=allc.goal[index];bypass=bool(allc.bypass[index])
        current=raw if bypass else np.asarray(gs.current_reference)
        gov=gov.replace(current_reference=jp.asarray(current),last_goal=jp.asarray(goal))
        # Decision describes the exact next 5ms normalized residual, not a target proxy.
        preview=s.replace(governor=gov)
        from .closed_loop_kernel import controls
        m,_,_=self.predictor.env.observe(s.data)
        governed=raw if bypass else current+np.clip(goal-current,-np.array([1.,.6])*.005,np.array([1.,.6])*.005)
        _,_,log=controls(s.controller,s.actuator,m,s.raw,jp.asarray(governed),s.physical_tick*.005>3.,bypass,self.predictor.env.cc,self.predictor.env.ac)
        jax.block_until_ready(log)
        self.last_diagnostics=dict(candidates=allc,results=r,index=index,raw=rr)
        summary={k:np.asarray(v[index]).tolist() for k,v in r.items()}
        ids=np.flatnonzero(r['feasible'])
        for label,key in [('best_steer','steer_rmse'),('best_speed','speed_rmse')]:
            i=int(ids[np.argmin(r[key][ids])]) if len(ids) else index
            summary[label]={k:np.asarray(r[k][i]).tolist() for k in ['steer_rmse','speed_rmse','heading_cost','smoothness']}
        counts=dict(coarse=coarse_count,refinement=fine_count,valid=int(np.sum(allc.valid)),feasible=int(np.sum(r['feasible'])),
            nonfinite=int(np.sum(allc.valid&~r['finite'])),physical_failure=int(np.sum(allc.valid&r['physical_failure'])),
            working_rejection=int(np.sum(allc.valid&~r['working'])),terminal_rejection=int(np.sum(allc.valid&~r['terminal'])))
        return Decision(goal,np.asarray(log['normalized_residual']),mode,bypass,raw_ok,reserve,
            bool(r['feasible'][index]),mode==EMERGENCY,blocked,summary,counts,used,
            (time.perf_counter()-start)*1000,cold,gov)
