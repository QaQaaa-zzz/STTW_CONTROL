"""V5.2 fixed-protocol physical selection; no training-return ranking."""
from pathlib import Path
import hashlib,json,os,shutil
import numpy as np
from .lower_tracking_audit import held,intervals,stats

COUNT_NAMES=('physical_failure_cases','working_limit_failure_cases','primary_failure_cases','joint_final_hold_failure_cases')
CASES=('straight_hold','gentle_positive','gentle_negative','fast_turn','steer_reversal','speed_changes')
DT=.005

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config_hash(config):return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic_json(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)

def selection_key(s):
    if not s.get('all_declared_traces_complete'):raise ValueError('partial validation cannot select best')
    counts=tuple(s[k] for k in COUNT_NAMES);q=float(s['physical_quality_score'])
    if any(type(x) is not int or x<0 for x in counts) or not np.isfinite(q) or q<0:raise ValueError('invalid selection metric')
    return counts+(q,)

def is_better(candidate,incumbent):
    new=selection_key(candidate)
    if incumbent is None:return True
    old=selection_key(incumbent)
    return new[:4]<old[:4] if new[:4]!=old[:4] else new[4]<old[4]-max(1e-6,.005*old[4])

def qualified(s):return selection_key(s)[:4]==(0,0,0,0)
def huber(x):return x*x if abs(x)<=1 else 2*abs(x)-1

def trace_metrics(d,alpha,case):
    """Use same-index interval command and post-interval physical measurement."""
    t=np.asarray(d['time'],float);n=len(t);raw=np.asarray(d['limited_command'],float);gov=np.asarray(d['governed'],float)
    actual=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)
    task=actual-raw;lower=actual-gov;upper=gov-raw
    np.testing.assert_allclose(task,lower+upper,rtol=0,atol=1e-10)
    peak=np.maximum(np.abs(d['phi']),d['peak_roll']);heading=np.asarray(d['e_psi_unwrapped'])
    finite=all(np.isfinite(x).all() for x in (t,raw,gov,actual,peak,heading))
    failure=any(np.any(d.get(k,False)) for k in ('physical_failure','policy_fault','lower_fault')) or not finite
    horizon=16 if case=='fast_turn' else 10
    complete=bool(n and not bool(np.asarray(d.get('partial',False))) and (failure or t[-1]+DT>=horizon-1e-4))
    # An engineering/nonfinite failure is not silently converted to a score.
    if not finite:raise ValueError(f'nonfinite fixed validation {case}/alpha{alpha}')
    rate=np.vstack([np.zeros(2),np.diff(raw,axis=0)/DT]);stable=held((abs(rate[:,0])<=.1)&(abs(rate[:,1])<=.02),DT,.3)
    window=(t>=1-1e-6)&(t<6-1e-6)&stable
    conflict=window&(np.asarray(d['chi'])>=.8);ordinary=window&~conflict
    rms=lambda values,mask:float(np.sqrt(np.mean(values[mask]**2))) if mask.any() else None
    mode={};weighted=[];primary_failed=False
    for name,mask in [('conflict',conflict),('ordinary',ordinary)]:
        count=int(mask.sum());v=rms(task[:,0],mask);s=rms(task[:,1],mask)
        if count<50:
            mode[name]=dict(count=count,status='N/A',speed_rmse=v,steer_rmse=s)
            # Absent modes are legitimate. Small present groups cannot certify a case.
            if count:primary_failed=True
            continue
        if name=='conflict':
            main,secondary,tol,secondary_tol=(s,v,.05,.60) if alpha==0 else (v,s,.08,.15)
            passed=main<=tol;q=huber(main/tol)+.2*huber(secondary/secondary_tol)
        else:passed=v<=.10 and s<=.04;q=huber(v/.10)+huber(s/.04)
        primary_failed|=not passed;weighted.append((count,q));mode[name]=dict(count=count,status='computed',speed_rmse=v,steer_rmse=s,passed=bool(passed),quality=q)
    if not weighted:primary_failed=True
    final=(t>=9.5-1e-6)&(t<10-1e-6)
    hold=bool(final.sum()>=100 and np.all((abs(task[:,0])<=.10)&(abs(task[:,1])<=.04)&(abs(heading)<=.05)&(peak<=.302) | ~final) and not failure)
    # Early physical failure ranks first; Q explicitly uses its observed tail if main tail is absent.
    tail=final if final.any() else t>=t[-1]-.5+DT-1e-6
    qmain=sum(c*q for c,q in weighted)/sum(c for c,_ in weighted) if weighted else 0.
    q=qmain+huber(rms(heading,tail)/.05)+huber(rms(task[:,0],tail)/.10)+huber(rms(task[:,1],tail)/.04)+2*float(np.mean(peak>.302))/.01
    recovery=np.maximum.accumulate(abs(raw[:,1])>.01)&(np.asarray(d['g'])>.5)
    recovered=held(recovery&(abs(heading)<=.05)&(abs(task[:,0])<=.1)&(abs(task[:,1])<=.04)&(peak<=.302),DT,.5)
    starts=np.flatnonzero(recovery);ends=np.flatnonzero(recovered)
    recovery_time=None if not len(starts) or not len(ends) else float(t[ends[0]]+DT-t[starts[0]])
    govrate=np.vstack([np.zeros(2),np.diff(gov,axis=0)/DT]);quiet=held((abs(govrate[:,0])<=.1)&(abs(govrate[:,1])<=.02),DT,.3)
    masks={'full':np.ones(n,bool),'legacy_1_6':(t>=1)&(t<6),'stable_2_4':(t>=2)&(t<4),'governed_quiet':quiet,'governed_transient':~quiet,'recovery':recovery}
    errors={name:{channel:{label:stats(values[:,j],mask,t,DT,tol) for label,values in [('upper',upper),('lower',lower),('task',task)]} for j,channel,tol in [(0,'speed',.1),(1,'steer',.04)]} for name,mask in masks.items()}
    extension=(t>=15.5-1e-6)&(t<16-1e-6)
    extension_hold=bool(extension.sum()>=100 and np.all((abs(task[:,0])<=.1)&(abs(task[:,1])<=.04)&(abs(heading)<=.05)&(peak<=.302) | ~extension)) if case=='fast_turn' else None
    return dict(complete=complete,physical_failure=bool(failure),working_limit_failure=bool(np.any(peak>.302)),primary_failure=bool(primary_failed),joint_final_hold_failure=not hold,
        primary_modes=mode,physical_quality_score=float(q),peak_roll=float(peak.max()),working_violation_fraction=float(np.mean(peak>.302)),working_intervals=intervals(peak>.302,t,DT),
        joint_hold10=hold,joint_hold16=extension_hold,recovery_time_s=recovery_time,errors=errors,quality_tail='main9.5_10' if final.any() else 'observed_failure_tail',
        smoothness=dict(reference_rate_rms=np.sqrt(np.mean(govrate**2,axis=0)).tolist()),observed_ticks=n)

def summarize(traces,alpha):
    if set(traces)!=set(CASES):raise ValueError('fixed six case coverage required')
    cases={case:trace_metrics(d,alpha,case) for case,d in traces.items()}
    summary={name:sum(int(c[name.removesuffix('_cases')]) for c in cases.values()) for name in COUNT_NAMES}
    summary.update(physical_quality_score=float(np.mean([c['physical_quality_score'] for c in cases.values()])),all_declared_traces_complete=all(c['complete'] for c in cases.values()),cases=cases)
    summary['score_tuple']=list(selection_key(summary));summary['qualified']=qualified(summary)
    return summary

def persist_best(root,checkpoint,actor,summary,identity,candidate_set):
    """Immutable generation + atomic pointer; stable aliases are conveniences only."""
    root=Path(root);checkpoint=Path(checkpoint);sha=digest(checkpoint)
    generation=root/'best_generations'/sha;generation.mkdir(parents=True,exist_ok=True)
    for src,name in [(checkpoint,'best_model.pt'),(Path(actor),'best_actor.pkl')]:
        dest=generation/name
        if not dest.exists():
            tmp=dest.with_suffix('.tmp');shutil.copyfile(src,tmp);os.replace(tmp,dest)
    assert digest(generation/'best_model.pt')==sha
    record=dict(identity,source_checkpoint=str(checkpoint.resolve()),checkpoint_sha256=sha,
        checkpoint=str((generation/'best_model.pt').resolve()),actor=str((generation/'best_actor.pkl').resolve()),
        actor_sha256=digest(generation/'best_actor.pkl'),metrics=summary,qualified=qualified(summary),score_tuple=list(selection_key(summary)),candidate_set=candidate_set)
    atomic_json(generation/'validation_metrics.json',summary);atomic_json(generation/'best_model.json',record)
    for name in ('best_model.pt','best_actor.pkl','validation_metrics.json'):
        link=root/(name+'.tmp');link.unlink(missing_ok=True);link.symlink_to((generation/name).relative_to(root));os.replace(link,root/name)
    atomic_json(root/'best_model.json',record)
    return record

def load_best(root,policy,device='cuda'):
    import torch
    root=Path(root);record=json.loads((root/'best_model.json').read_text());p=Path(record['checkpoint'])
    if digest(p)!=record['checkpoint_sha256']:raise ValueError('best checkpoint SHA mismatch')
    saved=torch.load(p,map_location=device,weights_only=False)
    if saved['update']!=record['source_update']:raise ValueError('best source update mismatch')
    policy.load_state_dict(saved['policy']);return record

def convergence_decision(update,maximum,validations,windows,discordant_used):
    out=dict(additional_updates=update,extend_by=0,reason='budget_reached',converged=False,discordant_extension_used=discordant_used)
    if update>=maximum:return out
    if len(validations)>=2 and all(qualified(x) for x in validations[-2:]):return {**out,'reason':'two_qualified_validations_target_met'}
    costs=[w.get('family_task_cost') for w in windows[-4:]]
    # Equal-family comparison on actually observed families; never replace missing data by zero.
    def improvement(x,y):
        if not x or not y or set(x)!=set(y):return None
        return float(np.mean([(x[k]-y[k])/max(abs(x[k]),1e-12) for k in x]))
    gain=improvement(costs[-2],costs[-1]) if len(costs)>=2 else None
    if len(validations)<2:return {**out,'reason':'insufficient_validation_no_blind_extension'}
    old,new=map(selection_key,validations[-2:]);quality_gain=(old[4]-new[4])/max(old[4],1e-12)
    safety_worse=new[:2]>old[:2]
    if len(validations)>=3 and len(costs)==4 and all(x for x in costs):
        keys=[selection_key(x) for x in validations[-3:]]
        qgain=(keys[0][4]-min(x[4] for x in keys))/max(keys[0][4],1e-12)
        changes=[improvement(costs[i],costs[i+1]) for i in range(3)]
        if all(k[:4]==keys[0][:4] for k in keys) and qgain<.01 and all(x is not None and abs(x)<.02 for x in changes):
            return {**out,'reason':'plateau_unqualified' if not qualified(validations[-1]) else 'empirical_plateau_qualified'}
    if not safety_worse and gain is not None and gain>=.03 and (new[:4]<old[:4] or (new[:4]==old[:4] and quality_gain>=.02)):
        return {**out,'extend_by':min(100,maximum-update),'reason':'physical_training_and_validation_improving','training_gain':gain,'validation_gain':quality_gain}
    if not safety_worse and gain is not None and gain>=.03 and not discordant_used:
        return {**out,'extend_by':min(100,maximum-update),'reason':'one_discordant_confirmation_block','discordant_extension_used':True,'training_gain':gain}
    return {**out,'reason':'no_supported_extension','training_gain':gain,'validation_gain':quality_gain}
