"""Bounded physical replay audit with independently stepped execution comparison."""
import json,pickle,time
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from .closed_loop_predictor import ClosedLoopPredictor
from .governor_state import Candidate
from .preference_governor import PreferenceGovernor


def copy_diagnostics(before,after,summary):
    preserved=len(before)==len(after) and all(np.array_equal(x,np.asarray(y),equal_nan=True) for x,y in zip(before,after))
    fields={k:float(np.max(np.abs(v-v[0]))) for k,v in summary.items() if np.issubdtype(v.dtype,np.floating)}
    return dict(state_preserved=preserved,per_field_duplicate_max_abs=fields)


def physical_contracts(env,prepared,budget,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    predictor=ClosedLoopPredictor(env,budget);results={};passed=True
    # One step and full 1.2s from an unchanged complete snapshot; no resets.
    for label,delta,bypass,goal_value in [('left',.08,True,[2.3,.08]),('right',-.08,True,[2.3,-.08]),('governed',0.,False,[1.5,.35])]:
        raw=jp.array([2.3,delta]);s=prepared.replace(raw=raw)
        goal=jp.asarray(goal_value)
        budget.charge(predictor=480,contracts=2,reason=f'{label} scan vs explicit 240-step replay')
        start=time.perf_counter();end,trace=predictor.replay(s,goal,bypass);jax.block_until_ready(end)
        scan_s=time.perf_counter()-start
        actual=s;logs=[]
        for _ in range(240):
            actual,l=env.step(actual,goal,bypass);logs.append(jax.device_get(l))
        jax.block_until_ready(actual)
        errors={k:float(np.max(np.abs(np.asarray(trace[k])-np.asarray([l[k] for l in logs]))))
            for k in ('actual_forward_speed','actual_delta','phi','phi_dot','yaw_unwrapped','actual_xy','applied_residual','final_ctrl')}
        for key in ('qpos','qvel','qacc_warmstart'):
            errors[key]=float(np.max(np.abs(np.asarray(getattr(end.data,key))-np.asarray(getattr(actual.data,key)))))
        errors['eso']=float(np.max(np.abs(np.asarray(end.controller.eso)-np.asarray(actual.controller.eso))))
        direction=float(end.yaw_unwrapped-s.yaw_unwrapped)
        ok=max(errors.values())<=1e-5 and (direction*delta>0 if bypass else True)
        results[label]=dict(max_abs_errors=errors,passed=ok,yaw_increment=direction,scan_compile_and_run_s=scan_s,physical_failure=bool(end.failed),bypass=bypass)
        np.savez_compressed(out/f'{label}_replay.npz',**{k:np.asarray(v) for k,v in trace.items()})
        passed=passed and ok
        (out/'replay_progress.json').write_text(json.dumps(results,indent=2)+'\n')
        if not ok:break
    # 144 copied candidates must not mutate warmstart/ESO/actuator/physical tick.
    before=[np.array(x) for x in jax.tree.leaves(prepared)]
    goals=np.tile(np.asarray(prepared.raw),(144,1));c=Candidate(goals,np.ones(144,bool),np.ones(144,bool))
    result,cold=predictor.predict(prepared,c)
    preserved=all(np.array_equal(x,np.asarray(y),equal_nan=True) for x,y in zip(before,jax.tree.leaves(prepared)))
    duplicates=max(float(np.max(np.abs(v-v[0]))) for k,v in result.items() if np.issubdtype(v.dtype,np.floating))
    results['144_copy_isolation']=dict(passed=preserved and duplicates==0,duplicate_max_error=duplicates)
    passed=passed and results['144_copy_isolation']['passed']
    gov=PreferenceGovernor(predictor)
    # No future data exists in API/state; identical current prefix/snapshot, different
    # evaluator-owned future lists cannot be consumed by this method.
    d0=gov.decide(prepared,0);d1=gov.decide(prepared,1)
    same=np.array_equal(d0.goal,d1.goal) and d0.bypass==d1.bypass and d0.raw_feasible==d1.raw_feasible
    results['nonconflict_alpha_identity']=dict(passed=same and d0.bypass and d1.bypass)
    passed=passed and results['nonconflict_alpha_identity']['passed']
    results['passed']=bool(passed);results['cpu_mjx_audit']='pending B0 left/right snapshots'
    results['predictor_compile_seconds']=predictor.compile_seconds
    (out/'physical_contracts.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def independent_summary(initial,end,logs,env):
    """NumPy reconstruction from actual executed steps, independent of scan sums."""
    l={k:np.asarray(v) for k,v in logs.items()};H=160
    raw=np.asarray(initial.raw);command=np.asarray(l['final_command'])
    previous=np.vstack([np.asarray(initial.actuator.previous),command[:-1]])
    smooth=.02*np.mean(np.sum(((command[:H]-previous[:H])/np.array([3.,60.]))**2,axis=1))
    smooth+=.001*np.mean(np.sum((l['applied_residual'][:H]/np.array([1.5,10.]))**2,axis=1))
    cc=env.cc;v=max(cc.minimum_speed,float(l['wheel_speed_proxy'][-1]))
    m2=-(cc.mass*v*v*cc.cg_height-cc.mass*cc.cg_forward*cc.trail*cc.gravity)*np.cos(cc.caster)/cc.wheelbase
    m2=(-1 if m2<0 else 1)*max(abs(m2),.1)
    ratio=(-cc.mass*cc.gravity*cc.cg_height)/m2
    eq=abs(float(l['phi'][-1]-l['actual_delta'][-1]/ratio))
    rate=abs(float(l['phi_dot'][-1]));pr=float(l['peak_roll'].max());pd=float(l['peak_roll_rate'].max())
    vr=np.maximum((l['peak_roll']-.30)/.06,0);vd=np.maximum((l['peak_roll_rate']-1.5)/1.5,0)
    violation=float(np.maximum(vr,vd).max()+np.mean(vr**2+vd**2)+max((rate-.30)/.30,0)**2+max((eq-.08)/.08,0)**2)
    finite=not bool(np.any(l['nonfinite']));failure=bool(end.failed);working=pr<=.30 and pd<=1.5
    terminal=rate<=.30 and eq<=.08;command_ok=bool(np.all(np.abs(command)<=np.array([3.,60.])))
    feasible=finite and not failure and working and terminal and command_ok
    return dict(steer_rmse=float(np.sqrt(np.mean((l['actual_delta'][:H]-raw[1])**2))),
        speed_rmse=float(np.sqrt(np.mean((l['actual_forward_speed'][:H]-raw[0])**2))),
        heading_cost=float(np.mean((l['e_psi_unwrapped'][:H]/.1)**2)+2*(l['e_psi_unwrapped'][H-1]/.1)**2),
        smoothness=float(smooth),peak_roll=pr,peak_roll_rate=pd,terminal_rate=rate,equilibrium_error=eq,
        finite=finite,physical_failure=failure,working=working,terminal=terminal,command_ok=command_ok,
        feasible=feasible,reserve=feasible and pr<=.24 and pd<=.8,violation=violation,
        terminal_qpos=np.asarray(end.data.qpos),terminal_qvel=np.asarray(end.data.qvel))


def production_batch_contracts(env,prepared,budget,out):
    """Actual prediction summaries vs independent 240-tick execution, all shapes."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    predictor=ClosedLoopPredictor(env,budget)
    goals=np.array([[2.3,0.],[1.5,.35],[1.5,-.35],[2.3,.08]],dtype=np.asarray(prepared.raw).dtype)
    bypass=np.array([True,False,False,False])
    before=[np.asarray(x).copy() for x in jax.tree.leaves(prepared)]
    summaries={};traces={};results={'replay':{},'batches':{},'precision':str(prepared.data.qpos.dtype),'device':str(jax.devices())}
    passed=True
    for i,(goal,bp) in enumerate(zip(goals,bypass)):
        budget.charge(predictor=480,contracts=2,reason=f'production contract {i}: actual 240 steps plus scan replay')
        state=prepared;rows=[]
        for _ in range(240):
            state,row=env.step(state,jp.asarray(goal),bool(bp));rows.append(jax.device_get(row))
        l={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
        summaries[i]=independent_summary(prepared,state,l,env);traces[i]=l
        scan_end,scan_logs=predictor.replay(prepared,jp.asarray(goal),bool(bp));jax.block_until_ready(scan_end)
        fields=['actual_forward_speed','actual_delta','phi','phi_dot','yaw_unwrapped','actual_xy','final_ctrl','applied_residual']
        err={k:float(np.max(np.abs(l[k]-np.asarray(scan_logs[k])))) for k in fields}
        for k in ['qpos','qvel','qacc_warmstart']:err[k]=float(np.max(np.abs(np.asarray(getattr(state.data,k))-np.asarray(getattr(scan_end.data,k)))))
        err['eso']=float(np.max(np.abs(np.asarray(state.controller.eso)-np.asarray(scan_end.controller.eso))))
        ok=all(np.isfinite(v) and v<=1e-5 for v in err.values());passed &= ok
        results['replay'][str(i)]=dict(passed=ok,max_abs_errors=err,goal=goal.tolist(),bypass=bool(bp))
        np.savez_compressed(out/f'executed_{i}.npz',**l)
    for n in [1,144,80]:
        indices=np.arange(n)%4;c=Candidate(goals[indices],bypass[indices],np.ones(n,bool))
        r,cold=predictor.predict(prepared,c);np.savez_compressed(out/f'batch_{n}.npz',**r)
        errors={};flags_equal=True
        for k,v in r.items():
            reference=np.asarray([summaries[int(i)][k] for i in indices])
            if np.issubdtype(v.dtype,np.bool_):flags_equal &= bool(np.array_equal(v,reference))
            else:errors[k]=float(np.max(np.abs(v-reference)))
        ok=flags_equal and all(np.isfinite(v) and v<=1e-5 for v in errors.values());passed &= ok
        results['batches'][str(n)]=dict(passed=ok,max_abs_errors=errors,flags_equal=flags_equal)
        (out/'production_progress.json').write_text(json.dumps(results,indent=2)+'\n')
    preserved=copy_diagnostics(before,jax.tree.leaves(prepared),{})['state_preserved'];passed &= preserved
    results['state_preserved']=preserved;results['passed']=bool(passed)
    results['predictor_compile_seconds']=predictor.compile_seconds
    results['cpu_mjx_audit']='pending B0 left/right snapshots; not included in pre-panel acceptance'
    (out/'physical_contracts.json').write_text(json.dumps(results,indent=2)+'\n')
    return results
