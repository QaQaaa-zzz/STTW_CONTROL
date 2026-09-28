"""Bounded physical replay audit with independently stepped execution comparison."""
import json,pickle,time
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
from .closed_loop_predictor import ClosedLoopPredictor
from .governor_state import Candidate
from .preference_governor import PreferenceGovernor


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
