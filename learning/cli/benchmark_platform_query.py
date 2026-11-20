"""Time synchronized single-robot candidate evaluations on real platform handback states."""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pandas as pd
import torch

from evaluate_platform_review import read
from sttw_control.platform_online import PlatformDepartureQuery, CapturedDepartureQuery, velocity_state
from sttw_control.playground_history_model import load_history_model


@torch.inference_mode()
def measure(out, graph_out=None, compiled=False):
    plan=read(out/'phase2_relative/plan.json');raw=Path(plan['p2']);wide=raw.parent
    config=read(wide/'primary/roots_0008/rb_run1/resolved_config.json')
    phys=mujoco.MjModel.from_binary_path(str(wide/'primary/roots_0008/rb_run1/terrain.mjb'))
    model=load_history_model(Path(plan['ground_model']))
    archive=pd.read_csv(raw/'physical_scores.csv')
    # Calibration only. Some R9 approaches have no physical handback at a particular root.
    # Use the three first distinct successful prefix states, and their actual three published branches.
    valid=archive[(archive.layout=='primary')&(archive.skill=='rb_run1')&(archive.section=='near')&
                  (archive.root<78)&archive.verified_jump].sort_values(['approach','root'])
    groups=list(valid.groupby(['approach','root'],sort=True))
    # Prefer distinct approaches. Fill the third slot from a different calibration root if necessary.
    unique={};extras=[]
    for key,g in groups:
        if key[0] not in unique:unique[key[0]]=g
        else:extras.append(g)
    samples=(list(unique.values())+extras)[:3]
    q=[];v=[];lastq=[];lastv=[];refs=[];seconds=[]
    bank=dict(np.load(wide/'primary/departure_bank.npz'))
    index={row.tobytes():i for i,row in enumerate(bank['xy'])}
    for group in samples:
        ids=[]
        for direction in ('left','straight','right'):
            row=group[group.departure==direction].iloc[0]
            with np.load(row.trajectory) as z:
                ids.append(index[z['published_reference_xy'].tobytes()])
                if direction=='straight':
                    k=round(row.handoff_s/.005)+1
                    q.append(z['qpos'][k]);lastq.append(z['qpos'][k-4])
                    v.append(z['qvel'][k]);lastv.append(z['qvel'][k-4])
                    seconds.append(10.-k*.005)
        refs.append(ids)
    def gpu(x):return torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
    route=SimpleNamespace(s=gpu(bank['s']),**{k:gpu(bank[k][np.array(refs).ravel()]) for k in ('xy','heading','curvature')})
    q=gpu(q);u=velocity_state(q,gpu(v));prev=velocity_state(gpu(lastq),gpu(lastv))
    relative=dict(raster=str(raw/'primary_terrain.npz'),height=.16,initializer=str(out/'phase2_relative/initializer.npz'))
    hand=config['ground_handoff'];rows=[]
    dest=graph_out or out
    dest.mkdir(exist_ok=True,parents=True)
    for name,setup in [('kinematic',None),('phase2_relative',relative)]:
        if compiled and not setup:continue
        query=PlatformDepartureQuery(phys,model,hand['goal'],plan['margin_m'],hand['road_half_width_m'],setup)
        for n in (1,3):
            candidates=3*n
            refids=torch.arange(candidates,device='cuda').reshape(n,3)
            request=torch.full((n,),2.,device='cuda');section=torch.full_like(request,13.6)
            old=torch.ones(n,device='cuda',dtype=torch.long);side=torch.ones_like(request)
            args=(q[:n],u[:n,0],request,route,refids,section,side,old,seconds[0],u[:n],prev[:n])
            runner=query.choose
            errors={}
            if graph_out:
                if compiled:
                    eager_step=model.step
                    compiled_step=torch.compile(eager_step,fullgraph=True,dynamic=False,options={'use_fast_math':False,'triton.cudagraphs':False,'emulate_precision_casts':True})
                    model.step=compiled_step
                runner=CapturedDepartureQuery(query,args)
                if compiled:model.step=eager_step
                errors=dict(max_pose_abs=0.,max_velocity_abs=0.,max_domain_abs=0.,decision_mismatches=0)
                for k in range(3):
                    idx=(torch.arange(n,device='cuda')+k)%len(q)
                    check=(q[idx],u[idx,0],request,route,(3*idx[:,None]+torch.arange(3,device='cuda')),section,side,old,seconds[0],u[idx],prev[idx])
                    ref=query.choose(*check,return_forecast=True);got=runner(*check)
                    for key,a,b in zip(list(errors)[:3],ref[3:],got[3:]):
                        errors[key]=max(errors[key],float((a-b).abs().amax()))
                    errors['decision_mismatches']+=sum(int((a!=b).sum()) for a,b in zip(ref[:3],got[:3]))
                print(name,candidates,errors,flush=True)
            samples_ms=[]
            for i in range(1050):
                start=time.perf_counter_ns()
                runner(*args)
                torch.cuda.synchronize()
                elapsed=(time.perf_counter_ns()-start)/1e6
                if i>=50:samples_ms.append(elapsed)
                if i in (9,99,549):
                    print(name,candidates,'iteration',i+1,'last synchronized ms',round(elapsed,2),flush=True)
            path=dest/f'latency_{name}_{candidates}.csv'
            np.savetxt(path,samples_ms,delimiter=',',header='ms',comments='')
            rows.append(dict(method=name,robots=1,candidates=candidates,horizon_steps=int(seconds[0]/.02),
                             n=1000,warmup=50,**errors,p50_ms=np.median(samples_ms),p95_ms=np.percentile(samples_ms,95)))
            pd.DataFrame(rows).to_csv(dest/'latency.csv',index=False)
            print(rows[-1],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--graph-out',type=Path);p.add_argument('--compile-step',action='store_true');a=p.parse_args();measure(a.out,a.graph_out,a.compile_step)
