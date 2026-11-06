"""Execute six connection candidates from one exposed closed-loop exit snapshot.

A fixed source route/terrain and both fixed downstream routes are prepared before
execution. Replanning is active for every downstream candidate. This is nominal
original STTW; hidden PhysX contact caches are not serialized by the state API.
"""
import os,argparse,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-snapshot',type=Path);p.add_argument('--resume-from',type=Path);a=p.parse_args()
spec=json.loads(a.config.read_text());candidates=spec['candidates']
if len(candidates)!=6 or len({(c['speed_m_s'],c['alpha']) for c in candidates})!=6:
    p.error('six distinct speed/alpha candidates required')
if spec['replanning']['goal_after_turn_m']<5:p.error('goal must be at least 5 m beyond turn end')
a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.isaac_path_rollout import IsaacPathRollout
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.torch_path_reference import TorchPathReference,wrap
from sttw_control.goal_replanning import IsaacGoalReplanning
s=json.loads(Path(spec['settings']).read_text());s.update(num_envs=6,alphas=[spec['source_alpha']]*6,zero_middle=[False]*6,user_speed_m_s=spec['source_speed_m_s'],terrains={'source':spec['terrain']},episode_s=spec['source_seconds'],auto_reset_horizon=False,diagnostics=False)
torch.manual_seed(spec['seed']);torch.set_num_threads(2)
bundle=Path(spec['source_bundle']);identity=json.loads((bundle/'identity.json').read_text());cfg=json.loads((bundle/'config.json').read_text())
policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],cfg,6,'cuda:0')
plant=IsaacLabVectorPlant(s['asset'],[spec['terrain']],[0]*6,a.output,**s['plant_options']);env=IsaacPathRollout(bundle,s,policy,plant=plant)
source_route=TorchPathReference(np.load(spec['source_route']),env.config,env.device)
env.begin_connection(source_route,env.speed.clone(),env.alpha.clone())
(a.output/'protocol.json').write_text(json.dumps(dict(spec=spec,settings=s,plant=plant.metadata,source_identity=identity,source_snapshot=str(a.source_snapshot) if a.source_snapshot else None,role='same exposed exit state paired DEV; not exact hidden PhysX cache replay'),indent=2))
start=time.monotonic()
records=[]
if a.resume_from or a.source_snapshot:
    import shutil
    if a.resume_from:
        parent=json.loads((a.resume_from/'protocol.json').read_text())
        if parent['spec']!=spec or parent['settings']!=s:raise ValueError('resume requires unchanged collection contract')
        record_path=a.resume_from/'records.jsonl'
        records=[json.loads(line) for line in record_path.read_text().splitlines() if line.strip()] if record_path.exists() else []
    source=a.source_snapshot if a.source_snapshot else a.resume_from/'exit_snapshot.pt'
    shutil.copy2(source,a.output/'exit_snapshot.pt')
    snapshot=torch.load(a.output/'exit_snapshot.pt',map_location='cpu',weights_only=False)
    env.restore_connection_state(snapshot)
    m,pose,v=plant.observe()
    feature=records[0]['state'] if records else [float(v[0]),float(m[0,0]),float(m[0,1]),float(env.middle.projection['cross_track'][0]),float(wrap(env.middle.projection['heading'][0]-pose[0,2])),float(m[0,2]),float(m[0,3]),float(env.previous_reference[0,0])]
else:
    source_trace=[];found=False
    for _ in range(round(spec['source_seconds']/.02)):
        result=env.step();source_trace.append(result['lower_trace'].clone())
        if not bool(env.alive[0]):break
        if float(env.middle.projection['progress'][0])>=spec['source_exit_progress_m']:
            found=True;break
    np.savez(a.output/'source_trajectory.npz',trace=torch.stack(source_trace).cpu().numpy())
    if not found:raise RuntimeError('Source lane did not produce the declared live exit; no candidate records emitted')
    snapshot=env.capture_connection_state(0);torch.save(snapshot,a.output/'exit_snapshot.pt')
    m,pose,v=plant.observe();feature=[float(v[0]),float(m[0,0]),float(m[0,1]),float(env.middle.projection['cross_track'][0]),float(wrap(env.middle.projection['heading'][0]-pose[0,2])),float(m[0,2]),float(m[0,3]),float(env.previous_reference[0,0])]
snapshot_id=hashlib.sha256((a.output/'exit_snapshot.pt').read_bytes()).hexdigest()
skill_id=env.upper_sha256+':'+('ecbc_zero_residual' if bool(env.zero_lower.all()) else env.lower_sha256)
if spec['replanning'].get('arrival_reference')=='original_route':skill_id+=':arrival_original_route'
speeds=torch.tensor([c['speed_m_s'] for c in candidates],device=env.device);alphas=torch.tensor([c['alpha'] for c in candidates],device=env.device)
completed_tasks={r['task_id'] for r in records}
for task in spec['tasks']:
    if task['task_id'] in completed_tasks:continue
    folder=a.output/task['task_id'];folder.mkdir()
    data=np.load(task['route']);original=TorchPathReference(data,env.config,env.device)
    env.restore_connection_state(snapshot);env.horizon=round(task['seconds']/.02);env.begin_connection(original,speeds,alphas)
    planner=IsaacGoalReplanning(env,{**spec['replanning'],'original_route':task['route']},folder)
    goal=torch.as_tensor(planner.goal,device=env.device);arrival=torch.full((6,),float('nan'),device=env.device)
    rows=[];original_errors=[]
    # Original fixed-path distance is measured independently of replanned references.
    points=original.xy;segments=points[1:]-points[:-1];segment_norm=segments.square().sum(-1)
    for k in range(env.horizon):
        planner.update();result=env.step();trace=result['lower_trace'];rows.append(trace.clone())
        xy=trace[:,:,:2]
        delta=xy[:,:,None,:]-points[None,None,:-1,:]
        u=(delta*segments).sum(-1)/segment_norm
        projection=points[:-1]+u.clamp(0,1)[...,None]*segments
        error=(xy[:,:,None,:]-projection).square().sum(-1).amin(-1).sqrt();original_errors.append(error)
        _,pose,_=plant.observe();at_goal=env.alive&(torch.linalg.vector_norm(pose[:,:2]-goal,dim=-1)<=spec['replanning']['arrival_radius_m'])
        arrival=torch.where(torch.isnan(arrival)&at_goal,(k+1)*.02,arrival)
        if not bool(env.alive.any()):break
    trace=torch.cat(rows).cpu().numpy();errors=torch.cat(original_errors).cpu().numpy();arrivals=arrival.cpu().numpy()
    np.savez(folder/'trajectory.npz',trace=trace,original_path_error=errors,original_xy=data['xy'],original_s=data['s'],arrival_time=arrivals,goal=planner.goal)
    for i,candidate in enumerate(candidates):
        valid=trace[:,i,6]>0;path_errors=errors[valid,i]
        completed=bool(env.alive[i]) and np.isfinite(arrivals[i]) and bool((path_errors<=spec['corridor_half_width_m']).all())
        records.append(dict(origin_id=spec['origin_id'],snapshot_id=snapshot_id,task_id=task['task_id'],skill_id=skill_id,candidate_id=candidate['candidate_id'],state=feature,complete=int(completed),success_cost=float(arrivals[i]) if completed else None,path_rmse_m=float(np.sqrt(np.mean(path_errors**2))) if len(path_errors) else None,observed_lower_steps=int(valid.sum()),physical_or_controller_failure=not bool(env.alive[i]),trace=str((folder/'trajectory.npz').resolve()),snapshot=str((a.output/'exit_snapshot.pt').resolve())))
    (a.output/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
(a.output/'status.json').write_text(json.dumps(dict(status='complete',rows=len(records),elapsed_s=time.monotonic()-start,optimizer_updates=0),indent=2))
os.sync();os._exit(0)
