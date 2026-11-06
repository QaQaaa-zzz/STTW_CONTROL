"""Execute six connection candidates from one exposed closed-loop exit snapshot.

A fixed source route/terrain and both fixed downstream routes are prepared before
execution. Replanning is active for every downstream candidate. This is nominal
original STTW; hidden PhysX contact caches are not serialized by the state API.
"""
import os,argparse,hashlib,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-snapshot',type=Path);p.add_argument('--source-only',action='store_true');p.add_argument('--replay-check-only',action='store_true');p.add_argument('--resume-from',type=Path);a=p.parse_args()
if a.replay_check_only and (a.source_only or a.source_snapshot or a.resume_from):p.error('Replay comparison must start from live uninterrupted source execution')
spec=json.loads(a.config.read_text());candidates=spec['candidates']
if len(candidates)!=6 or len({(c['speed_m_s'],c['alpha']) for c in candidates})!=6:
    p.error('six distinct speed/alpha candidates required')
if spec['replanning']['goal_after_turn_m']<5:p.error('goal must be at least 5 m beyond turn end')
pulse=spec.get('connection_force_pulse')
if pulse and (pulse['kind']!='external_force_N' or pulse['body']!='chasis' or not spec.get('common_task_reference') or pulse['duration_s']<=0 or not 0<=pulse['trigger_progress_m']<spec['connection_end_progress_m']):p.error('force pulse requires external_force_N, positive duration and trigger inside connection')
a.output.mkdir(parents=True,exist_ok=False)
from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
from sttw_control.isaac_path_rollout import IsaacPathRollout
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.torch_path_reference import TorchPathReference,wrap
from sttw_control.goal_replanning import IsaacGoalReplanning
s=json.loads(Path(spec['settings']).read_text());s.update(num_envs=6,alphas=[spec['source_alpha']]*6,zero_middle=[False]*6,user_speed_m_s=spec['source_speed_m_s'],terrains={'source':spec['terrain']},episode_s=spec['source_seconds'],auto_reset_horizon=False,diagnostics=False,record_control_commands=True,record_body_poses=spec.get('record_body_poses',False))
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
    source_trace=[];source_commands=[];found=False
    for _ in range(round(spec['source_seconds']/.02)):
        result=env.step();source_trace.append(result['lower_trace'].clone());commands=result['executed_commands'].clone();commands[:,:,0]+=(len(source_trace)-1)*.02;source_commands.append(commands)
        if not bool(env.alive[0]):break
        if float(env.middle.projection['progress'][0])>=spec['source_exit_progress_m']:
            found=True;break
    np.savez_compressed(a.output/'source_trajectory.npz',trace=torch.stack(source_trace).cpu().numpy(),executed_commands=torch.cat(source_commands).cpu().numpy())
    if not found:
        (a.output/'status.json').write_text(json.dumps(dict(status='source_failed',origin_id=spec['origin_id'],protocol_id=spec.get('protocol_id','legacy_full_suffix'),reached_decision=False,root_complete=False,reason='physical_or_controller_failure' if not bool(env.alive[0]) else 'exit_not_reached_in_window',rows=0,elapsed_s=time.monotonic()-start,optimizer_updates=0),indent=2))
        os.sync();os._exit(0)
    snapshot=env.capture_connection_state(0);torch.save(snapshot,a.output/'exit_snapshot.pt')
    m,pose,v=plant.observe();feature=[float(v[0]),float(m[0,0]),float(m[0,1]),float(env.middle.projection['cross_track'][0]),float(wrap(env.middle.projection['heading'][0]-pose[0,2])),float(m[0,2]),float(m[0,3]),float(env.previous_reference[0,0])]
# Decision-time inputs only; retain legacy eight-feature order for its kernel baseline.
m,pose,v=plant.observe();root=plant.roots[0];yaw=pose[0,2]
feature_names=['speed','roll','roll_rate','cross_track','heading_error','steer','steer_rate','previous_speed_reference']
state_named=dict(zip(feature_names,feature))
state_named.update(lateral_velocity_heading_frame=float(-torch.sin(yaw)*root[7]+torch.cos(yaw)*root[8]),world_yaw_rate=float(root[12]),previous_alpha=float(env.alpha[0]),previous_alpha_duration_s=snapshot['rollout_tick']*.02)
(a.output/'decision_state.json').write_text(json.dumps(dict(schema='ground_connection_exit_v1',origin_id=spec['origin_id'],state=state_named,history_scope='Same constant alpha since source episode start; saved warm-state before episode excluded',snapshot='exit_snapshot.pt',future_outcomes_excluded=True),indent=2)+'\n')
if a.source_only:
    (a.output/'status.json').write_text(json.dumps(dict(status='source_complete',origin_id=spec['origin_id'],reached_decision=True,state=feature,elapsed_s=time.monotonic()-start,optimizer_updates=0),indent=2))
    os.sync();os._exit(0)
snapshot_id=hashlib.sha256((a.output/'exit_snapshot.pt').read_bytes()).hexdigest()
skill_id=env.upper_sha256+':'+('ecbc_zero_residual' if bool(env.zero_lower.all()) else env.lower_sha256)
if spec['replanning'].get('arrival_reference')=='original_route':skill_id+=':arrival_original_route'
if a.replay_check_only:
    # First continuation uses the live source scene; only the second restores it.
    # This compares uninterrupted execution with replay, not replay with replay.
    task=spec['tasks'][0];route=TorchPathReference(np.load(task['route']),env.config,env.device)
    reference=spec.get('common_task_reference',dict(speed_m_s=spec['source_speed_m_s'],alpha=spec['source_alpha']))
    traces=[];endpoints=[]
    for restored in (False,True):
        if restored:env.restore_connection_state(snapshot)
        env.horizon=round(task['seconds']/.02)
        env.begin_connection(route,torch.full_like(env.speed,reference['speed_m_s']),torch.full_like(env.alpha,reference['alpha']))
        folder=a.output/('restored' if restored else 'uninterrupted');folder.mkdir()
        planner=IsaacGoalReplanning(env,{**spec['replanning'],'original_route':task['route']},folder)
        history=[]
        for _ in range(env.horizon):
            planner.update();history.append(env.step()['lower_trace'].clone())
            if not bool(env.alive.any()):break
        traces.append(torch.cat(history).cpu().numpy());endpoints.append(env.alive.cpu().tolist())
    np.savez_compressed(a.output/'replay_comparison.npz',uninterrupted=traces[0],restored=traces[1],original_xy=route.xy.cpu().numpy())
    n=min(len(t) for t in traces);both=(traces[0][:n,0,6]>0)&(traces[1][:n,0,6]>0)
    delta=traces[0][:n,0]-traces[1][:n,0]
    comparison=dict(status='replay_compared',origin_id=spec['origin_id'],reference=reference,task_id=task['task_id'],seconds=task['seconds'],lane=0,
        comparison='Live uninterrupted source continuation versus restored exposed state; hidden PhysX caches not serialized',
        common_valid_lower_steps=int(both.sum()),uninterrupted_alive=endpoints[0],restored_alive=endpoints[1],
        max_xy_difference_m=float(np.linalg.norm(delta[both,:2],axis=1).max()),max_speed_difference_m_s=float(abs(delta[both,2]).max()),
        elapsed_s=time.monotonic()-start,optimizer_updates=0,adoption='Inspect trajectories and task behavior; no automatic numerical pass threshold')
    (a.output/'status.json').write_text(json.dumps(comparison,indent=2)+'\n')
    os.sync();os._exit(0)
speeds=torch.tensor([c['speed_m_s'] for c in candidates],device=env.device);alphas=torch.tensor([c['alpha'] for c in candidates],device=env.device)
capability=None
if spec.get('capability_records'):
    from sttw_control.empirical_capability import ConnectionCapability
    from sttw_control.continuous_task import ContinuousTask
    cc=json.loads(Path(spec['capability_config']).read_text())
    evidence=[json.loads(line) for line in Path(spec['capability_records']).read_text().splitlines() if line.strip()]
    evidence=[r for r in evidence if r['origin_id']!=spec['origin_id']]
    capability=ConnectionCapability(evidence,cc['feature_scales'],cc['bandwidth'],cc['minimum_support'])
completed_tasks={r['task_id'] for r in records}
for task in spec['tasks']:
    if task['task_id'] in completed_tasks:continue
    folder=a.output/task['task_id'];folder.mkdir()
    decision=None
    if capability is not None:
        route=dict(route_id=task['task_id'],goal_id=task['task_id'],task_id=task['task_id'],skill_id=skill_id,candidates=[c['candidate_id'] for c in candidates],reference_path=task['route'])
        manager=ContinuousTask(dict(goals=[dict(goal_id=task['task_id'],allowed_routes=[task['task_id']])]),[route],candidates,capability)
        decision=manager.choose(feature,[task['task_id']])
        decision.update(role='Prospective recommendation before candidate rollouts; selected lane is executed in the paired batch',evidence_rows=len(evidence),evidence_records=spec['capability_records'])
        (folder/'prospective_decision.json').write_text(json.dumps(decision,indent=2))
    data=np.load(task['route']);original=TorchPathReference(data,env.config,env.device)
    env.restore_connection_state(snapshot);env.horizon=round(task['seconds']/.02);env.begin_connection(original,speeds,alphas)
    planner=IsaacGoalReplanning(env,{**spec['replanning'],'original_route':task['route']},folder)
    goal=torch.as_tensor(planner.goal,device=env.device);arrival=torch.full((6,),float('nan'),device=env.device)
    rows=[];command_rows=[];body_rows=[];original_errors=[];entry_events=[];common=spec.get('common_task_reference')
    original_progress=torch.zeros(6,device=env.device);entered_b=torch.zeros(6,device=env.device,dtype=torch.bool)
    # Original fixed-path distance is measured independently of replanned references.
    points=original.xy;segments=points[1:]-points[:-1];segment_norm=segments.square().sum(-1)
    if pulse:
        from sttw_control.torch_observation import rotate_inverse
        pulse_started=torch.zeros(6,device=env.device,dtype=torch.bool);remaining=torch.zeros(6,device=env.device,dtype=torch.long)
        pulse_start_time=torch.full((6,),float('nan'),device=env.device);pulse_base_tick=env.physical_tick.clone();pulse_rows=[]
        body_id=torch.tensor([plant.bodies.index(pulse['body'])],device=env.device)
        site=torch.tensor(pulse['site_body_position_m'],device=env.device).expand(6,1,3).contiguous()
        zero=torch.zeros(6,1,3,device=env.device);force_world=torch.tensor(pulse['force_world_N'],device=env.device).expand(6,3)
        def apply_connection_force(physics_tick):
            force=force_world*((remaining>0)&env.alive)[:,None]
            local=rotate_inverse(plant.roots[:,3:7],force)
            plant.robot.permanent_wrench_composer.set_forces_and_torques(forces=local[:,None].contiguous(),torques=zero,positions=site,body_ids=body_id,is_global=False)
            t=(env.physical_tick-pulse_base_tick)*.005+physics_tick*plant.physics_dt
            pulse_rows.append(torch.cat((t[:,None],force),dim=-1).clone())
            remaining.sub_(1).clamp_(min=0)

    for k in range(env.horizon):
        if common is not None:
            measurement,pose,velocity=plant.observe()
            original_progress=original.project(pose[:,:2],original_progress,velocity.abs()*.02)['progress']
            crossing=env.alive&~entered_b&(original_progress>=spec['connection_end_progress_m'])
            for lane in crossing.nonzero(as_tuple=False).flatten().tolist():
                entry_events.append(dict(lane=lane,candidate_id=candidates[lane]['candidate_id'],time_s=k*.02,original_progress_m=float(original_progress[lane]),speed_m_s=float(velocity[lane]),lateral_velocity_heading_frame_m_s=float(-torch.sin(pose[lane,2])*plant.roots[lane,7]+torch.cos(pose[lane,2])*plant.roots[lane,8]),world_yaw_rate_rad_s=float(plant.roots[lane,12]),measurement=measurement[lane].cpu().tolist(),pose=pose[lane].cpu().tolist(),root=plant.roots[lane].cpu().tolist(),previous_reference=env.previous_reference[lane].cpu().tolist(),previous_alpha=float(env.alpha[lane]),common_reference=common,history='retained; no reset'))
            env.speed.copy_(torch.where(crossing,common['speed_m_s'],env.speed));env.alpha.copy_(torch.where(crossing,common['alpha'],env.alpha));entered_b|=crossing
        if pulse:
            trigger=env.alive&~pulse_started&(original_progress>=pulse['trigger_progress_m'])
            remaining[trigger]=round(pulse['duration_s']/plant.physics_dt);pulse_start_time[trigger]=k*.02;pulse_started|=trigger
            env.training_pulse=apply_connection_force if bool((remaining>0).any()) else None
        planner.update();result=env.step();trace=result['lower_trace'];rows.append(trace.clone())
        if pulse and env.training_pulse is not None:
            plant.robot.permanent_wrench_composer.set_forces_and_torques(forces=zero,torques=zero,positions=site,body_ids=body_id,is_global=False)
            env.training_pulse=None

        commands=result['executed_commands'].clone();commands[:,:,0]+=k*.02;command_rows.append(commands)
        if s['record_body_poses']:body_rows.append(result['body_poses'])
        xy=trace[:,:,:2]
        delta=xy[:,:,None,:]-points[None,None,:-1,:]
        u=(delta*segments).sum(-1)/segment_norm
        projection=points[:-1]+u.clamp(0,1)[...,None]*segments
        error=(xy[:,:,None,:]-projection).square().sum(-1).amin(-1).sqrt();original_errors.append(error)
        _,pose,_=plant.observe();at_goal=env.alive&(torch.linalg.vector_norm(pose[:,:2]-goal,dim=-1)<=spec['replanning']['arrival_radius_m'])
        arrival=torch.where(torch.isnan(arrival)&at_goal,(k+1)*.02,arrival)
        if not bool(env.alive.any()):break
    if pulse:
        applied=torch.stack(pulse_rows).cpu().numpy() if pulse_rows else np.empty((0,6,4))
        np.savez_compressed(folder/'external_force.npz',samples=applied,columns=np.array(['time_since_connection_s','world_Fx_N','world_Fy_N','world_Fz_N']),trigger_time_s=pulse_start_time.cpu().numpy(),physics_dt_s=plant.physics_dt)
        (folder/'external_force.json').write_text(json.dumps(dict(protocol=pulse,triggered=pulse_started.cpu().tolist(),impulse_world_Ns=(applied[:,:,1:].sum(0)*plant.physics_dt).tolist(),scope='Prescribed rigid-body external force, not command residual or velocity replacement; omitted inactive intervals have zero force'),indent=2)+'\n')
    trace=torch.cat(rows).cpu().numpy();errors=torch.cat(original_errors).cpu().numpy();arrivals=arrival.cpu().numpy()
    commands=torch.cat(command_rows).cpu().numpy()
    np.savez_compressed(folder/'executed_commands.npz',commands=commands,columns=np.array(['time_since_connection_s','steering_rate_rad_s','rear_axle_speed_rad_s','alive_before_step']))
    np.savez(folder/'trajectory.npz',trace=trace,original_path_error=errors,original_xy=data['xy'],original_s=data['s'],arrival_time=arrivals,goal=planner.goal)
    if body_rows:np.savez_compressed(folder/'body_poses.npz',poses=torch.cat(body_rows).cpu().numpy(),body_names=np.array(plant.bodies),env_origins=plant.origin.cpu().numpy(),quaternion_order=np.array('xyzw'),sample_dt_s=np.array(.005),valid=trace[:,:,6]>0,urdf=np.array(str(Path(s['asset']).resolve())))
    if common is not None:(folder/'b_entry_events.json').write_text(json.dumps(entry_events,indent=2)+'\n')
    for i,candidate in enumerate(candidates):
        valid=trace[:,i,6]>0;path_errors=errors[valid,i]
        live_commands=commands[commands[:,i,3]>0,i,1:3]
        input_limit={}
        for column,name in enumerate(['cmd_steering_v','cmd_rearwheel_v']):
            actuator=next(a for a in plant.contract['actuators'] if a['name']==name);lo,hi=actuator['ctrlrange']
            count=int(((live_commands[:,column]<=lo+1e-6)|(live_commands[:,column]>=hi-1e-6)).sum()) if actuator['control_limited'] else 0
            input_limit[name]=dict(at_limit_samples=count,active_physics_samples=len(live_commands),fraction=count/len(live_commands) if len(live_commands) else None,ctrlrange=actuator['ctrlrange'],scope='Plant input target limit, not measured torque saturation')
        completed=bool(env.alive[i]) and (common is None or bool(entered_b[i])) and np.isfinite(arrivals[i]) and bool((path_errors<=spec['corridor_half_width_m']).all())
        records.append(dict(disturbance_id=pulse['sequence_id'] if pulse else 'D0',protocol_id=spec.get('protocol_id','legacy_full_suffix'),b_entry=next((e for e in entry_events if e['lane']==i),None),common_task_reference=common,origin_id=spec['origin_id'],snapshot_id=snapshot_id,task_id=task['task_id'],skill_id=skill_id,candidate_id=candidate['candidate_id'],state=feature,state_named=state_named,geometry=dict(connection_length_m=spec.get('connection_end_progress_m'),corridor_half_width_m=spec['corridor_half_width_m'],route=task['route']),complete=int(completed),success_cost=float(arrivals[i]) if completed else None,input_target_limit=input_limit,max_original_path_error_m=float(path_errors.max()) if len(path_errors) else None,path_rmse_m=float(np.sqrt(np.mean(path_errors**2))) if len(path_errors) else None,observed_lower_steps=int(valid.sum()),physical_or_controller_failure=not bool(env.alive[i]),trace=str((folder/'trajectory.npz').resolve()),snapshot=str((a.output/'exit_snapshot.pt').resolve())))
    if decision is not None and decision['status']=='reference_selected':
        selected=next(r for r in records if r['task_id']==task['task_id'] and r['candidate_id']==decision['reference']['candidate_id'])
        lane=next(i for i,c in enumerate(candidates) if c['candidate_id']==selected['candidate_id'])
        manager.observe_goal(task['task_id'],arrived=bool(np.isfinite(arrivals[lane])),upright=not selected['physical_or_controller_failure'],within_allowed_region=bool((errors[trace[:,lane,6]>0,lane]<=spec['corridor_half_width_m']).all()))
        (folder/'prospective_outcome.json').write_text(json.dumps(dict(decision=decision,observed=selected,completed=manager.completed,events=manager.events),indent=2))
    (a.output/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
(a.output/'status.json').write_text(json.dumps(dict(status='complete',rows=len(records),elapsed_s=time.monotonic()-start,optimizer_updates=0),indent=2))
os.sync();os._exit(0)
