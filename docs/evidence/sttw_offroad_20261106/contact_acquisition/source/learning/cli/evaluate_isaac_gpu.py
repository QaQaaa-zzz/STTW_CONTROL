"""Collect fixed-alpha H75 and Zero trajectories using the training GPU plant."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--case-manifest',type=Path);p.add_argument('--configured-lanes',action='store_true');p.add_argument('--channel-protocol',type=Path);p.add_argument('--protocol-alpha',type=float,choices=(0,.25,.5,.75,1),default=.5);p.add_argument('--protocol-phase',choices=('entry_projection','exit_projection'),default='entry_projection');p.add_argument('--lower-training-run',type=Path);p.add_argument('--lower-checkpoint',type=Path);p.add_argument('--backend',choices=['gym','lab'],default='gym');p.add_argument('--asset',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=6);a=p.parse_args()
if (a.lower_training_run is None)!=(a.lower_checkpoint is None):p.error('--lower-training-run and --lower-checkpoint are required together')
if a.backend=='gym':
    from isaacgym import gymapi
else:
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.isaac_path_rollout import IsaacPathRollout
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
s=json.loads(a.config.read_text());torch.manual_seed(s.get('seed',1))
if not a.configured_lanes and not a.case_manifest:s.update(num_envs=6,alphas=[0,.25,.5,.75,1,0],zero_middle=[False]*5+[True])
cases=json.loads(a.case_manifest.read_text()) if a.case_manifest else None
if cases:
    from sttw_control.isaac_evaluation_batch import batch_settings,split_case
    s=batch_settings(s,cases)
if not 1<=s['num_envs']<=4096:p.error('evaluation supports 1 to 4096 lanes')
if cases and a.channel_protocol:p.error('case batching is for fixed-case evaluation, not shared-state interventions')
if a.channel_protocol:
    protocol=json.loads(a.channel_protocol.read_text());s.update(num_envs=5,alphas=[a.protocol_alpha]*5,zero_middle=[False]*5,case_labels=protocol['case_labels'],evaluation_role=protocol['evaluation_role'],channel_intervention=dict(progress_m=protocol['fork_phases_m'][a.protocol_phase],hold_s=protocol['hold_s'],hold_channels=protocol['hold_channels'],phase=a.protocol_phase,kind=protocol.get('kind','hold'),post_fork_seconds=protocol.get('post_fork_seconds'),target_alpha=protocol.get('target_alpha'),execution_modes=protocol.get('execution_modes')))
    s['plant_options']['track_wheel_contacts']=protocol.get('track_wheel_contacts',True)
intervention=s.get('channel_intervention')
if intervention and (s.get('randomization') or s['num_envs']!=5 or len(set(s['alphas']))!=1):p.error('channel intervention requires five nominal lanes at one common alpha')
if not cases:s['terrains']={'evaluation':str(a.terrain)}
s.update(diagnostics=True,episode_s=a.seconds)
if a.asset:s['asset']=str(a.asset.resolve())
identity=json.loads((a.bundle/'identity.json').read_text());cfg=json.loads((a.bundle/'config.json').read_text());policy=TorchPathPolicy(a.bundle/'actor.npz',identity['files']['actor.npz'],cfg,s['num_envs'],'cuda');plant=None
if a.backend=='lab':
    from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
    plant=IsaacLabVectorPlant(s['asset'],list(s['terrains'].values()),s.get('lane_terrain_ids',[0]*s['num_envs']),a.output,**s.get('plant_options',{}))
lower=None
if a.lower_checkpoint:
    from sttw_control.torch_go2_lower import TorchGo2Lower
    lower=TorchGo2Lower(a.lower_training_run,a.lower_checkpoint,s['num_envs'],'cuda:0')
env=IsaacPathRollout(a.bundle,s,policy,plant=plant,lower=lower)
if s.get('fixed_randomization_groups'):
    from sttw_control.isaaclab_randomization import apply_fixed_evaluation_groups
    apply_fixed_evaluation_groups(env,s['fixed_randomization_groups'])
if intervention:env.policy.channel_intervention_mode=intervention.get('kind','hold')
contract=dict(backend=a.backend,settings=s,bundle=str(a.bundle.resolve()),identity=identity,terrain=env.plant.terrain_identity,bodies=env.plant.bodies,origin=env.plant.origin.cpu().tolist(),role=s.get('evaluation_role','fixed preference DEV; Zero is frozen LOCAL300 plus ECBC/ESO, no H75 correction'),physics_dt=env.physics_dt,lower_dt=.005,policy_dt=.02)
if lower is not None:
    contract['lower_identity']=lower.identity
    contract['role']='fixed preference DEV; Zero uses the same selected frozen lower plus ECBC/ESO, with no H75 correction'
if a.backend=='lab':contract['lab_physics']=env.plant.metadata
if env.randomization is not None:
    contract['randomization_readback']=env.randomization_properties
    contract['delay_steps']={key:value.time_lags.cpu().tolist() for key,value in env.randomization.buffers.items()}
(a.output/'contract.json').write_text(json.dumps(contract,indent=2));rows=[];detail=[];row_chunks=[];detail_chunks=[];started=time.time()
def transfer(records):
    return {key:torch.stack([item[key] for item in records]).cpu().numpy() for key in records[0]}
planner=None
if s.get('replanning'):
    from sttw_control.goal_replanning import IsaacGoalReplanning
    planner=IsaacGoalReplanning(env,s['replanning'],a.output)
    (a.output/'replanning_decisions.json').write_text('[]')
if s.get('physics_contact_recording'):
    from sttw_control.physics_contact_recording import PhysicsContactRecorder
    env.physics_recorder=PhysicsContactRecorder(a.output/'physics_contacts',env,s['physics_contact_recording'])
fork_step=None
for step in range(env.horizon):
    if intervention:
        if fork_step is None and bool(env.alive[0]) and float(env.middle.projection['progress'][0])>=intervention['progress_m']:
            from sttw_control.isaac_path_snapshot import capture,restore
            state=capture(env,0);restore(env,state,[1,2,3,4]);fork_step=step
            if intervention.get('kind')=='switch':
                target=torch.tensor(intervention['target_alpha'],device=env.device)
                env.switched.copy_(target!=env.alpha);env.switch_tick[env.switched]=env.tick*4
                env.alpha.copy_(target);env.policy.execution_mode.copy_(torch.tensor(intervention['execution_modes'],device=env.device))
            np.savez_compressed(a.output/'fork_snapshot.npz',**{key:value.cpu().numpy() for key,value in state.items()})
            (a.output/'fork.json').write_text(json.dumps(dict(policy_step=step,time_s=step*.02,source_lane=0,restored_lanes=[1,2,3,4],fields=list(state),root_pose='link pose wxyz in local terrain coordinates',root_velocity='root COM linear/angular world velocity',signal_root='root link state xyzw; local pose, world velocity; policy signal cache',physical_caches_copied=False,analysis_window='source lane defines prefix; compare all branches only after fork; sibling pre-fork traces retained but not branch outcomes',protocol=intervention),indent=2))
        if fork_step is not None:
            active=(step-fork_step)*.02<intervention['hold_s'];env.policy.hold_channels.copy_(torch.tensor(intervention['hold_channels'],device=env.device,dtype=torch.bool) if active else torch.zeros_like(env.policy.hold_channels))
    if planner is not None:planner.update()
    result=env.step(None)
    rows.append({key:result[key] for key in ['reward','valid','done','failure','lower_trace']})
    detail.extend(result['diagnostics'])
    if (step+1)%50==0:
        row_chunks.append(transfer(rows));detail_chunks.append(transfer(detail));rows.clear();detail.clear()
    if (step+1)%50==0:print(json.dumps(dict(step=step+1,alive=int(env.alive.sum()),seconds=time.time()-started)),flush=True)
    if intervention and fork_step is not None and intervention.get('post_fork_seconds') is not None and (step+1-fork_step)*.02>=intervention['post_fork_seconds']:break
if env.physics_recorder is not None:env.physics_recorder.close()
if rows:
    row_chunks.append(transfer(rows));detail_chunks.append(transfer(detail))
arrays={key:np.concatenate([chunk[key] for chunk in chunks]) for chunks in (row_chunks,detail_chunks) for key in chunks[0]}
summary=dict(post_fork_failures=arrays['failure'][fork_step:].sum(0).tolist() if fork_step is not None else None,fork_policy_step=fork_step,stage='complete',policy_steps=len(arrays['reward']),physics_steps=len(arrays['reward'])*s['num_envs']*4*env.physics_substeps,elapsed_s=time.time()-started,failures=arrays['failure'].sum(0).tolist(),valid_steps=arrays['valid'].sum(0).tolist(),push_events=env.push_events)
if cases:
    for index,case in enumerate(cases):
        run=Path(case['output']);run.mkdir(parents=True,exist_ok=False)
        part,part_contract=split_case(arrays,contract,{**case,'batch_output':str(a.output.resolve())},index)
        np.savez_compressed(run/'trajectory.npz',**part)
        (run/'contract.json').write_text(json.dumps(part_contract,indent=2))
        part_summary=dict(summary,physics_steps=len(part['reward'])*6*4*env.physics_substeps,failures=part['failure'].sum(0).tolist(),valid_steps=part['valid'].sum(0).tolist(),elapsed_scope='shared batch elapsed, not additive per case')
        (run/'status.json').write_text(json.dumps(part_summary,indent=2))
else:
    np.savez_compressed(a.output/'trajectory.npz',**arrays)
summary['elapsed_with_output_s']=time.time()-started
(a.output/'status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if a.backend=='lab':app.close(wait_for_replicator=False,skip_cleanup=True)
else:env.plant.close()
