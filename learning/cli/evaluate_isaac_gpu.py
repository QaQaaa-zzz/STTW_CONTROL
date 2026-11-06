"""Collect fixed-alpha H75 and Zero trajectories using the training GPU plant."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--configured-lanes',action='store_true');p.add_argument('--channel-protocol',type=Path);p.add_argument('--protocol-alpha',type=float,choices=(0,.25,.5,.75,1),default=.5);p.add_argument('--protocol-phase',choices=('entry_projection','exit_projection'),default='entry_projection');p.add_argument('--lower-training-run',type=Path);p.add_argument('--lower-checkpoint',type=Path);p.add_argument('--backend',choices=['gym','lab'],default='gym');p.add_argument('--asset',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=6);a=p.parse_args()
if (a.lower_training_run is None)!=(a.lower_checkpoint is None):p.error('--lower-training-run and --lower-checkpoint are required together')
if a.backend=='gym':
    from isaacgym import gymapi
else:
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.isaac_path_rollout import IsaacPathRollout
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(1)
s=json.loads(a.config.read_text())
if not a.configured_lanes:s.update(num_envs=6,alphas=[0,.25,.5,.75,1,0],zero_middle=[False]*5+[True])
if not 1<=s['num_envs']<=15:p.error('diagnostic evaluation supports 1 to 15 lanes')
if a.channel_protocol:
    protocol=json.loads(a.channel_protocol.read_text());s.update(num_envs=5,alphas=[a.protocol_alpha]*5,zero_middle=[False]*5,case_labels=protocol['case_labels'],evaluation_role=protocol['evaluation_role'],channel_intervention=dict(progress_m=protocol['fork_phases_m'][a.protocol_phase],hold_s=protocol['hold_s'],hold_channels=protocol['hold_channels'],phase=a.protocol_phase,kind=protocol.get('kind','hold'),post_fork_seconds=protocol.get('post_fork_seconds'),target_alpha=protocol.get('target_alpha'),execution_modes=protocol.get('execution_modes')))
    s['plant_options']['track_wheel_contacts']=protocol.get('track_wheel_contacts',True)
intervention=s.get('channel_intervention')
if intervention and (s.get('randomization') or s['num_envs']!=5 or len(set(s['alphas']))!=1):p.error('channel intervention requires five nominal lanes at one common alpha')
s.update(terrains={'evaluation':str(a.terrain)},diagnostics=True,episode_s=a.seconds)
if a.asset:s['asset']=str(a.asset.resolve())
identity=json.loads((a.bundle/'identity.json').read_text());cfg=json.loads((a.bundle/'config.json').read_text());policy=TorchPathPolicy(a.bundle/'actor.npz',identity['files']['actor.npz'],cfg,s['num_envs'],'cuda');plant=None
if a.backend=='lab':
    from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
    plant=IsaacLabVectorPlant(s['asset'],list(s['terrains'].values()),[0]*s['num_envs'],a.output,**s.get('plant_options',{}))
lower=None
if a.lower_checkpoint:
    from sttw_control.torch_go2_lower import TorchGo2Lower
    lower=TorchGo2Lower(a.lower_training_run,a.lower_checkpoint,s['num_envs'],'cuda:0')
env=IsaacPathRollout(a.bundle,s,policy,plant=plant,lower=lower)
if intervention:env.policy.channel_intervention_mode=intervention.get('kind','hold')
contract=dict(backend=a.backend,settings=s,bundle=str(a.bundle.resolve()),identity=identity,terrain=env.plant.terrain_identity,bodies=env.plant.bodies,origin=env.plant.origin.cpu().tolist(),role=s.get('evaluation_role','fixed preference DEV; Zero is frozen LOCAL300 plus ECBC/ESO, no H75 correction'),physics_dt=.0002,lower_dt=.005,policy_dt=.02)
if lower is not None:
    contract['lower_identity']=lower.identity
    contract['role']='fixed preference DEV; Zero uses the same selected frozen lower plus ECBC/ESO, with no H75 correction'
if a.backend=='lab':contract['lab_physics']=env.plant.metadata
if env.randomization is not None:
    contract['randomization_readback']=env.randomization_properties
    contract['delay_steps']={key:value.time_lags.cpu().tolist() for key,value in env.randomization.buffers.items()}
(a.output/'contract.json').write_text(json.dumps(contract,indent=2));rows=[];detail=[];started=time.time()
planner=None
if s.get('replanning'):
    from sttw_control.goal_replanning import IsaacGoalReplanning
    planner=IsaacGoalReplanning(env,s['replanning'],a.output)
    (a.output/'replanning_decisions.json').write_text('[]')
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
    rows.append({key:result[key].cpu().numpy() for key in ['reward','valid','done','failure','lower_trace']})
    detail.extend({key:value.cpu().numpy() for key,value in item.items()} for item in result['diagnostics'])
    if (step+1)%50==0:print(json.dumps(dict(step=step+1,alive=int(env.alive.sum()),seconds=time.time()-started)),flush=True)
    if intervention and fork_step is not None and intervention.get('post_fork_seconds') is not None and (step+1-fork_step)*.02>=intervention['post_fork_seconds']:break
np.savez_compressed(a.output/'trajectory.npz',**{key:np.stack([r[key] for r in rows]) for key in rows[0]},**{key:np.stack([r[key] for r in detail]) for key in detail[0]})
summary=dict(post_fork_failures=np.stack([r['failure'] for r in rows[fork_step:]]).sum(0).tolist() if fork_step is not None else None,fork_policy_step=fork_step,stage='complete',policy_steps=len(rows),physics_steps=len(rows)*s['num_envs']*100,elapsed_s=time.time()-started,failures=np.stack([r['failure'] for r in rows]).sum(0).tolist(),valid_steps=np.stack([r['valid'] for r in rows]).sum(0).tolist(),push_events=env.push_events)
(a.output/'status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if a.backend=='lab':app.close(wait_for_replicator=False,skip_cleanup=True)
else:env.plant.close()
