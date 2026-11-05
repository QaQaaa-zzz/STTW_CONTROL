"""Collect fixed-alpha H75 and Zero trajectories using the training GPU plant."""
import argparse,json,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['gym','lab'],default='gym');p.add_argument('--asset',type=Path);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=6);a=p.parse_args()
if a.backend=='gym':
    from isaacgym import gymapi
else:
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
import torch
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.isaac_path_rollout import IsaacPathRollout
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(1)
s=json.loads(a.config.read_text());s.update(num_envs=6,alphas=[0,.25,.5,.75,1,0],terrains={'bend_bump':str(a.terrain)},zero_middle=[False]*5+[True],diagnostics=True,episode_s=a.seconds)
if a.asset:s['asset']=str(a.asset.resolve())
identity=json.loads((a.bundle/'identity.json').read_text());cfg=json.loads((a.bundle/'config.json').read_text());policy=TorchPathPolicy(a.bundle/'actor.npz',identity['files']['actor.npz'],cfg,6,'cuda');plant=None
if a.backend=='lab':
    from sttw_control.isaaclab_vector_env import IsaacLabVectorPlant
    plant=IsaacLabVectorPlant(s['asset'],list(s['terrains'].values()),[0]*s['num_envs'],a.output)
env=IsaacPathRollout(a.bundle,s,policy,plant=plant)
contract=dict(backend=a.backend,settings=s,bundle=str(a.bundle.resolve()),identity=identity,terrain=env.plant.terrain_identity,bodies=env.plant.bodies,origin=env.plant.origin.cpu().tolist(),role='fixed preference DEV; Zero is frozen LOCAL300 plus ECBC/ESO, no H75 correction',physics_dt=.0002,lower_dt=.005,policy_dt=.02)
if a.backend=='lab':contract['lab_physics']=env.plant.metadata
(a.output/'contract.json').write_text(json.dumps(contract,indent=2));rows=[];detail=[];started=time.time()
for step in range(env.horizon):
    result=env.step(None)
    rows.append({key:result[key].cpu().numpy() for key in ['reward','valid','done','failure','lower_trace']})
    detail.extend({key:value.cpu().numpy() for key,value in item.items()} for item in result['diagnostics'])
    if (step+1)%50==0:print(json.dumps(dict(step=step+1,alive=int(env.alive.sum()),seconds=time.time()-started)),flush=True)
np.savez_compressed(a.output/'trajectory.npz',**{key:np.stack([r[key] for r in rows]) for key in rows[0]},**{key:np.stack([r[key] for r in detail]) for key in detail[0]})
summary=dict(stage='complete',policy_steps=env.horizon,physics_steps=env.horizon*6*100,elapsed_s=time.time()-started,failures=np.stack([r['failure'] for r in rows]).sum(0).tolist(),valid_steps=np.stack([r['valid'] for r in rows]).sum(0).tolist())
(a.output/'status.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));env.plant.close()

if a.backend=='lab':app.close()
