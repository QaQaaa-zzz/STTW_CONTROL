"""Paired deterministic parent/final and zero-middle evaluation on two terrains."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
root=Path.cwd();parent=a.training.resolve()/'policy_000';final=a.training.resolve()/'policy_002';alphas=[0,.25,.5,.75,1]
env=os.environ.copy();env['PATH']='/home/qy/miniconda3/envs/go2_isaacgym/bin:'+env['PATH'];env['LD_LIBRARY_PATH']='/home/qy/miniconda3/envs/go2_isaacgym/lib';env['PYTHONPATH']=str(root/'learning/src');reward_env=env.copy();reward_env.pop('LD_LIBRARY_PATH');reward_env['JAX_PLATFORMS']='cpu'
status=dict(stage='running',pid=os.getpid(),completed=0,physics_steps=0,max_episodes=22,max_physics_steps=550000,seed=20261105,role='paired DEV; trained terrain, single initial state; no optimizer',started=time.time())
for terrain in ['slope','rough_slope']:
    folder=a.output/terrain;folder.mkdir();tile=root/'runs/isaac_training_terrain'/terrain;digest=json.loads((tile/'identity.json').read_text())['sha256']
    cases=[('baseline',parent,.5,True)]+[(method+'_a'+str(alpha),policy,alpha,False) for method,policy in [('parent',parent),('final',final)] for alpha in alphas]
    for name,policy,alpha,baseline in cases:
        receipt=folder/(name+'.json');status.update(terrain=terrain,case=name);(a.output/'status.json').write_text(json.dumps(status,indent=2))
        command=[sys.executable,'learning/cli/isaac_asset_probe.py','--asset',str(root/'runs/isaac_urdf_drive/vehicle.urdf'),'--output',str(receipt),'--steps','25000','--ground','trimesh','--terrain-mesh',str(tile/'terrain.npz'),'--terrain-sha256',digest,'--controller-config',str(root/'runs/isaac_signals_parity/dynamic.json'),'--lower-actor',str(root/'runs/local300_torch_portable_parity/actor.npz'),'--lower-sha256','517f41fe825282ac4d4260a9757802e2258c644a1b235d0c328cd12e7b45f125','--path-bundle',str(policy),'--alpha',str(alpha),'--contact-offset','.01','--rest-offset','0','--seed','20261105']
        if baseline:command.append('--zero-middle')
        with receipt.with_suffix('.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for score_alpha in alphas if baseline else [alpha]:
            prefix=folder/('baseline_a'+str(score_alpha)+'.json') if baseline else receipt
            command=['/home/qy/miniconda3/envs/mjp/bin/python','learning/cli/reconstruct_isaac_reward.py','--receipt',str(receipt),'--bundle',str(policy),'--output-prefix',str(prefix)]
            if baseline:command+=['--alpha',str(score_alpha)]
            with prefix.with_suffix('.reward.log').open('w') as log:subprocess.run(command,env=reward_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['completed']+=1;status['physics_steps']+=json.loads(receipt.read_text())['physics_steps'];(a.output/'status.json').write_text(json.dumps(status,indent=2))
status.update(stage='completed',finished=time.time());(a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
