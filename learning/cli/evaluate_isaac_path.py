"""Stage-end paired evaluation with verified reuse of immutable trajectories."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
import numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
root=Path.cwd();spec=json.loads(a.config.read_text());training=a.training.resolve();training_status=json.loads((training/'status.json').read_text())
if training_status['stage']!='training_complete' and not a.dry_run:raise ValueError('stage-end evaluation requires completed training')
parent=(root/spec['reference_bundle']).resolve()
final=Path(json.loads((training/'best_training.json').read_text())['policy']) if spec['selection']=='best_training' else Path(training_status['final_policy'])
for bundle in (parent,final):
 identity=json.loads((bundle/'identity.json').read_text())
 for name,digest in identity['files'].items():
  if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=digest:raise ValueError('bundle identity mismatch: '+str(bundle/name))
alphas=spec['alphas'];expected_config=json.loads((root/'runs/isaac_signals_parity/dynamic.json').read_text());asset=root/'runs/isaac_urdf_drive/vehicle.urdf';asset_sha=hashlib.sha256(asset.read_bytes()).hexdigest()
plan=[]
for terrain,tile_name in spec['terrains'].items():
 tile=root/tile_name;digest=json.loads((tile/'identity.json').read_text())['sha256']
 for name,policy,alpha,baseline in [('baseline',parent,.5,True)]+[(method+'_a'+str(alpha),policy,alpha,False) for method,policy in [('parent',parent),('final',final)] for alpha in alphas]:
  old=root/spec['reuse_run']/terrain/(name+'.json');reuse=False
  if not name.startswith('final') and old.exists():
   receipt=json.loads(old.read_text());identity=json.loads((policy/'identity.json').read_text())
   conditions=[receipt['asset_sha256']==asset_sha,receipt['controller_config']==expected_config,receipt['requested_physics_steps']==spec['physics_steps'],np.isclose(receipt['dt_s'],.0002,rtol=0,atol=1e-11),receipt['drive_speed']==2.3,receipt['terrain_mesh']['sha256']==digest,receipt['path_contract']['alpha']==alpha,receipt['path_contract']['zero_middle']==baseline,receipt['path_contract']['bundle']['files']==identity['files'],receipt['physx_config']['contact_offset']==spec['contact_offset'],receipt['physx_config']['rest_offset']==spec['rest_offset']]
   # Float32 PhysX readback is compared with its declared value, not string rounding.
   conditions[-2:]=[np.isclose(receipt['physx_config']['contact_offset'],spec['contact_offset'],rtol=0,atol=1e-9),np.isclose(receipt['physx_config']['rest_offset'],spec['rest_offset'],rtol=0,atol=1e-9)]
   conditions += [receipt['sampling']==dict(seed=spec['seed'],policy_std=None),receipt['lower_actor']['sha256']=='517f41fe825282ac4d4260a9757802e2258c644a1b235d0c328cd12e7b45f125']
   if not all(conditions):raise ValueError('cached case contract mismatch: '+str(old))
   with np.load(old.with_suffix('.npz')) as data:
    if len(data['body_states'])!=receipt['physics_steps']:raise ValueError('incomplete cached physics')
   reuse=True
  plan.append(dict(terrain=terrain,tile=str(tile),digest=digest,name=name,policy=str(policy),alpha=alpha,baseline=baseline,reuse=str(old) if reuse else None))
manifest=dict(reference_bundle=str(parent),candidate_bundle=str(final),selection=spec['selection'],training_complete=training_status['stage']=='training_complete',cases=plan,new_episodes=sum(r['reuse'] is None for r in plan),reused_episodes=sum(r['reuse'] is not None for r in plan),max_new_physics_steps=sum(r['reuse'] is None for r in plan)*spec['physics_steps'])
if a.dry_run:print(json.dumps(manifest,indent=2));sys.exit(0)
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
env=os.environ.copy();env['PATH']='/home/qy/miniconda3/envs/go2_isaacgym/bin:'+env['PATH'];env['LD_LIBRARY_PATH']='/home/qy/miniconda3/envs/go2_isaacgym/lib';env['PYTHONPATH']=str(root/'learning/src');reward_env=env.copy();reward_env.pop('LD_LIBRARY_PATH');reward_env['JAX_PLATFORMS']='cpu'
status=dict(stage='running',pid=os.getpid(),completed=0,physics_steps=0,reused_physics_steps=0,max_episodes=len(plan),max_physics_steps=manifest['max_new_physics_steps'],seed=spec['seed'],role='paired DEV; trained terrain, single initial state; no optimizer',started=time.time())
for case in plan:
        terrain=case['terrain'];name=case['name'];policy=Path(case['policy']);alpha=case['alpha'];baseline=case['baseline'];tile=Path(case['tile']);digest=case['digest']
        folder=a.output/terrain;folder.mkdir(exist_ok=True)
        receipt=folder/(name+'.json');status.update(terrain=terrain,case=name);(a.output/'status.json').write_text(json.dumps(status,indent=2))
        if case['reuse']:
            old=Path(case['reuse'])
            for suffix in ['.json','.npz']:shutil.copy2(old.with_suffix(suffix),receipt.with_suffix(suffix))
            status['reused_physics_steps']+=json.loads(receipt.read_text())['physics_steps']
        command=[sys.executable,'learning/cli/isaac_asset_probe.py','--asset',str(root/'runs/isaac_urdf_drive/vehicle.urdf'),'--output',str(receipt),'--steps',str(spec['physics_steps']),'--ground','trimesh','--terrain-mesh',str(tile/'terrain.npz'),'--terrain-sha256',digest,'--controller-config',str(root/'runs/isaac_signals_parity/dynamic.json'),'--lower-actor',str(root/'runs/local300_torch_portable_parity/actor.npz'),'--lower-sha256','517f41fe825282ac4d4260a9757802e2258c644a1b235d0c328cd12e7b45f125','--path-bundle',str(policy),'--alpha',str(alpha),'--contact-offset',str(spec['contact_offset']),'--rest-offset',str(spec['rest_offset']),'--seed',str(spec['seed'])]
        if baseline:command.append('--zero-middle')
        if not case['reuse']:
            with receipt.with_suffix('.log').open('w') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for score_alpha in alphas if baseline else [alpha]:
            prefix=folder/('baseline_a'+str(score_alpha)+'.json') if baseline else receipt
            command=['/home/qy/miniconda3/envs/mjp/bin/python','learning/cli/reconstruct_isaac_reward.py','--receipt',str(receipt),'--bundle',str(policy),'--output-prefix',str(prefix)]
            if baseline:command+=['--alpha',str(score_alpha)]
            with prefix.with_suffix('.reward.log').open('w') as log:subprocess.run(command,env=reward_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        status['completed']+=1;status['physics_steps']+=0 if case['reuse'] else json.loads(receipt.read_text())['physics_steps'];(a.output/'status.json').write_text(json.dumps(status,indent=2))
status.update(stage='completed',finished=time.time());(a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
