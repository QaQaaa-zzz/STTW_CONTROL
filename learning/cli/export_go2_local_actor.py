"""Export native Go2 weights through the existing Torch-to-Flax layer exporter."""
import argparse
import hashlib
import shutil
import json
import pickle
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--training-run',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--h75-reference-bundle',type=Path,help='Export native H75 checkpoint into the existing fixed-route evaluation bundle')
p.add_argument('--portable-local',action='store_true',help='Export Local300-shaped actor directly as CPU NPZ for existing Isaac ground evaluator')
a=p.parse_args()
if a.portable_local:
    a.output.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=checkpoint['model_state_dict'];arrays={}
    for i,index in enumerate((0,2,4)):
        arrays[f'Dense_{i}_kernel']=state[f'actor.{index}.weight'].numpy().T
        arrays[f'Dense_{i}_bias']=state[f'actor.{index}.bias'].numpy()
    np.savez(a.output/'actor.npz',**arrays)
    identity=dict(path=str(a.output/'actor.npz'),sha256=hashlib.sha256((a.output/'actor.npz').read_bytes()).hexdigest(),
        source_checkpoint=str(a.checkpoint.resolve()),source_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        training_run=str(a.training_run.resolve()),raw_upstream_iteration=checkpoint['iter'],infos=checkpoint.get('infos'),
        actor_dim=210,hidden_sizes=[256,128],activation='elu',normalized_action='tanh logits',
        physics_steps=0,optimizer_updates=0,role='unchanged Local300-shaped actor export; not task qualification')
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2))
    print(json.dumps(identity,indent=2));raise SystemExit(0)
if a.h75_reference_bundle:
    reference=a.h75_reference_bundle
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=checkpoint['model_state_dict']
    cfg=json.loads((reference/'config.json').read_text())
    training_path=a.training_run/'config.json'
    task=json.loads(training_path.read_text())['task'] if training_path.exists() else {}
    reference_arch=cfg['network'].get('architecture', 'legacy')
    architecture=task.get('actor_architecture', reference_arch)
    if checkpoint.get('actor_architecture', architecture) != architecture:
        raise ValueError('Training and checkpoint actor architectures differ')
    modulation=('gain_2','shift_2','gain_3','shift_3')
    if architecture not in ('legacy','alpha_modulated_v1'):
        raise ValueError(f'Unsupported actor architecture: {architecture}')
    expected_modulation=modulation if architecture=='alpha_modulated_v1' else ()
    actual_modulation={name for name in modulation if 'actor.'+name in state}
    if actual_modulation != set(expected_modulation):
        raise ValueError(f'Checkpoint modulation parameters differ from architecture {architecture}')
    channel=task.get('reference_action_channel')
    if channel not in (None,'speed','steer') or checkpoint.get('reference_action_channel')!=channel:
        raise ValueError('Training and checkpoint reference action channels differ')
    arrays={}
    for i,index in enumerate((0,2,4,6)):
        arrays[f'Dense_{i}_kernel']=state[f'actor.{index}.weight'].numpy().T
        arrays[f'Dense_{i}_bias']=state[f'actor.{index}.bias'].numpy()
    for name in expected_modulation:
        arrays[name]=state['actor.'+name].numpy()
    if channel is not None:
        disabled=1 if channel=='speed' else 0
        arrays['Dense_3_kernel'][:,disabled]=0
        arrays['Dense_3_bias'][disabled]=0
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez(a.output/'actor.npz',**arrays)
    shutil.copyfile(reference/'route.npz',a.output/'route.npz')
    if architecture!=reference_arch or cfg.get('reference_action_channel')!=channel:
        cfg['network']['architecture']=architecture
        if channel is None:
            cfg.pop('reference_action_channel',None)
        else:
            cfg['reference_action_channel']=channel
        (a.output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    else:
        shutil.copyfile(reference/'config.json',a.output/'config.json')
    identity=json.loads((reference/'identity.json').read_text())
    identity['files']={name:hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in ('actor.npz','config.json','route.npz')}
    infos=checkpoint.get('infos') or {}
    if 'completed_updates' in infos:completed=infos['completed_updates']
    elif a.checkpoint.stem=='model_initial':completed=0
    else:
        label=int(a.checkpoint.stem.removeprefix('model_'))
        budget=task['updates']
        completed=label if label==budget else label+1
    identity['native_checkpoint']=dict(path=str(a.checkpoint.resolve()),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),training_run=str(a.training_run.resolve()),raw_upstream_iteration=checkpoint['iter'],completed_updates=completed,infos=infos,log_std=state['log_std'].tolist())
    identity['actor_architecture']=architecture
    identity['training_critic_architecture']=checkpoint.get('critic_architecture',reference_arch)
    identity['training_initialization']=task.get('policy_initialization')
    identity['reference_config_sha256']=hashlib.sha256((reference/'config.json').read_bytes()).hexdigest()
    if channel is not None:
        identity['reference_action_channel']=channel
        identity['exported_channel_semantics']='One Gaussian action during training; disabled exported output is identically zero.'
    identity['reference_bundle_source_actor'] = identity.get('source_actor')
    identity['source_actor'] = dict(source=str(a.checkpoint.resolve()),
        source_sha256=identity['native_checkpoint']['sha256'], artifact='actor.npz',
        artifact_sha256=identity['files']['actor.npz'],
        conversion='native actor arrays transposed into frozen evaluation format')
    identity['role']='native Go2 path actor export; frozen original route and lower controller; no physics or optimizer'
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2))
    print(f'Exported native {architecture} actor into the fixed-route bundle')
    raise SystemExit(0)
a.output.mkdir(parents=True,exist_ok=False)
from sttw_control.torch_go2_lower import TorchGo2Lower
from sttw_control.lower_command_validation import export_local_actor
lower=TorchGo2Lower(a.training_run,a.checkpoint,1,'cuda:0')
params=export_local_actor(lower.model)
with (a.output/'actor.pkl').open('wb') as f:pickle.dump(params,f)
rng=np.random.default_rng(501)
obs=np.clip(rng.normal(0,2,(2048,210)),-5,5).astype(np.float32)
obs[:,-10:]=rng.integers(0,2,(2048,10));obs[0]=0
with torch.no_grad():
 logits=lower.model.act_inference(torch.as_tensor(obs,device='cuda:0'))
 action=torch.tanh(logits)
np.savez_compressed(a.output/'inference_reference.npz',observation=obs,logits=logits.cpu().numpy(),action=action.cpu().numpy())
identity={**lower.identity,'hidden_sizes':lower.identity['policy']['actor_hidden_dims'],'actor_dim':210,'activation':'elu','normalized_action':'tanh logits','jax_default_matmul_precision':'highest','physics_steps':0,'optimizer_updates':0,'role':'conversion/inference check only; candidate not adopted'}
(a.output/'identity.json').write_text(json.dumps(identity,indent=2))
print('Exported unchanged native checkpoint with existing export_local_actor')
