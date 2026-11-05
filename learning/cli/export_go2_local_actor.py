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
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
if a.h75_reference_bundle:
    reference=a.h75_reference_bundle
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    state=checkpoint['model_state_dict']
    arrays={}
    for i,index in enumerate((0,2,4,6)):
        arrays[f'Dense_{i}_kernel']=state[f'actor.{index}.weight'].numpy().T
        arrays[f'Dense_{i}_bias']=state[f'actor.{index}.bias'].numpy()
    for name in ('gain_2','shift_2','gain_3','shift_3'):arrays[name]=state['actor.'+name].numpy()
    np.savez(a.output/'actor.npz',**arrays)
    for name in ('config.json','route.npz'):shutil.copyfile(reference/name,a.output/name)
    identity=json.loads((reference/'identity.json').read_text())
    identity['files']={name:hashlib.sha256((a.output/name).read_bytes()).hexdigest() for name in ('actor.npz','config.json','route.npz')}
    infos=checkpoint.get('infos') or {}
    if 'completed_updates' in infos:completed=infos['completed_updates']
    elif a.checkpoint.stem=='model_initial':completed=0
    else:
        label=int(a.checkpoint.stem.removeprefix('model_'))
        budget=json.loads((a.training_run/'config.json').read_text())['task']['updates']
        completed=label if label==budget else label+1
    identity['native_checkpoint']=dict(path=str(a.checkpoint.resolve()),sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),training_run=str(a.training_run.resolve()),raw_upstream_iteration=checkpoint['iter'],completed_updates=completed,infos=infos,log_std=state['log_std'].tolist())
    identity['role']='native Go2 H75 candidate export; frozen original route/config/lower; no physics or optimizer'
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2))
    print('Exported native H75 Actor into existing fixed-case bundle')
    raise SystemExit(0)
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
