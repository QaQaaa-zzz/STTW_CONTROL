"""Export native Go2 weights through the existing Torch-to-Flax layer exporter."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_go2_lower import TorchGo2Lower
from sttw_control.lower_command_validation import export_local_actor

p=argparse.ArgumentParser()
p.add_argument('--training-run',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
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
