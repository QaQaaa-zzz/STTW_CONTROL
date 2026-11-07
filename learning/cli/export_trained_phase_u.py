"""Export an explicitly selected Go2-runner Phase U checkpoint for frozen evaluation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.phase_u_actor_critic import PhaseUActorCritic
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--source-bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
model=PhaseUActorCritic(checkpoint['model_state_dict']['mean'].numel(),106,4,a.source_bundle)
model.load_state_dict(checkpoint['model_state_dict']);model.eval()
a.output.mkdir(parents=True,exist_ok=False)
arrays={k:getattr(model,k).detach().numpy() for k in ('mean','std','critic_mean','critic_std')}
for prefix,network in (('',model.actor),('critic_',model.critic)):
    for i,layer in enumerate(layer for layer in network if isinstance(layer,torch.nn.Linear)):
        arrays[f'{prefix}kernel_{i}']=layer.weight.detach().numpy().T
        arrays[f'{prefix}bias_{i}']=layer.bias.detach().numpy()
np.savez(a.output/'actor.npz',**arrays)
identity=dict(model.identity)
identity.update(skill_id=a.output.name,source_bundle=str(a.source_bundle.resolve()),training_checkpoint=str(a.checkpoint.resolve()),training_checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),stored_checkpoint_iter=checkpoint.get('iter'),scored_policy_update=(checkpoint.get('infos') or {}).get('update'),selection_evidence=checkpoint.get('infos'),artifact_sha256=hashlib.sha256((a.output/'actor.npz').read_bytes()).hexdigest(),conversion='exact selected Torch actor/critic/normalizer arrays; no extra optimization',source_note='source records identify original Phase U initialization; training_checkpoint identifies current weights')
(a.output/'identity.json').write_text(json.dumps(identity,indent=2));print(json.dumps({k:identity[k] for k in ('skill_id','stored_checkpoint_iter','scored_policy_update','training_checkpoint','artifact_sha256')},indent=2))
