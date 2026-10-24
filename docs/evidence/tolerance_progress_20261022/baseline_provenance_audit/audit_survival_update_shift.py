"""Describe matched candidate Gaussian shifts on their authenticated training batch."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.preference_best import digest,atomic_json
from sttw_control.path_command_policy import make_torch_path_network

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
output=root/'sampling_shift.json'
if output.exists():raise FileExistsError('preserve previous sampling audit')
plan=json.loads((root/'plan.json').read_text());results=json.loads((root/'results.json').read_text())
collection=Path(plan['collection']);original=json.loads((collection/'plan.json').read_text())
cfg=json.loads(Path(original['source_declaration']).read_text())['config']
assert digest(plan['model']['checkpoint'])==plan['model']['checkpoint_sha256']
parent=torch.load(plan['model']['checkpoint'],map_location='cpu',weights_only=False)
old=make_torch_path_network(351,2,cfg);old.load_state_dict({k[6:]:v for k,v in parent['policy'].items() if k.startswith('actor.')})
rows=[];hashes={}
for arm in results['arms']:
    path=root/arm['arm']/'checkpoint.pt';assert digest(path)==arm['checkpoint_sha256'];hashes[str(path)]=digest(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    actor=make_torch_path_network(351,2,cfg);actor.load_state_dict(checkpoint['actor'])
    for case in json.loads((collection/'results.json').read_text())['cases']:
        raw=collection/f"case_{case['case']:03d}.npz";assert digest(raw)==case['raw_sha256'];hashes[str(raw)]=digest(raw)
        with np.load(raw) as z:
            valid=z['decision_valid'];obs=torch.tensor(z['decision_observation'][valid]);actions=torch.tensor(z['decision_raw_action'][valid]);saved=torch.tensor(z['decision_log_prob'][valid])
        with torch.no_grad():
            source=torch.distributions.Normal(old(obs),parent['policy']['log_std'].exp())
            target=torch.distributions.Normal(actor(obs),checkpoint['log_std'].exp())
            old_logp=source.log_prob(actions).sum(-1);new_logp=target.log_prob(actions).sum(-1)
            assert float((old_logp-saved).abs().max())<1e-3
            kl=torch.distributions.kl_divergence(source,target).sum(-1)
            ratio=(new_logp-saved).exp()
            rows.append(dict(arm=arm['arm'],case=case['case'],legal_actions=int(valid.sum()),
                exact_diagonal_Gaussian_KL_parent_to_candidate_mean=float(kl.mean()),KL_max=float(kl.max()),
                likelihood_ratio_outside_clip_fraction=float(((ratio<.8)|(ratio>1.2)).float().mean()),
                raw_mean_shift_rms=float((target.loc-source.loc).square().mean().sqrt()),
                source_std=source.scale[0].tolist(),candidate_std=target.scale[0].tolist()))
atomic_json(output,dict(rows=rows,input_sha256=hashes,source_sha256=digest(__file__),new_physics_ticks=0,optimizer_steps=0,
    scope='Train-state local distribution shift only; no new acceptance gate, risk calibration or rollout efficacy inference.'))
print(json.dumps(rows))
