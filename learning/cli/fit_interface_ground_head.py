"""Fit one bounded ground-only outcome branch; freeze both handoff interfaces."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sttw_control.interface_world_model import network

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
start = time.monotonic()
torch.set_num_threads(2)
contract = dict(
    change='Independent ground-only BCE fitting and DEV Brier selection; frozen upstream/downstream/direct models',
    budget='3 seeds x250 CPU epochs;0 physics/0 PPO;30-minute stop threshold',
    roles='Original TRAIN/DEV splits only; historical transfer is reported, not selected; previous online30 roots not used',
    source=str(a.source), seeds=[71201, 71202, 71203],
    architecture='582-128-64-1 SiLU;Adam lr.001 weight_decay.0001',
    selection='Ground-only grouped DEV Brier; no search over architectures or budgets',
)
(a.output/'ground_declaration.json').write_text(json.dumps(contract, indent=2))
meta = json.loads((a.source/'model.json').read_text())
meta['ground_directory'] = str(a.output.resolve())
meta['ground_fit'] = contract
(a.output/'model.json').write_text(json.dumps(meta, indent=2))
for path in list(a.source.glob('upstream_*.pt')) + list(a.source.glob('direct_*.pt')):
    (a.output/path.name).symlink_to(path.resolve())
z = np.load(a.source/'dataset.npz')
x, y = z['x'], z['goal']
ground = ~z['jump']
train, dev = z['train'] & ground, z['dev'] & ground
mean = x[train].mean(0)
std = x[train].std(0).clip(.01)
xx, yy = torch.tensor((x-mean)/std), torch.tensor(y)
values, selected = [], {}
for seed in contract['seeds']:
    torch.manual_seed(seed)
    model = network(x.shape[1], 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    best, curve = float('inf'), []
    for epoch in range(1, 251):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xx[train])[:,0], yy[train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(xx)[:,0]).numpy()
        score = float(np.mean((probabilities[dev]-y[dev])**2))
        curve.append([epoch, float(loss.detach()), score])
        if score < best:
            best, chosen = score, epoch
            state = {k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    with torch.no_grad():
        values.append(torch.sigmoid(model(xx)[:,0]).numpy())
    torch.save(dict(state_dict=state,input_mean=mean,input_std=std,seed=seed,selected_epoch=chosen), a.output/f'ground_{seed}.pt')
    (a.output/f'ground_{seed}_curve.json').write_text(json.dumps(curve))
    selected[str(seed)] = dict(epoch=chosen, dev_brier=best)
pred = np.mean(values, axis=0)
old = np.load(a.source/'predictions.npz')['interface']
cases = json.loads((a.source/'results.json').read_text())['cases']
groups=[]
for name, mask in [('TRAIN',train),('DEV',dev),('historical_source240',ground & np.array([r['dataset']=='source240' for r in cases]))]:
    groups.append(dict(role=name,n=int(mask.sum()),goals=int(y[mask].sum()),
        shared_brier=float(np.mean((old[mask]-y[mask])**2)),
        independent_brier=float(np.mean((pred[mask]-y[mask])**2))))
actions=[]
for candidate in sorted({r['candidate'] for r,m in zip(cases,train) if m}):
    mask=train & np.array([r['candidate']==candidate for r in cases])
    actions.append(dict(candidate=candidate,n=int(mask.sum()),goals=int(y[mask].sum()),
        shared_mean=float(old[mask].mean()),independent_mean=float(pred[mask].mean())))
np.savez_compressed(a.output/'ground_predictions.npz',probability=pred,old=old,goal=y,ground=ground,train=train,dev=dev)
result=dict(contract=contract,selection=selected,groups=groups,train_actions=actions,elapsed_s=time.monotonic()-start)
(a.output/'ground_results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
