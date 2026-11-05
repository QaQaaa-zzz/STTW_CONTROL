"""Compare H75 actor, filter state and reference constraints to original JAX."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_path_policy import TorchPathPolicy
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
m=json.loads((a.fixture/'identity.json').read_text());c=json.loads((a.fixture/'config.json').read_text());d=np.load(a.fixture/'oracle.npz')
net=TorchPathPolicy(a.fixture/'actor.npz',m['actor_sha256'],c,m['lanes'],a.device)
errors={k:0. for k in ('latent','governed','filtered','offsets')}
for t in range(m['ticks']):
    tensor=lambda key:torch.as_tensor(d[key][t],device=a.device)
    net.reset(tensor('reset'));z=net.latent(tensor('observation'));g,flags=net.correct(z,tensor('nominal'))
    for key,actual in [('latent',z),('governed',g),('filtered',net.filtered),('offsets',net.offsets)]:
        value=actual.cpu().numpy();expected=d[key][t];np.testing.assert_allclose(value,expected,atol=4e-5,rtol=2e-5)
        errors[key]=max(errors[key],float(np.max(np.abs(value-expected))))
    actual=np.stack([flags[k].cpu().numpy() for k in ('reference_clipped','rate_clipped','policy_fault')],axis=-1)
    np.testing.assert_array_equal(actual,d['flags'][t])
report=dict(device=a.device,max_abs_error=errors,flag_equality=True,ticks=m['ticks'],lanes=m['lanes'],physics_steps=0,optimizer_updates=0)
(a.fixture/('torch_'+a.device.replace(':','_')+'.json')).write_text(json.dumps(report,indent=2));print(report)
