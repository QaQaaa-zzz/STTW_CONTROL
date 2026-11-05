"""Replay the same physical state stream through the combined Torch scheduler."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_controller import TorchController
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.torch_path_reference import TorchPathReference
from sttw_control.torch_middle_control import TorchMiddleControl
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
m=json.loads((a.fixture/'identity.json').read_text());d=np.load(a.fixture/'oracle.npz');base=Path('runs/h75_torch_parity');c=json.loads((base/'config.json').read_text());identity=json.loads((base/'identity.json').read_text());n=len(m['alphas'])
policy=TorchPathPolicy(base/'actor.npz',identity['actor_sha256'],c,n,a.device);route=TorchPathReference(np.load(a.fixture/'route.npz'),c,a.device)
controller=TorchController(json.load(open('runs/isaac_signals_parity/dynamic.json')),n,a.device)
alpha=torch.tensor(m['alphas'],device=a.device);v=torch.full((n,),2.3,device=a.device);middle=TorchMiddleControl(policy,route,m['frame_scales'],alpha,v)
tensor=lambda key,t:torch.as_tensor(d[key][t],device=a.device)
raw=torch.stack((v,torch.zeros_like(v)),dim=-1);controller.disturbance=tensor('disturbance',0)
middle.reset(tensor('measurement',0),tensor('poses',0),tensor('speed',0),raw,raw,tensor('command',0),controller)
errors={k:0. for k in ('observation','latent','governed','nominal')}
for t in range(m['ticks']):
    g=middle.before(tensor('poses',t))
    for key,value in [('observation',middle.obs),('latent',middle.latent),('governed',g),('nominal',middle.nominal)]:
        actual=value.cpu().numpy();expected=d[key][t];np.testing.assert_allclose(actual,expected,atol=4e-5,rtol=2e-5)
        errors[key]=max(errors[key],float(np.max(np.abs(actual-expected))))
    controller.disturbance=tensor('disturbance',t+1)
    middle.after(tensor('measurement',t+1),tensor('poses',t+1),tensor('speed',t+1),g,tensor('command',t+1),tensor('applied',t+1),controller)
assert not middle.fault.any() and not middle.domain_exit.any()
report=dict(device=a.device,max_abs_error=errors,ticks=m['ticks'],alphas=m['alphas'],physics_steps=0,optimizer_updates=0)
(a.fixture/('torch_'+a.device.replace(':','_')+'.json')).write_text(json.dumps(report,indent=2));print(report)
