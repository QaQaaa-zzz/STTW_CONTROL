"""Check full causal histories and real frozen actions against a JAX oracle."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_local_lower import TorchLocalLower
from sttw_control.torch_controller import TorchController
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
meta=json.loads((a.fixture/'identity.json').read_text());d=np.load(a.fixture/'oracle.npz')
lower=TorchLocalLower(meta['actor']['original'],meta['actor']['sha256'],meta['lanes'],device=a.device)
controller=TorchController(meta['controller'],meta['lanes'],device=a.device)
errors={'observation':0.,'action':0.}
for t in range(meta['ticks']):
    lower.reset(torch.as_tensor(d['reset'][t],device=a.device))
    def tensor(key):return torch.as_tensor(d[key][t],device=a.device)
    controller.disturbance=tensor('disturbance')
    action,obs,finite=lower.prepare(tensor('measurement'),tensor('speed'),tensor('reference'),tensor('previous'),tensor('command'),controller)
    assert finite.all()
    for key,value in [('action',action),('observation',obs)]:
        actual=value.cpu().numpy();expected=d[key][t]
        np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=2e-5)
        errors[key]=max(errors[key],float(np.max(np.abs(actual-expected))))
    lower.finish(tensor('applied'),tensor('yaw'))
receipt=dict(device=a.device,max_abs_error=errors,ticks=meta['ticks'],lanes=meta['lanes'],physics_steps=0,optimizer_updates=0)
(a.fixture/('torch_'+a.device.replace(':','_')+'.json')).write_text(json.dumps(receipt,indent=2));print(receipt)
