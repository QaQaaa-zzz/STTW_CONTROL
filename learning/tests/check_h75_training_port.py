"""Check saved native H75 TRAIN route/state observations against Torch; no steps."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_path_reference import TorchPathReference
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.torch_middle_control import TorchMiddleControl
p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--checkpoint',type=Path);p.add_argument('--go2-checkout',type=Path);a=p.parse_args();torch.set_num_threads(2)
s=np.load(a.bank/'states.npz');o=np.load(a.bank/'route_oracle.npz');meta=json.loads((a.bundle/'identity.json').read_text());cfg=json.loads((a.bundle/'config.json').read_text());route=TorchPathReference(np.load(a.bank/'routes.npz'),cfg,a.device);n=len(s['source_bank_index']);tensor=lambda value:torch.as_tensor(value,device=a.device)
errors={}
def check(name,actual,expected):
    value=actual.detach().cpu().numpy();np.testing.assert_allclose(value,expected,atol=4e-5,rtol=2e-5);errors[name]=float(np.max(abs(value-expected)))
for name,value,expected in zip(['route_xy','route_heading','route_curvature'],route.at(tensor(o['query'])),[o['points'],o['heading'],o['curvature']]):check(name,value,expected)
projection=route.project(tensor(o['pose_xy']),tensor(o['previous']),torch.full((n,),.025,device=a.device))
for key,value in projection.items():check('projection_'+key,value,o['projection_heading' if key=='heading' else key])
policy=TorchPathPolicy(a.bundle/'actor.npz',meta['files']['actor.npz'],cfg,n,a.device)
policy.offsets[:]=tensor(s['closedloop.offsets']);policy.filtered[:]=tensor(s['closedloop.filtered'])
middle=TorchMiddleControl(policy,route,meta['frame_scales'],tensor(s['closedloop.alpha']),tensor(np.load(a.bank/'routes.npz')['speed']))
middle.frames[:]=tensor(s['closedloop.history.frames']);middle.mask[:]=tensor(s['closedloop.history.mask'])
for name in ('projection','follower'):
 prefix='closedloop.'+name+'.';setattr(middle,name,{key[len(prefix):]:tensor(s[key]) for key in s.files if key.startswith(prefix)})
obs,fault=middle.observation(tensor(s['native_pose']));check('observation',obs,s['native_observation']);check('actor_mean',policy.latent(obs),s['native_mean'])
assert not bool(fault.any()) and not bool(np.any(s['native_fault']))
native_import=None
if a.checkpoint:
    import sys
    sys.path.insert(0,str(a.go2_checkout/'rsl_rl'))
    from sttw_control.go2_ppo_adapter import H75ActorCritic
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    imported=H75ActorCritic.from_original_policy(cfg,saved['policy'],a.device)
    check('original_checkpoint_actor',imported.act_inference(obs),s['native_mean'])
    torch.testing.assert_close(imported.log_std.cpu(),saved['policy']['log_std'],atol=0,rtol=0)
    sampled=imported.act(obs);values=imported.evaluate(tensor(s['native_critic_observation']))
    assert bool(torch.isfinite(sampled).all()) and bool(torch.isfinite(values).all())
    assert imported.get_actions_log_prob(sampled).shape==(n,)
    native_import=dict(checkpoint=str(a.checkpoint.resolve()),actor_critic_logstd_restored=True,std=imported.std.detach().cpu().tolist(),optimizer_restored=False,mean_value=float(values.detach().mean()))
# Route selection must follow lane IDs after an independent reset/permutation.
permutation=torch.arange(n-1,-1,-1,device=a.device);route.route_ids=permutation
for name,value,expected in zip(['permuted_xy','permuted_heading','permuted_curvature'],route.at(tensor(o['query'])[permutation]),[o['points'][::-1].copy(),o['heading'][::-1].copy(),o['curvature'][::-1].copy()]):check(name,value,expected)
report=dict(schema='h75_training_port_forward_v1',cases=n,device=a.device,max_abs_errors=errors,original_checkpoint_import=native_import,physics_steps=0,optimizer_updates=0,scope='authentic saved route interpolation/projection/observation/H75 forward plus lane permutation; no physics continuation or training claim')
(a.bank/('check_'+a.device.replace(':','_')+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
