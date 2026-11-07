"""Clearance confirmation, preference ramp and no state restoration contract."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
import torch
from sttw_control.gravel_recovery_preference import GravelRecoveryPreference

class RecoveryPreferenceTests(unittest.TestCase):
    def test_continuous_support_then_next_decision_ramp_without_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);np.savez(path/'terrain.npz',vertices=np.array([[1.,-.1,.01],[1.,.1,.01],[.9,0,.01]]));(path/'identity.json').write_text(json.dumps(dict(terrain_support_end_s_m=1.)))
            positions=torch.zeros(5,1,3);positions[:,:,0]=1.2;force=torch.zeros(5,1,3);force[:,:,2]=1.
            plant=S(bodies=['rearwheel'],origin=torch.zeros(5,3),roots=torch.zeros(5,13),dofs=torch.zeros(5,2,2),contact=force,robot=S(data=S(body_link_pos_w=positions)))
            env=S(n=5,device='cpu',physics_dt=.0002,plant=plant,alive=torch.ones(5,dtype=torch.bool),alpha=torch.full((5,),.5),middle=S(obs=torch.zeros(5,351)),policy=S(filtered=torch.zeros(5,2),offsets=torch.zeros(5,2)),controller=S(eso=torch.zeros(5,3)),route=S(at=lambda s:(torch.tensor([[1.,0.]]),torch.zeros(1),torch.zeros(1))))
            cfg=dict(groups=[dict(lanes=list(range(5)),entry_alpha=.5,target_alphas=[0,.25,.5,.75,1])],terrain=str(path),confirmation_s=.02,rear_support_force_N=.1,wheel_bound_radius_m=.1,ramp_s=.1)
            manager=GravelRecoveryPreference(env,cfg,path,None);before=plant.roots.clone()
            for _ in range(50):manager.record(env,0,0)
            force[:,:,2]=0.;manager.record(env,0,0);force[:,:,2]=1.
            for _ in range(99):manager.record(env,0,0)
            manager.decision(1);self.assertEqual(len(manager.logs),0)
            manager.record(env,0,0);manager.decision(2);torch.testing.assert_close(env.alpha,torch.full((5,),.5))
            manager.decision(7);torch.testing.assert_close(env.alpha,torch.tensor([0,.25,.5,.75,1]))
            torch.testing.assert_close(plant.roots,before);self.assertFalse(manager.logs[0]['physics_state_restored'])

if __name__=='__main__':unittest.main()
