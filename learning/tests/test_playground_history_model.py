import unittest
import numpy as np
import torch
from sttw_control.playground_prediction import NominalRollout
from sttw_control.playground_history_model import HistoryDynamics


class HistoryModelTest(unittest.TestCase):
    def test_zero_correction_preserves_nominal_and_gradients(self):
        follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=.3,lookahead_min_m=.3,lookahead_max_m=2.)
        weights=np.zeros((34,9));weights[:9]=np.eye(9)*.99
        nominal=NominalRollout(dict(mean=np.zeros(33),scale=np.ones(33),weights=weights),follower,'cpu')
        model=HistoryDynamics(nominal,np.zeros(381),np.ones(381),np.zeros(26),np.ones(26),np.ones(9))
        s=torch.linspace(0,10,501);xy=torch.stack((s,torch.zeros_like(s)),1)[None];heading=torch.zeros(1,501)
        pose=torch.zeros(1,12);pose[:,3]=1;velocity=torch.zeros(1,9);velocity[:,0]=1.8
        predicted,_=model.rollout(pose,velocity,velocity,torch.zeros(1,381),s,xy,heading,heading,5)
        expected,_=nominal(pose,velocity,velocity,s,xy,heading,heading,5)
        torch.testing.assert_close(predicted,expected)
        predicted[:,:,:3].square().mean().backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().max()>0 for g in gradients))


if __name__=='__main__':unittest.main()
