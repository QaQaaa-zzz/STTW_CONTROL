"""Target interpolation and bitwise zero-bias regression on the frozen PCRP H75 bundle."""
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from sttw_control.pcrp_oracle import OracleTargetBias
from sttw_control.torch_path_policy import TorchPathPolicy


BUNDLE = Path(__file__).resolve().parents[2]/'runs/h75_selected_local300_flat/bundle'
BOUNDS = dict(speed_negative_scale_m_s=1.,speed_positive_scale_m_s=.25,steer_scale_rad=.20)


class OracleTests(unittest.TestCase):
    def test_interpolation_and_clipping_are_in_physical_units(self):
        offsets=np.array([[[1.25,-.4],[0,0],[-1.25,.4]]])
        bias=OracleTargetBias([0,2.,10.],offsets,1,500,BOUNDS,'cpu')
        target=torch.tensor([[.12,-.06]])
        torch.testing.assert_close(bias(target),torch.tensor([[.25,-.20]]),rtol=0,atol=0)
        bias.set_step(100)
        self.assertTrue(torch.equal(bias(target),target))
        bias.set_step(150)  # 3 s, 1/8 of the distance between the last two knots.
        torch.testing.assert_close(bias(target),target+torch.tensor([[-1.25/8,.4/8]]),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            OracleTargetBias([0,2.,2.],offsets,1,500,BOUNDS,'cpu')

    @unittest.skipUnless((BUNDLE/'identity.json').is_file(),'requires the frozen H75 artifact')
    def test_zero_bias_and_shaper_exactly_match_original(self):
        identity=json.loads((BUNDLE/'identity.json').read_text())
        config=json.loads((BUNDLE/'config.json').read_text())
        policies=[TorchPathPolicy(BUNDLE/'actor.npz',identity['files']['actor.npz'],config,3)
                  for _ in range(2)]
        bias=OracleTargetBias([0.,10.],np.zeros((3,2,2)),3,500,config['residual_interface'],'cpu')
        policies[1].evaluation_target_bias=bias
        generator=torch.Generator().manual_seed(293)
        for step in range(12):
            obs=torch.rand((3,351),generator=generator)*4.-2.
            latent=policies[0].latent(obs)
            bias.set_step(step)
            for _ in range(4):
                nominal=torch.rand((3,2),generator=generator)*torch.tensor([3.,.6])+torch.tensor([1.,-.3])
                (original,flags),(new,new_flags)=[p.correct(latent,nominal) for p in policies]
                self.assertEqual(original.numpy().tobytes(),new.numpy().tobytes())
                for k in flags:
                    self.assertEqual(flags[k].numpy().tobytes(),new_flags[k].numpy().tobytes(),k)
                for k in ('filtered','offsets'):
                    self.assertEqual(getattr(policies[0],k).numpy().tobytes(),
                                     getattr(policies[1],k).numpy().tobytes(),k)
        # Applying the bias at the raw 50 Hz physical target leaves the original 200 Hz low-pass in force.
        policies[1].reset(slice(None))
        bias=OracleTargetBias([0.,10.],np.broadcast_to([-.30,.12],(3,2,2)),3,500,BOUNDS,'cpu')
        policies[1].evaluation_target_bias=bias
        nominal=torch.tensor([[2.6,0.]]).repeat(3,1)
        _,flags=policies[1].correct(torch.zeros((3,2)),nominal)
        expected=torch.tensor([[-.30,.12]]).repeat(3,1)
        torch.testing.assert_close(flags['target'],expected,rtol=0,atol=0)
        torch.testing.assert_close(policies[1].filtered,(1-policies[1].decay)*expected,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
