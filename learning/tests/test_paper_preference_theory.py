"""Source-backed identities and explicit limits of the paper's nominal theory."""
import json, unittest
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_controller import TorchController
from sttw_control.torch_path_reward import costs
ROOT=Path(__file__).resolve().parents[2]

class PaperTheoryTest(unittest.TestCase):
    def test_source_coefficients_and_pole_placement(self):
        c=json.loads((ROOT/'learning/tests/fixtures/paper_preference/controller.json').read_text());v=torch.tensor([1.5,1.8,2.25,2.567,2.6,3.],dtype=torch.float64);ctrl=TorchController(c,len(v),dtype=torch.float64);a1,a2,a4,_,k=ctrl._system(v)
        np.testing.assert_allclose(a1.numpy(),-1.51526*v.numpy(),rtol=1e-5)
        np.testing.assert_allclose(a2.numpy(),1.78195-9.23941*v.numpy()**2,rtol=1e-5)
        kd,kphi,kp=k.T;poly=torch.stack([kd+a1*kp,a1*kphi+a2*kp-a4,a2*kphi-a4*kd],1)
        np.testing.assert_allclose(poly.numpy(),np.tile([15,75,125],(len(v),1)),atol=1e-10)

    def test_effective_reward_affine_with_caps(self):
        c=json.loads((ROOT/'learning/tests/fixtures/paper_preference/reward.json').read_text());x=torch.linspace(-20,20,101,dtype=torch.float64);kw=dict(chi=torch.sigmoid(x),ev=x,ey=x*.3,eh=x*.1,peak_roll=abs(x)*.1,roll_rate=x,speed=x+2,offsets=torch.stack([x,x*.1],1),offset_rate=torch.stack([x*.4,x],1),config=c)
        left=costs(alpha=torch.zeros_like(x),**kw)['reward'];right=costs(alpha=torch.ones_like(x),**kw)['reward']
        for alpha in [.1,.25,.5,.75,.9]:
            actual=costs(alpha=torch.full_like(x,alpha),**kw)['reward'];np.testing.assert_allclose(actual.numpy(),((1-alpha)*left+alpha*right).numpy(),atol=1e-12)

    def test_monotonicity_is_for_complete_endpoint_costs(self):
        # Two policies: common cost changes; both bare task costs increase at the switch.
        common=np.array([10.,0.]);j0=np.array([0.,15.]);j1=np.array([10.,16.]);f0=common+j0;f1=common+j1
        choices=[np.argmin((1-a)*f0+a*f1) for a in [0.,1.]]
        self.assertEqual(choices,[0,1]);self.assertGreater(j0[1],j0[0]);self.assertGreater(j1[1],j1[0]);self.assertGreater(f0[1],f0[0]);self.assertLess(f1[1],f1[0])

    def test_equal_tolerances_do_not_fix_balanced_alpha(self):
        alpha=np.linspace(0,1,10001);ev=1-alpha;ep=alpha**2;i=np.argmin(abs(ev-ep));self.assertAlmostEqual(alpha[i],(np.sqrt(5)-1)/2,places=3);self.assertGreater(abs(alpha[i]-.5),.1)

    def test_impulse_peak_from_source_gains(self):
        c=json.loads((ROOT/'learning/tests/fixtures/paper_preference/controller.json').read_text());v=torch.tensor([2.25,2.567],dtype=torch.float64);k=TorchController(c,2,dtype=torch.float64)._system(v)[-1][:,0].numpy();t=np.linspace(0,1,200001);peaks=[np.max(np.exp(-5*t)*(t+(x-5)*t*t/2)) for x in k];self.assertAlmostEqual(100*(peaks[1]/peaks[0]-1),4.87,places=1)

    def test_recovery_kernel_is_negative_in_declared_range(self):
        c=json.loads((ROOT/'learning/tests/fixtures/paper_preference/controller.json').read_text());v=torch.linspace(.5,3,201,dtype=torch.float64);a1,a2,a4,*_=TorchController(c,len(v),dtype=torch.float64)._system(v);t=torch.linspace(0,1,101,dtype=torch.float64);lam=a4.sqrt()[:,None];kernel=a1[:,None]*torch.cosh(lam*t)+a2[:,None]/lam*torch.sinh(lam*t);self.assertTrue(bool((kernel<0).all()))

if __name__=='__main__':unittest.main()
