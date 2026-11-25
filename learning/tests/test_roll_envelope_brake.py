"""Match both signed extreme trajectories and the actual reference shaper."""
import json
from pathlib import Path
import numpy as np
import torch

from sttw_control.envelope_intervention import RollEnvelopeBrake
from sttw_control.margin_theory import coefficients, constant_rate_segment
from sttw_control.torch_controller import TorchController

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'fixtures/paper_preference/controller.json').read_text())
SETTINGS = dict(enter_margin_rad=.07, release_margin_rad=.14, horizon_s=.25, grid_s=.001)


def test_two_sided_limits_match_piecewise_constant_dynamics():
    rng=np.random.default_rng(127)
    states=rng.uniform([-.65,-7,-.8],[.65,7,.8],(100,3))
    speeds=rng.uniform(.6,3.5,len(states))
    ctrl=TorchController(CONFIG,len(states),'cpu',dtype=torch.float64)
    guard=RollEnvelopeBrake(ctrl,.1,{'speed_negative_scale_m_s':1},len(states),'cpu',SETTINGS)
    margins,domain=guard.margins(*torch.from_numpy(states.T),torch.from_numpy(speeds))
    assert domain.all()
    t=guard.t.numpy().astype(float)
    for i,(state,v) in enumerate(zip(states,speeds)):
        c,b,a=coefficients(v,CONFIG)
        for side in (1,-1):
            x=side*state
            switch=(.8-x[2])/3
            first=constant_rate_segment(x,np.minimum(t,switch),c,b,a,3)
            x1=constant_rate_segment(first,np.maximum(t-switch,0),c,b,a,0)
            np.testing.assert_allclose(margins[i,int(side<0)],.7-max(x1[0]),atol=2e-9)


def test_causal_arming_release_and_off_lane_speed():
    ctrl=TorchController(CONFIG,2,'cpu')
    guard=RollEnvelopeBrake(ctrl,.1,{'speed_negative_scale_m_s':.8},2,'cpu',SETTINGS)
    m=torch.zeros(2,7);m[:,5]=26
    alive=torch.ones(2,dtype=torch.bool)
    m[1,0]=.52
    assert not guard.observe(m,alive)['guard_valid'].any()
    m[1,0]=.65
    row=guard.observe(m,alive)
    assert row['guard_active'].tolist()==[False,True]
    torch.testing.assert_close(row['guard_inputs'][1,1],torch.tensor(6.5))
    target=torch.tensor([[.25,.12],[.25,-.11]])
    torch.testing.assert_close(guard(target),torch.tensor([[.25,.12],[-.8,-.11]]))
    m[1,0]=0
    guard.observe(m,alive)
    assert not guard.observe(m,alive)['guard_active'].any()
    torch.testing.assert_close(guard(target),target)
