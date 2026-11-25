"""Nominal governor selects bounded actions on the frozen PCRP reference interface."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sttw_control.predictive_reference_governor import PredictiveReferenceGovernor
from sttw_control.torch_controller import TorchController


def test_equilibrium_symmetry_and_infeasible_fallback():
    cfg = json.loads((Path(__file__).resolve().parents[2] / 'docs/evidence/h75_recent_20261109/inputs/bundle/config.json').read_text())
    n = 5
    alpha = torch.full((n,), .5)
    speed = torch.full((n,), 2.6)
    curvature = torch.tensor([0., .35, -.35, 0., 0.])
    controller_config = dict(mass=5.4, cg_height=.2, cg_forward=.164, wheelbase=.408,
                             caster=.436332, trail=.024, gravity=9.8, minimum_speed=.5, wc=5, wo=40)
    controller = TorchController(controller_config, n)
    nominal = torch.stack((speed, curvature * .4), dim=-1)
    interface = cfg['residual_interface']
    dt = cfg['timing']['lower_dt_s']
    policy = SimpleNamespace(offsets=torch.zeros(n, 2), filtered=torch.zeros(n, 2),
                             low=torch.tensor([1.5, -.35]), high=torch.tensor([3.5, .35]),
                             decay=torch.exp(-dt / torch.tensor(interface['filter_tau_s'])),
                             increment_low=dt * torch.tensor([-1., -.8]),
                             increment_high=dt * torch.tensor([.5, .8]))
    middle = SimpleNamespace(pose=torch.zeros(n, 3), nominal=nominal,
                             projection=dict(progress=torch.zeros(n), heading=torch.zeros(n),
                                             cross_track=torch.tensor([0., .15, -.15, 0., 0.])))
    env = SimpleNamespace(n=n, device='cpu', config=cfg, cc=controller_config, speed=speed, alpha=alpha,
                          policy=policy, middle=middle, controller=controller,
                          route=SimpleNamespace(at=lambda s: (None, None, curvature)))
    governor = PredictiveReferenceGovernor(env, dict(preview_steps=15, speed_response_s=.2))
    measurement = torch.zeros(n, 7)
    measurement[:, 0] = torch.tensor([0., .2, -.2, .69, -.69])
    measurement[:, 1] = torch.tensor([0., .2, -.2, 8., -8.])
    action, result = governor.action(measurement, speed)
    assert torch.isfinite(action).all()
    assert torch.equal(action[0], torch.zeros(2))
    assert action[1, 0] == action[2, 0] and action[1, 1] == -action[2, 1]
    assert torch.all(result['governor_feasible_count'][:3] > 0)
    assert torch.all(result['governor_feasible_count'][3:] == 0)
    assert torch.all(result['governor_margin_rad'][3:] < 0)
    assert not policy.offsets.any() and not policy.filtered.any() and not controller.disturbance.any()
