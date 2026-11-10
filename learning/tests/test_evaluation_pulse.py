from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sttw_control.evaluation_pulse import EvaluationPulse


@pytest.mark.parametrize('force',[0.,-8.,8.])
def test_pulse_timing_failure_cutoff_and_removal(tmp_path,force):
    class Composer:
        def set_forces_and_torques(self,**values):
            self.last=values['forces'].clone()
    composer=Composer()
    plant=SimpleNamespace(bodies=['chasis'],roots=torch.tensor([[0.,0,0,0,0,0,1]]).repeat(2,1),robot=SimpleNamespace(permanent_wrench_composer=composer))
    env=SimpleNamespace(n=2,device='cpu',physics_dt=.0002,alive=torch.tensor([True,True]),plant=plant)
    pulse=EvaluationPulse(env,dict(start_s=.01,duration_s=.1,body='chasis',site_body_position_m=[0.,0.,0.],world_force_N=[0.,force,0.]))
    for tick in range(600):
        if tick==300:env.alive[1]=False
        pulse(tick%25)
    pulse.save(tmp_path)
    data=np.load(tmp_path/'applied_pulse.npz')
    assert data['force_world_N'].shape==(500,2,3)
    np.testing.assert_allclose(data['impulse_Ns'][:,1],[force*.1,force*.05],atol=1e-6)
    np.testing.assert_allclose(data['time_s'][[0,-1]],[.01,.1098])
    assert not composer.last.any()
