"""Offline analysis conventions, exact dynamics and failure-window masking."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('roll_contacts',Path(__file__).parents[1]/'cli/analyze_roll_contacts.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_exact_unforced_solution_and_failure_mask():
    t=np.arange(100)*.005;phi=np.exp(2*t)[:,None];rate=2*phi;valid=np.ones(phi.shape,bool)
    pred=m.replay(phi,rate,np.zeros_like(phi),valid,4.,.1)
    np.testing.assert_allclose(pred[20:],phi[20:],rtol=1e-12)
    valid[40:]=False
    pred=m.replay(phi,rate,np.zeros_like(phi),valid,4.,.1)
    assert np.isnan(pred[40:]).all()


def test_roll_derivative_uses_euler_kinematics():
    # Nonzero pitch/yaw makes the Euler derivative differ from body-x gyro.
    from scipy.spatial.transform import Rotation
    euler=np.array([1.4,.2,.3]);rate=np.array([.4,-.1,.25]);eps=1e-6
    q=Rotation.from_euler('xyz',euler).as_quat()
    nxt=Rotation.from_euler('xyz',euler+eps*rate)
    omega=(nxt*Rotation.from_quat(q).inv()).as_rotvec()/eps
    body=np.zeros(13);body[3:7]=q;body[10:13]=omega
    phi,derivative=m.roll_state(body)
    np.testing.assert_allclose(phi,np.pi/2-euler[0],atol=1e-12)
    np.testing.assert_allclose(derivative,-rate[0],atol=1e-6)


def test_constant_forcing_equilibrium():
    phi=np.ones((60,2))*.2;rate=np.zeros_like(phi);force=np.full_like(phi,-.8)
    pred=m.replay(phi,rate,force,np.ones_like(phi,dtype=bool),4.,.1)
    np.testing.assert_allclose(pred[20:],phi[20:],atol=1e-12)
