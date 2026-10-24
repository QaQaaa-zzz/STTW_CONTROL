"""Verify reduced-model algebra and its distinction from fixed feedback."""
from pathlib import Path
import json
import hashlib
import numpy as np
import sympy as s
from sttw_control.controller import ControllerConfig, _system, initial_controller

ROOT = Path(__file__).resolve().parents[4]
out = ROOT/'runs/formula_applicability_audit_v1'
out.mkdir(exist_ok=True)
lam, phi, omega, delta, u, a1, a2, d, dh, dhdot, a2dot, gamma, F = s.symbols(
    'lam phi omega delta u a1 a2 d dh dhdot a2dot gamma F', real=True)
lam = s.Symbol('lam', positive=True)
z = omega+lam*phi+(a2*delta+dh)/lam
zdot = lam**2*phi+a2*delta+a1*u+d+gamma*F+lam*omega+(a2dot*delta+a2*u+dhdot)/lam
complete = lam*z+(a1+a2/lam)*u+gamma*F+d-dh+(a2dot*delta+dhdot)/lam
assert s.simplify(zdot-complete) == 0
t, duration, delay = s.symbols('t duration delay', positive=True)
impulse = s.integrate(gamma*s.exp(-lam*t), (t,delay,delay+duration))
# Check away from lambda=0 (our actual lambda is strictly positive).
assert s.simplify(impulse-gamma*s.exp(-lam*delay)*(1-s.exp(-lam*duration))/lam) == 0
declaration = ROOT/'runs/tolerance_recovery_opportunity_v1/declaration.json'
cc = json.loads(declaration.read_text())['source']['controller']
cfg = ControllerConfig(**cc)
rows = []
for speed in (1.5,1.8,2.55):
    aa1,aa2,aa4,ratio,gains = _system(speed,initial_controller(cfg).gains,cfg)
    aa1,aa2,aa4 = map(float,(aa1,aa2,aa4))
    A = np.array([[0.,0.,0.],[0.,0.,1.],[aa2,aa4,0.]])
    B = np.array([1.,0.,aa1]); K = np.asarray(gains,dtype=float)
    C = np.array([aa2/np.sqrt(aa4),np.sqrt(aa4),1.])
    assert np.allclose(C@A,np.sqrt(aa4)*C)
    Acl = A-np.outer(B,K)
    eig = np.linalg.eigvals(Acl)
    assert np.all(eig.real < 0)
    # z alone does not close the fixed-feedback dynamics.
    rank = int(np.linalg.matrix_rank(np.vstack((C,C@Acl))))
    assert rank == 2
    rows.append(dict(speed_proxy_m_s=speed,open_unstable_lambda=np.sqrt(aa4),
        closed_loop_poles=[dict(real=float(x.real),imag=float(x.imag)) for x in eig],
        ideal_feedback_gains=K.tolist(), projected_dynamics_rank=rank,
        note='Ideal unsaturated ECBC only. ESO, local300, delay and plant mismatch omitted.'))
sources = [declaration,Path(__file__),ROOT/'learning/src/sttw_control/controller.py',
           ROOT/'learning/src/sttw_control/tolerance_state_features.py']
receipt = dict(complete_derivative_verified=True,delayed_pulse_factor_verified=True,
    open_coordinate_projection_verified=True,rows=rows,
    derivative_residual='(d-dhat)+(a2dot*delta+dhatdot)/lambda',
    sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
    new_physics_steps=0,optimization_steps=0,
    scope='Mathematical applicability audit, not closed-loop system identification or evidence of a new controller.')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
