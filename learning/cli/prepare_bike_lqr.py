"""Design speed-scheduled ground LQR gains; no physics or policy training.

State is [steering angle, left-positive roll, roll rate]; input is steering
rate. Reuse the vehicle's linear roll coefficients, with an explicit steering
integrator. scipy CARE is the standard solver; these are not copied bike gains.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from scipy.linalg import solve_continuous_are,expm
p=argparse.ArgumentParser();p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
c=json.loads(a.controller.read_text());rows=[];radii=[]
Q=np.diag(1/np.square([.08,.08,.5]));R=np.array([[1/3.**2]])
J=c['mass']*c['cg_height']**2+c['roll_inertia_com']
for v in np.arange(.5,3.5001,.05):
 a1=-c['mass']*c['cg_forward']*c['cg_height']*v*math.cos(c['caster'])/c['wheelbase']/J
 a2=-(c['mass']*v*v*c['cg_height']-c['mass']*c['cg_forward']*c['trail']*c['gravity'])*math.cos(c['caster'])/c['wheelbase']/J
 a4=c['mass']*c['gravity']*c['cg_height']/J
 A=np.array([[0,0,0],[0,0,1],[a2,a4,0.] ]);B=np.array([[1.],[0.],[a1]])
 P=solve_continuous_are(A,B,Q,R);K=np.linalg.solve(R,B.T@P)
 block=np.zeros((4,4));block[:3,:3]=A;block[:3,3:]=B;disc=expm(block*c['dt'])
 radii.append(float(max(abs(np.linalg.eigvals(disc[:3,:3]-disc[:3,3:]@K)))))
 rows.append([float(v),*K[0].tolist()])
c.update(scheduled_feedback=rows,use_eso_correction=False,design=dict(method='continuous LQR; held-input discrete stability checked',state=['steer','roll','roll_rate'],input='steer_rate',Q=Q.tolist(),R=R.tolist(),maximum_nominal_discrete_spectral_radius=max(radii),scope='linear nominal model only; no saturation, delay, contact or closed-loop tracking guarantee'))
assert max(radii)<1, 'Nominal sampled linear controller is unstable'
a.output.write_text(json.dumps(c,indent=2)+'\n');print(json.dumps(c['design']))
