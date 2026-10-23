"""Symbolic identity plus retrospective speed-variation term; not a predictor."""
import json,math,hashlib
from pathlib import Path
import numpy as np
import sympy as sp
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
v,vd,delta,phi,omega,u,d,dh,dhd,F,gamma,lam=sp.symbols('v vd delta phi omega u d dh dhd F gamma lam',real=True)
a1=sp.Function('a1')(v);a2=sp.Function('a2')(v)
z=omega+lam*phi+a2*delta/lam+dh/lam
zdot=(a1*u+a2*delta+lam**2*phi+d+gamma*F)+lam*omega+sp.diff(a2,v)*vd*delta/lam+a2*u/lam+dhd/lam
expected=lam*z+(a1+a2/lam)*u+sp.diff(a2,v)*vd*delta/lam+dhd/lam+d-dh+gamma*F
assert sp.simplify(zdot-expected)==0
# Frozen speed, constant exact disturbance estimate recovers the original scalar model.
assert sp.simplify((zdot-lam*z-(a1+a2/lam)*u-gamma*F).subs({vd:0,dhd:0,d:dh}))==0
# Product rule term is essential whenever a2 changes and steer is nonzero.
assert sp.simplify((zdot-expected+sp.diff(a2,v)*vd*delta/lam))!=0
a20=sp.symbols('a20',real=True)
fixed_z=omega+lam*phi+a20*delta/lam+dh/lam
fixed_dot=a1*u+a2*delta+lam**2*phi+d+gamma*F+lam*omega+a20*u/lam+dhd/lam
assert sp.simplify(fixed_dot-(lam*fixed_z+(a1+a20/lam)*u+(a2-a20)*delta+dhd/lam+d-dh+gamma*F))==0
# Verify the midpoint Stieltjes discretization on a smooth prescribed toy trajectory.
from scipy.integrate import quad
truth=quad(lambda t:math.exp(-7*t)*(.1*math.sin(t))/7*(-4*(2+.2*t)*.2),0,2,epsabs=1e-12)[0]
errors=[]
for n in [1001,2001]:
 tt=np.linspace(0,2,n);aa=-2*(2+.2*tt)**2;dd=.1*np.sin(tt)
 estimate=np.sum(np.exp(-7*(tt[1:]+tt[:-1])/2)*(dd[1:]+dd[:-1])/14*np.diff(aa))
 errors.append(abs(estimate-truth))
assert errors[1]<errors[0]/3 and errors[1]<1e-8
records=[]
for name in ['tolerance_teaching_search_pilot_v1','tolerance_recovery_opportunity_v1']:
 root=ROOT/'runs'/name;decl=json.loads((root/'declaration.json').read_text());c=decl['source']['controller'];spec=decl['recovery_intervention']
 m=c['mass'];h=c['cg_height'];a=c['cg_forward'];L=c['wheelbase'];g=c['gravity'];inertia=m*h*h+.043929*m/5.434
 lambda_=math.sqrt(m*g*h/inertia)
 for row in json.loads((root/'evaluation/results.json').read_text())['cases']:
  if row['policy'] not in ['control','speed_off'] or row['force_N']==0:continue
  path=root/'evaluation'/(row['name']+'.npz')
  with np.load(path) as q:
   mask=(q['time']>=spec['start_s']-1e-5)&(q['time']<spec['end_s']-1e-5)
   t=q['time'][mask].astype(float);vel=np.maximum(q['wheel_speed_proxy'][mask].astype(float),c['minimum_speed']);steer=q['actual_delta'][mask].astype(float)
  if len(t)<2:raise ValueError('insufficient executed intervention samples')
  a1=-m*a*h*vel*math.cos(c['caster'])/L/inertia
  raw2=-(m*vel**2*h-m*a*c['trail']*g)*math.cos(c['caster'])/L
  a2=np.where(raw2<0,-1.,1.)*np.maximum(abs(raw2),.1)/inertia
  b=a1+a2/lambda_;weights=np.exp(-lambda_*((t[1:]+t[:-1])/2-spec['start_s']))
  Sv=float(np.sum(weights*(steer[1:]+steer[:-1])/2/lambda_*np.diff(a2)))
  U=float(np.sum(weights*(b[1:]+b[:-1])/2*np.diff(steer)))
  W0=b[0]**2*(-math.expm1(-2*lambda_))/ (2*lambda_)
  reference_scale=math.sqrt(.030839085298575167*W0)
  records.append(dict(run=name,policy=row['policy'],context=[row[k] for k in ['route','speed','alpha','force_N']],task=row['task_recovered'],ticks=len(t),full_window=len(t)==round((spec['end_s']-spec['start_s'])/.005),Sv=Sv,steering_integral=U,abs_Sv_over_old_budget_scale=abs(Sv)/reference_scale,b_start=float(b[0]),b_end=float(b[-1]),speed_start=float(vel[0]),speed_end=float(vel[-1]),model_floor_active=bool(np.any(abs(raw2)<=.1)),source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
full=[r for r in records if r['full_window']]
summary=dict(traces=len(records),full_window_traces=len(full),median_abs_Sv_over_old_budget_scale=float(np.median([r['abs_Sv_over_old_budget_scale'] for r in full])),max_abs_Sv_over_old_budget_scale=max(r['abs_Sv_over_old_budget_scale'] for r in full),model_floor_active=sum(r['model_floor_active'] for r in records))
(OUT/'verification.json').write_text(json.dumps(dict(sympy_version=sp.__version__,symbolic_identity=True,frozen_limit=True,fixed_coordinate_identity=True,toy_quadrature_errors=errors,summary=summary,records=records,limitations='Retrospective discrete discounted product-rule term; actual future wheel proxy and steer used. Does not include ESO derivative/model error or prove predictive or causal improvement.'),indent=2)+'\n')
print(json.dumps(summary,indent=2))
