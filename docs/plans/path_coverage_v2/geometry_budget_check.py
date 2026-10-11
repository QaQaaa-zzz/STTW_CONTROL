"""CPU-only arithmetic/nominal-route coverage check, not a controller simulation.
Uses the audited ECBC equilibrium ratio and immutable route curvature formula.
"""
import math, json
from pathlib import Path
import numpy as np

DT=.005
L=.408; C=math.cos(math.radians(25)); G=9.8; B=.164; TRAIL=.024; H=.2

def q(v):
    return G*L/((v*v-B*TRAIL*G/H)*C)

def chi(v,k):
    return np.clip((np.abs(np.arctan(L*np.asarray(k)/C)/q(v))-.18)/.12,0,1)

def smooth(x):
    z=np.clip(x,0,1);return z*z*(3-2*z)

def one_route(v,R,deg,is_s):
    # Approximate progress=vt; excludes physical dynamics and starting-speed slew.
    s=np.arange(4000)*DT*v
    angle=math.radians(deg);ramp=.5;start=3.;end=start+angle*R+ramp
    def bend(st,en):
        return smooth((s-st)/ramp)*smooth((en-s)/ramp)/R
    k=bend(start,end)
    if is_s:
        st=end+3.;en=st+angle*R+ramp;k-=bend(st,en)
    h=chi(v,k)
    return float(np.mean(h>0)),float(np.mean(h>=.5)),float(np.mean(np.abs(k)>1e-4))

rng=np.random.default_rng(910201)
new=[]
for _ in range(1000):
    v=rng.uniform(2.4,2.6);R=rng.uniform(2.,2.2);deg=rng.uniform(90,100)
    single=one_route(v,R,deg,False);ss=one_route(v,R,deg,True)
    new.append(np.array(single)*(204/512)+np.array(ss)*(206/512))
predict=np.mean(new,axis=0)
result={
 'kind':'nominal_geometry_only_no_physics_no_learning',
 'audited_commit':'e6af0cae12790320b4d42cef2e582bcc7bade701',
 'new_group_sizes':{'retention_old_sampler':102,'demand_single':204,'demand_S':206},
 'nominal_prediction_excluding_retention_group_contribution':{
  'chi_positive_fraction':float(predict[0]),'chi_ge_half_fraction':float(predict[1]),
  'nonzero_curvature_fraction':float(predict[2])},
 'nominal_plateau_chi_min_in_demand_box':float(chi(2.4,1/2.2)),
 'updates100_both_endpoints':{
  'upper_transitions':512*128*100*2,'lower_steps':512*128*100*2*4},
 'notes':['Assumes progress=vt; actual exposure must come from active training logs.',
          'chi is a nominal-demand surrogate, not a proven safety boundary.',
          'Same reward/20s horizon/physics/control structure; only sampler distribution changed.']}
assert sum(result['new_group_sizes'].values())==512
assert result['nominal_plateau_chi_min_in_demand_box']>=.5
out=Path(__file__).with_name('arithmetic_check.json')
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
