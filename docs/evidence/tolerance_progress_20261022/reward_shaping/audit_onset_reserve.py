import pathlib,json,pickle
import numpy as np
from dataclasses import asdict
from sttw_control.teleop_env import TeleopEnv
from sttw_control.controller import _system
from sttw_control.tolerance_shaping import model_reserve
w=pathlib.Path('/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration');r=w/'runs/tolerance_shaping_post_v1/evaluation';c=w/'runs/tolerance_shaping_train_v1';plan=json.loads((c/'baseline_alpha0/plan.json').read_text());source=json.loads(pathlib.Path(plan['parent_manifest']).read_text())['source'];env=TeleopEnv(backend='cpu');assert asdict(env.cc)==source['controller'];rows=[]
for p in sorted(r.glob('*_snapshot.pkl')):
 s=pickle.load(p.open('rb'));m=env.helpers.measure(s.physical.data);ctrl=s.physical.controller;a1,a2,a4,_,_=_system(m[5]*.1,ctrl.gains,env.cc);lam=np.sqrt(float(a4));z=float(a2)/lam*float(m[2])+lam*float(m[0])+float(m[1])+float(ctrl.disturbance)/lam
 result=model_reserve(z=z,b=a1+a2/lam,lam=lam,remaining=20-float(s.tick)*env.cc.dt,effort_budget=.030839085298575167,gamma=.815560907125473,duration=.1,target_force=6.,weight=10.)
 assert bool(result['valid']);name=p.name[:-len('_snapshot.pkl')];rows.append(dict(name=name,z=z,cost=float(result['cost_rate']),plus_N=float(result['plus_N']),minus_N=float(result['minus_N']),roll=float(m[0]),speed=float(m[5])*.1,disturbance=float(ctrl.disturbance),tick=int(s.tick)))
(r/'report/onset_reserve_audit.json').write_text(json.dumps({'scope':'exact saved snapshot measurement and ESO; original controller verified; no physical steps; hypothetical common dynamic shaping on all policies','cases':rows},indent=2))
for a in [0,1]:
 for pol in ['baseline_update50','static_update50','dynamic_update50']:
  g=[x for x in rows if x['name'].startswith(pol+'_') and f'_a{a}_' in x['name']];print(a,pol,'n',len(g),'cost mean',np.mean([x['cost'] for x in g]),'saturated',sum(x['cost']>=9.999 for x in g),'reserve median',np.median([min(x['plus_N'],x['minus_N']) for x in g]))
