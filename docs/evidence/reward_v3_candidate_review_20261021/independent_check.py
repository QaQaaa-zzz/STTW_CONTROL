"""Validate the supplied candidate directly on raw saved physics; no simulator."""
from pathlib import Path
import json,sys,importlib.util,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'learning/src'))
from sttw_control.path_command_reporting import _interval_context
BASE=ROOT/'docs/evidence/reward_preference_audit_20261021'
OUT=Path(__file__).parent
RUN=ROOT/'runs/reward_persistent_preference_v3_audit'
PACKAGE=Path(sys.argv[1])
spec=importlib.util.spec_from_file_location('candidate',PACKAGE/'reward_reference.py');candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
cfg=json.loads((PACKAGE/'reward_config.json').read_text());c=cfg;caps=c['component_caps'];eps=c['path_secondary_ratio']
audit=json.loads((BASE/'results.json').read_text());rows=audit['records'];source={(x['file'],x['native_alpha']) for x in audit['audits']}
original=np.load(ROOT/'runs/reward_preference_audit_20261021/reward_components.npz')
external=list(csv.DictReader((RUN/'discounted/new_returns.csv').open()))
external={(x['dataset'],x['case'],x['method'],float(x['score_alpha'])):x for x in external}
H=lambda x:np.where(abs(x)<=1,x*x,2*abs(x)-1)
P=lambda x:np.maximum(x,0)
checks=[];summaryrows=[];path_caps=[];max_tick=0.;max_total=0.;max_disc=0.
# Mathematical identity with the actual current safety configuration.
for key,value in c.items():
 if key not in ('reward_version','path_secondary_ratio'):assert audit['scoring_config'][key]==value,key
for item in rows:
 group,case,method,a=item['dataset'],item['case'],item['method'],item['score_alpha']
 prefix=f'{group}/{case}/{method}'
 if group=='coverage_dual_u200':
  base=Path(audit['source_manifests'][0]).parent
  folder=base/'zero_upper' if method=='B0' else base/('alpha'+str(int(item['behavior_alpha'])))/'final_best'
 else:
  base=Path(audit['source_manifests'][1]).parent
  key={0:'alpha_0',.25:'alpha_0p25',.5:'alpha_0p5',.75:'alpha_0p75',1:'alpha_1'}[item['behavior_alpha']]
  folder=base/'zero_upper' if method=='B0' else base/'shared'/('final_best' if group=='high_demand_u150' else 'validation/update_0200')/key
 d=dict(np.load(folder/(case+'.npz')));ctx=_interval_context(d,{'timing':{'lower_dt_s':.005,'upper_dt_s':.02}})
 ev=np.asarray(d['actual_forward_speed'],float)-np.asarray(d['v_user'],float);ey=np.asarray(d['path_cross_track'],float);eh=np.asarray(d['path_heading_error'],float);chi=np.asarray(d['chi'],float)
 p=10*(H(ey/.1)+.5*H(eh/.1));v=H(ev/.1);qp=20*chi*H(P(abs(ey)-.1)/.1);qv=20*chi*H(P(abs(ev)-.05)/.1)
 independent=dict(speed=(1-a)*np.minimum((10-9*chi)*v,200)+a*np.minimum(10*v,200),path=(1-.9*a)*np.minimum(p,200),primary=(1-a)*np.minimum(qp,400)+a*np.minimum(qv,400),
 roll=np.minimum(80*H(P(np.asarray(d['peak_roll'],float)-.26)/.04),1800),working_roll=np.minimum(240*H(P(np.asarray(d['peak_roll'],float)-.30)/.02),6000),
 roll_rate=np.minimum(H(P(abs(np.asarray(d['phi_dot'],float))-.8)/.8),40),overspeed=np.minimum(2*H(P(ev-.05)/.1),160),low_speed=np.minimum(4*H(P(1.5-np.asarray(d['actual_forward_speed'],float))/.2),160),
 offset_magnitude=np.minimum(.02*np.sum((np.asarray(d['offsets'],float)/[1,.2])**2,axis=-1),.2),offset_rate=np.minimum(.03*np.sum((ctx['offset_rate']/[1,.4])**2,axis=-1),20))
 cand=candidate.costs(alpha=a,chi=chi,ev=ev,ey=ey,eh=eh,peak_roll=np.asarray(d['peak_roll'],float),roll_rate=np.asarray(d['phi_dot'],float),speed=np.asarray(d['actual_forward_speed'],float),offsets=np.asarray(d['offsets'],float),offset_rate=ctx['offset_rate'],config=cfg)
 for k,vv in independent.items():assert np.allclose(vv,cand['effective'][k],rtol=1e-12,atol=1e-10),(prefix,k)
 r=-.0005*sum(independent.values());r[ctx['interval_failed']]=0
 terminal=ctx['interval_failed']&(ctx['ticks']==ctx['interval_end_tick'])
 remaining=1000-ctx['interval_start_tick'][terminal]//4
 r[terminal]=-5-.002*8980.2*(1-.997**remaining)/(1-.997)
 def cached(alpha):return {k:original[f'{prefix}/score{alpha:g}/{k}'].astype(float) for k in item['components']}
 x0=cached(0);x1=cached(1);new0=sum(x0.values());new1=sum(x1.values())-x1['path']+.1*x0['path'];trans=(1-a)*new0+a*new1
 delta=float(max(abs(r-trans)));max_tick=max(max_tick,delta);assert delta<.0006,(prefix,a,delta)
 discount=.997**(ctx['ticks']//4);g=float(sum(r));gd=float(r@discount)
 exp=external[(group,case,method,a)];max_total=max(max_total,abs(g-float(exp['new_total_return'])));max_disc=max(max_disc,abs(gd-float(exp['new_discounted_return'])))
 assert np.isclose(g,float(exp['new_total_return']),atol=.002,rtol=1e-6)
 assert np.isclose(gd,float(exp['new_discounted_return']),atol=.002,rtol=1e-6)
 summaryrows.append(dict(dataset=group,case=case,method=method,score_alpha=a,return_raw_rebuilt=g,discounted_raw_rebuilt=gd))
 if a==1:path_caps.append(dict(dataset=group,case=case,method=method,cap_count=int(np.sum(p>=200)),samples=len(p)))
# Input-only probes: these are cost-function tests, not feasible robot rollouts.
probes=[]
for ey in [.0,.1,.3,1.,1.05,1.1,2.,10.]:
 x=candidate.costs(alpha=1.,chi=0.,ev=0.,ey=ey,eh=0.,peak_roll=0.,roll_rate=0.,speed=2.6,offsets=np.zeros(2),offset_rate=np.zeros(2),config=cfg)
 probes.append(dict(ey=ey,path_cost=float(x['effective']['path']),reward_per_second=-.1*float(x['cost_rate'])))
assert np.allclose([p['path_cost'] for p in probes[-3:]],20,rtol=0,atol=1e-12)
# Constant off-path stream vs a late restoration illustrates discounted motivation,
# without claiming dynamic feasibility or using it as a trained-policy result.
info=dict(raw_trajectory_rescores=len(summaryrows),config_common_items_match_current=True,reference_matches_independent_raw_formula=True,max_abs_tick_difference_from_float32_component_transform=max_tick,max_abs_total_difference=max_total,max_abs_discounted_difference=max_disc,simulator_steps=0,training_updates=0,path_cap_stats=path_caps,algebraic_path_probes=probes,
 failure_bound_alpha0=8980.2,failure_bound_alpha1=8800.2,path_cap_alpha1=20.,path_saturation_pure_lateral_m=1.05,objective_scope='offline fixed-trajectory candidate reward, no integration or adoption')
(OUT/'independent_results.json').write_text(json.dumps(info,indent=2)+'\n')
with (OUT/'raw_rescores.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=summaryrows[0]);w.writeheader();w.writerows(summaryrows)
print(json.dumps({k:v for k,v in info.items() if k not in ('path_cap_stats','algebraic_path_probes')},indent=2))
