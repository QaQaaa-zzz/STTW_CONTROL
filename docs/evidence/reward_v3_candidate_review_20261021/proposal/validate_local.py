"""Executed local validation: real published pair integrals + synthetic unit tests.
Synthetic inputs are not robot simulations and are never labeled as performance.
"""
from pathlib import Path
import csv,json,importlib.util
import numpy as np
from reward_reference import costs,failure_reward
ROOT=Path(__file__).parent
cfg=json.loads((ROOT/'reward_config.json').read_text())
fixtures=json.loads((ROOT/'historical_pair_inputs.json').read_text())['pairs']
checks=[]
def check(name,condition,detail=''):
    ok=bool(condition);checks.append(dict(name=name,passed=ok,detail=detail))
    if not ok:raise AssertionError(name+': '+str(detail))

rows=[];sensitivity=[]
for p in fixtures:
    d0=p['old_delta0'];d1=p['old_delta1']-p['path_delta1']+.1*p['path_delta0']
    check(p['case']+' new endpoint ranking',d0<0<d1,str((d0,d1)))
    for a in (0.,.25,.5,.75,1.):
        rows.append(dict(case=p['case'],score_alpha=a,new_delta_policy1_minus_policy0=(1-a)*d0+a*d1,
            new_discounted_delta='UNVERIFIED',source='exact affine transform of published component integrals'))
    for eps in (.05,.075,.1,.125,.15,.2):
        val=p['old_delta1']-p['path_delta1']+eps*p['path_delta0']
        sensitivity.append(dict(case=p['case'],secondary_ratio=eps,new_alpha1_delta=val,speed_trajectory_preferred=val>0))
    p['new_delta0']=d0;p['new_delta1']=d1
    p['eps_upper_for_positive_delta']=(p['old_delta1']-p['path_delta1'])/(-p['path_delta0'])
    p['alpha_crossover_on_this_pair']=-d0/(d1-d0)
(ROOT/'validated_historical_endpoints.json').write_text(json.dumps(fixtures,indent=2)+'\n')
for name,data in [('historical_five_alpha_scores.csv',rows),('secondary_ratio_sensitivity.csv',sensitivity)]:
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)

# Independent scalar expression of the baseline unversioned Coverage formula.
def legacy_scalar(x,alpha):
    c=cfg;H=lambda v: v*v if abs(v)<=1 else 2*abs(v)-1
    ev,ey,eh,q=x['ev'],x['ey'],x['eh'],x['chi']
    raw={'speed':(10-9*q*(1-alpha))*H(ev/.1),
         'path':(10-9*q*alpha)*(H(ey/.1)+.5*H(eh/.1)),
         'primary':20*q*((1-alpha)*H(max(abs(ey)-.1,0)/.1)+alpha*H(max(abs(ev)-.05,0)/.1)),
         'roll':80*H(max(x['peak_roll']-.26,0)/.04),
         'working_roll':240*H(max(x['peak_roll']-.30,0)/.02),
         'roll_rate':H(max(abs(x['roll_rate'])-.8,0)/.8),
         'overspeed':2*H(max(ev-.05,0)/.1),
         'low_speed':4*H(max(1.5-x['speed'],0)/.2),
         'offset_magnitude':.02*sum((np.asarray(x['offsets'])/[1.,.2])**2),
         'offset_rate':.03*sum((np.asarray(x['offset_rate'])/[1.,.4])**2)}
    return {k:min(v,c['component_caps'][k]) for k,v in raw.items()}

rng=np.random.default_rng(911401);n=10000
x=dict(chi=rng.uniform(0,1,n),ev=rng.uniform(-3,3,n),ey=rng.uniform(-3,3,n),eh=rng.uniform(-2,2,n),
       peak_roll=rng.uniform(0,.9,n),roll_rate=rng.uniform(-6,6,n),speed=rng.uniform(0,4,n),
       offsets=rng.uniform(-1,1,(n,2)),offset_rate=rng.uniform(-6,6,(n,2)))
y0=costs(alpha=0.,config=cfg,**x);y1=costs(alpha=1.,config=cfg,**x)
old0=np.array([sum(legacy_scalar({k:v[i] for k,v in x.items()},0).values()) for i in range(n)])
check('alpha0 equals independent baseline, including cap-active states',np.allclose(y0['cost_rate'],old0,atol=1e-10),f'{n} synthetic states')
a=rng.uniform(0,1,n);ya=costs(alpha=a,config=cfg,**x)
check('affine conditional cost for arbitrary alpha',np.allclose(ya['cost_rate'],(1-a)*y0['cost_rate']+a*y1['cost_rate'],atol=1e-10))
check('global bound 8980.2 remains conservative',np.all((ya['cost_rate']>=0)&(ya['cost_rate']<=8980.2+1e-9)))
for key in ('roll','working_roll','roll_rate','overspeed','low_speed','offset_magnitude','offset_rate'):
    check('common component unchanged: '+key,np.array_equal(y0['effective'][key],y1['effective'][key]))
zero=dict(chi=0.,ev=0.,ey=0.,eh=0.,peak_roll=0.,roll_rate=0.,speed=2.6,offsets=np.zeros(2),offset_rate=np.zeros(2))
check('perfect straight tracking has zero cost for all alpha',all(float(costs(alpha=a,config=cfg,**zero)['cost_rate'])==0 for a in (0,.25,.5,.75,1)))
# Same speed/safety, path offset positive: no-return is never free.
drift=dict(zero,ey=.3)
drift_cost=float(costs(alpha=1.,config=cfg,**drift)['cost_rate'])
check('alpha1 straight path-offset penalty remains strictly positive',np.isclose(drift_cost,5.))
check('one extra second at ey=.3 adds 0.5 reward loss',np.isclose(.1*drift_cost,.5))
stop=dict(zero,speed=0.,ev=-2.6)
check('stopping is penalized at both endpoints',all(float(costs(alpha=a,config=cfg,**stop)['cost_rate'])>float(costs(alpha=a,config=cfg,**zero)['cost_rate']) for a in (0,1)))
unsafe=dict(zero,peak_roll=.32)
check('safety cost at .32 rad identical at both endpoints',np.isclose(costs(alpha=0.,config=cfg,**unsafe)['cost_rate'],costs(alpha=1.,config=cfg,**unsafe)['cost_rate']))
check('failure terminal replacement matches pinned formula',np.isclose(failure_reward(1000,cfg),-5-.1*.02*8980.2*(1-.997**1000)/(1-.997)))
# Test the aggregate transform helper using SYNTHETIC records, not the real full file.
from rescore_from_published import convert
base=dict(dataset='SYNTHETIC',case='unit',method='policy0',samples=4,failed=False,final_hold=True,behavior_alpha=0.,physical={})
fake=dict(scoring_config={**cfg,'reward_version':None},records=[dict(base,score_alpha=0.,total_return=-8.,discounted_return=-8.,components={'speed':-1.,'path':-4.,'primary':-2.,'roll':-1.}),dict(base,score_alpha=1.,total_return=-10.,discounted_return=-10.,components={'speed':-5.,'path':-1.,'primary':-3.,'roll':-1.})])
converted=convert(fake)
check('aggregate transform helper on synthetic data',np.isclose(converted[-1]['new_total_return'],-9.4))
# Optional independent backend agreement with this module's exact same algebra.
if importlib.util.find_spec('jax') is not None:
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    f=jax.jit(lambda aa,xx:costs(alpha=aa,config=cfg,xp=jnp,**xx)['cost_rate'])
    out=np.asarray(f(jnp.asarray(a),{k:jnp.asarray(v) for k,v in x.items()}))
    check('JAX JIT / NumPy numerical agreement',np.allclose(out,ya['cost_rate'],atol=1e-9,rtol=1e-10),f'max_abs={np.max(np.abs(out-ya["cost_rate"]))}')
else:
    checks.append(dict(name='JAX JIT / NumPy numerical agreement',passed=None,detail='JAX unavailable; not run'))
summary=dict(checks=checks,synthetic_states=n,real_historical_pairs=4,
    exact_new_integrated_scores_on_real_pairs=20,new_discounted_real_scores_verified=False,
    original_physical_npz_accessible=False,robot_simulation_steps=0,ppo_updates=0,
    summary='Full-episode undiscounted historical preferences aligned; not a PPO success claim.')
(ROOT/'local_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'checks_passed':sum(c['passed'] is True for c in checks),'checks_not_run':sum(c['passed'] is None for c in checks),'real_pairs':[{k:p[k] for k in ('case','new_delta0','new_delta1','eps_upper_for_positive_delta','alpha_crossover_on_this_pair')} for p in fixtures]},indent=2))
