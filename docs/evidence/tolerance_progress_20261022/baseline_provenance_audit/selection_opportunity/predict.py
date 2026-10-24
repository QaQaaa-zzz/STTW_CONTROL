"""Retrospective, campaign-held-out check of fixed pre-disturbance scores."""
from pathlib import Path
import json
import hashlib
import numpy as np
from sttw_control.tolerance_state_features import unstable_features

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT / 'runs/selection_opportunity_audit_v1'
parents = [('low_speed_reference_authority_v1', 'low_speed_balance_v1'),
           ('reference_authority_expansion_v1', 'reference_authority_expansion_prefix_v1')]
sources, rows = [], []
for campaign, prefix in parents:
    def read(path):
        raw = path.read_bytes()
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        return json.loads(raw)
    plan = read(ROOT/'runs'/campaign/'plan.json')
    declaration = read(Path(plan['source_declaration']))
    cc, ac = declaration['source']['controller'], declaration['source']['actuator']
    declared_states = {c['state_id'] for c in plan['cases']}
    for r in read(ROOT/'runs'/prefix/'prefix_results.json')['cases']:
        if r['state_id'] not in declared_states:
            continue
        f = unstable_features(speed_proxy=r['wheel_proxy_m_s'], steer=r['steer_rad'],
            roll=r['roll_rad'], roll_rate=r['roll_rate_rad_s'],
            disturbance=r['controller_disturbance'], controller=cc)
        lam, b, delta, T = f['lambda_u'], f['b_u'], r['steer_rad'], .3
        lo = max(-ac['steer_rate_limit'], (-ac['steer_limit']-delta)/T)
        hi = min(ac['steer_rate_limit'], (ac['steer_limit']-delta)/T)
        q = -np.expm1(-lam*T)/lam
        margin = min(-q*min(b*lo,b*hi)-f['z_u'], f['z_u']+q*max(b*lo,b*hi))
        K = .7192241548921705 * (-np.expm1(-lam*.1))/lam
        rows.append(dict(campaign=campaign, state_id=r['state_id'],
            scores=dict(formula=-margin/K, speed=-r['wheel_proxy_m_s'],
                        roll=abs(r['roll_rad']), roll_rate=abs(r['roll_rate_rad_s'])),
            predicted_min_force_N=margin/K))
# Features above use actual prepared prefix only, one second before pulse.
# Historical labels were previously inspected: this is not prospective registration.
(OUT/'prefix_scores.json').write_text(json.dumps(dict(rows=rows,sources=sources),indent=2)+'\n')
opportunity = json.loads((OUT/'receipt.json').read_text())
lookup = {(r['campaign'],r['state_id']):r for r in opportunity['records']}
for r in rows:
    r['modes'] = lookup[r['campaign'],r['state_id']]['modes']
    r['hold_unsafe'] = r['modes']['hold']['safe_count'] < 4
assert len(rows) == 12

def evaluate(choices):
    choices = [bool(x) for x in choices]
    selected = [r['modes']['react_accel' if a else 'hold'] for r,a in zip(rows,choices)]
    return dict(safe=sum(r['safe_count'] for r in selected), total=48,
                interventions=sum(choices),
                command_exposure_m=sum(r['speed_command_exposure_m'] for r in selected),
                missed_risky_states=sum(r['hold_unsafe'] and not a for r,a in zip(rows,choices)))

results = {}
for name in ('formula','speed','roll','roll_rate'):
    pos = [r['scores'][name] for r in rows if r['hold_unsafe']]
    neg = [r['scores'][name] for r in rows if not r['hold_unsafe']]
    auc = float(np.mean([float(a>b)+.5*float(a==b) for a in pos for b in neg]))
    choices, folds = [], []
    for r in rows:
        train = [x for x in rows if x['campaign'] != r['campaign']]
        # Fixed objective: miss as few risky TRAIN states as possible, then intervene least.
        thresholds = [-np.inf]+sorted({x['scores'][name] for x in train})+[np.inf]
        def objective(t):
            return (sum(x['hold_unsafe'] and x['scores'][name]<t for x in train),
                    sum(x['scores'][name]>=t for x in train))
        threshold = min(thresholds, key=objective)
        act = r['scores'][name] >= threshold
        choices.append(bool(act));folds.append(dict(state_id=r['state_id'],
            train_campaign=train[0]['campaign'],threshold=str(threshold),intervene=bool(act)))
    results[name] = dict(descriptive_state_AUC=auc, campaign_held_out=evaluate(choices), folds=folds)
results['formula_fixed_56N'] = evaluate([r['predicted_min_force_N']<56 for r in rows])
results['always_accelerate'] = evaluate([True]*12)
results['never_accelerate'] = evaluate([False]*12)
out = dict(results=results, rows=rows, sources=sources,
           new_physics_steps=0, optimization_steps=0,
           scope='Retrospective DEV. Two related campaigns, not independent test or deployment. Threshold fitting uses opposite campaign only; formula coefficients unchanged.')
(OUT/'prediction_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k: {a:b for a,b in v.items() if a!='folds'} for k,v in results.items()},indent=2))
