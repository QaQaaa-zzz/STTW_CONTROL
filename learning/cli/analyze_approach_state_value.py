"""Whole-root exploratory validation and all-root XY for short approach data."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.jump_connection import decision_features

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
a = p.parse_args()
data = json.loads((a.run/'comparison.json').read_text())
roots = list(dict.fromkeys(r['origin_id'] for r in data['rows']))
actions = sorted(set((r['speed_m_s'], r['alpha']) for r in data['rows']))
names = ['cross_track', 'heading_error', 'speed', 'remaining_path_to_release',
         'lateral_velocity_heading_frame', 'roll', 'roll_rate']
samples = []; prefix = []; figures = []
colors = ['#0072B2', '#D55E00', '#009E73']
for root in roots:
    rows = sorted([r for r in data['rows'] if r['origin_id']==root],
                  key=lambda r:(r['speed_m_s'], r['alpha']))
    fig, ax = plt.subplots(figsize=(7, 6), layout='constrained')
    traces = []; decisions = []
    for j, r in enumerate(rows):
        out = Path(r['summary']).parent
        protocol = json.loads((out/'protocol.json').read_text()); spec = protocol['spec']
        d = np.load(out/'trajectory.npz'); trace = d['trace'][d['valid'][:,0],0]
        traces.append(trace)
        if j==0:
            route=np.load(out/'ground_route.npz')['xy']; ax.plot(*route.T, 'k:', label='Original reference')
        ax.plot(trace[:,1], trace[:,2], color=colors[j%3], linestyle='-' if j>=3 else '--',
                label=f"v={r['speed_m_s']}, alpha={r['alpha']}; success={int(r['complete'])}")
        ax.plot(trace[-1,1], trace[-1,2], 'o' if r['complete'] else 'x', color=colors[j%3])
        event = out/'connection_decision.json'
        if event.exists():
            decision=json.loads(event.read_text()); decisions.append(decision['time'])
            if j==0:
                ax.plot(decision['root'][0][0],decision['root'][0][1],'ks',markersize=7,label='Action selection; before: v=2, alpha=.5')
            _, named=decision_features(decision,out,protocol,spec)
            samples.append(dict(root=root, action=j, state=[named[k] for k in names], y=int(r['complete'])))
        else: decisions.append(None)
    cutoff=min(t for t in decisions if t is not None) if any(t is not None for t in decisions) else min(t[-1,0] for t in traces)
    pre=[t[t[:,0]<=cutoff+1e-7] for t in traces]
    equal=all(x.shape==pre[0].shape and np.array_equal(x,pre[0]) for x in pre)
    prefix.append(dict(root=root, exact_shared_prefix=equal, decision_times=decisions, prefix_end_s=float(cutoff)))
    ax.axvline(spec['connection_candidate']['decision_x_m'],color='gray',linestyle=':',linewidth=.7)
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f'{root}\nShared upstream v=2, alpha=.5; six actions only AFTER decision\nTRAIN update2; endpoint x = incomplete, o = complete')
    ax.set_aspect('equal',adjustable='box'); ax.legend(fontsize=7)
    path=a.run/(root+'_xy_all.png'); fig.savefig(path,dpi=140); plt.close(fig); figures.append(path.name)

# Fixed regularization; every normalization and fit excludes the entire held-out root.
# Action-specific linear state interactions, no threshold or hyperparameter search.
def fit_predict(train, test, count):
    x=np.array([s['state'][:count] for s in train]); q=np.array([s['state'][:count] for s in test])
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-8]=1
    def design(rows, values):
        one=np.eye(6)[[s['action'] for s in rows]]
        return np.column_stack([np.ones(len(rows)),one,(one[:,:,None]*((values-mean)/scale)[:,None,:]).reshape(len(rows),-1)])
    X=design(train,x); Q=design(test,q); y=np.array([s['y'] for s in train])
    penalty=np.ones(X.shape[1]); penalty[0]=0
    def objective(w):
        z=X@w
        return np.logaddexp(0,z).sum()-y@z+.5*np.sum(penalty*w*w), X.T@(expit(z)-y)+penalty*w
    result=minimize(objective,np.zeros(X.shape[1]),jac=True,method='L-BFGS-B')
    if not result.success: raise RuntimeError(result.message)
    return expit(Q@result.x)

predictions=[]
for root in roots:
    train=[s for s in samples if s['root']!=root]; test=[s for s in samples if s['root']==root]
    if not test: continue
    geom=fit_predict(train,test,4); dynamic=fit_predict(train,test,7)
    fixed=np.array([np.mean([s['y'] for s in train if s['action']==j]) for j in range(6)])
    for i,s in enumerate(test): predictions.append(dict(s, geometry=float(geom[i]), dynamics=float(dynamic[i]), fixed=float(fixed[s['action']])))
scores={}
for method in ('fixed','geometry','dynamics'):
    chosen=[]
    for root in roots:
        rows=[r for r in predictions if r['root']==root]
        chosen.append(dict(root=root, reached=bool(rows), choice=max(rows,key=lambda r:r[method]) if rows else None))
    scores[method]=dict(successes=sum(c['choice']['y'] for c in chosen if c['reached']), denominator=len(roots),
                        brier=float(np.mean([(r[method]-r['y'])**2 for r in predictions])), choices=chosen)
result=dict(role='Exploratory leave-one-training-root-out; not independent TEST; direct Q baseline, not two-stage entry predictor',
            features=names, regularization=1, samples=samples,predictions=predictions,scores=scores,
            prefix=prefix,figures=figures,upstream_failed_roots=[r for r in roots if not any(s['root']==r for s in samples)])
(a.run/'state_value.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Short approach: whole-root state-value diagnostic','',
       'All 20 TRAIN roots, six actions, frozen update2. Direct outcome models; not independent TEST or a two-stage planner.',
       'L2 logistic action/state interactions, strength 1 fixed without tuning; normalization fitted inside each root fold.',
       'Brier uses reached decision rows; full-route selection denominator retains all roots. Ties use declared action order.', '',
       '| Model | Whole-route success | Brier on reached rows |','|---|---:|---:|']
for k,v in scores.items(): lines.append(f"| {k} | {v['successes']}/{len(roots)} | {v['brier']:.5f} |")
lines+=['',f"Exact shared prefixes: {sum(r['exact_shared_prefix'] for r in prefix)}/{len(roots)}.",'',
        '## All-root XY', *[f'![{r}]({f})' for r,f in zip(roots,figures)]]
(a.run/'STATE_VALUE.md').write_text('\n'.join(lines)+'\n')
fig, ax=plt.subplots(figsize=(8,7),layout='constrained')
root_evidence=[]
for root in roots:
    r=next(r for r in data['rows'] if r['origin_id']==root and r['speed_m_s']==2 and r['alpha']==.5)
    out=Path(r['summary']).parent; d=np.load(out/'trajectory.npz');t=d['trace'][d['valid'][:,0],0]
    event=out/'connection_decision.json';cut=json.loads(event.read_text())['time'] if event.exists() else float(t[-1,0])
    t=t[t[:,0]<=cut+1e-7]
    ax.plot(t[:,1],t[:,2],label=root[-7:]);ax.plot(t[-1,1],t[-1,2],'o' if event.exists() else 'x',color=ax.lines[-1].get_color())
    root_evidence.append(dict(root=root,start_xy=t[0,1:3].tolist(),decision_reached=event.exists(),end_time_s=cut))
ax.plot(*np.load(out/'ground_route.npz')['xy'].T,'k:',label='Original reference')
ax.set(xlim=(-5.5,-1),ylim=(-9.3,1.5),xlabel='World X (m)',ylabel='World Y (m)',
       title='20 distinct TRAIN roots, upstream only\nAll use v=2 m/s, alpha=.5; o: decision reached, x: upstream exit')
ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=7,bbox_to_anchor=(1.02,1),loc='upper left')
fig.savefig(a.run/'upstream_roots_xy.png',dpi=140);plt.close(fig)
(a.run/'upstream_roots.json').write_text(json.dumps(root_evidence,indent=2)+'\n')
print(json.dumps({k:{n:v[n] for n in ('successes','denominator','brier')} for k,v in scores.items()}))
