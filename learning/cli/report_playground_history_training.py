"""Checkpoint and tail-error comparison for the bounded history-model pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
stages=['training','continue600'];logs=[];summaries=[]
for stage in stages:
    root=a.root/stage
    logs.extend(dict(json.loads(line),stage=stage) for line in (root/'scalars.jsonl').read_text().splitlines())
    windows=json.loads((root/'dev_windows.json').read_text())
    best=json.loads((root/'best.json').read_text())
    for horizon in [1.,3.]:
        rows=[r for r in windows if r['horizon_s']==horizon];errors=[r['max_xy_m'] for r in rows]
        summaries.append(dict(stage=stage,best_update=best['update'],horizon_s=horizon,windows=len(rows),cases=len({r['case'] for r in rows}),
                              mean_max_xy_m=float(np.mean(errors)),p95_max_xy_m=float(np.quantile(errors,.95)),max_xy_m=max(errors)))
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for h in [5,15,50,150]:
    rows=[r for r in logs if r['kind']=='training' and r['horizon_steps']==h]
    axes[0,0].plot([r['update'] for r in rows],[r['loss'] for r in rows],'.-',ms=2,lw=.6,label=f'{h*.02:g}s')
axes[0,0].set(title='Actual training loss, separated by horizon',yscale='log',ylabel='Dimensionless loss');axes[0,0].legend()
rows=[r for r in logs if r['kind']=='validation']
for h in ['1.0','3.0']:axes[0,1].plot([r['update'] for r in rows],[r['dev_mean_max_xy_m'][h] for r in rows],'.-',label=h+'s DEV')
axes[0,1].set(title='Fixed DEV windows: mean maximum XY error',ylabel='m');axes[0,1].legend()
rows=[r for r in logs if r['kind']=='training']
axes[1,0].plot([r['update'] for r in rows],[r['gradient_norm'] for r in rows],lw=.7);axes[1,0].set(title='Gradient norm before clipping',yscale='log')
for h in [1.,3.]:
    rows=[r for r in summaries if r['horizon_s']==h]
    axes[1,1].plot([r['best_update'] for r in rows],[r['p95_max_xy_m'] for r in rows],'o-',label=f'{h:g}s P95')
    axes[1,1].plot([r['best_update'] for r in rows],[r['max_xy_m'] for r in rows],'x--',label=f'{h:g}s worst')
axes[1,1].set(title='Selected checkpoints: tails are not the mean',ylabel='m');axes[1,1].legend()
for ax in axes.flat:ax.axvline(150,color='black',ls=':',lw=.8);ax.set_xlabel('Cumulative optimizer updates');ax.grid(alpha=.2)
fig.suptitle('At150: weights-only continuation with fresh Adam; balanced cases and mixed horizons')
fig.savefig(a.root/'training_curves.png',dpi=160);plt.close(fig)
(a.root/'checkpoint_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
late=[r for r in logs if r['kind']=='validation'][-6:]
decision=dict(selected_checkpoint='continue600/best.pt',selected_update=json.loads((a.root/'continue600/best.json').read_text())['update'],
              formal_convergence_verified=False,long_horizon_adopted=False,
              reason='Lower mean but1s worst nominal error increased; late DEV means fluctuate; pure-flat continuation stopped',
              last_evaluated_checkpoints=late,scope='One nominal-history initialization, not full terrain-conditioned M1 or P3 multi-seed comparison')
(a.root/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
lines=['# Phase2 history model: mean improves, long-horizon adoption rejected','',
       '![Nominal worst1s actual/predicted XY](nominal_tail/prediction_xy.png)','',
       '![Rough DEV actual/predicted XY](rough_dev/prediction_xy.png)','',
       '![Loss, fixed DEV, gradient and tail curves](training_curves.png)','',
       'A frozen nominal ARX plus GRU64 correction initializes from the actual381D controller/history state. Rollout uses predicted states and evolving hidden state, never future real controller history. RB, H75, physics and physical success criteria stay unchanged.','',
       '|Stage best|Horizon s|Windows/cases|Mean max XY m|P95 max XY m|Worst XY m|','|---|---:|---|---:|---:|---:|']
for r in summaries:lines.append(f"|{r['best_update']}|{r['horizon_s']}|{r['windows']}/{r['cases']}|{r['mean_max_xy_m']:.4f}|{r['p95_max_xy_m']:.4f}|{r['max_xy_m']:.4f}|")
lines+=['','The first150 updates took40.14s;600 additional updates took115.00s. Seven and25 checkpoints were evaluated at25-update intervals, with the shared initial150 boundary counted in both stages. Best means best among evaluated candidates, not every update or task-qualified. The continuation used weights only and fresh Adam because the first stage had not saved optimizer state; later stages now save it.','',
        'Nominal1s and3s use216 windows/17 cases and53 windows/14 cases. Horizons have different coverage:3s is not evidence that prediction errors decrease with time. Failed physical trajectories remain in the source; no future states are padded.','',
        'The continuation1s mean improved to0.0678m, but worst error rose from0.8120m to1.2085m. Late validation means fluctuate. Formal convergence and long-horizon suitability are not established; no more pure-flat updates are launched.','',
        '[All rough-domain rows and support errors](rough_dev/RESULTS.md), [nominal tail cases](nominal_tail/RESULTS.md). Rough0.3s worst XY is0.0958m, while1s/2s worst are0.5845/0.8424m. Short-horizon evidence is not whole-plan accuracy or safety.','',
        'Next: use measured executable motion primitives for the first segmented-versus-joint planning comparison; preserve the long-horizon model failures. Terrain-conditioned M1/M2 and independent new-map tests remain unfinished.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summaries,indent=2))
