from pathlib import Path
import json,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).parent;d=Path('docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p4_sequence/phase_order');screen=json.loads((r/'screen/checkpoints.json').read_text());summary=json.loads((r/'training_summary.json').read_text());fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,layout='constrained')
for seed,color in enumerate(['#0072B2','#D55E00','#009E73']):
 logs=json.loads((r/'training'/f'seed{seed}'/'log.json').read_text());origin=logs[0]['update'];x=[v['update']-origin for v in logs];ready={v['update']:v['dev_ready'] for v in screen if v['seed']==seed}
 axes[0].plot(x,[v['dev_score'] for v in logs],color=color,label=f'Seed{seed}');axes[1].plot(x,[ready[v['update']] for v in logs],color=color);axes[2].plot(x,[v['learning_rate'] for v in logs],color=color)
 last=[ready[v['update']] for v in logs[-10:]];summary[seed]['last10_ready_range']=[min(last),max(last)];summary[seed]['last_ready']=last[-1]
axes[0].legend();axes[0].set_ylabel('DEV supervised loss');axes[1].set_ylabel('Handoff ready / 72');axes[1].set_ylim(-1,73);axes[2].set_ylabel('Learning rate');axes[2].set_xlabel('Additional supervised updates');axes[0].set_title('Ordered-phase continuation: all41 checkpoints per seed, original DEV only')
for a in axes:a.grid(alpha=.25)
fig.savefig(d/'learning_curves.png',dpi=150);plt.close(fig);(r/'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
