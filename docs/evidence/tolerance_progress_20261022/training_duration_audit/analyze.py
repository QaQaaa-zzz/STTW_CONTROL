"""Read-only audit of optimization exposure; not a convergence test."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
names=['preference_recovery_train_v1','preference_recovery_calibrated_v1','preference_coverage_recovery_v1','preference_full_tail_v1']
records=[]
for name in names:
 for path in sorted((ROOT/'runs'/name).glob('*/metrics.jsonl')):
  rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
  assert [r['update'] for r in rows]==list(range(1,28))
  ppo=[r for r in rows if r['training_stage']=='ppo']; assert len(ppo)==25
  cfg=json.loads((path.parent/'effective_config.json').read_text())
  def block(rr):
   return {k:float(np.mean([r[k] for r in rr])) for k in ['mean_step_reward','domain_failures','physical_failures']}
  record=dict(run=name,arm=path.parent.name,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),warmup_updates=2,ppo_updates=25,policy_transitions=rows[-1]['policy_transitions_total'],lower_ticks=rows[-1]['training_control_ticks_total'],snippet_s=cfg['snippet_s'],parent_anchor_kl=cfg['parent_anchor_kl'],pulse_fired_total=sum(r['pulse']['fired'] for r in rows),pulse_fired_ppo=sum(r['pulse']['fired'] for r in ppo),first5=block(ppo[:5]),last5=block(ppo[-5:]),warmup_actor_unchanged=all(r['optimizer'].get('actor_bitwise_unchanged') is True for r in rows[:2]),hard_kl_stops=sum(bool(r['optimizer'].get('hard_kl_stop')) for r in ppo),nonfinite_stops=sum(bool(r['optimizer'].get('nonfinite_stop')) for r in ppo),accepted_epochs_total=sum(r['optimizer']['accepted_epochs'] for r in ppo),force_counts=np.sum([r['pulse']['force_histogram'] for r in rows],axis=0).tolist(),series=[{k:r[k] for k in ['update','mean_step_reward','domain_failures']} for r in rows])
  assert sum(record['force_counts'])==record['pulse_fired_total']
  records.append(record)
(OUT/'analysis.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records: print(r['run'],r['arm'],'pulse',r['pulse_fired_total'],'first/last',r['first5'],r['last5'],'stops',r['hard_kl_stops'])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(2,1,figsize=(11,8),sharex=True)
for r in records:
 x=[v['update'] for v in r['series']]
 label=r['run'].replace('preference_','').replace('_v1','')+'/'+r['arm']
 ax[0].plot(x,[v['mean_step_reward'] for v in r['series']],label=label)
 ax[1].plot(x,[v['domain_failures'] for v in r['series']])
ax[0].set_ylabel('Mean training step reward');ax[1].set_ylabel('Domain failures / update');ax[1].set_xlabel('Update (1-2 critic warmup)')
ax[0].legend(fontsize=8);fig.suptitle('Training diagnostics: changing sampled states; not fixed-panel success')
fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=140)
