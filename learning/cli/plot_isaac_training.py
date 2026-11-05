"""Read saved training samples; export grouped diagnostics without physics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--updates',type=int,nargs='*');a=p.parse_args()
history=json.loads((a.run/'history.json').read_text());contract=json.loads((a.run/'contract.json').read_text())
selected=set(a.updates) if a.updates is not None else {history[0]['update'],history[-1]['update'],max(history,key=lambda r:r['mean_step_reward'])['update']}
records=[];panels=[]
for row in history:
    update=row['update'];bundle=Path(row['scored_policy']);route=np.load(bundle/'route.npz');cfg=json.loads((bundle/'config.json').read_text());groups={}
    for ep in row['episodes']:
        terrain=ep.get('terrain',contract.get('terrain_schedule',['unspecified']*len(history))[update] if 'terrain_schedule' in contract else 'unspecified')
        receipt=Path(ep['receipt']) if 'receipt' in ep else a.run/('u%03d_alpha%s.json'%(update,ep['alpha']))
        d=np.load(receipt.with_suffix('.rewards.npz'));physical=json.loads(receipt.read_text());meta=json.loads(receipt.with_suffix('.rewards.json').read_text())
        if not np.isfinite(d['reward']).all():raise ValueError('nonfinite reward in '+str(receipt))
        np.testing.assert_allclose(d['reward'].sum(),d['tick_reward'].sum(),atol=1e-4,rtol=1e-5)
        record=dict(update=update,terrain=terrain,alpha=ep['alpha'],seed=physical['sampling']['seed'],policy=str(bundle),seconds=physical['physics_steps']*physical['dt_s'],policy_steps=len(d['reward']),reward_total=float(d['reward'].sum()),mean_step_reward=float(d['reward'].mean()),terminal_penalty=meta['terminal_penalty'],termination=physical['termination'],speed_rmse=float(np.sqrt(np.mean((d['speed']-physical['drive_speed'])**2))),cross_rmse=float(np.sqrt(np.mean(d['cross']**2))),max_abs_roll=float(d['peak_roll'].max()),role='stochastic training sample; actual duration; not matched DEV')
        component_sums=-cfg['path_reward']['scale']*cfg['timing']['lower_dt_s']*d['effective_components'].sum(0)
        record.update({'normal_reward_'+str(k):float(v) for k,v in zip(d['component_names'],component_sums)});records.append(record)
        if update in selected:groups.setdefault(terrain,[]).append((ep,d,physical))
    for terrain,episodes in groups.items():
        fig,axes=plt.subplots(2,3,figsize=(15,8));keep=route['s']<=12;axes[0,0].plot(*route['xy'][keep].T,'k--',label='Original reference')
        for ep,d,physical in episodes:
            xy=d['xy'];label='alpha=%s (%s)'%(ep['alpha'],ep['termination']);line,=axes[0,0].plot(*xy.T,label=label);axes[0,0].plot(*xy[-1],marker='x',color=line.get_color())
            steps=np.arange(1,len(d['reward'])+1);time=np.minimum(np.arange(1,len(d['speed'])+1)*.005,physical['physics_steps']*.0002)
            axes[0,1].plot(steps,d['reward'],color=line.get_color());axes[0,2].plot(steps,np.cumsum(d['reward']),color=line.get_color())
            for ax,key in zip(axes[1],['speed','cross','peak_roll']):ax.plot(time,d[key],color=line.get_color())
        axes[0,0].set(xlabel='World X [m]',ylabel='World Y [m]',aspect='equal');axes[0,0].legend(fontsize=6)
        axes[1,0].axhline(physical['drive_speed'],color='grey',linestyle='--')
        for ax,title in zip(axes.flat,['Actual/reference XY','Reward per policy transition','Cumulative reward','True speed [m/s]','Signed cross-track [m]','Peak roll magnitude [rad]']):ax.set_title(title);ax.grid(alpha=.2)
        for ax in axes[0,1:]:ax.set_xlabel('Policy step (50 Hz)')
        for ax in axes[1]:ax.set_xlabel('Time [s]')
        fig.suptitle('Training samples from policy_%03d | %s | stochastic, no matched baseline\nX marks actual endpoint; terrain begins at world X=2m (except flat); no imposed external force pulse'%(update,terrain))
        fig.tight_layout();name='sampling_%03d_%s.png'%(update,terrain);fig.savefig(a.run/name,dpi=150);plt.close(fig);panels.append(name)
with (a.run/'training_groups.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
fig,axes=plt.subplots(2,3,figsize=(14,7));updates=[r['update'] for r in history]
for ax,(name,values) in zip(axes.flat,[('Policy loss',[r['loss_mean'][0] for r in history]),('Value loss',[r['loss_mean'][1] for r in history]),('Full-batch Gaussian KL',[r['kl'] for r in history]),('Training mean step reward',[r['mean_step_reward'] for r in history]),('Training failure fraction',[sum(e['termination']!='budget_end' for e in r['episodes'])/len(r['episodes']) for r in history]),('Accepted updates (cumulative)',np.cumsum([r['accepted'] for r in history]))]):
    ax.plot(updates,values,'o-');ax.set(xlabel='Stage update (zero based)',ylabel=name);ax.grid(alpha=.2)
fig.suptitle('Training diagnostics | pre-update samples belong to their generating policy\nTraining failure and reward curves are not fixed DEV recovery evaluation');fig.tight_layout();fig.savefig(a.run/'training_diagnostics.png',dpi=150);plt.close(fig)
terrains=list(dict.fromkeys(r['terrain'] for r in records));fig,axes=plt.subplots(len(terrains),3,figsize=(14,3*len(terrains)),squeeze=False)
for i,terrain in enumerate(terrains):
 for alpha in contract['alphas']:
  rows=[r for r in records if r['terrain']==terrain and r['alpha']==alpha];x=[r['update'] for r in rows]
  for ax,key in zip(axes[i],['mean_step_reward','speed_rmse','cross_rmse']):ax.plot(x,[r[key] for r in rows],'.-',label='alpha='+str(alpha));ax.set_title(terrain+' / '+key);ax.set_xlabel('Stage update');ax.grid(alpha=.2)
 axes[i,0].legend(fontsize=7)
fig.suptitle('Training groups: five exact alpha levels per terrain | actual-duration RMSE\nDifferent random actions and failure durations: not paired improvement evidence');fig.tight_layout();fig.savefig(a.run/'alpha_terrain_diagnostics.png',dpi=150);plt.close(fig)
index=dict(completed_updates=len(history),episodes=len(records),plotted_updates=sorted(selected),panels=panels,data='training_groups.csv',role='saved training samples only; no new physics',failures=sum(r['termination']!='budget_end' for r in records))
(a.run/'diagnostics_index.json').write_text(json.dumps(index,indent=2));print(json.dumps(index))
