"""All-prefix XY and exhaustive recent paired-change diagnostics, no new physics."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_best import atomic_json,digest
from sttw_control.path_command_reward import configured_cost_keys
from sttw_control.survival_endpoint_figures import reward_blocks


def render(review):
    receipt=json.loads((review/'receipt.json').read_text());plan=json.loads((review/'plan.json').read_text())
    training=Path(plan['training_run']);through=plan['through_update']
    out=review/'figures';out.mkdir(exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),stage='read_only_plots',completed_updates=0,physics_steps=0,optimizer_steps=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(review/'plot_status.json',status)
    update()
    import sys
    sys.path.insert(0,str(Path('learning/cli').resolve()))
    from prepare_short_recovery_campaign import register
    register(review/'PLOT_ACTIVE_RUN.json','STTW_CONTROL 训练中期全轨迹与配对变化绘图',review/'plot_status.json')
    try:
        for p,sha in receipt['source_sha256'].items():
            if digest(p)!=sha:raise ValueError('review source changed: '+p)
        train=json.loads((training/'plan.json').read_text());cfg=json.loads((training/'effective_config.json').read_text())
        cases=train['cases'];dt=cfg['timing']['lower_dt_s'];scale=cfg['path_reward']['scale']*dt;keys=configured_cost_keys(cfg)
        rows=json.loads((review/'episodes.json').read_text());strata=json.loads((review/'strata.json').read_text())
        paths={}
        for c in cases:
            if c['route'] not in paths:
                with np.load(training/(c['route']+'_path.npz')) as z:paths[c['route']]=z['xy']
        gradients=json.loads((review/'gradient_diagnostics.json').read_text())
        rawhashes={};changes=[];covered=[]
        for update_index in range(1,through+1):
            directory=training/'collections'/f'update_{update_index:04d}';complete=json.loads((directory/'collection_complete.json').read_text())
            logs=[]
            fields=['active_tick','actual_xy','contact_failure','domain_exit','fallback_active','time','task_reward_valid',
                'actual_forward_speed','wheel_speed_proxy','reference','actual_delta','phi','peak_roll','phi_dot',
                'path_cross_track','path_heading_error','scored_tick_reward','tick_reward','effective_costs','applied_force_impulse']
            for name,sha in sorted(complete['raw_chunks'].items()):
                path=directory/name
                if digest(path)!=sha:raise ValueError('raw chunk changed: '+str(path))
                rawhashes[str(path)]=sha
                with np.load(path) as z:logs.append({k:z['lower_'+k] for k in fields})
            raw={k:np.concatenate([p[k] for p in logs]) for k in fields}
            fig,axes=plt.subplots(4,3,figsize=(16,18),squeeze=False)
            for context in range(4):
                for ai,alpha in enumerate([0.,.5,1.]):
                    ax=axes[context,ai];indices=[i for i,c in enumerate(cases) if c['source_case']==context and c['alpha']==alpha]
                    c=cases[indices[0]];ref=paths[c['route']];ax.plot(*ref.T,'--',color='.65',lw=.7,label='Original path')
                    allxy=[]
                    for i in indices:
                        case=cases[i];active=raw['active_tick'][:,i];xy=raw['actual_xy'][active,i];allxy.append(xy)
                        color='tab:blue' if case['arm']=='task_only' else 'tab:orange';style=['-','--','-.',':'][case['replicate']]
                        ax.plot(*xy.T,color=color,ls=style,lw=.8,label=f'{case["arm"]} draw{case["replicate"]}')
                        held=active & raw['fallback_active'][:,i]
                        if held.any():ax.plot(*raw['actual_xy'][held,i].T,':',color=color,lw=1.6)
                        contact=(raw['contact_failure'][:,i] & active).any();ax.scatter(*xy[-1],marker='x' if contact else 'o',color=color,s=15)
                        exited=np.flatnonzero(active & raw['domain_exit'][:,i])
                        if exited.size:ax.scatter(*raw['actual_xy'][exited[0],i],marker='s',s=18,facecolors='none',edgecolors=color)
                        covered.append((update_index,i))
                    xy=np.concatenate(allxy);center=(xy.max(0)+xy.min(0))/2;span=max(float(np.ptp(xy,axis=0).max())*.56,.1)
                    ax.set(xlim=(center[0]-span,center[0]+span),ylim=(center[1]-span,center[1]+span),
                        title=f'{c["route"]} v={c["speed"]} F={c["force_N"]}N alpha={alpha}',xlabel='World X (m)',ylabel='World Y (m)')
                    ax.set_aspect('equal');ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=5,ncol=2)
            fig.suptitle(f'TRAINING collection{update_index}: all96 actual episodes; square=domain, thick dotted=hold, x=contact\nFour action draws are not independent training seeds; own trajectories differ before the pulse')
            fig.tight_layout();fig.savefig(out/f'update_{update_index:04d}_all_xy.png',dpi=110);plt.close(fig)
            # Exhaust every gain AND loss in the fixed latest-four diagnostic window.
            if update_index>=through-3:
                local=[r for r in rows if r['update']==update_index and r['force_N']]
                pairs={}
                for row in local:pairs.setdefault((row['source_case'],row['alpha'],row['replicate']),{})[row['arm']]=row
                for pairkey,pair in pairs.items():
                    a,b=pair['task_only'],pair['physical_credit']
                    if a['outcome']==b['outcome']:continue
                    change='gain' if b['outcome']=='survived' else 'loss'
                    filename=f'update_{update_index:04d}_context{pairkey[0]}_alpha{pairkey[1]:g}_draw{pairkey[2]}_{change}.png'
                    fig,axes=plt.subplots(9,2,figsize=(12,22),squeeze=False)
                    for ci,row in enumerate([a,b]):
                        i=row['case_index'];case=cases[i];p={k:v[:,i] for k,v in raw.items()};active=p['active_tick'];valid=active & p['task_reward_valid'];t=p['time'][active]
                        line=lambda ri,key,label,mask=active:axes[ri,ci].plot(p['time'][mask],p[key][mask],label=label,lw=.9)
                        line(0,'actual_forward_speed','True forward');line(0,'wheel_speed_proxy','Wheel proxy')
                        axes[0,ci].plot(t,p['reference'][active,0],label='Given reference');axes[0,ci].axhline(case['speed'],ls=':',color='.4',label='Original request')
                        line(1,'actual_delta','Actual steer');axes[1,ci].plot(t,p['reference'][active,1],label='Given reference')
                        line(2,'phi','Signed roll');line(2,'peak_roll','Peak abs roll');line(3,'phi_dot','Roll rate')
                        line(4,'path_cross_track','Path error',valid);line(5,'path_heading_error','Heading error',valid)
                        reward=reward_blocks(p,scale=scale)
                        for name in ['scored_reward','running_reward','terminal_charge']:axes[6,ci].plot(reward['time'],reward[name],label=name,lw=.8)
                        for k,name in enumerate(keys):axes[7,ci].plot(reward['time'],reward['signed_components'][:,k],label='-'+name,lw=.7)
                        for k,name in enumerate(['X','Y','Z']):axes[8,ci].plot(t,p['applied_force_impulse'][active,k]/dt,label='Applied '+name)
                        domain=np.flatnonzero(active & p['domain_exit']);contact=np.flatnonzero(active & p['contact_failure'])
                        for ri,unit in enumerate(['Speed (m/s)','Steer (rad)','Roll (rad)','Roll rate (rad/s)','Path error (m)',
                            'Heading error (rad)','Reward / actual policy interval','Signed reward components','Applied mean force (N)']):
                            ax=axes[ri,ci];onset=case['pulse_onset_task_tick']*dt;ax.axvspan(onset,onset+case['pulse_duration_s'],color='orange',alpha=.15)
                            if domain.size:ax.axvline(p['time'][domain[0]],ls='--',color='.4')
                            if contact.size:ax.axvline(p['time'][contact[0]],ls=':',color='red')
                            ax.set_ylabel(unit);ax.grid(alpha=.2);ax.legend(fontsize=5,ncol=2 if ri==7 else 1)
                            if ri==8:ax.set_xlabel('Task time (s)')
                        axes[0,ci].set_title(f'{case["arm"]}: {row["outcome"]}; hold={row["hold_ticks"]*dt:.2f}s')
                    for rowaxes in axes:
                        bounds=[ax.get_ylim() for ax in rowaxes]
                        for ax in rowaxes:ax.set_ylim(min(b[0] for b in bounds),max(b[1] for b in bounds))
                    fig.suptitle(f'TRAINING pair {change}: update{update_index}, context{pairkey[0]}, alpha{pairkey[1]}, draw{pairkey[2]}\nOriginal same task0/noise keys, NOT identical disturbed onset states; no endpoint test or best selection')
                    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/filename,dpi=105);plt.close(fig)
                    changes.append(dict(update=update_index,source_case=pairkey[0],alpha=pairkey[1],replicate=pairkey[2],change=change,figure=filename))
            update(completed_updates=update_index)
        fig,axes=plt.subplots(4,3,figsize=(16,14),squeeze=False)
        for context in range(4):
            c=next(c for c in cases if c['source_case']==context)
            for ai,alpha in enumerate([0.,.5,1.]):
                ax=axes[context,ai]
                for arm in ['task_only','physical_credit']:
                    x=sorted([r for r in strata if r['route']==c['route'] and r['force_N']==c['force_N'] and r['alpha']==alpha and r['arm']==arm],key=lambda r:r['update'])
                    ax.plot([r['update'] for r in x],[r['survived']/r['episodes'] for r in x],marker='.',label=arm)
                ax.set(title=f'{c["route"]} F={c["force_N"]} alpha={alpha}',ylim=(-.03,1.03),xlabel='Collection index',ylabel='Observed training survival');ax.grid(alpha=.2)
        axes[0,0].legend();fig.suptitle('All completed training batches,4 action draws/arm/stratum; no independent seed or final test')
        fig.tight_layout();fig.savefig(out/'stratified_training_survival.png',dpi=120);plt.close(fig)
        atomic_json(out/'receipt.json',dict(raw_chunks_sha256=rawhashes,covered_update_case_pairs=covered,paired_changes=changes,
            figures_sha256={p.name:digest(p) for p in out.glob('*.png')},physics_steps=0,optimizer_updates=0,
            visual_reviewed=False,scope='All prefix training trajectories; exhaustive latest-four paired changes selected for diagnosis, not unbiased endpoint efficacy.'))
        update(phase='completed',stage='generated_pending_visual_review')
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,required=True);render(p.parse_args().review.resolve())
