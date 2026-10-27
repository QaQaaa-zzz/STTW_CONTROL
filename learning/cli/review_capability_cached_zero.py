"""Join only exact common cached zero events to capability90; no new physics."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sttw_control.goal_arrival_analysis import digest
from sttw_control.path_command_reward import configured_cost_keys
from sttw_control.preference_trace_rescore import rescore_trace_alpha
from sttw_control.survival_endpoint_task_metrics import goal_arrival_metrics


def main(root,out):
    assert json.loads((out/'plan.json').read_text())['expected_actual_unique_traces']==33
    grid=root/'runs/preference_capability_grid_v1';old=root/'runs/early_support_prospective_v1/right_batch64'
    a=grid/'evaluation';p=json.loads((a/'plan.json').read_text());q=json.loads((old/'plan.json').read_text())
    assert p['source_sha256']==q['source_sha256']
    for key in ('source_initial_state_sha256','path_sha256'):
        assert p[key]==q[key]
    for name in ('common_environment_config.json','source_declaration.json'):
        assert (a/name).read_bytes()==(old/name).read_bytes()
    assert digest(p['source_initial_state'])==p['source_initial_state_sha256']
    assert digest(p['path_file'])==p['path_sha256']
    for base,plan in ((a,p),(old,q)):
        for file,sha in plan['sealed_files'].items():assert digest(file)==sha
        for file,sha in plan['source_sha256'].items():
            matches=[f for f in (base/'source_snapshot').iterdir()
                     if f.name.partition('_')[0].isdigit()
                     and f.name.partition('_')[2]==Path(file).name]
            assert len(matches)==1 and digest(matches[0])==sha
    cfg=json.loads((a/'common_environment_config.json').read_text())
    task_columns=[configured_cost_keys(cfg).index(k) for k in ('speed','path','primary')]
    rows=list(csv.DictReader((grid/'analysis/all_cases.csv').open()))
    rows=[r for r in rows if float(r['force_N']) in (0.,-32.,32.)];assert len(rows)==30
    inventory=json.loads((old/'reentry/results.json').read_text())
    indices={r['case_id']:r for r in json.loads((old/'reentry/case_index.json').read_text())}
    zero=[r for r in inventory if r['role']=='zero_middle' and (r['force_N']==0 or r['pulse_onset_task_tick']==456)]
    assert len(zero)==3 and {r['force_N'] for r in zero}=={0.,-32.,32.}
    for r in zero:
        assert r['zero_mode'] and r['profile']==0 and r['reference_speed_floor_m_s']==1.
        item=indices[r['case_id']]
        rows.append(dict(r,alpha=None,raw_file=str(old/item['file']),raw_sha256=item['sha256']))
    with np.load(p['path_file']) as z:path={k:z[k] for k in z.files}
    goal=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,k]) for k in (0,1)])
    records=[];traces={};scores=[]
    for r in rows:
        file=Path(r['raw_file']);assert digest(file)==r['raw_sha256']
        with np.load(file) as z:raw={k:z[k] for k in z.files}
        active=raw['active_tick'].astype(bool)
        arrived=goal_arrival_metrics(raw,goal_xy=goal,radius_m=1.,dt=.005,horizon_s=10.)
        contact=bool(np.any(active & raw['contact_failure']))
        fault=bool(np.any(active & (raw['nonfinite']|raw['policy_fault']|raw['lower_fault'])))
        outcome='contact' if contact and not fault else 'survived' if active.sum()==2000 and not fault else 'unknown'
        assert outcome==r['physical_outcome']
        role=r['role'];alpha=None if role=='zero_middle' else float(r['alpha']);force=float(r['force_N'])
        valid=bool(np.all(raw['task_reward_valid'][active]));exposed=np.flatnonzero(active & (np.linalg.norm(raw['applied_force_impulse'],axis=1)>0))
        if not force:assert not len(exposed)
        key=(role,alpha,force);assert key not in traces;traces[key]=raw
        row=dict(role=role,alpha=alpha,force_N=force,raw_file=str(file.resolve()),raw_sha256=r['raw_sha256'],
            physical_outcome=outcome,active_ticks=int(active.sum()),task_cost_valid_entire_active_prefix=valid,
            nonzero_impulse_ticks=raw['physical_tick'][exposed].tolist(),
            integrated_force_impulse_Ns=np.asarray(raw['applied_force_impulse'][active],float).sum(axis=0).tolist(),
            domain_invalid_ticks=int(np.sum(active & raw['domain_invalid'])),**arrived)
        before=np.flatnonzero(active & (raw['physical_tick']==455))
        row['comparison_time_2p28s_state']=None if not len(before) else {
            k:np.asarray(raw[k][before[0]]).tolist() for k in
            ('actual_xy','actual_yaw','actual_forward_speed','actual_delta','phi','phi_dot','path_progress')}
        records.append(row)
        users=[0.,.25,.5,.75,1.] if alpha is None else [alpha]
        for user in users:
            score=rescore_trace_alpha(raw,alpha=user,config=cfg) if valid else None
            if valid and (alpha is not None or user==0):
                np.testing.assert_allclose(score['running_reward'][active],raw['tick_reward'][active],rtol=3e-5,atol=2e-6)
                np.testing.assert_allclose(score['scored_reward'][active],raw['scored_tick_reward'][active],rtol=3e-5,atol=2e-5)
            scores.append(dict(role=role,execution_alpha=alpha,user_alpha=user,force_N=force,
                physical_outcome=outcome,goal_arrived=arrived['goal_arrived'],valid=valid,
                running_cost=score['running_cost'] if valid else None,
                task_channel_cost=float(cfg['path_reward']['scale']*.005*score['effective_costs'][:,task_columns].sum()) if valid else None,
                scored_cost=score['scored_cost'] if valid else None,
                ranking_eligible=valid and outcome=='survived'))
    assert len(records)==33 and len({r['raw_sha256'] for r in records})==33
    comparisons=[]
    for r in records:
        if r['role']=='zero_middle':continue
        z=next(x for x in records if x['role']=='zero_middle' and x['force_N']==r['force_N'])
        own=next(x for x in scores if (x['role'],x['user_alpha'],x['force_N'])==(r['role'],r['alpha'],r['force_N']))
        zs=next(x for x in scores if (x['role'],x['user_alpha'],x['force_N'])==('zero_middle',r['alpha'],r['force_N']))
        eligible=own['ranking_eligible'] and zs['ranking_eligible']
        comparisons.append(dict(role=r['role'],user_alpha=r['alpha'],force_N=r['force_N'],
            learned_arrived=r['goal_arrived'],zero_arrived=z['goal_arrived'],
            learned_survived=r['physical_outcome']=='survived',zero_survived=z['physical_outcome']=='survived',
            common_complete_valid_survivors=eligible,
            learned_minus_zero_running_cost=own['running_cost']-zs['running_cost'] if eligible else None,
            learned_minus_zero_task_channel_cost=own['task_channel_cost']-zs['task_channel_cost'] if eligible else None))
    report=dict(records=records,scores=scores,comparisons=comparisons,actual_unique_traces=33,
        zero_actual_unique_traces=3,zero_cross_evaluation_rows=15,new_physics_ticks=0,new_training_steps=0,
        identity='Same187 archived source hashes, config bytes, source declaration bytes, complete initial state/hash and original world path/hash; zero disables Actor. Nominal onset356 metadata preserved.',
        zero_missing_force_grid=[-64,-48,-16,16,48,64],
        scope='One seen DEV context. Thirty logical learned-versus-zero comparisons reuse only3 zero physical traces, not independent trials. No extrapolation to untested force grids or native H75.',
        source_sha256={str(f.resolve()):digest(f) for f in (out/'plan.json',a/'plan.json',old/'plan.json',Path(__file__))})
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,data in [('records',records),('scores',scores),('comparisons',comparisons)]:
        with (out/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']
    for ri,role in enumerate(['task_only','task_constant']):
        for ci,force in enumerate([0.,-32.,32.]):
            ax=axes[ri,ci];points=[goal[None,:]+[-1,-1],goal[None,:]+[1,1]]
            ax.plot(*path['xy'].T,':',color='.6',label='Original path')
            for alpha,color in zip([0.,.25,.5,.75,1.],colors):
                raw=traces[(role,alpha,force)];mask=raw['active_tick'].astype(bool);xy=raw['actual_xy'][mask];points.append(xy)
                ax.plot(*xy.T,color=color,label=f'execution alpha{alpha:g}')
                pulse=np.flatnonzero(mask & (np.linalg.norm(raw['applied_force_impulse'],axis=1)>0))
                if len(pulse):
                    ax.plot(*raw['actual_xy'][pulse].T,color=color,lw=3)
                    ax.scatter(*raw['actual_xy'][pulse[0]],facecolors='none',edgecolors=color,s=24,marker='o')
                ax.scatter(*xy[-1],color=color,marker='x' if raw['contact_failure'][mask].any() else 's',s=22)
            raw=traces[('zero_middle',None,force)];xy=raw['actual_xy'][raw['active_tick'].astype(bool)];points.append(xy)
            ax.plot(*xy.T,'k--',lw=1.5,label='Zero middle (one actual trace)');ax.scatter(*xy[-1],color='black',marker='s',s=20)
            pulse=np.flatnonzero(raw['active_tick'].astype(bool) & (np.linalg.norm(raw['applied_force_impulse'],axis=1)>0))
            if len(pulse):
                ax.plot(*raw['actual_xy'][pulse].T,color='black',lw=3)
                ax.scatter(*raw['actual_xy'][pulse[0]],facecolors='none',edgecolors='black',s=24,marker='o')
            ax.scatter(*goal,marker='*',color='black');ax.add_patch(Circle(goal,1,fill=False,color='.5'))
            pts=np.concatenate(points);lo=pts.min(axis=0)-.4;hi=pts.max(axis=0)+.4
            ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
            ax.set(title=f'{role}, F={force:+g}N',xlabel='World X (m)',ylabel='World Y (m)')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Cached common-interface10s / task64 / right90/R1.1/requested1.5m/s\nSigned32N onset2.28s /0.1s: open circle+thick segment; x=contact stop; zero3 unique reused across alpha views')
    for ext in ('png','pdf'):fig.savefig(out/('common3_xy.'+ext),dpi=160)
    plt.close(fig)
    plot_series(out,traces,cfg)
    print('Completed33 cached traces, no new physics; '+str(out/'review.json'))


def plot_series(out,traces,cfg):
    import matplotlib.pyplot as plt
    folder=out/'paired_series';folder.mkdir(exist_ok=True)
    roles=['task_only','task_constant','zero_middle'];colors=['#0072B2','#D55E00','black'];styles=['-','-.','--']
    keys=list(configured_cost_keys(cfg));scale_dt=cfg['path_reward']['scale']*.005
    index=[]
    for alpha in [0.,.25,.5,.75,1.]:
        for force in [0.,-32.,32.]:
            name=f'a{alpha:g}_F{force:+g}';index.append(name)
            fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
            component_fig,component_axes=plt.subplots(5,4,figsize=(15,15),layout='constrained')
            for role,color,style in zip(roles,colors,styles):
                raw=traces[(role,None if role=='zero_middle' else alpha,force)];mask=raw['active_tick'].astype(bool)
                score=rescore_trace_alpha(raw,alpha=alpha,config=cfg);t=(raw['physical_tick'][mask]+1)*.005
                series=[score['scored_reward'][mask],-np.cumsum(score['scored_reward'][mask],dtype=float),
                    raw['path_cross_track'][mask],raw['actual_forward_speed'][mask],raw['phi'][mask],raw['path_heading_error'][mask]]
                for ax,value in zip(axes.flat,series):ax.plot(t,value,color=color,ls=style,lw=1,label=role)
                axes.flat[3].plot(t,raw['wheel_speed_proxy'][mask],color=color,ls=':',lw=.7)
                axes.flat[3].plot(t,raw['reference'][mask,0],color=color,ls='--',lw=.6)
                for k,key in enumerate(keys):
                    row,col=divmod(k,2);inst=scale_dt*score['effective_costs'][mask,k]
                    for axis,value,label in [(component_axes[row,col*2],inst,'Cost/lower tick'),
                        (component_axes[row,col*2+1],np.cumsum(inst,dtype=float),'Cumulative running cost')]:
                        axis.plot(t,value,color=color,ls=style,lw=.8,label=role)
                        axis.set(title=key,xlabel='Task elapsed(s)',ylabel=label);axis.grid(alpha=.2)
                        if force:axis.axvspan(2.28,2.38,color='.5',alpha=.12)
            for ax,label in zip(axes.flat,['Scored reward/lower tick','Cumulative original scored cost','Path error(m)',
                'Forward speed(m/s)','Roll(rad)','Path heading error(rad)']):
                ax.set(xlabel='Task elapsed(s)',ylabel=label);ax.grid(alpha=.2)
                if force:ax.axvspan(2.28,2.38,color='.5',alpha=.12)
            axes.flat[3].axhline(1.5,color='.4',lw=.7,ls=':')
            axes.flat[3].text(.02,.03,'Thin dashed: issued; dotted: wheel proxy',
                transform=axes.flat[3].transAxes,fontsize=7,bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
            title=f'Cached common-interface task64 / useralpha{alpha:g} / F{force:+g}N / original goal and request unchanged'
            fig.suptitle(title+'\nAll actual stops; no extrapolation; shaded=.1s force; zero physically one trace re-evaluated at original useralpha')
            component_fig.suptitle(title+'\nAll10 capped running components, per tick and cumulative; terminal replacement excluded here and retained in scored plot')
            for current,filename in [(fig,name),(component_fig,name+'_components')]:
                handles,labels=current.axes[0].get_legend_handles_labels();current.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
                for ext in ('png','pdf'):current.savefig(folder/(filename+'.'+ext),dpi=130)
                plt.close(current)
    (folder/'INDEX.md').write_text('# All15 original useralpha/event pairs\n\nZero3 unique actual traces, reweighted without new physics. Each pair has actual reward/cumulative/kinematic plots and all10 per-tick/cumulative components; physical endpoints unextended.\n\n'+'\n'.join(f'- [{s}]({s}.png) / [components]({s}_components.png)' for s in index)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    main(Path.cwd(),parser.parse_args().output.resolve())
