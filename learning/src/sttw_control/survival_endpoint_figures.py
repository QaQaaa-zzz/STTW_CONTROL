"""Read-only factual XY and per-decision diagnostics; aliases are not episodes."""
import json
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .path_command_reward import configured_cost_keys
from .survival_endpoint_analysis import rescore_v3

ROLES=('parent','task_only','physical_credit','parent_speed_floor','zero_middle')
COLORS=('tab:blue','tab:orange','tab:green','tab:red','tab:purple')


def reward_blocks(physical, *, scale, scored=None, components=None):
    """Sum actual lower costs within logged policy intervals, including partial ends."""
    active=np.asarray(physical['active_tick'],bool)
    valid=active & np.asarray(physical['task_reward_valid'],bool)
    if valid.size%4:raise ValueError('complete four-slot logged policy intervals required')
    v=valid.reshape(-1,4);keep=v.any(axis=1)
    time=np.where(valid,physical['time'],np.inf).reshape(-1,4).min(axis=1)[keep]
    sums=lambda values:np.where(valid,values,0.).reshape(-1,4).sum(axis=1)[keep]
    if components is None:components=np.asarray(physical['effective_costs'])
    signed=np.where(valid[:,None],-scale*components,0.).reshape(-1,4,components.shape[-1]).sum(axis=1)[keep]
    factual_terminal=np.asarray(physical['scored_tick_reward'])-physical['tick_reward']
    return dict(time=time,scored_reward=sums(physical['scored_tick_reward'] if scored is None else scored),
        running_reward=signed.sum(axis=1),terminal_charge=sums(factual_terminal),signed_components=signed)


def comparison_panels(rows):
    zeros={};panels={}
    for row in rows:
        if row['role']=='zero_middle':
            key=(row['panel'],row['context_index'])
            if key in zeros:raise ValueError('duplicate actual zero episode')
            zeros[key]=row['case_id'];continue
        key=(row['panel'],row['context_index'],row['alpha'],row['replicate'])
        panel=panels.setdefault(key,dict(panel=key[0],context_index=key[1],alpha=key[2],replicate=key[3],case_ids={}))
        if row['role'] in panel['case_ids']:raise ValueError('duplicate case in comparison panel')
        panel['case_ids'][row['role']]=row['case_id']
    for panel in panels.values():
        zeropanel='physical_deterministic' if panel['panel']=='physical_stochastic' else panel['panel']
        zero=zeros.get((zeropanel,panel['context_index']))
        if set(panel['case_ids'])!=set(ROLES[:-1]) or zero is None:raise ValueError('missing comparison role/zero alias')
        panel['case_ids']['zero_middle']=zero
    return list(panels.values())


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def draw_xy(ax, records, reference, *, title):
    ax.plot(reference[:,0],reference[:,1],'--',color='.6',lw=1,label='Original path')
    positions=[]
    for i,record in enumerate(records):
        d=record['data'];active=d['active_tick'];xy=d['actual_xy'][active]
        if not len(xy):continue
        color=COLORS[i%len(COLORS)];positions.append(xy)
        style=('-', '--', '-.', ':', (0,(5,1,1,1)))[i%5]
        ax.plot(xy[:,0],xy[:,1],color=color,ls=style,label=record['label'],lw=1)
        end='x' if d['contact_failure'][active].any() else 'o'
        ax.scatter(*xy[-1],marker=end,color=color,s=22)
        exited=np.flatnonzero(active & d['domain_exit'])
        if exited.size:ax.scatter(*d['actual_xy'][exited[0]],marker='s',facecolors='none',edgecolors=color,s=28)
        held=active & d['fallback_active']
        if held.any():ax.plot(*d['actual_xy'][held].T,ls=':',color=color,lw=1.7)
    if positions:
        allxy=np.concatenate(positions);lo=allxy.min(axis=0);hi=allxy.max(axis=0)
        center=(hi+lo)/2;half=max(float((hi-lo).max())*.56,.02)
        ax.set_xlim(center[0]-half,center[0]+half);ax.set_ylim(center[1]-half,center[1]+half)
    ax.set(xlabel='World X (m)',ylabel='World Y (m)');ax.set_title(title,fontsize=8)
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)


def render_diagnostics(records, output, *, config, user_alpha, title):
    """All given records; scores/components use the SAME user preference weight."""
    plt=_pyplot();fig,axes=plt.subplots(9,len(records),figsize=(5*len(records),24),squeeze=False)
    names=configured_cost_keys(config);scale=config['path_reward']['scale']*config['timing']['lower_dt_s']
    for ci,record in enumerate(records):
        d=record['data'];case=record['case'];active=d['active_tick'];valid=active & d['task_reward_valid']
        t=d['time'][active];tv=d['time'][valid]
        def line(row,key,label,style='-',mask=active):
            axes[row,ci].plot(d['time'][mask],d[key][mask],style,label=label,lw=.9)
        line(0,'actual_forward_speed','True forward');line(0,'wheel_speed_proxy','Wheel proxy','--')
        axes[0,ci].plot(t,d['reference'][active,0],label='Given reference')
        axes[0,ci].axhline(case['speed'],ls=':',color='.3',label='Original request')
        line(1,'actual_delta','Actual steering');axes[1,ci].plot(t,d['reference'][active,1],label='Given reference')
        axes[1,ci].plot(tv,d['nominal_reference'][valid,1],ls=':',label='Follower request')
        line(2,'phi','Signed roll');line(2,'peak_roll','Peak absolute roll','--');line(3,'phi_dot','Roll rate')
        line(4,'path_cross_track','Path error',mask=valid);line(5,'path_heading_error','Heading error',mask=valid)
        score=rescore_v3(d,user_alpha=user_alpha,config=config,include_components=True)
        reward=reward_blocks(d,scale=scale,scored=score['per_lower_tick_reward'],components=score['per_lower_tick_effective_costs'])
        axes[6,ci].plot(reward['time'],reward['scored_reward'],color='black',label='Total score',lw=1.2)
        axes[6,ci].plot(reward['time'],reward['running_reward'],'--',color='.5',label='Running score')
        axes[6,ci].plot(reward['time'],reward['terminal_charge'],':',color='red',label='Terminal charge')
        for k,name in enumerate(names):axes[7,ci].plot(reward['time'],reward['signed_components'][:,k],label='-'+name,lw=.8)
        impulse=d['applied_force_impulse'][active]
        for k,label in enumerate(['X','Y','Z']):axes[8,ci].plot(t,impulse[:,k]/config['timing']['lower_dt_s'],label='Applied '+label)
        onset=case.get('pulse_onset_task_tick',0)*config['timing']['lower_dt_s'];duration=case.get('pulse_duration_s',0.)
        domain=np.flatnonzero(active & d['domain_exit']);contact=np.flatnonzero(active & d['contact_failure'])
        for ri,unit in enumerate(['Speed (m/s)','Steering (rad)','Roll (rad)','Roll rate (rad/s)',
                                 'Path error (m)','Heading error (rad)','Score / actual policy interval',
                                 'Signed score components','Applied mean force (N)']):
            ax=axes[ri,ci];ax.set_ylabel(unit);ax.grid(alpha=.2)
            if duration and case.get('force_N',0.):ax.axvspan(onset,onset+duration,color='orange',alpha=.13)
            if domain.size:ax.axvline(d['time'][domain[0]],color='.4',ls='--',lw=.8)
            if contact.size:ax.axvline(d['time'][contact[0]],color='red',ls=':',lw=1)
            ax.legend(fontsize=5,ncol=2 if ri==7 else 1)
            if ri==8:ax.set_xlabel('Actual task time (s)')
        model=record.get('model_id','unknown')
        axes[0,ci].set_title(record['label']+'\n'+model[:16],fontsize=9)
    for row in axes:
        bounds=[ax.get_ylim() for ax in row]
        for ax in row:ax.set_ylim(min(v[0] for v in bounds),max(v[1] for v in bounds))
    fig.suptitle(title+f' | utility alpha={user_alpha:g}; common row scales; task curves stop at domain exit',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.985));fig.savefig(output,dpi=110);plt.close(fig)


def render_endpoint_run(root):
    root=Path(root).resolve();status=json.loads((root/'status.json').read_text())
    if status['phase']!='completed' or not status['endpoint_authenticated']:
        raise ValueError('authenticated completed collection required')
    plan=json.loads((root/'plan.json').read_text());cfg=json.loads((root/'common_environment_config.json').read_text())
    rows=json.loads((root/'results.json').read_text());index=json.loads((root/'case_index.json').read_text())
    if len(rows)!=plan['protocol_plan']['declared_actual_episodes'] or {r['case_id'] for r in rows}!={i['case_id'] for i in index}:
        raise ValueError('full actual episode matrix required for all-case figures')
    for item in index:
        if digest(root/item['file'])!=item['sha256']:raise ValueError('raw case changed: '+item['case_id'])
    out=root/'figures';out.mkdir(exist_ok=False)
    byrow={r['case_id']:r for r in rows};byindex={i['case_id']:i for i in index}
    panels=comparison_panels(rows);sheets={};coverage=[];plt=_pyplot()
    def records(panel):
        result=[]
        for role in ROLES:
            cid=panel['case_ids'][role];item=byindex[cid]
            with np.load(root/item['file']) as z:data={k:z[k] for k in z.files if not k.startswith('decision_')}
            result.append(dict(label=role,data=data,case=byrow[cid],model_id=item['policy_id']))
        return result
    for panel in panels:
        key=(panel['panel'],panel['replicate']);sheets.setdefault(key,[]).append(panel)
    for (name,rep),selected in sheets.items():
        fig,axes=plt.subplots(4,5,figsize=(22,17),squeeze=False)
        for panel,ax in zip(sorted(selected,key=lambda x:(x['context_index'],x['alpha'])),axes.flat):
            item=byindex[panel['case_ids']['parent']]
            with np.load(root/item['reference']) as z:reference=z['xy']
            source=byrow[panel['case_ids']['parent']]
            pulse=(f' {source.get("pulse_duration_s",0):g}s@{source.get("pulse_onset_task_tick",0)*cfg["timing"]["lower_dt_s"]:g}s'
                   if source.get('force_N',0) else '')
            draw_xy(ax,records(panel),reference,title=f'{source["route"]} v={source["speed"]:g} alpha={panel["alpha"]:g}\nF={source.get("force_N",0):g}N'+pulse)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{name} draw={rep} | square=domain exit; dotted=held-reference tail; x=contact; circle=observed end')
        fig.tight_layout();fig.savefig(out/f'{name}_draw{rep}_xy.png',dpi=130);plt.close(fig)
        for panel in selected:
            filename=f'{name}_context{panel["context_index"]}_alpha{panel["alpha"]:g}_draw{rep}_diagnostics.png'
            render_diagnostics(records(panel),out/filename,config=cfg,user_alpha=panel['alpha'],
                title=f'{name} context={panel["context_index"]} draw={rep}; zero reuses factual deterministic case')
            coverage.append(dict(panel,diagnostics=filename,xy=f'{name}_draw{rep}_xy.png'))
    figures=sorted(out.glob('*.png'))
    atomic_json(out/'receipt.json',dict(actual_episode_count=len(rows),comparison_panel_count=len(panels),
        coverage=coverage,physics_steps=0,optimizer_updates=0,
        figures_sha256={p.name:digest(p) for p in figures},
        source_sha256={str(p):digest(p) for p in [root/'results.json',root/'case_index.json',
            root/'common_environment_config.json',Path(__file__),Path(__file__).with_name('survival_endpoint_analysis.py')]},
        note='Zero aliases do not create new episodes. Generated figures still require visual review.'))
    lines=['# Fixed endpoint factual figures','',
        f'{len(rows)} actual DEV episodes; {len(panels)} comparison panels. No new physics or optimizer updates.',
        'Zero-middle trajectories are reused as aliases. All reward columns use the same user alpha.',
        'Square: domain exit; dotted segment: held-reference physics; x: contact; circle: observed endpoint.',
        'Task errors and reward stop at domain exit. Survival is separate from completing the requested task.',
        'Files are generated; visual review remains pending.','']
    for (name,rep),selected in sheets.items():
        lines.extend([f'## {name}, draw {rep}','',f'[Actual XY overview]({name}_draw{rep}_xy.png)',''])
        for item in coverage:
            if item['panel']==name and item['replicate']==rep:
                lines.append(f'- [Context {item["context_index"]}, alpha {item["alpha"]:g}]({item["diagnostics"]})')
        lines.append('')
    (out/'INDEX.md').write_text('\n'.join(lines),encoding='utf-8')
    # Visual review is a separate explicit delivery step, not a plotting side effect.
    status.update(figures_generated=True,figures_visual_reviewed=False,figures_complete=False)
    atomic_json(root/'status.json',status)
