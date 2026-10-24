"""Analyze every declared nominal case, retaining failures and input limits."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dual-run',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    from sttw_control.preference_best import atomic_json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=args.run.resolve();plan=json.loads((root/'plan.json').read_text())
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=plan['total_cases']:
        raise ValueError('nominal physics incomplete')
    out=args.output.resolve() if args.output else root/'analysis';out.mkdir(exist_ok=False)
    dual_plan=None
    if args.dual_run:
        dual_plan=json.loads((args.dual_run/'plan.json').read_text())
        dual_status=json.loads((args.dual_run/'status.json').read_text())
        if dual_status['phase']!='completed' or dual_status['completed_cases']!=16:
            raise ValueError('specialist review incomplete')
        if Path(dual_plan['source_nominal'])!=root:raise ValueError('specialist source identity mismatch')
    rows=[];traces={};zero_checks=[];panels=[];summaries=[]
    fig,axes=plt.subplots(2,4,figsize=(19,9),layout='constrained')
    contexts=[]
    for panel in plan['panels']:
        ev=Path(panel['output']);manifest=json.loads((ev/'manifest.json').read_text())
        decl=manifest['declaration'];cfg=decl['config'];keys=manifest['reward_components']
        cases=json.loads((ev/'results.json').read_text())['cases']
        if len(cases)!=panel['cases']:raise ValueError('case roster incomplete')
        sources=[(row,ev) for row in cases]
        if dual_plan:
            for extra in dual_plan['panels']:
                if extra['radius_m']!=panel['radius_m']:continue
                extra_ev=Path(extra['output']);extra_m=json.loads((extra_ev/'manifest.json').read_text())
                if extra_m['physical_identity']!=manifest['physical_identity'] or extra_m['source']!=manifest['source']:
                    raise ValueError('specialist physical/source mismatch')
                other=extra_m['config'];other['network']['architecture']=cfg['network']['architecture']
                if other!=cfg:raise ValueError('specialist common effective config mismatch')
                extra_rows=json.loads((extra_ev/'results.json').read_text())['cases']
                if len(extra_rows)!=4:raise ValueError('specialist roster incomplete')
                sources.extend((row,extra_ev) for row in extra_rows)
        for row,trace_dir in sources:
            with np.load(trace_dir/(row['name']+'.npz')) as f:trace={k:f[k] for k in f.files}
            n=len(trace['time']);indices=np.arange(0,n,4)
            if n!=row['actual_suffix_ticks']:raise ValueError('raw cost mismatch')
            step_reward=np.add.reduceat(trace['scored_tick_reward'],indices)
            signed=-cfg['path_reward']['scale']*.005*np.add.reduceat(trace['effective_costs'],indices,axis=0)
            if not np.isclose(step_reward.sum(),row['logged_return'],atol=1e-5):raise ValueError('return mismatch')
            np.savez_compressed(out/(panel['name']+'_'+row['name']+'_rewards.npz'),
                time=trace['time'][indices],step_reward=step_reward,signed_components=signed,
                component_names=np.array(keys),failure_remainder=step_reward-signed.sum(axis=1))
            turn=(trace['path_progress']>=cfg['path']['straight_prefix_m'])&(trace['path_progress']<=cfg['path']['straight_prefix_m']+np.pi/2*panel['radius_m']+cfg['path']['curvature_transition_length_m'])
            row.update(panel=panel['name'],radius_m=panel['radius_m'],endpoint_s=float(trace['time'][-1]+.005),
                physical_outcome='FLOOR_CONTACT' if row['physical_failure'] else 'COMPLETE_NO_CONTACT' if row['complete_horizon'] else 'UNKNOWN',
                speed_rmse_m_s=float(np.sqrt(np.mean((trace['actual_forward_speed']-row['speed'])**2))),
                path_rmse_m=float(np.sqrt(np.mean(trace['path_cross_track']**2))),
                end_progress_m=float(trace['path_progress'][-1]),turn_ticks=int(turn.sum()),
                turn_speed_rmse_m_s=float(np.sqrt(np.mean((trace['actual_forward_speed'][turn]-row['speed'])**2))) if turn.any() else None,
                turn_path_rmse_m=float(np.sqrt(np.mean(trace['path_cross_track'][turn]**2))) if turn.any() else None,
                chi_mean=float(trace['chi'].mean()),chi_max=float(trace['chi'].max()),
                governed_speed_min=float(trace['governed'][:,0].min()),
                governed_steer_max_abs=float(np.abs(trace['governed'][:,1]).max()),
                upper_input_at_clip_fraction=float(np.mean(np.abs(trace['upper_observation'][::4])>=5)),
                upper_input_at_clip_by_dimension=np.mean(np.abs(trace['upper_observation'][::4])>=5,axis=0).tolist(),
                lower_input_at_clip_fraction=float(np.mean(np.abs(trace['lower_observation'][:,:200])>=5)),
                lower_input_at_clip_by_dimension=np.mean(np.abs(trace['lower_observation'][:,:200])>=5,axis=0).tolist())
            if np.any(trace['applied_force_impulse']):raise ValueError('nominal panel received force')
            if trace['upper_observation'].shape!=(n,351) or trace['lower_observation'].shape!=(n,210):raise ValueError('wrong observation identity')
            traces[(panel['name'],row['name'])]=trace;rows.append(row)
        for route in decl['routes']:
            with np.load(ev/(route['name']+'_path.npz')) as f:path={k:f[k] for k in f.files}
            for speed in decl['speeds']:
                group=[r for r in rows if r['panel']==panel['name'] and r['route']==route['name'] and r['speed']==speed]
                contexts.append((panel,route,speed,group,path,keys,cfg['path_reward']['scale']))
                zero=[r for r in group if r['policy']=='zero'];a,b=[traces[(r['panel'],r['name'])] for r in sorted(zero,key=lambda r:r['alpha'])]
                same=len(a['time'])==len(b['time'])
                zero_checks.append(dict(panel=panel['name'],route=route['name'],speed=speed,same_length=same,
                    max_absolute_differences={k:float(np.max(np.abs(a[k]-b[k]))) if same else None for k in ['time','actual_xy','phi','actual_forward_speed','governed']},
                    both_zero_latent=bool(not np.any(a['latent_z']) and not np.any(b['latent_z']))))
    for index,(panel,route,speed,group,path,keys,reward_scale) in enumerate(contexts):
        ax=axes.flat[index];detail,aa=plt.subplots(7,1,figsize=(11,16),layout='constrained')
        components,cc=plt.subplots(len(group)//2,2,figsize=(16,4*(len(group)//2)),layout='constrained')
        colors={('H75',0.):'tab:blue',('H75',1.):'tab:orange',('zero',0.):'black',('zero',1.):'gray',('dual',0.):'tab:green',('dual',1.):'tab:purple'}
        max_progress=max(r['end_progress_m'] for r in group)
        xy=path['xy'][path['s']<=max_progress+1]
        ax.plot(*xy.T,'--',color='green',label='fixed task path')
        for slot,row in enumerate(sorted(group,key=lambda r:(r['policy'],r['alpha']))):
            trace=traces[(row['panel'],row['name'])];t=trace['time'];color=colors[(row['policy'],row['alpha'])]
            label=f"{row['policy']} a={row['alpha']:g} {row['physical_outcome']}";marker='x' if row['physical_failure'] else 'o' if row['complete_horizon'] else 's'
            ax.plot(*trace['actual_xy'].T,color=color,label=label);ax.scatter(*trace['actual_xy'][-1],color=color,marker=marker)
            for axis,key in zip(aa[:5],['actual_forward_speed','path_cross_track','path_heading_error','phi','actual_delta']):
                axis.plot(t,trace[key],color=color,label=label);axis.scatter(t[-1],trace[key][-1],color=color,marker=marker)
            aa[0].plot(t,trace['governed'][:,0],color=color,ls='--',alpha=.5)
            aa[0].plot(t,trace['wheel_speed_proxy'],color=color,ls=':',alpha=.5)
            aa[4].plot(t,trace['governed'][:,1],color=color,ls='--',alpha=.5)
            ix=np.arange(0,len(t),4);reward=np.add.reduceat(trace['scored_tick_reward'],ix)
            signed=-reward_scale*.005*np.add.reduceat(trace['effective_costs'],ix,axis=0)
            aa[5].plot(t[ix],reward,color=color,label=label);aa[6].plot(t[ix],np.cumsum(reward),color=color,label=label)
            for k,name in enumerate(keys):cc.flat[slot].plot(t[ix],signed[:,k],label=name)
            cc.flat[slot].plot(t[ix],reward-signed.sum(1),'k--',label='failure remainder');cc.flat[slot].set_title(label);cc.flat[slot].legend(fontsize=7,ncol=3)
        ax.set_title(f"{route['name']} v={speed:g}");ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)');ax.legend(fontsize=6)
        for axis,label in zip(aa,['speed (m/s); dashed reference, dotted wheel proxy','cross track (m)','heading error (rad)','roll (rad)','steer (rad); dashed reference','reward / 20ms','cumulative reward']):axis.set_ylabel(label);axis.grid(alpha=.2)
        aa[0].axhline(speed,color='green',ls='--');aa[0].legend(fontsize=7);aa[-1].set_xlabel('task time (s); zero external force')
        title=f"{route['name']}_v{speed:g}";detail.suptitle(title+'; frozen H75 u77/local300; shared bank3 seed77001')
        detail.savefig(out/(title+'_timeseries.png'));components.savefig(out/(title+'_components.png'));plt.close(detail);plt.close(components)
    fig.suptitle(f'All {len(rows)} declared slots; endpoint x=contact, square=incomplete, circle=10s; no external force')
    fig.savefig(out/'all_paths.png');plt.close(fig)
    for panel in plan['panels']:
        for policy in ['H75','zero']+(['dual'] if dual_plan else []):
            for alpha in [0.,1.]:
                rr=[r for r in rows if r['panel']==panel['name'] and r['policy']==policy and r['alpha']==alpha]
                summaries.append(dict(panel=panel['name'],policy=policy,alpha=alpha,cases=len(rr),
                    task=sum(r['task_recovered'] for r in rr),contact=sum(r['physical_failure'] for r in rr),
                    complete_no_contact=sum(r['physical_outcome']=='COMPLETE_NO_CONTACT' for r in rr),
                    domain_exit=sum(r['domain_failure'] for r in rr),unknown=sum(r['physical_outcome']=='UNKNOWN' for r in rr)))
    total_ticks=status['active_lower_ticks']+(dual_status['active_lower_ticks'] if dual_plan else 0)
    atomic_json(out/'analysis.json',dict(cases=rows,summary=summaries,zero_alpha_physical_checks=zero_checks,
        active_lower_ticks=total_ticks,training_steps=0,
        input_boundary_note='at absolute normalized input 5; pre-clip magnitudes unavailable, not proof each value was truncated',
        inference='deterministic new-contract development diagnostic; not independent replicates or tolerance gain'))
    lines=['# 低速紧弯：冻结偏好与零修正名义对照','','目的：确认共同扩大参考权限后，现有冻结中层能否提供有用任务取舍，再决定是否施扰。',
           f'方法：R0.9/1.1、左右90度、v1.5/1.8、alpha0/1、10秒、同完整bank3；{len(rows)}执行槽（H75/zero'+('/dual' if dual_plan else '')+'），零力、零训练。旧权重/物理/奖励不改；新参考协议不是单因素消融。',
           '', '[全部XY轨迹](all_paths.png)。全部逐例速度/路径/侧倾/每步与累计奖励见同目录 timeseries；signed分项与失败余项见 components 和 rewards.npz。',
           '', '|半径面板|策略|alpha|任务达标|触地|完整未触地|域退出|未完整|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append('|'+ '|'.join(str(r[k]) for k in ['panel','policy','alpha','task','contact','complete_no_contact','domain_exit','unknown'])+'|')
    lines+=['','任务与不摔倒分别报告；提前域退出不能算10秒未摔。zero两alpha是评分上下文，不是独立重复。完整config/diff和执行源归档在父目录。新几何/参考范围外推失败不能否定旧训练能力。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summaries,lower_ticks=total_ticks)))


if __name__=='__main__':main()
