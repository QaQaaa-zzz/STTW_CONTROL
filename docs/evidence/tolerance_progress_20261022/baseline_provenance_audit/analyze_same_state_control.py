"""Analyze the predeclared same-state intervention test, with all physical outcomes."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from analyze_matched_control import load_trace
from sttw_control.preference_best import atomic_json, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();out=args.output.resolve()
    status=json.loads((root/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=48:
        raise ValueError('Wait for all48 new outcomes; no partial success claim')
    plan=json.loads((root/'plan.json').read_text())
    for path,sha in plan['source_sha256'].items():
        if digest(path)!=sha:raise ValueError('Declared source changed '+path)
    if status['active_lower_ticks']>38400:raise ValueError('New physics budget exceeded')
    baseline=json.loads((root/'baseline_reuse.json').read_text())['cases']
    current=json.loads((root/'results.json').read_text())['cases']
    lineage=json.loads((root/'snapshot_lineage.json').read_text())['cases']
    assert len(baseline)==24 and len(current)==48 and len(lineage)==8
    sources={str(p.resolve()):digest(p) for p in [root/'plan.json',root/'baseline_reuse.json',root/'results.json',root/'snapshot_lineage.json',Path(__file__),Path(__file__).with_name('analyze_matched_control.py')]}
    groups={};records=[]
    for mode,rows in [('continue_alpha0',baseline),('new',current)]:
        for row in rows:
            actual_mode=mode if mode!='new' else row['policy']
            file=Path(row['trace']) if mode!='new' else root/(row['name']+'.npz')
            trace=load_trace(file,row,sources)
            if actual_mode=='zero_latent':
                np.testing.assert_array_equal(trace['executed_policy_latent'][trace['upper_active']],0.)
            key=(row['route'],row['requested_speed'],row['fraction'],row['force_N'])
            group=groups.setdefault(key,{})
            if actual_mode in group:raise ValueError('Duplicate same-state comparison')
            group[actual_mode]=(row,trace)
            records.append(dict(mode=actual_mode,**row))
    assert len(groups)==24 and len(records)==72
    modes=['continue_alpha0','switch_alpha1','zero_latent']
    if any(set(g)!=set(modes) for g in groups.values()):raise ValueError('Missing paired arm')
    contrasts={}
    for mode in modes[1:]:
        gain=[];loss=[]
        for key,group in sorted(groups.items()):
            old,new=group[modes[0]][0],group[mode][0]
            record=dict(route=key[0],speed=key[1],fraction=key[2],force_N=key[3],
                        original=old['balance_label'],new=new['balance_label'],
                        new_fallback=new['fallback_triggered'])
            if old['balance_label']=='FLOOR_CONTACT' and new['balance_label']=='NO_FLOOR_CONTACT':gain.append(record)
            if old['balance_label']=='NO_FLOOR_CONTACT' and new['balance_label']=='FLOOR_CONTACT':loss.append(record)
        contrasts[mode]=dict(gains=gain,losses=loss,net=len(gain)-len(loss))
    totals={mode:dict(nonzero=dict(Counter(r['balance_label'] for r in records if r['mode']==mode and r['force_N'])),
                      zero=dict(Counter(r['balance_label'] for r in records if r['mode']==mode and not r['force_N'])),
                      fallback=sum(r['fallback_triggered'] for r in records if r['mode']==mode)) for mode in modes}
    out.mkdir(parents=True,exist_ok=False);(out/'cases').mkdir()
    atomic_json(out/'analysis.json',dict(cases=records,contrasts=contrasts,totals=totals,source_sha256=sources,
                new_lower_ticks=status['active_lower_ticks'],reused_cases=24,new_cases=48,training_steps=0,
                scope='Same complete alpha0 physical/history state; only future preference request or latent target changes. Common domain-hold fallback. No switching-time sweep.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(continue_alpha0='#0072B2',switch_alpha1='#D55E00',zero_latent='#009E73')
    links=[]
    for key,group in sorted(groups.items()):
        route,speed,fraction,force=key
        sid=group[modes[0]][0]['state_id'];origin=next(x for x in lineage if x['state_id']==sid)['genuine_prefix']
        old_root=Path(group[modes[0]][0]['source_run'])
        file=old_root/f'prefix_v{speed:g}_a0.npz';sources[str(file)]=digest(file)
        with np.load(file,allow_pickle=False) as z:prefix=z['actual_xy'][z['time']<origin['task_time_s']-1e-6]
        fig,axes=plt.subplots(9,1,figsize=(11,17),layout='constrained')
        axes[0].plot(*prefix.T,color='gray',label='shared genuine alpha0 prefix')
        all_xy=[prefix]
        for mode in modes:
            row,trace=group[mode];color=colors[mode];t=trace['time'];all_xy.append(trace['actual_xy'])
            axes[0].plot(*trace['actual_xy'].T,color=color,label=mode+' '+row['balance_label'])
            axes[0].scatter(*trace['actual_xy'][0],color=color,marker='^',s=20)
            axes[0].scatter(*trace['actual_xy'][-1],color=color,marker='x' if row['balance_label']=='FLOOR_CONTACT' else 'o')
            for ax,field in zip(axes[1:7],['actual_forward_speed','path_cross_track','path_heading_error','phi','phi_dot','actual_delta']):ax.plot(t,trace[field],color=color,label=mode)
            axes[1].plot(t,trace['reference'][:,0],color=color,ls='--',alpha=.5)
            axes[6].plot(t,trace['reference'][:,1],color=color,ls='--',alpha=.5)
            axes[7].plot(t,np.linalg.norm(trace['applied_force_impulse'],axis=1)/.005,color=color,label=mode)
            axes[8].plot(t,trace['upper_active'].astype(float),color=color,label=mode)
            if row['fallback_start_time_s'] is not None:
                for ax in axes[1:]:ax.axvline(row['fallback_start_time_s'],color=color,ls=':',alpha=.5)
        with np.load(root/(route+'_path.npz'),allow_pickle=False) as z:xy=z['xy']
        all_xy=np.concatenate(all_xy);lo,hi=all_xy.min(0)-.5,all_xy.max(0)+.5
        visible=np.where(((xy>=lo)&(xy<=hi)).all(1)[:,None],xy,np.nan)
        axes[0].plot(*visible.T,'k:',label='original fixed route');axes[0].set(xlabel='world X [m]',ylabel='world Y [m]',aspect='equal')
        axes[1].axhline(speed,color='gray',ls=':')
        for ax,label in zip(axes[1:],['speed/ref [m/s]','signed path [m]','heading [rad]','roll [rad]','roll rate [rad/s]','steer/ref [rad]','force magnitude [N]','follower/middle active']):
            ax.set(xlabel='time since common capture [s]',ylabel=label)
            if force:ax.axvspan(0,.1,color='gray',alpha=.15)
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=6)
        tag=f'{route}_v{speed:g}_phase{fraction:g}_F{force:+g}';fig.suptitle(tag+' | same physical/history state; future control only')
        fig.savefig(out/'cases'/(tag+'.png'),dpi=95);plt.close(fig);links.append(tag)
    # Prefix plot inputs were added during rendering; retain their identities.
    result=json.loads((out/'analysis.json').read_text());result['source_sha256']=sources;atomic_json(out/'analysis.json',result)
    lines=['# 同一状态后续控制权限判别','',
           '同8个真实α0完整快照、同0/±80N/.1s/4s，继续α0复用24，切α1/零latent新48。物理/历史/滤波carry不重置，世界推力方向一致；退出后共用hold-last。','',
           '|后续控制|受扰无触地/16|受扰触地/16|零力无触地/8|曾fallback|','|---|---:|---:|---:|---:|']
    for mode,x in totals.items():lines.append(f"|{mode}|{x['nonzero'].get('NO_FLOOR_CONTACT',0)}|{x['nonzero'].get('FLOOR_CONTACT',0)}|{x['zero'].get('NO_FLOOR_CONTACT',0)}|{x['fallback']}|")
    lines+=['',f"新增{status['active_lower_ticks']}lower ticks，0新前缀/优化；继续α0不重跑。",'']
    for mode,x in contrasts.items():lines.append(f"- {mode} 相对同状态继续α0：新增无触地{len(x['gains'])}、丢失{len(x['losses'])}；逐例及fallback见analysis.json。")
    lines+=['','这是固定一次时点/网格的控制权限证据，非估计器、触发器或完整任务成功；不扫切换时刻。零目标也必须经过原滤波与slew，不能把目标归零误写为瞬间归零实际修正。','','## 全部24配对工况','']
    lines += [f'- [{x}](cases/{x}.png)' for x in links]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(totals=totals,contrasts=contrasts,new_lower_ticks=status['active_lower_ticks'],figures=24),ensure_ascii=False))


if __name__=='__main__':main()
