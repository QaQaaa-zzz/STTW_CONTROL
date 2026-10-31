"""Compare authenticated completed native online/fixed runs without new physics."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_best import digest
from analyze_native_goal_confirmation import render_paired_load_diagnostics


def compare(online, fixed, output):
    roots=[online.resolve(),fixed.resolve()];plans=[];tables=[]
    for root in roots:
        plan=json.loads((root/'plan.json').read_text());receipt=json.loads((root/'analysis/receipt.json').read_text())
        if receipt['plan_sha256']!=digest(root/'plan.json'):raise ValueError('analysis plan differs')
        if json.loads((root/'evaluation/status.json').read_text())['phase']!='completed':raise ValueError('complete run required')
        rows=json.loads((root/'analysis/results.json').read_text())
        keyed={(r['bank_index'],r['route'],r['alpha'],r['force_N']):r for r in rows}
        if len(keyed)!=len(rows):raise ValueError('duplicate paired condition')
        plans.append(plan);tables.append(keyed)
    for field in ('models','config','initial_source','contexts','replanning','horizon_s'):
        if plans[0][field]!=plans[1][field]:raise ValueError('paired control identity differs: '+field)
    if tables[0].keys()!=tables[1].keys():raise ValueError('paired conditions differ')
    output.mkdir(exist_ok=False);pairs=[];links=[]
    for i,key in enumerate(sorted(tables[0])):
        rows=[table[key] for table in tables]
        for field in ('speed','onset_s','duration_s','original_goal_xy'):
            if rows[0][field]!=rows[1][field]:raise ValueError('paired task/event differs: '+field)
        arrays=[];endpoints=[]
        for root,row in zip(roots,rows):
            if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('raw trace changed')
            with np.load(row['raw_file']) as z:arrays.append(dict(z))
            endpoints.append(row.get('endpoint',row.get('horizons',{}).get('10')))
        paths=[np.load(root/'evaluation'/(row['path_key']+'_path.npz')) for root,row in zip(roots,rows)]
        for name in ('s','xy','heading','curvature','goal'):np.testing.assert_array_equal(paths[0][name],paths[1][name])
        fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained');path=paths[0]
        axes[0].plot(*path['xy'][path['s']<=path['goal']].T,'k:',label='Original reference')
        for name,color,style,row,raw,end in zip(('Coordinator','Fixed user'),('#0072B2','#D55E00'),('-','--'),rows,arrays,endpoints):
            n=int(raw['active_tick'].sum());t=raw['time'][:n];xy=raw['actual_xy'][:n]
            axes[0].plot(*xy.T,color=color,ls=style,label=name);axes[0].scatter(*xy[-1],color=color,marker='x' if end['physical_outcome']=='contact' else 's')
            axes[1].step(t,raw['alpha_exec'][:n] if 'alpha_exec' in raw else np.full(n,row['alpha']),where='post',color=color,ls=style,label=name)
            axes[2].plot(t,raw['actual_forward_speed'][:n],color=color,ls=style,label=name)
        axes[0].scatter(*rows[0]['original_goal_xy'],marker='*',color='black');axes[0].set(xlabel='World X (m)',ylabel='World Y (m)');axes[0].set_aspect('equal',adjustable='datalim')
        axes[1].set(xlabel='Task time (s)',ylabel='Executed alpha');axes[2].set(xlabel='Task time (s)',ylabel='True forward speed (m/s)')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        if key[3]:
            for ax in axes[1:]:ax.axvspan(rows[0]['onset_s'],rows[0]['onset_s']+rows[0]['duration_s'],color='gray',alpha=.2)
        fig.suptitle(f'{key[1]} | bank{key[0]} | user={key[2]:g} | worldY {key[3]:+g}N | U170/lower200 | actual endpoints')
        name=f'pair_{i:02d}_xy.png';fig.savefig(output/name,dpi=140);plt.close(fig);links.append(name)
        for path in paths:path.close()
        diagnostic_rows=[dict(row,active_lower_ticks=int(raw['active_tick'].sum()),endpoint=end,comparison_label=label) for row,raw,end,label in zip(rows,arrays,endpoints,('Coordinator','Fixed user'))]
        component_names=json.loads((online/'evaluation/manifest.json').read_text())['reward_components']
        render_paired_load_diagnostics(output,diagnostic_rows,{row['case_id']:raw for row,raw in zip(rows,arrays)},component_names,f'pair_{i:02d}')
        pairs.append(dict(bank=key[0],route=key[1],alpha=key[2],force_N=key[3],online=endpoints[0],fixed=endpoints[1],online_observed_s=float(arrays[0]['active_tick'].sum()*.005),fixed_observed_s=float(arrays[1]['active_tick'].sum()*.005),online_user_cost=rows[0]['original_user_three_running_costs'],fixed_user_cost=rows[1]['original_user_three_running_costs'],figure=name))
    summary={}
    for name in ('online','fixed'):
        summary[name]=dict(n=len(pairs),arrivals=sum(p[name]['goal_arrived'] for p in pairs),survivals=sum(p[name]['physical_outcome']=='survived' for p in pairs),arrival_and_survival=sum(p[name]['goal_arrived'] and p[name]['physical_outcome']=='survived' for p in pairs))
    (output/'comparison.json').write_text(json.dumps(dict(summary=summary,pairs=pairs,scope='Paired related DEV; costs use actual observed windows and must be read alongside early failure.'),indent=2)+'\n')
    (output/'README.md').write_text('# Matched online versus fixed-user pilot\n\n'+'\n\n'.join(f'![Matched actual/reference XY]({link})\n\n[Reward, speed and error traces]({link.replace("_xy.png","_traces.png")}) · [Signed reward components]({link.replace("_xy.png","_components.png")})' for link in links)+'\n\n'+json.dumps(summary,indent=2)+'\n\nPaired related DEV. Original1m goal entry and full10s survival are separate. Per-case costs retain actual observation lengths; early-failure costs are not comparable full-task performance. Full per-run diagnostic indices remain in the source analyses.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('online',type=Path);parser.add_argument('fixed',type=Path);parser.add_argument('output',type=Path);a=parser.parse_args();compare(a.online,a.fixed,a.output)
