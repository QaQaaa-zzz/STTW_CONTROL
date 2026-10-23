"""Summarize completed matched perturbation traces and render all XY comparisons."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.tolerance_comparison import paired_counts,comparison_layout
from sttw_control.preference_best import atomic_json


def report(root):
 root=Path(root);manifest=json.loads((root/'manifest.json').read_text());protocol=manifest['protocol']
 if manifest.get('engineering_only'):raise ValueError('smoke is not a performance result')
 status=json.loads((root/'status.json').read_text())
 if status.get('phase')!='completed':raise ValueError('evaluation incomplete')
 rows=json.loads((root/'results.json').read_text())['cases'];lanes,_=comparison_layout(protocol)
 fields=('policy','route','speed','alpha','force_N','onset_s')
 expected={tuple(dict(c,onset_s=t)[k] for k in fields) for c in lanes for t in protocol['onsets']}
 found=[tuple(c[k] for k in fields) for c in rows]
 if len(found)!=len(set(found)) or set(found)!=expected:raise ValueError('missing or duplicated evaluation cases')
 out=root/'report';out.mkdir(exist_ok=True);groups=defaultdict(list)
 for row in rows:groups[tuple(row[k] for k in ('route','speed','onset_s','force_N'))].append(row)
 colors=dict(zip(protocol['policies'],['#555555','#0072B2','#E69F00','#009E73']))
 figures=[]
 for (route,speed,onset,force),cases in sorted(groups.items()):
  fig,ax=plt.subplots(figsize=(8,6));traces=[];progress=0.
  for row in cases:
   with np.load(root/(row['name']+'.npz')) as archive:d={k:archive[k] for k in ('actual_xy','path_progress','time')}
   xy=d['actual_xy'];traces.append((row,d));progress=max(progress,float(np.max(d['path_progress'])))
  with np.load(root/(route+'_path.npz')) as path:
   ref=path['xy'][path['s']<=progress+1.];ax.plot(ref[:,0],ref[:,1],'k:',lw=2,label='Original path')
  for row,d in traces:
   xy=d['actual_xy'];color=colors[row['policy']];style='-' if row['alpha']==0 else '--'
   label=f"{row['policy']} a{row['alpha']} {'pass' if row['success'] else 'fail'}"
   ax.plot(xy[::4,0],xy[::4,1],color=color,ls=style,lw=1.3,label=label)
   ax.plot(xy[-1,0],xy[-1,1],marker='o' if row['success'] else 'x',color=color,ms=5)
   pulse=np.flatnonzero((d['time']>=onset)&(d['time']<onset+protocol['pulse_duration']))
   if len(pulse):ax.plot(xy[pulse,0],xy[pulse,1],color=color,lw=3,alpha=.55)
  ax.set(xlabel='World x (m)',ylabel='World y (m)',title=f'{route}, {speed:g} m/s, pulse {force:+g} N at {onset:g} s')
  ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
  fig.text(.02,.01,protocol.get('figure_caption','Parent u200; candidates u50; seed 910301 + alpha. Thick: pulse; x: failed trace endpoint.'),fontsize=7)
  fig.tight_layout(rect=(0,.03,1,1));name=f'xy_{route}_v{speed:g}_t{onset:g}_f{force:+g}.png';fig.savefig(out/name,dpi=140);plt.close(fig);figures.append(name)
 summaries=[]
 for alpha in protocol['endpoints']:
  for policy in protocol['policies']:
   group=[r for r in rows if r['alpha']==alpha and r['policy']==policy];nominal=[r for r in group if r['force_N']==0]
   summaries.append(dict(alpha=alpha,policy=policy,n=len(group),success=sum(r['success'] for r in group),nominal_n=len(nominal),nominal_success=sum(r['success'] for r in nominal),physical_failures=sum(r['physical_failure'] for r in group),domain_failures=sum(r['domain_failure'] for r in group),prefix_invalid=sum(not r['prefix_valid'] for r in group),speed_rmse_mean=float(np.mean([r['speed_rmse'] for r in group])),cross_track_rmse_mean=float(np.mean([r['cross_track_rmse'] for r in group]))))
 pairs={f'a{a}_{b}_vs_{c}':paired_counts(rows,b,c,a) for a in protocol['endpoints'] for b,c in protocol.get('comparisons',[('parent','baseline_update50'),('baseline_update50','static_update50'),('baseline_update50','dynamic_update50'),('static_update50','dynamic_update50')])}
 atomic_json(out/'summary.json',dict(summaries=summaries,paired=pairs,figures=figures,training_seeds=1))
 lines=['# Matched tolerance comparison','', 'One training seed; finite development panel. Original task labels retained as descriptive outcomes. Nominal rows repeat across pulse-onset groups and are not independent episodes. RMSE averages use actual trace length; early failures must be read alongside failure counts.','', '| Policy | Alpha | Full task | Zero force | Physical failures | Domain failures | Invalid prefix |','|---|---:|---:|---:|---:|---:|---:|']
 for r in summaries:lines.append(f"| {r['policy']} | {r['alpha']} | {r['success']}/{r['n']} | {r['nominal_success']}/{r['nominal_n']} | {r['physical_failures']} | {r['domain_failures']} | {r['prefix_invalid']} |")
 lines+=['','## Paired gains and losses','','| Comparison | Newly successful | Lost successes | Net |','|---|---:|---:|---:|']
 for k,v in pairs.items():lines.append(f"| {k} | {v['improved']} | {v['regressed']} | {v['difference']} |")
 lines+=['','## XY trajectories','']+[f'- [{name}]({name})' for name in figures]
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();print(report(a.run))
