"""Summarize frozen jump candidates on identical executed approach tasks."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);r=p.parse_args().run
rows=json.loads((r/'evaluation_index.json').read_text());models=list(dict.fromkeys(x['model'] for x in rows));origins=sorted({x['origin'] for x in rows})
for row in rows:
 d=Path(row['output']);row['chain']=json.loads((d/'chain_summary.json').read_text());row['landing']=json.loads((d/'landing.json').read_text())['cases'][0];row['arrival']=(d/'handoff.json').exists()
summary={m:dict(n=sum(x['model']==m for x in rows),arrival=sum(x['arrival'] for x in rows if x['model']==m),landing=sum(x['landing']['landing_hold_1s'] for x in rows if x['model']==m),contact_filtered_landing=sum(x['landing']['contact_filtered_20ms_hold_1s'] for x in rows if x['model']==m),complete=sum(x['chain']['complete'] for x in rows if x['model']==m)) for m in models}
for m in models:
 cases=[x for x in rows if x['model']==m]
 summary[m]['stage_counts']={k:sum(x['chain']['stages'][k] for x in cases) for k in cases[0]['chain']['stages']}
 summary[m]['failed_checks']={k:sum(not x['chain']['checks'][k] for x in cases) for k in cases[0]['chain']['checks']}
# Keep paired gains and losses rather than reporting only the net success count.
base={x['candidate']['source_case_id']:x for x in rows if x['model']=='update2'}
for m in models:
 pairs=[(base[x['candidate']['source_case_id']]['chain']['complete'],x['chain']['complete']) for x in rows if x['model']==m]
 summary[m].update(gained=sum(not a and b for a,b in pairs),lost=sum(a and not b for a,b in pairs))
for origin in origins:
 fig,axs=plt.subplots(1,len(models),figsize=(6*len(models),6),layout='constrained')
 for ax,m in zip(np.atleast_1d(axs),models):
  for x in [x for x in rows if x['origin']==origin and x['model']==m]:
   d=Path(x['output']);a=np.load(d/'trajectory.npz');t=a['trace'][:,0][a['valid'][:,0]];c=x['candidate'];color={0.:'tab:blue',.5:'tab:orange',1.:'tab:green'}[c['alpha']]
   ax.plot(t[:,1],t[:,2],color=color,ls='-' if c['speed_m_s']==2 else '--',label=f"v={c['speed_m_s']} alpha={c['alpha']}");ax.scatter(*t[-1,1:3],color=color,marker='o' if x['chain']['complete'] else 'x')
  route=np.load(d/'ground_route.npz')['xy'];ax.plot(*route.T,'k:',label='Original route');ax.set(title=m,xlabel='x (m)',ylabel='y (m)');ax.axis('equal');ax.legend(fontsize=7)
 fig.suptitle(origin+'; current chassis and Kp1 ground; matched DEV');fig.savefig(r/(origin+'_xy.png'),dpi=150);plt.close(fig)
(r/'summary.json').write_text(json.dumps(dict(summary=summary,cases=rows),indent=2))
lines=['# Current-chassis jump adaptation','','Matched DEV: six training origins, six connections per model; not independent TEST.','Original platform/contact/hold and full-chain criteria retained. Sampled training reward selects candidates, not qualification.','','|Model|Arrivals|Strict landing hold|20ms filtered landing (diagnostic)|Full chain (strict)|Gained / lost vs update2|','|---|---:|---:|---:|---:|---:|']
for m,s in summary.items():lines.append(f"|{m}|{s['arrival']}/{s['n']}|{s['landing']}/{s['n']}|{s['contact_filtered_landing']}/{s['n']}|{s['complete']}/{s['n']}|{s['gained']} / {s['lost']}|")
lines.extend(['','Failed conditions are overlapping counts, not mutually exclusive causes.'])
for m,s in summary.items():lines.extend(['',m+': stages '+json.dumps(s['stage_counts'])+'; failed conditions '+json.dumps(s['failed_checks'])])
for origin in origins:lines.extend(['',f'![{origin}]({origin}_xy.png)'])
(r/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary))
