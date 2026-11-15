"""All-case actual/reference/full-chain predictions, with optional paired I/C results."""
import argparse,csv,json,shutil,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser(description=__doc__)
for name in ['run','routes','output']:p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--extra',type=Path,action='append',default=[]);p.add_argument('--extra-label',action='append',default=[]);p.add_argument('--compare',type=Path);p.add_argument('--label');p.add_argument('--compare-label')
a=p.parse_args();
if len(a.extra)!=len(a.extra_label):p.error('Each extra comparison requires one label')
a.output.mkdir(parents=True,exist_ok=True)
contract=json.loads((a.run/'contract.json').read_text())
if contract.get('evaluation'):
 config=json.loads((Path(contract['evaluation'])/'resolved_config.json').read_text());world=json.loads(Path(config['world_map']).read_text())
else:world=json.loads((Path(contract['oracle'])/'map48/world.json').read_text())
paths={a.label or contract.get('method','C'):a.run}
if a.compare:paths[a.compare_label or json.loads((a.compare/'contract.json').read_text())['method']]=a.compare
for label,path in zip(a.extra_label,a.extra):paths[label]=path
records={k:json.loads((v/'cases.json').read_text()) for k,v in paths.items()};arrays={k:np.load(v/'predictions.npz') for k,v in paths.items()};rows=next(iter(records.values()));bank=np.load(a.routes)
y_limit=max(1.6,max(float(np.abs(z[k][:,:,1]).max()) for z in arrays.values() for k in ['approach','jump','departure'])+.1)
fig,axes=plt.subplots(math.ceil(len(rows)/3),3,figsize=(18,math.ceil(len(rows)/3)*max(1.7,y_limit*.65)),layout='constrained',squeeze=False)
for i,(row,ax) in enumerate(zip(rows,axes.flat)):
 prefix=np.load(row['prefix']);actual=prefix['qpos'][:,:2]
 if row['suffix']:
  tail=np.load(row['suffix']);mask=prefix['trace'][:,0]<tail['trace'][0,0]-1e-6;actual=np.r_[prefix['qpos'][mask,:2],tail['qpos'][:,:2]]
 reference=bank['xy'][row['route']];ax.plot(reference[:,0],reference[:,1],':',color='.6',label='Original reference');ax.plot(actual[:,0],actual[:,1],color='#0072B2',label='Actual');ax.scatter(*actual[-1],color='#0072B2',marker='x',s=16)
 for method,color,style in zip(paths,['#D55E00','#009E73','#CC79A7','#E69F00','#222222','#56B4E9'],['--','-.',':','--','-.',':']):
  z=arrays[method]
  for j,(stage,stop) in enumerate([('approach',int(z['entry_steps'][i])+1),('jump',int(z['jump_steps'][i])),('departure',int(z['departure_steps'][i])+1)]):
   points=z[stage][i,:stop,:2];ax.plot(points[:,0],points[:,1],style,color=color,lw=1,label=method if j==0 else None)
 for x0,x1,y0,y1 in world['terrain'].get('forbidden_rectangles',[]):ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='.5',alpha=.25))
 front=world['terrain']['front_x_m'];ax.axvspan(front,front+world['terrain']['width_m'],color='#56B4E9',alpha=.15)
 ax.set_title(f'root{row["root"]} route{row["route"]} v{row.get("speed",1.8):.1f} | actual{int(row["actual_success"])} '+ 'admit['+','.join(str(int(records[m][i]['admitted'])) for m in paths)+']',fontsize=8)
 ax.set_xlim(1.5,20.5);ax.set_ylim(-y_limit,y_limit);ax.set_aspect('equal',adjustable='box');ax.set_xlabel('World x (m)',fontsize=7);ax.set_ylabel('World y (m)',fontsize=7);ax.tick_params(labelsize=7);ax.grid(alpha=.2)
for ax in list(axes.flat)[len(rows):]:ax.set_visible(False)
axes[0,0].legend(fontsize=6,ncol=2);fig.suptitle(f'Known DEV: all{len(rows)} trajectories; travel +x\nActual stops at failure/goal; prediction stops at predicted events/goal or max horizon; stage gaps retained. Admit vector follows prediction legend order.',fontsize=11);fig.savefig(a.output/'xy.png',dpi=160);plt.close(fig)
summaries={}
for method,path in paths.items():
 summaries[method]=json.loads((path/'summary.json').read_text());shutil.copyfile(path/'contract.json',a.output/f'{method}_contract.json')
 with (a.output/f'{method}_cases.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=records[method][0],lineterminator='\n');writer.writeheader();writer.writerows(records[method])
(a.output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
text=f'# P4 full-chain DEV diagnostic\n\n[All{len(rows)} actual/reference/predicted XY](xy.png).\n\n|Method|True admit|False admit|False reject|True reject|\n|---|---:|---:|---:|---:|\n'
for method,s in summaries.items():text+='|'+method+'|'+ '|'.join(str(s[k]) for k in ['true_admit','false_admit','false_reject','true_reject'])+'|\n'
text+='\nSame frozen skills, original task gates and raw envelopes; no new calibration or physical episodes. C propagates predicted states/history from real initial state; I uses route/speed-conditioned TRAIN interfaces. These are known DEV conditions, not independent TEST. All future actual data are only offline labels. Empty forbidden-box terrain still checks world bounds; absence of negatives cannot establish obstacle rejection quality. P4 remains incomplete.\n'
(a.output/'README.md').write_text(text)
