"""All-case diagnostic XY and admission confusion for RB handoff interchange."""
import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.run/'cases.json').read_text());summary=json.loads((a.run/'summary.json').read_text());pred=np.load(a.run/'predictions.npz');cases=[r for r in rows if r['source']=='measured' and r['margin']=='raw']
contract=json.loads((a.run/'contract.json').read_text());world=json.loads((Path(contract['oracle'])/'map48/world.json').read_text())
fig,axes=plt.subplots(4,3,figsize=(15,12),layout='constrained');colors=['#0072B2','#D55E00','#009E73','#CC79A7'];styles=['-','--','-.',':']
for i,(ax,row) in enumerate(zip(axes.flat,cases)):
 actual=np.load(row['actual_trajectory']);ax.plot(actual['qpos'][:,0],actual['qpos'][:,1],color='black',lw=2,label='Actual suffix');ax.scatter(*actual['qpos'][-1,:2],color='black',marker='X',s=25)
 reference=actual['reference_xy'];reference=reference[reference[:,0]>=10.5];ax.plot(reference[:,0],reference[:,1],color='#999999',ls=':',lw=1,label='Original reference')
 for source,color,style in zip(['measured','seed0','seed1','seed2'],colors,styles):
  stop=int(pred[source+'_stop'][i]);xy=pred[source+'_pose'][i,:stop+1,:2];ax.plot(xy[:,0],xy[:,1],color=color,ls=style,lw=1,label=source)
 for rectangle in world['terrain']['forbidden_rectangles']:
  x0,x1,y0,y1=rectangle;ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#777777',alpha=.35))
 ax.set_title(f'Root{row["root"]} up{row["upstream"]} down{row["downstream"]}: '+('goal' if row['actual_success'] else 'collision'),fontsize=10);ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=7);fig.suptitle('Map48 DEV: same ground model, measured or predicted handoff\nActual trace stops at real terminal; predictions stop at predicted goal or max horizon; travel +x')
fig.savefig(a.output/'xy.png',dpi=160);plt.close(fig)
for name in ['summary.json','contract.json']:shutil.copyfile(a.run/name,a.output/name)
with (a.output/'cases.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
text='# P3 handoff interchange: decision diagnostic\n\n[All12 actual/reference/predicted XY](xy.png). Source: `'+str(a.run)+'`. No new physical episodes; actual labels reuse the frozen map48 Oracle. Known DEV, not independent TEST or full-chain prediction.\n\n|Input|Margin|True admit|False admit|False reject|True reject|\n|---|---|---|---|---|---|\n'
text+='\n'.join(f'|{r["source"]}|{r["margin"]}|{r["true_admit"]}|{r["false_admit"]}|{r["false_reject"]}|{r["true_reject"]}|' for r in summary)
text+='\n\nEach row is12 conditions:2 upstream approaches x2 roots x3 departures, with4 actual feasible and8 failed. Raw geometry is a diagnostic; frozen M1 margin is unchanged, including its false rejections. Neither is a safety certificate. `state_only` replaces physical/previous state and event time while keeping real history; `history_only` replaces controller history while keeping measured physical state/time. This is an offline error-isolation experiment, not a deployable use of true future state.\n'
(a.output/'README.md').write_text(text)
