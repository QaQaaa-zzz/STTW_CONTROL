from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
r=Path(__file__).parent;dest=Path('docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p4_sequence/phase_order');groups=json.loads((r/'audit/summary.json').read_text())['groups'];fig,axes=plt.subplots(3,3,figsize=(16,13),layout='constrained');cmap=ListedColormap(['#CC79A7','#0072B2','#009E73','#E69F00','#56B4E9']);cmap.set_bad('#dddddd');norm=BoundaryNorm(np.arange(-.5,5.5),5)
for g,ax in zip(groups,axes.flat):
 rows=json.loads((Path(g['prediction'])/'cases.json').read_text());z=np.load(r/'audit'/f"trace_{g['group']:02}.npz");a=np.full((2*len(rows),125),np.nan)
 for i,row in enumerate(rows):
  phase=z['phase_logits'][i].argmax(-1);n=int(z['event_step'][i]);a[2*i+1,:n]=phase[:n]
  truth=np.load(row['prefix']);start=np.flatnonzero(truth['trace'][:,-1]==1);end=np.flatnonzero(truth['trace'][:,-1]==2)
  if len(start):
   ids=start[0]+4*np.arange(1,126);valid=(ids<len(truth['trace']))
   if len(end):valid&=ids<=end[0]
   a[2*i,np.flatnonzero(valid)]=truth['trace'][ids[valid],-1]
 im=ax.imshow(a,cmap=cmap,norm=norm,aspect='auto',interpolation='nearest',extent=[.02,2.52,2*len(rows),0]);ax.set_title(f"Seed{g['group']//3} / {Path(g['prediction']).name} / {len(rows)}cases");ax.set_xlabel('Seconds from each own RB entry');ax.set_ylabel('Case order: actual row, predicted row');ax.set_yticks(np.arange(len(rows))*2+1);ax.set_yticklabels([f"{x['root']}/{x['route']}" for x in rows],fontsize=5)
bar=fig.colorbar(im,ax=axes.ravel().tolist(),ticks=range(5),shrink=.6);bar.ax.set_yticklabels(['0 approach','1 RB','2 ground','3 prepare','4 relock']);fig.suptitle('All270 known DEV conditions: actual vs frozen cosine predicted phase; grey = absent or past stop.\nOwn-entry alignment diagnoses ordering, not absolute event-time accuracy. Actual no-RB failures retained.',fontsize=12);fig.savefig(dest/'phase_sequences.png',dpi=150);plt.close(fig)
