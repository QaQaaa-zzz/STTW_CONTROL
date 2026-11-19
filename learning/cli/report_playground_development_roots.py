"""Export recorded approach trajectories and per-root counts after a finite screen."""
import argparse
import csv
import json
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('output',type=Path)
p.add_argument('label')
a=p.parse_args()
terminal=json.loads((a.run/'terminal.json').read_text())
rows=list(csv.DictReader((a.run/'prefixes.csv').open()))
assert len(rows)==terminal['scored']
by_root={}
for r in rows:
    by_root.setdefault(r['root'],[]).append(r)
fig,axes=plt.subplots((len(by_root)+1)//2,2,squeeze=False,figsize=(12,2*len(by_root)/2+1),layout='constrained')
for ax,(root,records) in zip(axes.flat,by_root.items()):
    refs=set()
    for r in records:
        folder=a.run/f"group_{r['group']}"/'screen' if 'group' in r else a.run/'screen'
        case=f"source_route{r['lane']}_h75_v{r['lane']}"
        with np.load(folder/case/'trajectory.npz') as z:
            tr=z['trace']; ref=z['reference_xy']
            identity=ref.tobytes()
            if identity not in refs:
                refs.add(identity);ax.plot(*ref.T,':',lw=.6,color='0.3')
            hand=r['qualified']=='True'
            col='#0072b2' if hand else '#d55e00' if r['flight']=='True' else '0.65'
            ix=np.r_[np.arange(0,len(tr)-1,10),len(tr)-1]
            ax.plot(tr[ix,1],tr[ix,2],color=col,alpha=.7,lw=1.1 if hand else .6)
            ax.scatter(*tr[-1,[1,2]],marker='x',s=9,color=col)
    q=sum(r['qualified']=='True' for r in records)
    ax.set(xlim=(1,18),ylim=(-1.7,1.7),aspect='equal',title=f'Root {root} | {q}/{len(records)} handbacks',xlabel='World x (m)',ylabel='World y (m)')
for ax in list(axes.flat)[len(by_root):]:ax.set_visible(False)
fig.suptitle('Actual and reference XY (dotted) | blue: handback, orange: flight only, gray: other | cross: endpoint')
fig.savefig(a.output/f'{a.label}_xy.png',dpi=140)
copyfile(a.run/'prefixes.csv',a.output/f'{a.label}_prefixes.csv')
