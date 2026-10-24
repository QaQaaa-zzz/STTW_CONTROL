"""Overlay every matched candidate/parent XY from completed authenticated data."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_best import digest,atomic_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run.resolve()
assert json.loads((root/'status.json').read_text())['phase']=='completed'
results=json.loads((root/'results.json').read_text());plan=json.loads((root/'plan.json').read_text())
output=root/'paired_xy.png'
if output.exists():raise FileExistsError('preserve prior reviewed plot')
colors=dict(parent='#777777',task_only='#0072B2',physical_credit='#D55E00')
fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained');hashes={}
for index,ax in enumerate(axes.flat):
    case=plan['cases'][index]
    ref=root/'parent'/(case['route']+'_path.npz')
    with np.load(ref) as z:
        xy=z['xy'][z['s']<=float(z['goal'])+2.]
        ax.plot(xy[:,0],xy[:,1],c='black',ls='--',alpha=.5,label='original requested path')
        goal=[np.interp(float(z['goal']),z['s'],z['xy'][:,k]) for k in range(2)]
        ax.scatter(*goal,c='black',marker='+',s=70)
    for model,color in colors.items():
        raw=root/model/f'case_{index:03d}.npz';row=results['models'][model][index]
        assert digest(raw)==row['raw_sha256'];hashes[str(raw)]=digest(raw)
        with np.load(raw) as z:
            active=z['active_tick'];xy=z['actual_xy'][active];t=z['time'][active]
            ax.plot(xy[:,0],xy[:,1],c=color,label=f"{model}: {row['outcome']}")
            ax.scatter(*xy[0],c=color,marker='.',s=35)
            ax.scatter(*xy[-1],c=color,marker='x' if row['outcome']=='contact' else 'o',s=45)
            hold=np.flatnonzero(z['fallback_active'][active])
            if len(hold):ax.scatter(*xy[hold[0]],c=color,marker='s',s=45)
            if case['force_N']:
                onset=np.flatnonzero(t>=case['pulse_onset_task_tick']*.005)
                if len(onset):ax.scatter(*xy[onset[0]],c=color,marker='*',s=95)
    ax.set_title(f"{case['route']} | requested v={case['speed']}m/s | F={case['force_N']}N | alpha0")
    ax.set(xlabel='world X (m)',ylabel='world Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=8)
fig.suptitle('New common-noise DEV trajectories after one shared training batch\nx: contact; circle: 10s end; square: path-domain hold; star: nonzero pulse onset')
fig.savefig(output,dpi=170);plt.close(fig)
atomic_json(root/'paired_xy_receipt.json',dict(raw_sha256=hashes,figure_sha256=digest(output),new_physics_ticks=0,
    scope='All12 actual trajectories, original path, equal world axes; survival is not task success.'))
print(str(output))
