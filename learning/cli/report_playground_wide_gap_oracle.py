"""Rebuild the wide-gap oracle matrix and paired evidence from recorded poses."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import mujoco
import numpy as np

from sttw_control.playground_actual_clearance import obstacle_aabb_bounds


def join(row):
    with np.load(row['prefix']) as z:
        data={k:z[k] for k in ('qpos','qvel','trace','reference_xy')}
    if row['suffix']:
        with np.load(row['suffix']) as z:
            idx=np.flatnonzero(np.rint(data['trace'][:,0]/.005)==round(float(z['trace'][0,0])/.005)).item()
            for k in ('qpos','qvel'):
                np.testing.assert_array_equal(data[k][idx],z[k][0])
            for k in ('qpos','qvel','trace'):
                data[k]=np.concatenate([data[k][:idx],z[k]])
            data['reference_xy']=z['reference_xy']
    return data


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('out',type=Path)
    args=ap.parse_args()
    out=args.out
    out.mkdir(exist_ok=True,parents=True)
    rows=list(csv.DictReader((args.root/'oracle_matrix.csv').open()))
    models={}
    cache={}
    fig,axes=plt.subplots(2,2,figsize=(14,5),layout='constrained')
    for row in rows:
        source=Path(row['initial_snapshot']).parent/'terrain.mjb'
        if source not in models:
            models[source]=mujoco.MjModel.from_binary_path(str(source))
        key=(row['prefix'],row['suffix'])
        if key not in cache:
            data=join(row)
            bounds,dist=obstacle_aabb_bounds(models[source],data['qpos'])
            cache[key]=dist
            ax=axes[int(row['map'].endswith('055')),int(row['skill']=='source')]
            ax.plot(*data['reference_xy'].T,':',color='0.65',lw=.6)
            ax.plot(*data['qpos'][:,:2].T,lw=.8,alpha=.65)
            ax.plot(*data['qpos'][-1,:2],'x',ms=3)
            ax.set(title=row['map']+' / '+row['skill'],xlabel='World x (m)',ylabel='World y (m)',xlim=(1,19),ylim=(-1.6,1.6),aspect='equal')
        row.pop('clearance_m',None)
        row['obstacle_distance_lower_bound_m']=cache[key]
    for ax in axes.flat:
        ax.add_patch(Rectangle((13.3,-.06),.2,.12,color='black'))
    fig.savefig(out/'oracle_xy.png',dpi=145)
    plt.close(fig)
    with (out/'oracle_matrix.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    def approach_key(r):
        return (r['initial_snapshot'],r['root'],r['approach'],r['speed_m_s'],r.get('activation_offset_m',''))
    groups={}
    for row in rows:
        groups.setdefault(approach_key(row),[]).append(row)
    feasible={k:any(r['full_success']=='True' for r in group) for k,group in groups.items()}
    def has_flight(r):
        return r['strict_flight']=='True' or r['rear_launch_solid']=='True'
    mechanism=[]
    for k,l in itertools.combinations(groups,2):
        if k[:2]!=l[:2] or feasible[k]==feasible[l]:continue
        trials=groups[k]+groups[l]
        if all(r['status'] in ('continuous','branched') and r['landing_s'] for r in trials):
            mechanism.append([k,l])
    controls=[]
    for k,l in itertools.combinations(groups,2):
        if k[:2]!=l[:2]:continue
        a,b=groups[k][0],groups[l][0]
        with np.load(a['prefix']) as x,np.load(b['prefix']) as y:
            n=min(len(x['trace']),len(y['trace']))
            use=(x['trace'][:n,19]==0)&(y['trace'][:n,19]==0)
            if not use.any():continue
            init={f:float(abs(x[f][0]-y[f][0]).max()) for f in ['qpos','qvel','rb_state']}
            xy=np.linalg.norm(x['qpos'][:n,:2][use]-y['qpos'][:n,:2][use],axis=1)
            delta=abs(x['applied_targets'][:n][use]-y['applied_targets'][:n][use])
            controls.append(dict(root=k[1],map=a['map'],skill=a['skill'],a=k[2:],b=l[2:],
                initial_difference=init,approach_xy_max_m=float(xy.max()),
                approach_applied_targets_max_by_channel=np.nanmax(delta,axis=0).tolist()))
    (out/'approach_controls.json').write_text(json.dumps(controls,separators=(',',':'))+'\n')
    counts=[]
    for name,skill in sorted({(r['map'],r['skill']) for r in rows}):
        subset=[g[0] for g in groups.values() if g[0]['map']==name and g[0]['skill']==skill]
        counts.append(dict(map=name,skill=skill,approaches=len(subset),flight=sum(has_flight(r) for r in subset),
            landing=sum(bool(r['landing_s']) for r in subset),
            with_feasible_departure=sum(feasible[approach_key(r)] for r in subset)))
    (out/'summary.json').write_text(json.dumps(dict(cells=len(rows),approach_groups=len(groups),
        executed_departure_branches=sum(r['status']=='branched' for r in rows),
        missing_loaded_snapshot=sum(r['status']=='missing_loaded_snapshot' for r in rows),
        counts=counts,paired_feasibility_cases=mechanism),indent=2)+'\n')
    candidates=[]
    for a,b in itertools.combinations(rows,2):
        if (a['initial_snapshot'],a['root'])!=(b['initial_snapshot'],b['root']):continue
        if approach_key(a)==approach_key(b):continue
        # Prefer two actual jumps and different full outcomes, then two loaded landings.
        flights=sum(has_flight(r) for r in (a,b))
        pair=[approach_key(a),approach_key(b)]
        score=(pair in mechanism or pair[::-1] in mechanism,flights,(a['full_success']!=b['full_success']),sum(bool(r['landing_s']) for r in (a,b)),
               sum(bool(r['handoff_s']) for r in (a,b)),sum(float(r['end_s']) for r in (a,b)))
        candidates.append((score,a,b))
    _,a,b=max(candidates,key=lambda p:p[0])
    video=out/'video'
    video.mkdir(exist_ok=True)
    decl=json.loads((Path(a['initial_snapshot']).parent/'declaration.json').read_text())
    (video/'declaration.json').write_text(json.dumps(decl))
    model=models[Path(a['initial_snapshot']).parent/'terrain.mjb']
    fig,axes=plt.subplots(3,1,figsize=(10,7),layout='constrained')
    rendered=[]
    for i,row in enumerate((a,b)):
        name=f"{row['skill']}_root{row['root']}_{row['approach']}_{row['speed_m_s']}_{row['departure']}"
        if row.get('activation_offset_m'):
            name+=f"_x{row['activation_offset_m']}"
        d=join(row);t=d['trace'][:,0];folder=video/name;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'trajectory.npz',**d)
        src=Path(row['suffix'] or row['prefix'])
        actual=next(r for r in json.loads((src.parents[1]/'status.json').read_text())['results'] if r['case_dir']==src.parent.name)
        rendered.append(dict(actual,case_dir=name))
        env,_=obstacle_aabb_bounds(model,d['qpos'][::20])
        color=['#006bb3','#c35a00'][i]
        axes[0].add_collection(PatchCollection([Rectangle((x,y),u-x,v-y) for x,y,u,v in env],facecolor=color,alpha=.12))
        axes[0].plot(*d['reference_xy'].T,':',color=color,lw=.8)
        axes[0].plot(*d['qpos'][:,:2].T,color=color,label=name+' '+row['termination'])
        axes[1].plot(t,np.degrees(d['trace'][:,4]),color=color)
        axes[2].plot(t,d['qpos'][:,2],color=color)
    axes[0].add_patch(Rectangle((13.3,-.06),.2,.12,color='black'))
    axes[0].set(xlim=(1,19),ylim=(-1.6,1.6),aspect='equal',xlabel='World x (m)',ylabel='World y (m)')
    axes[0].legend(fontsize=7)
    axes[1].set(ylabel='Roll (deg)',xlabel='Time (s)')
    axes[2].set(ylabel='Root height (m)',xlabel='Time (s)')
    fig.savefig(out/'paired_states_occupancy.png',dpi=145)
    (out/'paired.json').write_text(json.dumps([a,b],indent=2)+'\n')
    (video/'status.json').write_text(json.dumps(dict(results=rendered)))
    subprocess.run([sys.executable,'learning/cli/render_playground_failures.py',str(video)],check=True)


if __name__=='__main__':
    main()
