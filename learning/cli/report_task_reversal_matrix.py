"""Summarize complete measured DEV branches, wall minima and task outcomes."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from matplotlib.patches import Rectangle
from run_playground_task_reversal import write_rows
from sttw_control.playground_actual_clearance import update_wall_minimum

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
p.add_argument('output',type=Path)
a = p.parse_args()
d, out = a.run, a.output
assert (d/'complete.json').exists()
rows = list(csv.DictReader((d/'oracle_task_reversal.csv').open()))
model = mujoco.MjModel.from_binary_path(str(d/'repeat/terrain.mjb'))
wall = next(i for i in range(model.ngeom) if model.geom(i).name.startswith('planning_obstacle_'))
robot = np.flatnonzero((model.geom_bodyid>0)&((model.geom_contype|model.geom_conaffinity)!=0))
md = mujoco.MjData(model)
plan = json.loads((d/'plan.json').read_text())
lanes = [str(lane) for lane in plan['lanes']]
screen = Path(plan['input'])
prefixes = {r['lane']:r for r in csv.DictReader((screen/'prefixes.csv').open())}
cfg = json.loads((screen/'screen.json').read_text())
world = json.loads(Path(cfg['world_map']).read_text())
out.mkdir(exist_ok=True,parents=True)
fig, panels = plt.subplots(len(lanes),1,figsize=(10,2.9*len(lanes)),squeeze=False,layout='constrained')
axes = panels[:,0]
witnesses = {}
for row in rows:
    key = row['lane'], row['departure']
    if key in witnesses:
        continue
    with np.load(row['trajectory']) as z:
        q,t,ref = z['qpos'],z['trace'][:,0],z['published_reference_xy']
    hand = float(row['handoff_s'])
    idx = np.flatnonzero((t>=hand)&(abs(q[:,0]-model.geom_pos[wall,0])<=model.geom_size[wall,0]+1))
    best = None
    for k in idx:
        md.qpos[:] = q[k]
        mujoco.mj_forward(model,md)
        rot = md.geom_xmat[robot].reshape(-1,3,3)
        centers = md.geom_xpos[robot]+(rot@model.geom_aabb[robot,:3,None])[...,0]
        half = (abs(rot)@model.geom_aabb[robot,3:,None])[...,0]
        lb = np.linalg.norm(np.maximum(abs(centers-md.geom_xpos[wall])-half-model.geom_size[wall],0),axis=1)
        best = update_wall_minimum(model,md,robot,wall,rot,lb,best,t[k])
    witnesses[key] = best or (None,None,None)
    ax = axes[lanes.index(key[0])]
    color = {'left':'#d55e00','right':'#029e73','straight':'#0072b2'}[key[1]]
    ax.plot(*ref.T,':',color=color,lw=.7)
    ax.plot(*q[t>=hand,:2].T,label=key[1],color=color)
    ax.plot(*q[-1,:2],'x',color=color)
for ax,num in zip(axes,lanes):
    corner = model.geom_pos[wall,:2]-model.geom_size[wall,:2]
    ax.add_patch(Rectangle(corner,* (2*model.geom_size[wall,:2]),color='0.3'))
    for x in plan['contracts']['x_sections_m']:
        ax.axvline(x,ls='--',lw=.8,color='0.5')
    ax.set(title=f'Prefix lane {num}',xlim=(11,17.8),ylim=(-1.1,.9),aspect='equal',xlabel='World x (m)',ylabel='World y (m)')
    ax.legend(fontsize=8)
fig.suptitle('Post-handoff measured paths | dotted: reference | dashed: scoring sections')
fig.savefig(out/'task_reversal_xy.png',dpi=150)
audits=json.loads((d/'repeat/jump_validation.json').read_text())['cases']
for row in rows:
    lane=row['lane']
    row['min_clearance_m'],row['min_clearance_s'],row['closest_geometry']=witnesses[lane,row['departure']]
    row['unloaded_ms']=max(e['unloaded_ms'] for e in audits[int(lane)]['events'] if e['solid_bank_rear_launch'] or e['bank_to_bank_flight'])
    row['width_m']=world['terrain']['width_m']
    row['bend_scale']=prefixes[lane].get('bend_scale')
    row['trigger_offset_m']=prefixes[lane]['trigger_offset_m']
    if 'template' in prefixes[lane]:
        row['upstream_template_lane']=prefixes[lane]['template']
    row['speed_m_s']=cfg['oracle_initial_speed_m_s'][int(lane)]
    row['approach_alpha']=cfg['oracle_initial_alpha'][int(lane)]
    row['prefix']=str(Path(row['prefix']).relative_to(d))
    row['trajectory']=str(Path(row['trajectory']).relative_to(d))
write_rows(out/'oracle_task_reversal.csv',rows)
counts=[]
for label in ('near','mid','far'):
    scored=[r for r in rows if r['bin']==label]
    vectors={lane:tuple(int(any(r['success']=='True' for r in scored if r['lane']==lane and r['task']==task)) for task in ('left','right')) for lane in lanes}
    vec=set(vectors.values())
    category=('insufficient_prefixes' if len(lanes)<2 else 'two_way' if {(1,0),(0,1)}<=vec else 'both_universal' if vec=={(1,1)} else
              'both_infeasible' if vec=={(0,0)} else 'one_way_differentiation' if len(vec)>1 else 'shared_one_side')
    times=[float(r['measured_T_s']) for r in scored if r['measured_T_s'] and r['task']=='left']
    counts.append(dict(bin=label,map_root_pairs=1,**{f'prefix{lane}':str(vectors[lane]) for lane in lanes},category=category,
                       reached_section=len(times),physical_suffixes=sum(r['task']=='left' for r in scored),min_T_s=min(times,default=None),max_T_s=max(times,default=None)))
write_rows(out/'reversal_summary.csv',counts)

# Descriptive DEV travel scales (world y), from the same executed suffixes.
# Keep the centre-based task gate separate from clearance of a whole vehicle.
scales=[]
for lane in lanes:
    physical = [r for r in rows if r['lane']==lane and r['bin']=='near' and r['task']=='left']
    straight=next(r for r in physical if r['departure']=='straight')
    with np.load(d/straight['trajectory']) as z:
        base = z['trace'].copy()
        hand = float(straight['handoff_s'])
        k0 = np.argmin(abs(base[:,0]-hand))
        md.qpos[:] = z['qpos'][k0]
        mujoco.mj_forward(model,md)
        rot=md.geom_xmat[robot].reshape(-1,3,3)
        mid=md.geom_xpos[robot]+(rot@model.geom_aabb[robot,:3,None])[...,0]
        half=(abs(rot)@model.geom_aabb[robot,3:,None])[...,0]
        left_need=max(0.,model.geom_pos[wall,1]+model.geom_size[wall,1]-(mid-half)[:,1].min())
        right_need=max(0.,(mid+half)[:,1].max()-(model.geom_pos[wall,1]-model.geom_size[wall,1]))
    for command in physical:
        if command['departure']=='straight':
            continue
        with np.load(d/command['trajectory']) as z:
            tr=z['trace'];t=tr[:,0];vy=tr[:,8]
        # Compare same-prefix nominal at matching absolute times, up to either endpoint.
        same=t<=base[-1,0]
        delta=abs(vy[same]-np.interp(t[same],base[:,0],base[:,8]))
        starts=np.flatnonzero(np.convolve(delta>=.05,np.ones(11,int),'valid')==11)
        delay=float(t[starts[0]]-hand) if len(starts) else None
        for r in (r for r in rows if r['lane']==lane and r['task']=='left' and r['departure']==command['departure']):
            T=r['measured_T_s']
            sign=1 if command['departure']=='left' else -1
            amax=naive=delayed=None
            if T:
                T=float(T)
                vi=vy[t<=hand+T+1e-6]
                amax=max(0.,float(np.max(sign*(vi[20:]-vi[:-20])/.1)))
                v0=sign*base[k0,8]
                naive=v0*T+.5*amax*T*T
                if delay is not None:
                    delayed=v0*T+.5*amax*max(0.,T-delay)**2
            scales.append(dict(lane=lane,departure=command['departure'],bin=r['bin'],T_s=T,delay_s=delay,
                               max_signed_100ms_accel_m_s2=amax,point_shift_m=naive,delayed_shift_m=delayed,
                               whole_body_shift_to_clear_wall_m=left_need if sign==1 else right_need))
write_rows(out/'lateral_scales.csv',scales)
