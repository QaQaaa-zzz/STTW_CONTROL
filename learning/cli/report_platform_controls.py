"""Audit sampled flight, physical platform support, and paired chapter-three controls."""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from jit_dvgc.config import load_config
from jit_dvgc.descent_semantics import (DescentSignals, advance_descent_events,
    classify_descent_terminal, initial_descent_events)
from jit_dvgc.geometry import build_geometry_contract, collision_support_bounds, geometric_penetration_signals
from jit_dvgc.semantics import PhaseUSignals, TerminalInputs, advance_events, classify_terminal, initial_event_state
from evaluate_native_platform import ARCHIVE, POLICIES

DIRECT = {'implicitfast_1': '.', 'rk4_1': 'rk4_1', 'implicitfast_5': 'implicitfast_5', 'rk4_5': 'rk4_5'}
ROUTES = ['left','straight','right']


def save_rows(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def runs(mask):
    switch = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(switch == 1), np.flatnonzero(switch == -1))


def positions(model, poses, ids):
    data = mujoco.MjData(model)
    centers = np.empty((len(poses), len(ids), 3))
    rotations = np.empty((len(poses), len(ids), 3, 3))
    for k, q in enumerate(poses):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        centers[k] = data.geom_xpos[ids]
        rotations[k] = data.geom_xmat[ids].reshape(-1, 3, 3)
    return centers, rotations


def flight(model, q, loads, supported, time, near, far, height, begin):
    ids = [model.geom(w+'wheel_collision').id for w in ('front','rear')]
    centers, rotations = positions(model, q, ids)
    radius = np.linalg.norm(rotations * model.geom_size[ids][None,:,None,:], axis=-1)
    bottom = centers[:,:,2] - radius[:,:,2]
    front_x, rear_x = centers[:,:,0].T
    far_dx = np.maximum(far - centers[:,:,0] - radius[:,:,0], 0.)
    near_dx = np.maximum(centers[:,:,0] - radius[:,:,0] - near, 0.)
    clearance = np.minimum(np.hypot(far_dx, np.maximum(bottom-height, 0.)),
                           np.hypot(near_dx, np.maximum(bottom, 0.))).min(1)
    top = (rear_x > far) & (front_x < far+4) & (np.abs(centers[:,:,1])<1).all(1)
    top &= (np.abs(bottom-height)<.02).all(1) & supported
    n = loads.shape[1]
    free = (loads[:,:,:3]<=1e-5).all(2)
    if loads.shape[-1] == 4:
        free &= loads[:,:,3] == 0
    for start, stop in runs(free.flatten()):
        first, end = (start+n-1)//n, (stop-1)//n
        if end <= first or time[first]<begin or rear_x[first]>=near or front_x[end]<near:
            continue
        high = clearance[first:end+1] > .01
        if not (high[:-1] & high[1:]).any() or not np.any(np.diff(q[first:end+1,2])>0):
            continue
        land = np.flatnonzero(top & (time>time[end]))
        return float(time[first]), float(time[end]), float(time[land[0]]) if len(land) else None
    return None, None, None


def chapter2_endpoint(model):
    """Apply the archived upstream and descent state machines to saved 50 Hz poses."""
    config = json.loads(Path('/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/audit/runtime_config.json').read_text())['inputs']
    up, down = (load_config(Path(config[k+'_config_path']), runtime_only=True) for k in ('up','down'))
    geometry = build_geometry_contract(model)
    offset = geometry.obstacle_front_x - 3.6
    ids = np.array(geometry.robot_geom_ids)
    @jax.jit
    def replay(centers, rotations, state, start):
        def step(carry, frame):
            ue,de,descending,done,tick,last_x = carry
            center, rotation, values = frame
            x,z,vx,vz,roll,pitch,yaw = values
            x -= offset
            bounds = collision_support_bounds(center, rotation, geometry.robot_geom_types, geometry.robot_geom_sizes)
            g = geometric_penetration_signals(bounds, geometry)
            inputs = TerminalInputs(tick, False,roll,pitch,g.prohibited_contact,x < -up.physical_limits.max_backward_distance,ue.stuck,yaw,ue.jump_zone_seen)
            pre = classify_terminal(inputs, up)
            ue = advance_events(ue,PhaseUSignals(x,z,vx,vz,pre.terminated),up)
            ut = classify_terminal(inputs.replace(stuck=ue.stuck,jump_zone_seen=ue.jump_zone_seen),up)
            signals = DescentSignals(x,g.front_wheel_terrain_clearance,g.rear_wheel_terrain_clearance,g.maximum_wheel_penetration,g.prohibited_contact,jp.asarray(True),roll,pitch,x<last_x-down.physical_limits.max_backward_distance,vx)
            advanced = advance_descent_events(de, signals, down.descent)
            dt = classify_descent_terminal(signals, advanced, down.descent, down.physical_limits,tick+1,down.ppo.episode_horizon)
            success = ~done & descending & dt.success
            done |= jp.where(descending, dt.terminated|dt.truncated, ut.terminated|ut.truncated).astype(bool)
            de = jax.tree.map(lambda a,b: jp.where(descending,a,b),advanced,initial_descent_events(x))
            tick = jp.where(~descending & ue.apex_seen,0,tick+1)
            return (ue,de,descending|ue.apex_seen,done,tick,x),success
        x = start-offset
        return jax.lax.scan(step,(initial_event_state(x,up),initial_descent_events(x),jp.asarray(False),jp.asarray(False),0,x),(centers,rotations,state))[1]
    def score(q, trace):
        c,r = positions(model, q[1:], ids)
        values=trace[1:][:,[1,3,7,9,4,5,6]]
        # One fixed compilation for all episode lengths; remove padding from the verdict.
        pad=400-len(c)
        packed=[np.pad(a,[(0,pad)]+[(0,0)]*(a.ndim-1),mode='edge').astype('float32') for a in (c,r,values)]
        success=np.flatnonzero(np.asarray(replay(*packed,np.float32(q[0,0])))[:len(c)])
        return float(trace[1+success[0],0]) if len(success) else None
    return score


def report(root, out):
    out.mkdir(parents=True,exist_ok=True)
    direct_model=mujoco.MjModel.from_xml_path('/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml')
    direct=[]
    fig, axes=plt.subplots(2,4,figsize=(13,5),sharex=True,sharey=True)
    for col,(cell,subdir) in enumerate(DIRECT.items()):
        for ax,(skill,directory) in zip(axes[:,col],POLICIES.items()):
            a=np.load(root/subdir/(skill+'.npz'))
            with (ARCHIVE/directory/'evaluation/batch_000/episodes.csv').open() as f:
                archived=list(csv.DictReader(f))
            for lane,mask in enumerate(a['prefix_mask'].T):
                def val(key):return a[key][mask,lane]
                q=val('qpos'); t=val('time_after')
                load=np.stack([val('all_load'+str(i)) for i in range(3)],1)
                s,e,land=flight(direct_model,q,load[:,None],(val('step_load0')>1e-5)&(val('step_load1')>1e-5),t,3.6,3.6,.16,0.)
                success=bool(val('success').any())
                direct.append(dict(cell=cell,skill=skill,bank_row=lane,original_success=int(archived[lane]['success']),native_success=int(success),
                                   sampled_flight_start_s=s,sampled_flight_end_s=e,loaded_top_s=land,verified_recovery=int(success and land is not None),end_code=int(val('end_code')[-1])))
                ax.plot(q[:,0],q[:,1],color='#17624e' if success else '#ae5144',alpha=.6,lw=.65)
            ax.axvline(3.6,color='black',lw=.8); ax.set_aspect('equal',adjustable='box');ax.set_title(skill+' / '+cell);ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    fig.tight_layout();fig.savefig(out/'direct_xy.png',dpi=135);plt.close(fig)
    save_rows(out/'direct_cases.csv',direct)
    records=[]
    fig,axes=plt.subplots(5,3,figsize=(14,9),sharex=True,sharey=True)
    groups = [('test1','test1'),('test2','test2'),('test3','test3_near_anchor')]
    if (root/'test2_speeds/status.json').is_file():
        groups.append(('test2_speed','test2_speeds'))
    for row,(which,subdir) in enumerate(groups):
        path=root/subdir
        model=mujoco.MjModel.from_binary_path(str(path/'terrain.mjb'))
        score=chapter2_endpoint(model) if which=='test1' else None
        terrain=json.loads((path/'resolved_config.json').read_text())['ditch_terrain']
        near=terrain['front_x_m']; far=near+terrain['width_m']
        for case in json.loads((path/'status.json').read_text())['results']:
            a=np.load(path/case['case_dir']/'trajectory.npz');tr=a['trace'];q=a['qpos'];t=tr[:,0]
            loads=a['contact_substep']; supported=a['wheel_contact_flags'][:,2] & (loads[:,-1,:2]>1e-5).all(1)
            entry=case['rb_entry_s']
            s,e,land=flight(model,q,loads,supported,t,near,far,terrain['far_bank_height_m'],entry if entry is not None else np.inf)
            back=case['handoff_s']; verified=bool(land is not None and back is not None and land<=back+1e-5)
            rt=None
            if score and entry is not None:
                ids=np.flatnonzero((t>=entry-1e-5)&(np.rint(t/.005).astype(int)%4==0))
                if len(ids)>1:rt=score(q[ids],tr[ids])
            rid=case['route'];skill=case['skill']; root_id=case['variant']['bank_row']
            records.append(dict(test=which,skill=skill,bank_row=root_id,approach=ROUTES[rid],requested_speed_m_s=case['variant']['approach_speed_m_s'],case_dir=case['case_dir'],entry_s=entry,
                                actual_entry_vx_m_s=float(tr[np.argmin(abs(t-entry)),7]) if entry is not None else None,
                                flight_start_s=s,flight_end_s=e,loaded_top_s=land,original_recovery_replay_s=rt,verified_rb_recovery=int(rt is not None and land is not None and land<=rt) if score else None,
                                raw_handoff_s=back,verified_handoff=int(verified),verified_goal=int(verified and case['goal']),termination=case['termination']))
            panel = row + int(which=='test2_speed' and case['variant']['approach_speed_m_s']>2.)
            ax=axes[panel,rid];col='#17624e' if skill=='source' else '#a74828'
            ax.plot(q[:,0],q[:,1],color=col,alpha=.55,lw=.65)
            ax.plot(q[-1,0],q[-1,1],'.',color=col,ms=2)
            if root_id==0 and skill=='source':
                ref=a['reference_xy'];ax.plot(ref[:,0],ref[:,1],':',color='#163e69',lw=1);ax.axvline(near,color='k',lw=.6)
                ax.set_title(which+' / '+ROUTES[rid]+f' / {case["variant"]["approach_speed_m_s"]:.1f} m/s');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
                ax.set_xlim(.8,15);ax.set_ylim(-1.2,1.2);ax.set_aspect('equal',adjustable='box')
    fig.suptitle('Measured XY to termination | green: Source   brown: Full RB   dotted: reference')
    fig.tight_layout();fig.savefig(out/'approach_xy.png',dpi=145);plt.close(fig)
    save_rows(out/'approach_cases.csv',records)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    report(args.run,args.output)
