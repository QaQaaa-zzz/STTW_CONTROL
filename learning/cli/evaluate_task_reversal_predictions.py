"""Paired DEV predictions from measured handbacks with an identical occupancy test."""
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from run_playground_task_reversal import write_rows, crossings_by_section
from sttw_control.playground_arx import states
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_prediction import BodyEnvelope, euler, rotate
from sttw_control.playground_terrain_features import TerrainFeatures, raster_from_spec


def kinematic(model, initial, speed, request, route, steps):
    follower = model.nominal.follower
    wheelbase = follower['wheelbase_m_expected']
    x = initial.clone()
    yaw = euler(x[:,3:7])[:,2]
    zero = torch.zeros_like(yaw)
    x[:,3:7] = torch.stack(((yaw/2).cos(), zero, zero, (yaw/2).sin()), 1)
    # Locked joints are common geometry. This baseline observes no live steering,
    # roll, yaw rate, or controller history. Use geometry and the actual chord.
    x[:,7] = 0
    u = torch.zeros((len(x),9), device=x.device)
    u[:,0] = speed
    poses, speeds, domains = [], [], []
    for _ in range(steps+1):
        f = model.nominal.features(x, u, *route)
        poses.append(x.clone())
        speeds.append(u.clone())
        domains.append(f[:,9:12])
        chord2 = f[:,11:13].square().sum(1).clamp_min(1e-8)
        steer = torch.atan(2*wheelbase*f[:,12]/chord2).clamp(-follower['steer_reference_max_rad'], follower['steer_reference_max_rad'])
        dyaw = .02*u[:,0]*steer.tan()/wheelbase
        x[:,0] += .02*u[:,0]*(yaw+dyaw/2).cos()
        x[:,1] += .02*u[:,0]*(yaw+dyaw/2).sin()
        yaw = yaw+dyaw
        x[:,3:7] = torch.stack(((yaw/2).cos(), zero, zero, (yaw/2).sin()),1)
        x[:,7] = steer
        dv = .02*follower['nominal_speed_slew_m_s2']
        u[:,0] += (request-u[:,0]).clamp(-dv,dv)
    return tuple(torch.stack(values, 1) for values in (poses, speeds, domains))


@torch.inference_mode()
def evaluate():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('matrix','model','calibration','output'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--deadline', type=float, default=10.)
    parser.add_argument('--extended-truth', type=Path)
    parser.add_argument('--group')
    parser.add_argument('--fit-low-speed', action='store_true')
    parser.add_argument('--margin-name', default='shared_dev')
    a = parser.parse_args()
    rows = list(csv.DictReader(a.matrix.open()))
    if a.extended_truth:
        with a.extended_truth.open() as stream:
            truth = {(r['lane'],r['departure'],r['bin'],r['task']):r for r in csv.DictReader(stream) if r['group']==a.group}
        for r in rows:
            t = truth[r['lane'],r['departure'],r['bin'],r['task']]
            r.update(success=t['success_12s'],termination=t['termination_12s'],checkpoint=t['checkpoint_12s'])
    runs = {(r['lane'],r['departure']):r for r in rows}
    out = a.output
    out.mkdir(exist_ok=True,parents=True)
    samples = []
    for (lane, _), r in runs.items():
        lane = int(lane)
        prefix, trajectory = Path(r['prefix']), Path(r['trajectory'])
        case = trajectory.parents[1]
        cfg = json.loads((case/'resolved_config.json').read_text())
        schema = json.loads((case/'model_state_schema.json').read_text())
        handoff = float(r['handoff_s'])
        with np.load(prefix) as z:
            k = int(np.argmin(abs(z['trace'][:,0]-handoff))) + 1  # first already-locked tick
            p, u = states(z['qpos'][[k-4,k]],z['qvel'][[k-4,k]],schema)
            samples.append((p[-1],u[-1],u[0],z['controller_state'][k],z['trace'][k,18]))
        b = dict(np.load(cfg['ground_handoff']['reference_path']))
        samples[-1] += tuple(b[key][lane] for key in ('xy','heading','curvature'))
    tensor = lambda v: torch.as_tensor(np.asarray(v),device='cuda',dtype=torch.float32)
    p,u,previous,memory,request,xy,h,curvature = (tensor(v) for v in zip(*samples))
    route = tensor(b['s']), xy, h, curvature
    model = load_history_model(a.model)
    model.terrain = TerrainFeatures(**raster_from_spec(cfg['ditch_terrain']),device='cuda') if model.terrain is not None else None
    dt = .02
    steps = int(np.ceil(max(a.deadline-float(r['handoff_s']) for r in runs.values())/dt))
    state = model.initialize(p,u,previous,memory)
    dynamic = model.rollout_state(state,*route,steps,request,torch.zeros(len(p),dtype=torch.long,device='cuda'))
    methods = {'kinematic':kinematic(model,p,u[:,0].clone(),request,route,steps),
               'phase2':(dynamic['pose'],dynamic['velocity'],dynamic['domain'])}
    physics = mujoco.MjModel.from_binary_path(str(case/'terrain.mjb'))
    envelope = BodyEnvelope(physics,dimensions=3)
    obstacles = [i for i in range(physics.ngeom) if physics.geom(i).name.startswith('planning_obstacle_')]
    center, half = tensor(physics.geom_pos[obstacles]),tensor(physics.geom_size[obstacles])
    calibration = [r for r in json.loads(a.calibration.read_text()) if r['model']=='M1']
    margins = tensor(np.max([r['axis_margin_m'] for r in calibration],axis=0))
    boundaries = tensor(calibration[0]['edges'][1:-1])
    plan = json.loads((a.matrix.parent/'plan.json').read_text())['contracts']
    goal = cfg['ground_handoff']['goal']
    if a.fit_low_speed:
        # Fit exclusively on measured support residuals inside the predicted x range.
        # Root exclusion and pooling across fits are performed before scoring.
        edges = np.asarray(calibration[0]['edges'])
        fit = []
        predicted = dynamic['pose']
        lows, highs = [v.cpu().numpy() for v in envelope(predicted)]
        for i,((lane,dep),r) in enumerate(runs.items()):
            hand = float(r['handoff_s'])
            with np.load(r['trajectory']) as z:
                start = np.argmin(abs(z['trace'][:,0]-(hand+.005)))
                actual,_ = states(z['qpos'][start::4],z['qvel'][start::4],schema)
                vx = float(z['qvel'][start,0])
            if not 1.1 <= vx <= 1.5:
                continue
            n = np.count_nonzero(hand+.005+dt*np.arange(steps+1)<=a.deadline+1e-6)
            x = predicted[i,:n,0].cpu().numpy()
            order = np.argsort(x,kind='stable')
            actual = actual[(actual[:,0]>=x.min()) & (actual[:,0]<=x.max())]
            lo,hi = [v.cpu().numpy() for v in envelope(tensor(actual))]
            interpolated = []
            for support in (lows[i,:n],highs[i,:n]):
                flat = support[order].reshape(n,-1)
                interpolated.append(np.stack([np.interp(actual[:,0],x[order],col) for col in flat.T],axis=1).reshape(lo.shape))
            miss = np.maximum(0,np.maximum(interpolated[0]-lo,hi-interpolated[1]))
            regions = np.searchsorted(edges[1:-1],actual[:,0],side='right')
            for region in np.unique(regions):
                selected = miss[regions==region]
                fit.append(dict(root=r['root'],lane=lane,departure=dep,vx_m_s=vx,region=int(region),
                                samples=len(selected),peak_m=selected.max((0,1)).tolist()))
        (out/'support_fit.json').write_text(json.dumps(fit,indent=2))
    reports = []
    for method,(pose,velocity,domain) in methods.items():
        low, high = envelope(pose)
        angles = euler(pose[:,:,3:7].reshape(-1,4)).reshape(len(p),-1,3)
        world_v = rotate(pose[:,:,3:7].reshape(-1,4),velocity[:,:,:3].reshape(-1,3)).reshape(len(p),-1,3)
        speed = world_v[:,:,0]*angles[:,:,2].cos()+world_v[:,:,1]*angles[:,:,2].sin()
        for margin_name, margin in [('bare',torch.zeros_like(margins)),(a.margin_name,margins)]:
            eps = margin[torch.bucketize(pose[:,:,0].contiguous(),boundaries)]
            sep = torch.maximum((low-eps[:,:,None])[:,:,:,None]-(center+half),
                                (center-half)-(high+eps[:,:,None])[:,:,:,None]).amax(-1)
            clear = sep.amin((2,3))
            in_goal = ((pose[:,:,0]>=goal['x_min_m']) & (pose[:,:,0]<=goal['x_max_m']) &
                       (pose[:,:,1].abs()<=goal['abs_y_max_m']) & (world_v[:,:,0]>=goal['min_vx_m_s']) &
                       (domain[:,:,0].abs()<=goal['path_tolerance_m']) & (domain[:,:,1].abs()<=goal['heading_tolerance_rad']) &
                       ((speed-request[:,None]).abs()<=goal['speed_tolerance_m_s']))
            xy_cpu, clear_cpu, goal_cpu = pose[:,:,:2].cpu().numpy(), clear.cpu().numpy(), in_goal.cpu().numpy()
            bad = ((angles[:,:,0].abs()>.7) | (angles[:,:,1].abs()>1.309) |
                   (domain[:,:,0].abs()>cfg['ground_handoff']['road_half_width_m']) |
                   (domain[:,:,1].abs()>=np.pi/2) | (domain[:,:,2]<model.nominal.follower['preview_body_x_min_m']))
            bad_cpu = bad.cpu().numpy()
            for i,((lane,departure),r) in enumerate(runs.items()):
                hand = float(r['handoff_s'])
                t = hand+.005+dt*np.arange(steps+1)
                n = np.count_nonzero(t<=a.deadline+1e-6)
                hold = np.convolve(goal_cpu[i,:n].astype(int),np.ones(26,int),'valid')
                ids = np.flatnonzero(hold==26)  # 26 samples span 0.500 s
                stop = int(ids[0]+26) if len(ids) else n
                closest = float(clear_cpu[i,:stop].min())
                trace = np.column_stack((t[:stop],xy_cpu[i,:stop]))
                actual = {(s['bin'],s['task']):s for s in rows if (s['lane'],s['departure'])==(lane,departure)}
                for label, _, _, y in crossings_by_section(trace,hand,plan['x_sections_m']):
                    for side in ('left','right'):
                        lo,hi = plan[side+'_y_interval_m']
                        passed = bool(y is not None and lo<=y<=hi and len(ids) and closest>0 and not bad_cpu[i,:stop].any())
                        truth = actual[label,side]
                        success = truth['success']=='True'
                        reports.append(dict(method=method,margin=margin_name,lane=lane,departure=departure,
                               bin=label,task=side,predicted_pass=passed,actual_pass=success,
                               false_pass=passed and not success,false_reject=success and not passed,
                               predicted_hold=bool(len(ids)),minimum_axis_separation_m=closest,
                               predicted_collision_free=closest>0,actual_collision=truth['termination']=='forbidden_obstacle_contact',
                               actual_termination=truth['termination'],actual_checkpoint=truth['checkpoint']))
    write_rows(out/'offline_false_accepts.csv',reports)


if __name__=='__main__':
    evaluate()
