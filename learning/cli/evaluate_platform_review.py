"""Score held-out physical departures using common calibrated body envelopes on two platforms."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch
from evaluate_task_reversal_predictions import kinematic
from predict_platform_departures import physical_raster
from sttw_control.playground_arx import states
from sttw_control.playground_history_model import load_history_model, GroundPredictionState
from sttw_control.playground_prediction import BodyEnvelope, euler, rotate
from sttw_control.playground_terrain_features import TerrainFeatures


def read(path):
    return json.loads(path.read_text())


def tensor(values):
    return torch.as_tensor(np.asarray(values),device='cuda',dtype=torch.float32)


def hand_state(z, hand, schema):
    t=round(hand/.005)+1
    q=z['qpos'];v=z['qvel']
    pose,velocity=states(q[[t-4,t]],v[[t-4,t]],schema)
    return pose[-1],velocity[-1],velocity[0]


def initial_features(p,u,last):
    return np.concatenate((p[...,:8],p[...,10:12],u,last),axis=-1)


def fit_hidden(cap,model,schema):
    # Wide runs save physical state. Fit the missing GRU initializer only on archived
    # controller states from primary calibration roots 8..23 (same approaches/skill).
    xs=[]; targets=[]
    for row in read(cap/'status.json')['results']:
        hand=row['handoff_s']
        if hand is None:continue
        with np.load(cap/row['case_dir']/'trajectory.npz') as z:
            xs.append(initial_features(*hand_state(z,hand,schema)))
            targets.append(z['controller_state'][round(hand/.005)+1])
    x=np.stack(xs);mu=x.mean(0);scale=x.std(0).clip(.005)
    x=np.column_stack((np.ones(len(x)),(x-mu)/scale))
    h=model.initial((tensor(targets)-model.controller_mean)/model.controller_scale).cpu().numpy()
    ridge=np.eye(x.shape[1]);ridge[0,0]=0
    weight=np.linalg.solve(x.T@x+ridge,x.T@h)
    return mu,scale,weight


def decisions(p,v,domain,request,ends,world,halfwidth,lookmin,envelope,obs,sections):
    lo,hi=envelope(p)
    center,half=obs
    clear=torch.maximum(lo[:,:,:,None]-center-half,center-half-hi[:,:,:,None]).amax(-1).amin((2,3))
    angles=euler(p[:,:,3:7].reshape(-1,4)).reshape(*p.shape[:2],3)
    world_v=rotate(p[:,:,3:7].reshape(-1,4),v[:,:,:3].reshape(-1,3)).reshape(*p.shape[:2],3)
    speed=world_v[:,:,0]*angles[:,:,2].cos()+world_v[:,:,1]*angles[:,:,2].sin()
    g=world['goal']
    ingoal=(p[:,:,0]>=g['x_min_m'])&(p[:,:,0]<=g['x_max_m'])&(p[:,:,1].abs()<=g['abs_y_max_m'])&(world_v[:,:,0]>=g['min_vx_m_s'])&(domain[:,:,0].abs()<=g['path_tolerance_m'])&(domain[:,:,1].abs()<=g['heading_tolerance_rad'])&((speed-request[:,None]).abs()<=g['speed_tolerance_m_s'])
    bad=(angles[:,:,0].abs()>.7)|(angles[:,:,1].abs()>1.309)|(domain[:,:,0].abs()>halfwidth)|(domain[:,:,1].abs()>=np.pi/2)|(domain[:,:,2]<lookmin)
    xy=p[:,:,:2].cpu().numpy()
    results=[]
    for path,dist,stop,goal,fail in zip(xy,clear.cpu().numpy(),ends,ingoal.cpu().numpy(),bad.cpu().numpy()):
        hold=np.flatnonzero(np.convolve(goal[:stop].astype(int),np.ones(26),'valid')==26)
        if len(hold):stop=int(hold[0])+26
        item=dict(clear=float(dist[:stop].min()),reachable=bool(len(hold) and not fail[:stop].any()))
        for label,x in zip(('near','mid','far'),sections):
            ids=np.flatnonzero((path[:stop-1,0]<x)&(path[1:stop,0]>=x))
            y=np.nan
            if len(ids):
                k=ids[0];a,b=path[k:k+2];y=float(a[1]+(x-a[0])/(b[0]-a[0])*(b[1]-a[1]))
            item[label]=y
        results.append(item)
    return pd.DataFrame(results)


@torch.inference_mode()
def forecast(raw):
    wide=raw.parent; plan=read(raw/'plan.json')
    src=wide.parent/'replication/primary'
    schema=read(src/'source_capture/model_state_schema.json')
    truth=pd.read_csv(raw/'physical_scores.csv')
    model=load_history_model(wide.parents[2]/'p3_multimap_models/residual_seed2')
    for layout,mapfile in plan['maps'].items():
        case=wide/layout/'roots_0008'
        physics=mujoco.MjModel.from_binary_path(str(case/'source/terrain.mjb'))
        world=read(Path(mapfile));height=world['terrain']['far_bank_height_m']
        raster=raw/(layout+'_terrain.npz')
        if not raster.exists():physical_raster(case/'source/terrain.mjb',raster)
        terrain=dict(np.load(raster));env=BodyEnvelope(physics,dimensions=3)
        walls=[i for i in range(physics.ngeom) if physics.geom(i).name.startswith('planning_obstacle_')]
        obs=tensor(physics.geom_pos[walls]),tensor(physics.geom_size[walls])
        bank=dict(np.load(wide/layout/'departure_bank.npz'));arc=tensor(bank['s'])
        ids={hashlib.sha256(x.tobytes()).digest():i for i,x in enumerate(bank['xy'])}
        for skill in ('source','rb_run1'):
            dest=raw/f'forecasts_{layout}_{skill}.csv'
            if dest.exists():continue
            rows=truth[(truth.layout==layout)&(truth.skill==skill)&(truth.section=='near')&truth.verified_jump].copy().reset_index(drop=True)
            mu,scale,weight=fit_hidden(src/(skill+'_capture'),model,schema)
            ps=[];us=[];vs=[];refs=[];req=[]
            for row in rows.itertuples():
                with np.load(row.trajectory) as z:
                    p,u,v=hand_state(z,row.handoff_s,schema)
                    ps.append(p);us.append(u);vs.append(v);req.append(z['trace'][round(row.handoff_s/.005)+1,18])
                    refs.append(ids[hashlib.sha256(z['published_reference_xy'].tobytes()).digest()])
            p,u,v=map(tensor,(ps,us,vs));request=tensor(req)
            feat=(initial_features(np.stack(ps),np.stack(us),np.stack(vs))-mu)/scale
            hidden=tensor(np.column_stack((np.ones(len(feat)),feat))@weight)
            route=(arc,*(tensor(bank[k][refs]) for k in ('xy','heading','curvature')))
            ends=np.floor((10-rows.handoff_s.to_numpy()-.005+1.e-6)/.02).astype(int)+1
            output=[]
            for method in ('kinematic','phase2_raw','phase2_relative'):
                aligned=height if method=='phase2_relative' else 0.
                model.terrain=TerrainFeatures(**dict(terrain,height=terrain['height']-aligned),device='cuda')
                pose=p.clone();pose[:,2]-=aligned
                if method=='kinematic':
                    paths,vel,domain=kinematic(model,pose,u[:,0].clone(),request,route,int(ends.max())-1)
                else:
                    state=GroundPredictionState(pose,u,v,hidden)
                    f=model.rollout_state(state,*route,int(ends.max())-1,request,torch.zeros(len(p),dtype=torch.long,device='cuda'))
                    paths,vel,domain=f['pose'],f['velocity'],f['domain']
                paths[:,:,2]+=aligned
                d=decisions(paths,vel,domain,request,ends,world,.6,model.nominal.follower['preview_body_x_min_m'],env,obs,plan['contracts']['x_sections_m'])
                d['method']=method
                output.append(pd.concat((rows.drop(columns=['section','trajectory']),d),axis=1))
            result=pd.concat(output)
            for i,col in enumerate(('x','y','z','qw','qx','qy','qz','steer')):
                result[col]=np.tile(np.asarray(ps)[:,i],3)
            result['speed']=np.tile(np.asarray(us)[:,0],3)
            result.to_csv(dest,index=False)
            print(layout,skill,len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path)
    forecast(parser.parse_args().raw)
