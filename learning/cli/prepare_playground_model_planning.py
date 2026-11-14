"""Same finite full-motion candidates, model-specific DEV-calibrated departure envelopes."""
import argparse,copy,json,time,shutil,subprocess
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_arx import states
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_prediction import BodyEnvelope
from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter()
source=a.collection/'map00/evaluation';cfg=json.loads((source/'resolved_config.json').read_text());schema=json.loads((source/'model_state_schema.json').read_text());bank=dict(np.load(cfg['ground_handoff']['reference_path']));tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
envelope=BodyEnvelope(mujoco.MjModel.from_binary_path(str(source/'terrain.mjb')),dimensions=3)
items=[]
for row in json.loads((source/'status.json').read_text())['results']:
    with np.load(source/row['case_dir']/'trajectory.npz') as d:
        pose,xi=states(d['qpos'][::4],d['qvel'][::4],schema);tr=d['trace'][::4];anchor=int(np.flatnonzero(tr[:,-1]==2)[0]);steps=len(pose)-1-anchor
        with torch.no_grad():lo,hi=[x.cpu().numpy() for x in envelope(tensor(pose))]
        items.append(dict(row=row,pose=pose,lo=lo,hi=hi,xi=xi[anchor],previous=xi[anchor-1],controller=d['controller_state'][::4][anchor],anchor=anchor,steps=steps,speed=float(tr[anchor,18])))
steps=max(x['steps'] for x in items);s=tensor(bank['s']);xy=tensor(np.stack([bank['xy'][r['row']['route']] for r in items]));h=tensor(np.stack([bank['heading'][r['row']['route']] for r in items]));k=tensor(np.stack([bank['curvature'][r['row']['route']] for r in items]))
models=[('M0','residual_seed0','model_0000.pt'),('M1','residual_seed2','best.pt'),('M2','direct_seed0','best.pt')]
plans=[];entries=[];all_calibration=[];saved=[]
# Same candidate identity, source cost and eligibility for every model.
keys=[(route,speed) for route in range(9) for speed in [1.8,1.9]]
groups=[[i for i,r in enumerate(items) if r['row']['route']==route and abs(r['speed']-speed)<1e-4] for route,speed in keys]
cost=np.array([max(items[i]['row']['duration_s'] for i in ids) for ids in groups]);eligible=np.array([all(items[i]['row']['goal'] and items[i]['row']['nonwheel_contact_steps']==0 for i in ids) for ids in groups])
for map_id in [6,7]:
    target=a.collection/f'map{map_id:02d}/evaluation';base=json.loads((target/'resolved_config.json').read_text());world=base['world_map_spec'];raster=raster_from_spec(base['ditch_terrain']);edges=np.array([-5.,2.,8.3,12.6,16.5,25.])
    actual=[]
    for row in json.loads((target/'status.json').read_text())['results']:
        with np.load(target/row['case_dir']/'trajectory.npz') as d:
            pose,_=states(d['qpos'][::4],d['qvel'][::4],schema)
            with torch.no_grad():lo,hi=[x.cpu().numpy() for x in envelope(tensor(pose))]
            actual.append(dict(row=row,pose=pose,lo=lo,hi=hi,speed=float(d['trace'][0,18])))
    for name,folder,checkpoint in models:
        model=load_history_model(a.models/folder,checkpoint)
        if model.terrain is not None:model.terrain=TerrainFeatures(**raster,device='cuda')
        with torch.no_grad():
            pred,vel=model.rollout(tensor(np.stack([r['pose'][r['anchor']] for r in items])),tensor(np.stack([r['xi'] for r in items])),tensor(np.stack([r['previous'] for r in items])),tensor(np.stack([r['controller'] for r in items])),s,xy,h,k,steps,speed_request=tensor([r['speed'] for r in items]))
            low,high=[x.cpu().numpy() for x in envelope(pred)];prediction=pred.cpu().numpy()
        motions=[]
        for i,r in enumerate(items):
            end=r['steps']+1;anchor=r['anchor'];motions.append(dict(x=np.r_[r['pose'][:anchor,0],prediction[i,:end,0]],lo=np.concatenate([r['lo'][:anchor],low[i,:end]]),hi=np.concatenate([r['hi'][:anchor],high[i,:end]])))
        # Spatial DEV calibration is explicit, uses no obstacle outcome, and is not TEST evidence.
        margin=np.zeros((len(edges)-1,3));counts=np.zeros(len(margin),int)
        for r in actual:
            key=next(j for j,(route,speed) in enumerate(keys) if route==r['row']['route'] and abs(speed-r['speed'])<1e-4)
            lows=[];highs=[]
            for i in groups[key]:
                m=motions[i];order=np.argsort(m['x'],kind='stable')
                for field,dest in [('lo',lows),('hi',highs)]:
                    flat=m[field][order].reshape(len(order),-1);dest.append(np.stack([np.interp(r['pose'][:,0],m['x'][order],flat[:,j]) for j in range(flat.shape[1])],1).reshape(r['lo'].shape))
            miss=np.maximum(np.maximum(np.minimum.reduce(lows)-r['lo'],r['hi']-np.maximum.reduce(highs)),0.)
            regions=np.clip(np.searchsorted(edges,r['pose'][:,0],side='right')-1,0,len(margin)-1)
            for region in range(len(margin)):
                selected=regions==region
                if selected.any():margin[region]=np.maximum(margin[region],miss[selected].max((0,1)));counts[region]+=int(selected.sum())
        margin[counts==0]=margin.max(0);margin+=.02
        all_calibration.append(dict(map_id=map_id,model=name,edges=edges.tolist(),axis_margin_m=margin.tolist(),counts=counts.tolist(),rule='Same DEV source, per-region per-axis max spatial support miss plus0.02m; no obstacle outcomes used'))
        candidate_bounds=[]
        for ids in groups:
            lows=[];highs=[]
            for i in ids:
                m=motions[i];region=np.clip(np.searchsorted(edges,m['x'],side='right')-1,0,len(margin)-1);lows.append((m['lo']-margin[region,None,:]).reshape(-1,3));highs.append((m['hi']+margin[region,None,:]).reshape(-1,3))
            candidate_bounds.append((tensor(np.concatenate(lows)),tensor(np.concatenate(highs))))
        for layout,y in enumerate([-.08,0.,.08]):
            rect=[13.3,13.5,y-.06,y+.06];box=tensor([rect[0],rect[2],0.,rect[1],rect[3],.5]);bounds=world['bounds_xy_m'];clear=[]
            for lo,hi in candidate_bounds:
                road=torch.stack((lo[:,0]-bounds[0],bounds[1]-hi[:,0],lo[:,1]-bounds[2],bounds[3]-hi[:,1]),1).amin()
                obstacle=torch.maximum(box[:3]-hi,lo-box[3:]).amax(1).amin();clear.append(float(torch.minimum(road,obstacle)))
            feasible=eligible&(np.array(clear)>=0);chosen=int(np.argmin(np.where(feasible,cost,np.inf))) if feasible.any() else 0
            plans.append(dict(map_id=map_id,layout=layout,model=name,feasible=bool(feasible.any()),candidate=chosen,route=keys[chosen][0],speed=keys[chosen][1],predicted_cost_s=float(cost[chosen]),all_clearance_m=clear,eligible=eligible.tolist(),checkpoint=str(a.models/folder/checkpoint)))
        saved.append(dict(map_id=map_id,model=name,checkpoint=str(a.models/folder/checkpoint)))
    for layout,y in enumerate([-.08,0.,.08]):
        folder=a.output/f'map{map_id:02d}_layout{layout:02d}';folder.mkdir();w=copy.deepcopy(world);w['map_id']=folder.name;w['terrain']['forbidden_rectangles']=[[13.3,13.5,y-.06,y+.06]];w['terrain']['obstacle_height_m']=.5;(folder/'world.json').write_text(json.dumps(w,indent=2)+'\n')
        c=copy.deepcopy(base)
        for field in ['world_map_spec','ditch_terrain','snapshot_times_s']:c.pop(field,None)
        c.update(world_map=str(folder/'world.json'),task_evaluation=True,role='P3 model-specific calibrated finite-library DEV; static before initialization, no online or independent TEST claim')
        selected=[r for r in plans if r['map_id']==map_id and r['layout']==layout];lanes=[]
        for plan in selected:
            vi=[i for i,v in enumerate(c['ground_handoff']['handoff_variants']) if abs(v['requested_speed_m_s']-plan['speed'])<1e-4];plan['evaluation']=str(folder/'evaluation');plan['case_dirs']=[f"rb_run3_route{plan['route']}_h75_v{i}" for i in vi]
            lanes.extend((plan['route'],'h75',i) for i in vi)
        c['evaluation_lanes']=[list(x) for x in sorted(set(lanes))];(folder/'config.json').write_text(json.dumps(c,indent=2)+'\n');entries.append(dict(layout=folder.name,config=str(folder/'config.json'),evaluation=str(folder/'evaluation'),episodes=len(c['evaluation_lanes'])))
manifest=dict(role='Static finite-candidate DEV pilot: M0/M1/M2 compare model plus same-rule calibrated envelope; not independent TEST or neural online planning',entries=entries,routes=list(range(9)),seconds=10,ground_only=False,method_conditions=36,physical_episodes=sum(e['episodes'] for e in entries),models=saved,candidates=18,source=str(source),calibration_maps=[6,7],shared='Same18 route/speed candidates, activation8.55, two TRAIN-map00 histories each, full measured approach/RB prefix, source duration cost/eligibility, physical plant/goal/boxes; only predicted departure and same-rule error expansion differ',limitations='Candidate histories are TRAIN templates, not future actual DEV handoff states. Uses observed source departure horizons3.22-4s, not a timing predictor or predicted goal-hold certificate. Retained incumbent after no-plan counts planner failure regardless physical goal. DEV used for calibration, not independent map generalization.',selection='M1seed2 update575 and M2seed0 update600 selected by prior1s DEV error, not new full-departure tails or obstacle outcomes',expected_wallclock_s=[700,1200],hard_limit_s=1500,stop='One queue, first runtime error or25min; no retries or training',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),preparation_s=time.perf_counter()-started)
for name,data in [('manifest',manifest),('plans',plans),('calibration',all_calibration)]: (a.output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
(a.output/'source').mkdir();shutil.copyfile(__file__,a.output/'source'/Path(__file__).name)
print(json.dumps(dict(physical_episodes=manifest['physical_episodes'],feasible={name:sum(r['feasible'] for r in plans if r['model']==name) for name,_,_ in models},preparation_s=manifest['preparation_s']),indent=2))
