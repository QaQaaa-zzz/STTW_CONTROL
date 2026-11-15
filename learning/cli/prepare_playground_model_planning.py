"""Same finite full-motion candidates, model-specific DEV-calibrated departure envelopes."""
import argparse,copy,json,time,shutil,subprocess
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_constraint_log import clearance_witness,dynamic_witnesses
from sttw_control.playground_arx import states
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_prediction import BodyEnvelope,euler,rotate
from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--test-catalog',type=Path);p.add_argument('--frozen-calibration',type=Path)
p.add_argument('--predicted-terminal',action='store_true');p.add_argument('--collection',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if bool(a.test_catalog)!=bool(a.frozen_calibration) or (a.test_catalog and not a.predicted_terminal):p.error('TEST requires catalog, frozen calibration, and predicted-terminal together')
a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter()
source=a.collection/'map00/evaluation';cfg=json.loads((source/'resolved_config.json').read_text());schema=json.loads((source/'model_state_schema.json').read_text());bank=dict(np.load(cfg['ground_handoff']['reference_path']));tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
body_model=mujoco.MjModel.from_binary_path(str(source/'terrain.mjb'))
envelope=BodyEnvelope(body_model,dimensions=3)
geom_names=[body_model.geom(i).name for i in range(body_model.ngeom) if body_model.geom_bodyid[i]>0 and (body_model.geom_contype[i] or body_model.geom_conaffinity[i])]
items=[]
for row in json.loads((source/'status.json').read_text())['results']:
    with np.load(source/row['case_dir']/'trajectory.npz') as d:
        pose,xi=states(d['qpos'][::4],d['qvel'][::4],schema);tr=d['trace'][::4];anchor=int(np.flatnonzero(tr[:,-1]==2)[0]);steps=len(pose)-1-anchor
        with torch.no_grad():lo,hi=[x.cpu().numpy() for x in envelope(tensor(pose))]
        items.append(dict(row=row,phase=tr[:,-1],pose=pose,lo=lo,hi=hi,xi=xi[anchor],previous=xi[anchor-1],controller=d['controller_state'][::4][anchor],anchor=anchor,steps=steps,speed=float(tr[anchor,18])))
steps=max(x['steps'] for x in items);s=tensor(bank['s']);xy=tensor(np.stack([bank['xy'][r['row']['route']] for r in items]));h=tensor(np.stack([bank['heading'][r['row']['route']] for r in items]));k=tensor(np.stack([bank['curvature'][r['row']['route']] for r in items]))
models=[('M0','residual_seed0','model_0000.pt'),('M1','residual_seed2','best.pt'),('M2','direct_seed0','best.pt')]
plans=[];entries=[];all_calibration=[];saved=[];dynamics=[];constraint_records=[]
# Same candidate identity, source cost and eligibility for every model.
keys=[(route,speed) for route in range(9) for speed in [1.8,1.9]]
groups=[[i for i,r in enumerate(items) if r['row']['route']==route and abs(r['speed']-speed)<1e-4] for route,speed in keys]
cost=np.array([max(items[i]['row']['duration_s'] for i in ids) for ids in groups]);eligible=np.array([all(items[i]['row']['goal'] and items[i]['row']['nonwheel_contact_steps']==0 for i in ids) for ids in groups])
catalog=json.loads(a.test_catalog.read_text()) if a.test_catalog else None
frozen=json.loads(a.frozen_calibration.read_text()) if a.frozen_calibration else None
for map_id in ([r['map_id'] for r in catalog['entries']] if catalog else [6,7]):
    target=a.collection/f'map{(6 if catalog else map_id):02d}/evaluation';base=json.loads((target/'resolved_config.json').read_text());world=base['world_map_spec'];raster=raster_from_spec(base['ditch_terrain']);edges=np.array([-5.,2.,8.3,12.6,16.5,25.])
    if catalog:
        entry=next(r for r in catalog['entries'] if r['map_id']==map_id)
        base=json.loads(Path(entry['config']).read_text());world=json.loads(Path(base['world_map']).read_text());raster=raster_from_spec(world['terrain'])
    actual=[]
    for row in ([] if catalog else json.loads((target/'status.json').read_text())['results']):
        with np.load(target/row['case_dir']/'trajectory.npz') as d:
            pose,_=states(d['qpos'][::4],d['qvel'][::4],schema)
            with torch.no_grad():lo,hi=[x.cpu().numpy() for x in envelope(tensor(pose))]
            actual.append(dict(row=row,pose=pose,lo=lo,hi=hi,speed=float(d['trace'][0,18])))
    for name,folder,checkpoint in models:
        model=load_history_model(a.models/folder,checkpoint)
        if model.terrain is not None:model.terrain=TerrainFeatures(**raster,device='cuda')
        with torch.no_grad():
            pred,vel,domain=model.rollout(tensor(np.stack([r['pose'][r['anchor']] for r in items])),tensor(np.stack([r['xi'] for r in items])),tensor(np.stack([r['previous'] for r in items])),tensor(np.stack([r['controller'] for r in items])),s,xy,h,k,steps,speed_request=tensor([r['speed'] for r in items]),return_domain=True)
            low,high=[x.cpu().numpy() for x in envelope(pred)];prediction=pred.cpu().numpy();angles=euler(pred[:,:,3:7].reshape(-1,4)).reshape(len(items),steps+1,3).cpu().numpy();velocities=vel.cpu().numpy();domains=domain.cpu().numpy();world_vx=rotate(pred[:,:,3:7].reshape(-1,4),vel[:,:,:3].reshape(-1,3))[:,0].reshape(len(items),steps+1).cpu().numpy()
        np.savez_compressed(a.output/f'predictions_map{map_id:02d}_{name}.npz',pose=prediction,velocity=velocities,domain=domains,world_vx=world_vx,source_steps=np.array([r['steps'] for r in items]))
        motions=[];model_checks=[];model_times=[]
        for i,r in enumerate(items):
            end=r['steps']+1;anchor=r['anchor'];tail=slice(max(0,end-26),end);goal=world['goal']
            checks=dict(finite=bool(np.isfinite(prediction[i,:end]).all()),roll=bool(np.abs(angles[i,:end,0]).max()<=.7),pitch=bool(np.abs(angles[i,:end,1]).max()<=1.309),tracking=bool(np.abs(domains[i,:end,0]).max()<=base['ground_handoff']['road_half_width_m']),heading_domain=bool(np.abs(domains[i,:end,1]).max()<np.pi/2),preview=bool(domains[i,:end,2].min()>.05),forward=bool(velocities[i,:end,0].min()>=.5))
            hold=(prediction[i,tail,0]>=goal['x_min_m'])&(prediction[i,tail,0]<=goal['x_max_m'])&(np.abs(prediction[i,tail,1])<=goal['abs_y_max_m'])&(world_vx[i,tail]>=goal['min_vx_m_s'])&(np.abs(domains[i,tail,0])<=goal['path_tolerance_m'])&(np.abs(domains[i,tail,1])<=goal['heading_tolerance_rad'])&(np.abs(velocities[i,tail,0]-r['speed'])<=goal['speed_tolerance_m_s'])
            goal_all=(prediction[i,:,0]>=goal['x_min_m'])&(prediction[i,:,0]<=goal['x_max_m'])&(np.abs(prediction[i,:,1])<=goal['abs_y_max_m'])&(world_vx[i]>=goal['min_vx_m_s'])&(np.abs(domains[i,:,0])<=goal['path_tolerance_m'])&(np.abs(domains[i,:,1])<=goal['heading_tolerance_rad'])&(np.abs(velocities[i,:,0]-r['speed'])<=goal['speed_tolerance_m_s'])
            first_hold=np.flatnonzero(np.convolve(goal_all.astype(int),np.ones(26,dtype=int),'valid')==26)
            goal_time=float((first_hold[0]+25)*.02) if len(first_hold) else None
            dynamics.append(dict(map_id=map_id,model=name,route=r['row']['route'],speed=r['speed'],source_case=r['row']['case_dir'],checks=checks,dynamic_valid=all(checks.values()),predicted_last_halfsecond_goal=bool(hold.all()),predicted_goal_time_s=goal_time,available_prediction_horizon_s=steps*.02,unobserved_after_source_goal_s=max(0.,goal_time-r['steps']*.02) if goal_time is not None else None,source_horizon_s=r['steps']*.02,scope='Diagnostic only; current static pilot selection did not impose predicted dynamics or goal hold. Tail uses source observed endpoint, not a model-predicted arrival time.'))
            if a.predicted_terminal:
                end=round(goal_time/.02)+1 if goal_time is not None else steps+1
                checks=dict(finite=bool(np.isfinite(prediction[i,:end]).all()),roll=bool(np.abs(angles[i,:end,0]).max()<=.7),pitch=bool(np.abs(angles[i,:end,1]).max()<=1.309),tracking=bool(np.abs(domains[i,:end,0]).max()<=base['ground_handoff']['road_half_width_m']),heading_domain=bool(np.abs(domains[i,:end,1]).max()<np.pi/2),preview=bool(domains[i,:end,2].min()>.05),forward=bool(velocities[i,:end,0].min()>=.5))
            model_checks.append(all(checks.values()) and goal_time is not None and anchor*.02+goal_time<=10.)
            model_times.append(anchor*.02+goal_time if goal_time is not None else np.inf)
            dynamics[-1]['diagnostics']=dynamic_witnesses(prediction[i],velocities[i],angles[i],domains[i],world_vx[i],r['speed'],goal,base['ground_handoff']['road_half_width_m'],anchor)
            dynamics[-1].update(terminal_gate_checks=checks,terminal_gate_valid=model_checks[-1],selection_enforced=a.predicted_terminal)
            motions.append(dict(x=np.r_[r['pose'][:anchor,0],prediction[i,:end,0]],lo=np.concatenate([r['lo'][:anchor],low[i,:end]]),hi=np.concatenate([r['hi'][:anchor],high[i,:end]])))
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
        if frozen:margin=np.array(frozen[name]['axis_margin_m']);edges=np.array(frozen[name]['edges'])
        all_calibration.append(dict(map_id=map_id,model=name,edges=edges.tolist(),axis_margin_m=margin.tolist(),counts=counts.tolist(),rule='Same DEV source, per-region per-axis max spatial support miss plus0.02m; no obstacle outcomes used'))
        candidate_bounds=[]
        for ids in groups:
            lows=[];highs=[]
            for i in ids:
                m=motions[i];region=np.clip(np.searchsorted(edges,m['x'],side='right')-1,0,len(margin)-1);lows.append((m['lo']-margin[region,None,:]).reshape(-1,3));highs.append((m['hi']+margin[region,None,:]).reshape(-1,3))
            candidate_bounds.append((tensor(np.concatenate(lows)),tensor(np.concatenate(highs))))
        admitted=eligible & np.array([all(model_checks[i] for i in ids) for ids in groups]) if a.predicted_terminal else eligible
        costs=np.array([max(model_times[i] for i in ids) for ids in groups]) if a.predicted_terminal else cost
        for layout,y in ([(0,0.)] if catalog else enumerate([-.08,0.,.08])):
            rectangles=world['terrain']['forbidden_rectangles'] if catalog else [[13.3,13.5,y-.06,y+.06]]
            boxes=tensor([[r[0],r[2],0.,r[1],r[3],.5] for r in rectangles]);bounds=world['bounds_xy_m'];clear=[]
            for lo,hi in candidate_bounds:
                road=torch.stack((lo[:,0]-bounds[0],bounds[1]-hi[:,0],lo[:,1]-bounds[2],bounds[3]-hi[:,1]),1).amin()
                obstacle=torch.maximum(boxes[:,None,:3]-hi[None],lo[None]-boxes[:,None,3:]).amax(2).amin();clear.append(float(torch.minimum(road,obstacle)))
            for candidate,ids in enumerate(groups):
                witnesses=[]
                for i in ids:
                    motion=motions[i];region=np.clip(np.searchsorted(edges,motion['x'],side='right')-1,0,len(margin)-1)
                    witness=clearance_witness(motion['lo'],motion['hi'],margin[region],bounds,
                        np.array([[r[0],r[2],0.,r[1],r[3],.5] for r in rectangles]),geom_names)
                    witness.update(time_s=witness['step']*.02,source_case=items[i]['row']['case_dir'],
                        phase=2 if witness['step']>=items[i]['anchor'] else int(items[i]['phase'][witness['step']]),
                        margin_source=str(a.output/'calibration.json'),margin_rule=all_calibration[-1]['rule'])
                    witnesses.append(witness)
                constraint_records.append(dict(map_id=map_id,layout=layout,model=name,candidate=candidate,
                    admitted=bool(admitted[candidate] and clear[candidate]>=0),domain_and_goal_eligible=bool(admitted[candidate]),
                    witnesses=witnesses,limiting=min(witnesses,key=lambda r:r['expanded_clearance_m'])))
            feasible=admitted&(np.array(clear)>=0);chosen=int(np.argmin(np.where(feasible,costs,np.inf))) if feasible.any() else 0
            plans.append(dict(map_id=map_id,layout=layout,model=name,feasible=bool(feasible.any()),candidate=chosen,route=keys[chosen][0],speed=keys[chosen][1],predicted_cost_s=float(costs[chosen]) if np.isfinite(costs[chosen]) else None,all_clearance_m=clear,eligible=admitted.tolist(),checkpoint=str(a.models/folder/checkpoint)))
        saved.append(dict(map_id=map_id,model=name,checkpoint=str(a.models/folder/checkpoint)))
    for layout,y in ([(0,0.)] if catalog else enumerate([-.08,0.,.08])):
        folder=a.output/f'map{map_id:02d}_layout{layout:02d}';folder.mkdir();w=copy.deepcopy(world);w['map_id']=folder.name;w['terrain']['forbidden_rectangles']=world['terrain']['forbidden_rectangles'] if catalog else [[13.3,13.5,y-.06,y+.06]];w['terrain']['obstacle_height_m']=.5;(folder/'world.json').write_text(json.dumps(w,indent=2)+'\n')
        c=copy.deepcopy(base)
        for field in ['world_map_spec','ditch_terrain','snapshot_times_s']:c.pop(field,None)
        c.update(world_map=str(folder/'world.json'),task_evaluation=True,role='P3 model-specific calibrated finite-library DEV; static before initialization, no online or independent TEST claim')
        if catalog:c['role']='Frozen new-map/new-root TEST; static selection, no TEST calibration'
        selected=[r for r in plans if r['map_id']==map_id and r['layout']==layout];lanes=[]
        for plan in selected:
            vi=[i for i,v in enumerate(c['ground_handoff']['handoff_variants']) if abs(v['requested_speed_m_s']-plan['speed'])<1e-4];plan['evaluation']=str(folder/'evaluation');plan['case_dirs']=[f"rb_run3_route{plan['route']}_h75_v{i}" for i in vi]
            lanes.extend((plan['route'],'h75',i) for i in vi)
        c['evaluation_lanes']=[list(x) for x in sorted(set(lanes))];(folder/'config.json').write_text(json.dumps(c,indent=2)+'\n');entries.append(dict(layout=folder.name,config=str(folder/'config.json'),evaluation=str(folder/'evaluation'),episodes=len(c['evaluation_lanes'])))
manifest=dict(predicted_terminal=a.predicted_terminal,terminal_rule='First26 consecutive20ms samples meeting original goal; predicted physical/controller domain through that end; prefix elapsed plus goal time<=10s' if a.predicted_terminal else 'Original source endpoint, no predicted terminal gate',role='Static finite-candidate DEV pilot: M0/M1/M2 compare model plus same-rule calibrated envelope; not independent TEST or neural online planning',entries=entries,routes=list(range(9)),seconds=10,ground_only=False,method_conditions=36,physical_episodes=sum(e['episodes'] for e in entries),models=saved,candidates=18,source=str(source),calibration_maps=[6,7],shared='Same18 route/speed candidates, activation8.55, two TRAIN-map00 histories each, full measured approach/RB prefix, source duration cost/eligibility, physical plant/goal/boxes; only predicted departure and same-rule error expansion differ',limitations='Candidate histories are TRAIN templates, not future actual DEV handoff states. Uses observed source departure horizons3.22-4s, not a timing predictor or predicted goal-hold certificate. Retained incumbent after no-plan counts planner failure regardless physical goal. DEV used for calibration, not independent map generalization.',selection='M1seed2 update575 and M2seed0 update600 selected by prior1s DEV error, not new full-departure tails or obstacle outcomes',expected_wallclock_s=[700,1200],hard_limit_s=1500,stop='One queue, first runtime error or25min; no retries or training',terminal_limitation='Common maximum source departure horizon; after an individual source observed goal predictions are unobserved and not additional accuracy evidence. Existing DEV calibration is exploratory, not independent TEST.',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),preparation_s=time.perf_counter()-started)
if catalog:
    manifest.update(role='Frozen new-map/new-root TEST of static finite-candidate model-plus-calibration systems; not full P4 or online neural planning',method_conditions=len(plans)*len(catalog['roots']),calibration_maps=[6,7],test_catalog=str(a.test_catalog),frozen_calibration=str(a.frozen_calibration),expected_wallclock_s=[2400,4200],hard_limit_s=7200,stop='One queue, first runtime error or120min; no retries or training',limitations='Measured TRAIN approach/RB templates, learned departure only; frozen max-over-DEV regional margins. No new-map outcomes used in selection. No claim of online neural replanning or complete E1/E2/P4.')
    for row in all_calibration:row['rule']='Frozen max over DEV maps06/07 regional axis margins; no TEST trajectory access'
for name,data in [('manifest',manifest),('plans',plans),('calibration',all_calibration),('dynamics',dynamics),('candidate_constraints',constraint_records)]: (a.output/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
(a.output/'source').mkdir()
for path in [Path(__file__),Path('learning/src/sttw_control/playground_history_model.py'),Path('learning/src/sttw_control/playground_prediction.py'),Path('learning/src/sttw_control/playground_terrain_features.py')]:shutil.copyfile(path,a.output/'source'/path.name)
print(json.dumps(dict(physical_episodes=manifest['physical_episodes'],feasible={name:sum(r['feasible'] for r in plans if r['model']==name) for name,_,_ in models},preparation_s=manifest['preparation_s']),indent=2))
