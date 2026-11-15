"""Freeze a finite H0/H1 DEV pilot from measured complete motion primitives."""
import argparse,copy,json,time,subprocess,shutil
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import BodyEnvelope
from sttw_control.playground_joint_planner import MotionLibraryPlanner

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--test-catalog',type=Path)
p.add_argument('--sweep',type=Path,required=True)
p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
started=time.perf_counter();manifest=json.loads((a.sweep/'manifest.json').read_text())
first=Path(manifest['entries'][0]['evaluation']);base=json.loads((first/'resolved_config.json').read_text());world=copy.deepcopy(base['world_map_spec'])
world['forward_obstacle']['activation_offset_m']=-1.7
model=mujoco.MjModel.from_binary_path(str(first/'terrain.mjb'));schema=json.loads((first/'model_state_schema.json').read_text())
envelope=BodyEnvelope(model,device='cpu',dimensions=3)
motions=[];candidate_rows=[]
for entry in manifest['entries']:
    evaluation=Path(entry['evaluation']);config=json.loads((evaluation/'resolved_config.json').read_text());status=json.loads((evaluation/'status.json').read_text())
    for route in range(9):
        sources=[];lower=[];upper=[];prefix=[];centers=[];cost=[];prefix_cost=[];eligible=True
        for row in status['results']:
            if row['route']!=route:continue
            with np.load(evaluation/row['case_dir']/'trajectory.npz') as data:
                pose,velocity=states(data['qpos'],data['qvel'],schema)
                with torch.no_grad():lo,hi=[x.numpy() for x in envelope(torch.tensor(pose,dtype=torch.float32))]
                pre=data['trace'][:,0]<=row['handoff_s']+1e-6
                body_contacts=int(np.nansum(data['contact_substep'][:,:,3]))
                min_forward=float(velocity[:,0].min());min_world_forward=float(data['qvel'][:,0].min())
                eligible &= bool(row['goal'] and body_contacts==0 and min_forward>=.5 and min_world_forward>=.5)
                sources.append(dict(path=str(evaluation/row['case_dir']/'trajectory.npz'),root=row['variant']['root_variant'],min_body_forward_m_s=min_forward,min_world_forward_m_s=min_world_forward,body_contact_substeps=body_contacts,x=pose[:,0],lower=lo,upper=hi))
                lower.append(lo.reshape(-1,3));upper.append(hi.reshape(-1,3));prefix.append(np.repeat(pre,lo.shape[1]));centers.append(np.repeat(pose[:,0],lo.shape[1]))
                cost.append(row['duration_s']);prefix_cost.append(row['handoff_s'])
        motions.append(sources)
        candidate_rows.append(dict(index=len(candidate_rows),activation_id=entry['layout'],activation_x_m=config['ground_handoff']['approach']['jump_start_x_m'],upstream=route//3,downstream=route%3,route=route,cost_s=max(cost),prefix_cost_s=max(prefix_cost),eligible=eligible,lower=np.concatenate(lower),upper=np.concatenate(upper),prefix=np.concatenate(prefix),center_x=np.concatenate(centers)))
# Fixed DEV calibration of spatial envelopes, not a time-warped motion predictor.
# The measured8.6m starts are compared to the declared neighboring8.55m templates.
cal_status=json.loads((a.calibration/'status.json').read_text());calibration=[]
region_edges=np.array([world['bounds_xy_m'][0],world['terrain']['rough_x_intervals_m'][0][0],world['terrain']['rough_x_intervals_m'][0][1],world['terrain']['rough_x_intervals_m'][1][0],world['goal']['x_min_m'],world['bounds_xy_m'][1]])
region_max=np.zeros((len(region_edges)-1,3));region_counts=np.zeros(len(region_max),int)
for row in cal_status['results']:
    if row['variant']['data_split']=='TRAIN':continue
    with np.load(a.calibration/row['case_dir']/'trajectory.npz') as data:
        pose,_=states(data['qpos'],data['qvel'],schema)
        with torch.no_grad():actual_low,actual_high=[x.numpy() for x in envelope(torch.tensor(pose,dtype=torch.float32))]
        lows=[];highs=[]
        for source in motions[9+row['route']]:
            for field,destination in [('lower',lows),('upper',highs)]:
                flat=source[field].reshape(len(source['x']),-1)
                destination.append(np.stack([np.interp(pose[:,0],source['x'],flat[:,i]) for i in range(flat.shape[1])],1).reshape(actual_low.shape))
        low=np.minimum.reduce(lows);high=np.maximum.reduce(highs)
        miss=np.maximum(np.maximum(low-actual_low,actual_high-high),0.)
        calibration.append(dict(case=row['case_dir'],route=row['route'],root=row['variant']['root_variant'],max_spatial_support_m=float(miss.max()),max_axis_support_m=miss.max((0,1)).tolist(),physical_outcome=row['termination']))
        regions=np.clip(np.searchsorted(region_edges,pose[:,0],side='right')-1,0,len(region_max)-1)
        for region in range(len(region_max)):
            selected=regions==region
            if selected.any():region_max[region]=np.maximum(region_max[region],miss[selected].max((0,1)));region_counts[region]+=int(selected.sum())
isotropic_margin=max(r['max_spatial_support_m'] for r in calibration)+.02
region_max[region_counts==0]=np.max(region_max,axis=0)
region_margin=region_max+.02; margin=0.
size=max(len(r['lower']) for r in candidate_rows);count=len(candidate_rows)
low=np.zeros((count,size,3),np.float32);high=np.zeros_like(low);valid=np.zeros((count,size),bool);prefix=np.zeros_like(valid)
for i,row in enumerate(candidate_rows):
    n=len(row['lower']);regions=np.clip(np.searchsorted(region_edges,row.pop('center_x'),side='right')-1,0,len(region_max)-1)
    low[i,:n]=row.pop('lower')-region_margin[regions];high[i,:n]=row.pop('upper')+region_margin[regions];valid[i,:n]=True;prefix[i,:n]=row.pop('prefix')
tensor=lambda x:torch.as_tensor(x,device='cuda')
planner=MotionLibraryPlanner(tensor(low),tensor(high),tensor(valid),tensor(prefix),tensor(np.array([[r['activation_id'],r['upstream'],r['downstream']] for r in candidate_rows])),tensor(np.array([r['cost_s'] for r in candidate_rows],np.float32)),tensor(np.array([r['prefix_cost_s'] for r in candidate_rows],np.float32)),tensor(np.array([r['eligible'] for r in candidate_rows])),margin)
catalog=json.loads(a.test_catalog.read_text()) if a.test_catalog else None
layout_specs=[dict(rectangle=[x,x+.2,y-.06,y+.06],height_m=.5) for x in [12.9,13.3] for y in [-.08,0.,.08]]
(a.output/'calibration.json').write_text(json.dumps(dict(records=calibration,margin_m=margin,original_isotropic_margin_m=isotropic_margin,region_edges_x_m=region_edges.tolist(),region_counts=region_counts.tolist(),region_axis_expansion_m=region_margin.tolist(),rule='Per-axis maximum observed support miss in fixed physical map regions plus0.02m; expand each geometry before clearance' ,scope='Same-map DEV spatial comparison at common body x, with boundary support held only for measuring envelope miss; no timing prediction or probability guarantee'),indent=2)+'\n')
source_rows=[]
for i,sources in enumerate(motions):
    source_rows.extend(dict(candidate=i,**{k:v for k,v in s.items() if k not in ['x','lower','upper']}) for s in sources)
(a.output/'candidates.json').write_text(json.dumps(candidate_rows,indent=2)+'\n');(a.output/'sources.json').write_text(json.dumps(source_rows,indent=2)+'\n')
dev_variants=[copy.deepcopy(next(r['variant'] for r in cal_status['results'] if r['variant']['root_variant']==root)) for root in sorted({r['variant']['root_variant'] for r in cal_status['results'] if r['variant']['data_split']!='TRAIN'})]
if catalog:layout_specs=catalog['entries']
# Canonical prefix ordering used by the previously evaluated strong comparator.
group_order=np.argsort([r['prefix_cost_s'] for r in candidate_rows if r['downstream']==0],kind='stable');group_ranks=np.empty(len(group_order),int);group_ranks[group_order]=np.arange(len(group_order));rank_tensor=tensor(np.repeat(group_ranks,3))
plans=[];entries=[]
for layout,spec in enumerate(layout_specs):
    folder=a.output/f'layout{layout:02d}';folder.mkdir();rect=spec.get('rectangle')
    w=copy.deepcopy(world);w['map_id']=f'e2_dev_{layout:02d}'
    if catalog:
        test_config=json.loads(Path(spec['config']).read_text());w=json.loads(Path(test_config['world_map']).read_text());dev_variants=[copy.deepcopy(v) for v in test_config['ground_handoff']['handoff_variants'] if v['requested_speed_m_s']==1.8]
    else:w['terrain']['forbidden_rectangles']=[rect];w['terrain']['obstacle_height_m']=spec['height_m']
    (folder/'world.json').write_text(json.dumps(w,indent=2)+'\n')
    boxes=tensor(np.array([[r[0],r[2],0.,r[1],r[3],w['terrain']['obstacle_height_m']] for r in w['terrain']['forbidden_rectangles']],np.float32));bounds=tensor(np.array(w['bounds_xy_m'],np.float32))
    planner(bounds,boxes);torch.cuda.synchronize();clock=time.perf_counter();indices,feasible,clearance,prefix_clearance,first_choice=planner(bounds,boxes);torch.cuda.synchronize();elapsed=time.perf_counter()-clock
    methods=['segmented','joint']
    if catalog:
        eligible=(clearance>=0)&planner.eligible;first_rank=torch.where(eligible,rank_tensor,torch.inf).amin();back=torch.where(eligible&(rank_tensor==first_rank),planner.cost,torch.inf).argmin();indices=torch.cat((indices,torch.where(eligible.any(),back,torch.zeros_like(back))[None]));feasible=torch.cat((feasible,eligible.any()[None]));methods.append('backtracking')
    indices=indices.cpu().tolist();feasible=feasible.cpu().tolist();clearance=clearance.cpu().tolist()
    selected=[candidate_rows[i] for i in indices]
    for method_index,method in enumerate(methods):
        choice=selected[method_index]
        plans.append(dict(layout=layout,method=method,feasible=feasible[method_index],candidate=indices[method_index],activation_id=choice['activation_id'],activation_x_m=choice['activation_x_m'],route=choice['route'],predicted_full_cost_s=choice['cost_s'],clearance_m=clearance[indices[method_index]],margin_m=margin,shared_planning_ms=elapsed*1000,segmented_prefix_candidate=int(first_choice),all_candidate_clearance_m=clearance))
    for activation in sorted({r['activation_id'] for r in selected}):
        sub=folder/f'activation{activation}';sub.mkdir();c=copy.deepcopy(base)
        for key in ['world_map_spec','ditch_terrain','snapshot_times_s']:c.pop(key,None)
        c.update(world_map=str(folder/'world.json'),planned_activation_offset_m=next(r['activation_x_m'] for r in selected if r['activation_id']==activation)-w['terrain']['front_x_m'],task_evaluation=True,role='E2 static initial-plan DEV; no rolling publication or online latency claim',audit_actuator_owner=False)
        if catalog:c['role']='Frozen new-map E2; same prior DEV algorithms and margins; E3 outcomes previously inspected'
        c['ground_handoff']['handoff_variants']=copy.deepcopy(dev_variants)
        routes=sorted({r['route'] for r in selected if r['activation_id']==activation})
        c['evaluation_lanes']=[[route,'h75',root] for root in range(len(dev_variants)) for route in routes]
        (sub/'config.json').write_text(json.dumps(c,indent=2)+'\n')
        key=f'layout{layout:02d}_activation{activation}'
        entries.append(dict(layout=key,config=str(sub/'config.json'),evaluation=str(sub/'evaluation'),episodes=len(c['evaluation_lanes'])))
        for plan in plans:
            if plan['layout']==layout and plan['activation_id']==activation:
                plan['evaluation']=str(sub/'evaluation');plan['case_dirs']=[f"rb_run3_route{plan['route']}_h75_v{i}" for i in range(len(dev_variants))]
protocol=dict(role='E2 finite-motion DEV pilot, not continuous CEM, rolling publication or independent TEST',methods=['segmented','joint'],selection='H0 canonical straight-downstream prefix then conditional continuation; H1 complete-motion minimization; both use the same36 frozen candidate evaluations',cost='Worst full-chain completion time across two TRAIN sources',source_sweep=str(a.sweep),calibration=str(a.calibration),margin_m=margin,uncertainty='Geometry already expanded by fixed-region per-axis DEV maximum plus0.02m; clearance must remain nonnegative',candidates=36,initial_states='Existing DEV roots6/7, not optimizer variables',goal=world['goal'],seconds=10,ground_only=False,routes=list(range(9)),layouts=layout_specs,entries=entries,paired_conditions=12,method_episodes=24,physical_episodes=sum(r['episodes'] for r in entries),physics_budget_max_lane_steps=240000,expected_wallclock_s=[300,900],hard_limit_s=1500,stop='One frozen queue, first runtime error or25min; no retries or training',publication='Static complete plan before initialization; planning clock separate from physical episode. Actual controller history never reset mid-route.',no_plan='Keep original straight incumbent at first validated activation; count planner failure even if physical goal occurs',physical_map='Same terrain, ditch and2m flat buffers; new forbidden boxes start beyond12.6m; activation is a plan parameter, not a terrain change',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),preparation_s=time.perf_counter()-started)
if catalog:
    protocol.update(role='New-map E2 comparison on frozen E3 map/root catalog; protocol frozen after E3 outcomes were inspected; no E2 outcome-dependent tuning',test_catalog=str(a.test_catalog),methods=methods,paired_conditions=len(layout_specs)*4,method_episodes=len(layout_specs)*4*len(methods),physical_episodes=sum(r['episodes'] for r in entries),physics_budget_max_lane_steps=sum(r['episodes'] for r in entries)*2000,initial_states='New roots1001-1004 from catalog, shared with E3;1.8 request only',primary='Joint versus strong backtracking: all-condition full goal and paired penalized time; greedy secondary',selection='Same36 candidate full-envelope evaluations. Strong segmented comparator scans groups in canonical prefix-cost order, chooses minimum full cost in first feasible group; joint minimizes all feasible full costs.',calibration_scope='Existing DEV calibration and TRAIN motion templates unchanged; new terrain changes are physical tests, not new calibration. E3 outcomes already inspected; no algorithm/threshold change for E2.',expected_wallclock_s=[3600,6600],hard_limit_s=9000,stop='One frozen queue; first runtime error or150min budget; no retries/training; assistant waits over30min pause independently')
(a.output/'manifest.json').write_text(json.dumps(protocol,indent=2)+'\n');(a.output/'plans.json').write_text(json.dumps(plans,indent=2)+'\n')
(a.output/'source').mkdir()
for path in [Path(__file__),Path('learning/src/sttw_control/playground_joint_planner.py'),Path('learning/cli/evaluate_playground_handoff.py')]:shutil.copyfile(path,a.output/'source'/path.name)
print(json.dumps(dict(margin_m=margin,eligible=sum(r['eligible'] for r in candidate_rows),plans=[{k:r[k] for k in ['layout','method','feasible','route','activation_x_m','clearance_m']} for r in plans],physical_episodes=protocol['physical_episodes']),indent=2))
