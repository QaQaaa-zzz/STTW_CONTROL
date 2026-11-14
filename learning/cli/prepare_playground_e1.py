"""Freeze a 12-layout x 3-root E1 DEV pilot and equal-budget GPU plans."""
import argparse,copy,json,time,math
from pathlib import Path
import numpy as np
import torch
import mujoco
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import NominalRollout,BodyEnvelope
from sttw_control.playground_planner import SplineRoutes,GroundCEM
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('learning/configs/playground_sttw_rb_sttw.json'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
contract=json.loads((a.probe/'protocol.json').read_text())['contract'];base=json.loads(a.config.read_text());world=json.loads(Path(base['world_map']).read_text());source=a.collection/'flat_01_dev_root/evaluation';scfg=json.loads((source/'resolved_config.json').read_text());schema=json.loads((source/'model_state_schema.json').read_text());status=json.loads((source/'status.json').read_text());follower=json.loads((Path(scfg['ground_handoff']['bundle'])/'config.json').read_text())['follower'];follower.update(scfg['ground_handoff'].get('follower_overrides',{}))
roots=[];poses=[];speeds=[];previous=[]
for root in range(3):
 row=next(r for r in status['results'] if r['route']==0 and r['case_dir'].endswith('_v'+str(root)));data=np.load(source/row['case_dir']/'trajectory.npz');q,v=states(data['qpos'][::4],data['qvel'][::4],schema);poses.append(q[1]);speeds.append(v[1]);previous.append(v[0]);roots.append(row['variant'])
manifest=dict(constraint_ranking='Dimensionless violations normalized by their frozen physical margin/limit/tolerance; feasible candidates ordered by common fixed-endpoint length and smoothness.',role='E1 DEV pilot, not independent TEST or P1 acceptance',layouts=12,roots_per_layout=3,methods=['geometry','dynamic'],physical_episodes=72,seconds=4.5,contract=contract,candidates_per_topology=64,iterations=4,elites=8,topologies=['left','right'],source_model=str(a.collection/'model/nominal_arx.npz'),root_source=str(source),selection='Static pre-execution planning from saved matching flat reset20ms histories. Run verifies measured query-state mismatch. Not live online planning. Equal frozen noise and 2 incumbent candidates within each 64 budget.',publication='All methods/roots in a layout publish together at ceil(max measured planning time /20ms)+20ms; old straight plan continues and histories are retained.',failure='No feasible plan is retained in denominator and runs its existing straight plan; a least-violating candidate is never published.',budget='One 12-batch GPU queue only; 4.5s each, maximum72 episodes. No retries or training. Expected12-25min; >30min wait pauses goal with Bark. Original complete-chain terrain remains unchanged.',terrain='Original near-bank stones and2cm ridge, original ditch and far-bank4cm ridge retained. E1 ends before ditch. Fixed forbidden boxes and road bounds per layout, independent of selected route.')
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
all_pose,all_speed,all_previous=map(tensor,(np.array(poses),np.array(speeds),np.array(previous)));pose=all_pose[:1].clone();xi=all_speed[:1].clone();prev=all_previous[:1].clone();bounds=tensor([-5,25,-1.5,1.5]);obstacle=tensor([[4.3,4.5,.18,.42]]);noise=tensor(np.random.default_rng(20261115).normal(size=(4,64,6)));mean=torch.full((6,),.6,device='cuda');std=torch.full((6,),.35,device='cuda');routes=SplineRoutes(3.3,6.3);predictor=NominalRollout(dict(np.load(a.collection/'model/nominal_arx.npz')),follower);predictor.compile_step();envelope=BodyEnvelope(mujoco.MjModel.from_binary_path(str(source/'terrain.mjb')));graphs={}
for method in ['geometry','dynamic']:
 solver=GroundCEM(routes,predictor,envelope,contract,method);inputs=(pose,xi,prev,bounds,obstacle,noise,mean,std);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):solver(*inputs)
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=solver(*inputs)
 graphs[method]=(solver,graph,out)  # Keep captured constant tensors alive through every replay.
straight_xy,straight_h,straight_k=routes(torch.zeros((1,6),device='cuda'));straight=[x[0].cpu().numpy() for x in (straight_xy,straight_h,straight_k)];arc=routes.s.cpu().numpy();plans=[];entries=[]
for layout,(x,offset) in enumerate((x,y) for x in (4.2,4.6) for y in (-.7,-.5,-.3,.3,.5,.7)):
 folder=a.output/f'layout{layout:02d}';folder.mkdir();rect=[x,x+.2,offset-.12,offset+.12];obstacle.copy_(tensor([rect]));selected=[];times=[];rows=[]
 for root in range(3):
  pose.copy_(all_pose[root:root+1]);xi.copy_(all_speed[root:root+1]);prev.copy_(all_previous[root:root+1])
  for method in ['geometry','dynamic']:
   solver,graph,out=graphs[method];candidates=[];start=time.perf_counter()
   for side in (1,-1):
    mean.fill_(side*.6);graph.replay();candidates.append(tuple(v.clone() for v in out))
   torch.cuda.synchronize();elapsed=time.perf_counter()-start;times.append(elapsed)
   metrics=[(float(z[5]),float(z[6])) for z in candidates];choice=min(range(2),key=lambda i:metrics[i]);result=candidates[choice];feasible=metrics[choice][0]==0
   xy,h,k=[t.cpu().numpy() for t in result[1:4]] if feasible else straight
   selected.append((xy,h,k));record=dict(layout=layout,root=root,method=method,feasible=feasible,topology=['left','right'][choice],violation=metrics[choice][0],cost=metrics[choice][1],planning_s=elapsed,constraints=result[8].cpu().tolist(),margin_m=contract['margin_m']);rows.append(record);plans.append(record)
   np.savez_compressed(folder/f'{method}_root{root}_prediction.npz',predicted_pose=result[4].cpu().numpy(),proposed_xy=result[1].cpu().numpy(),published_xy=xy)
 np.savez_compressed(folder/'initial_routes.npz',s=arc,xy=np.stack([straight[0]]*6),heading=np.stack([straight[1]]*6),curvature=np.stack([straight[2]]*6))
 np.savez_compressed(folder/'planned_routes.npz',s=arc,xy=np.stack([z[0] for z in selected]),heading=np.stack([z[1] for z in selected]),curvature=np.stack([z[2] for z in selected]))
 w=copy.deepcopy(world);w['map_id']=f'e1_dev_{layout:02d}';w['bounds_xy_m']=[-5,25,-1.5,1.5];w['terrain']['forbidden_rectangles']=[rect];w['terrain']['obstacle_height_m']=.5;w['goal'].update(x_min_m=6.8,x_max_m=8.5,abs_y_max_m=.3);(folder/'world_map.json').write_text(json.dumps(w,indent=2))
 c=copy.deepcopy(base);c.update(task_evaluation=True,world_map=str(folder/'world_map.json'),contact_substep_telemetry=True,collect_model_state=True,audit_actuator_owner=False,evaluation_lanes=[[2*r+m,'h75',r] for r in range(3) for m in range(2)])
 c['ground_handoff']['reference_path']=str(folder/'initial_routes.npz');c['ground_handoff']['handoff_variants']=copy.deepcopy(roots);publish=.02+math.ceil(max(times)/.02)*.02;c['plan_updates']=[dict(time_s=publish,skills=['rb_run3'],reference_path=str(folder/'planned_routes.npz'),committed_until_m=8.25,expected_previous_version=0)]
 c['role']='E1 static DEV plans, shared delayed publication; initial saved-state replay mismatch must be reported';(folder/'config.json').write_text(json.dumps(c,indent=2));(folder/'plans.json').write_text(json.dumps(rows,indent=2));entries.append(dict(layout=layout,config=str(folder/'config.json'),evaluation=str(folder/'evaluation'),publish_s=publish,planned=sum(z['feasible'] for z in rows),episodes=6))
manifest['entries']=entries;(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.output/'plans.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in plans));np.savez_compressed(a.output/'query_states.npz',pose=np.array(poses),xi=np.array(speeds),previous=np.array(previous));print(json.dumps(entries,indent=2))
