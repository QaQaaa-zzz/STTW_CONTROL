"""Frozen bounded CEM engineering probe; not the E1 physical experiment."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from sttw_control.playground_arx import states
from sttw_control.playground_prediction import NominalRollout,BodyEnvelope
from sttw_control.playground_planner import SplineRoutes,GroundCEM
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--compile-step',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
c=dict(steps=150,iterations=4,elites=8,parameter_limit_m=1.4,minimum_std_m=.04,margin_m=.36,roll_limit_rad=.7,pitch_limit_rad=1.309,min_speed_m_s=.5,curvature_limit_per_m=1.4,exit_x_m=5.,tail_distance_m=1.8,exit_y_m=0.,exit_y_tolerance_m=.3,exit_heading_tolerance_rad=.2,exit_speed_tolerance_m_s=.4,wheelbase_m=.27903)
protocol=dict(contract=c,batch=64,topologies=['left','right'],methods=['geometry','dynamic'],role='Solver engineering DEV only; no new physics or E1 success claims',budget='4 graph captures; 5 timing repetitions each; identical frozen noise and candidate budgets',margin='Shared engineering margin .36m = maximum prior rough DEV 3s body-error P95 .339744m + .02m map/discretization, rounded up. No probabilistic guarantee, not independently calibrated geometry baseline.',road_bounds=[-5,25,-1.5,1.5],forbidden_rectangles=[[4.5,4.7,-.12,.12]],spline_x=[3.3,6.3])
(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
r=a.collection/'flat_01_dev_root/evaluation';cfg=json.loads((r/'resolved_config.json').read_text());schema=json.loads((r/'model_state_schema.json').read_text());row=json.loads((r/'status.json').read_text())['results'][0];z=np.load(r/row['case_dir']/'trajectory.npz');q,v=states(z['qpos'][::4],z['qvel'][::4],schema)
follower=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text())['follower'];follower.update(cfg['ground_handoff'].get('follower_overrides',{}));model=dict(np.load(a.collection/'model/nominal_arx.npz'));plant=mujoco.MjModel.from_binary_path(str(r/'terrain.mjb'))
tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
pose,xi,previous=map(tensor,(q[1:2],v[1:2],v[:1]));bounds=tensor(protocol['road_bounds']);obstacles=tensor(protocol['forbidden_rectangles']);noise=tensor(np.random.default_rng(20261114).normal(size=(4,64,6)));std=torch.full((6,),.35,device='cuda');routes=SplineRoutes(3.3,6.3);predictor=NominalRollout(model,follower);envelope=BodyEnvelope(plant)
if a.compile_step:predictor.compile_step()
protocol['compile_step']=a.compile_step;(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
results=[]
for method in ['geometry','dynamic']:
 for topology,side in [('left',1),('right',-1)]:
  mean=torch.full((6,),.6*side,device='cuda');solver=GroundCEM(routes,predictor,envelope,c,method);inputs=(pose,xi,previous,bounds,obstacles,noise,mean,std)
  start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):out=solver(*inputs)
  torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();warmup=time.perf_counter()-start
  start=time.perf_counter();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out=solver(*inputs)
  torch.cuda.synchronize();capture=time.perf_counter()-start;times=[]
  for _ in range(5):
   start=time.perf_counter();graph.replay();torch.cuda.synchronize();times.append(time.perf_counter()-start)
  arrays=[x.cpu().numpy() for x in out];params,xy,h,k,prediction,violation,cost,clearance,errors=arrays
  entry=dict(method=method,topology=topology,feasible=bool(violation==0),violation=float(violation),cost=float(cost),minimum_clearance_m=float(clearance),constraint_violation_components=errors.tolist(),warmup_s=warmup,capture_s=capture,median_ms=float(np.median(times)*1000),p95_ms=float(np.quantile(times,.95)*1000),maximum_ms=max(times)*1000)
  results.append(entry);np.savez_compressed(a.output/(method+'_'+topology+'.npz'),parameters=params,xy=xy,heading=h,curvature=k,predicted_pose=prediction,s=routes.s.cpu().numpy());(a.output/'result.json').write_text(json.dumps(results,indent=2));print(json.dumps(entry),flush=True)
