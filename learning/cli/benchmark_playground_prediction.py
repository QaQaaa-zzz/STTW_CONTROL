"""One bounded CPU/GPU parity and resident-graph timing check on saved DEV roots."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sttw_control.playground_arx import states,Reference,rollout
from sttw_control.playground_prediction import NominalRollout
p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--compile-step',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
contract=dict(role='Engineering check, saved DEV inputs only; no simulation, policy training or planning benefit claim',batch=64,steps=100,dt_s=.02,parity_cases=8,timing_repeats=20,budget='one benchmark; no retries; initial compilation/graph capture reported separately')
(a.output/'protocol.json').write_text(json.dumps(contract,indent=2))
r=a.collection/'flat_01_dev_root/evaluation';cfg=json.loads((r/'resolved_config.json').read_text());schema=json.loads((r/'model_state_schema.json').read_text());rows=json.loads((r/'status.json').read_text())['results'];follower=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text())['follower'];follower.update(cfg['ground_handoff'].get('follower_overrides',{}));model=dict(np.load(a.collection/'model/nominal_arx.npz'));bank=np.load(a.collection/'routes.npz')
pose=[];xi=[];prev=[];refs=[]
for row in rows:
 d=np.load(r/row['case_dir']/'trajectory.npz');q,v=states(d['qpos'][::4],d['qvel'][::4],schema);pose.append(q[50]);xi.append(v[50]);prev.append(v[49]);refs.append(row['route'])
pose,xi,prev=map(np.array,(pose,xi,prev));tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
inputs=[tensor(x) for x in (pose,xi,prev,bank['s'],bank['xy'][refs],bank['heading'][refs],bank['curvature'][refs])]
start=time.perf_counter();predictor=NominalRollout(model,follower)
if a.compile_step:predictor.compile_step()
stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
 for _ in range(2):predicted,velocity=predictor(*inputs)
torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();warmup=time.perf_counter()-start
start=time.perf_counter();graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):predicted,velocity=predictor(*inputs)
torch.cuda.synchronize();capture=time.perf_counter()-start
elapsed=[]
for _ in range(20):
 start=time.perf_counter();graph.replay();torch.cuda.synchronize();elapsed.append(time.perf_counter()-start)
actual=predicted.cpu().numpy();speed=velocity.cpu().numpy();parity=[]
for i in range(8):
 reference=Reference(bank,refs[i],follower);expected,ev=rollout(model,reference,pose[i:i+1],xi[i:i+1],prev[i:i+1],100)
 parity.append(dict(case=rows[i]['case_dir'],max_pose_abs=float(np.max(abs(actual[i]-expected[0]))),max_velocity_abs=float(np.max(abs(speed[i]-ev[0])))))
result=dict(compile_step=a.compile_step,contract=contract,warmup_s=warmup,capture_s=capture,graph_ms=dict(median=float(np.median(elapsed)*1e3),p95=float(np.quantile(elapsed,.95)*1e3),maximum=max(elapsed)*1e3),parity=parity,scope='Timing includes synchronization and 64 candidate 2s prediction; excludes candidate generation, geometry checks, CEM and publication. Not end-to-end 200ms acceptance.')
(a.output/'result.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.output/'prediction.npz',pose=actual,xi=speed);print(json.dumps(result,indent=2))
