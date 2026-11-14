"""Batch full observed STTW departure prediction, with true endpoints and controller histories."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
import torch,mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.playground_arx import states
from sttw_control.playground_history_model import load_history_model
from sttw_control.playground_terrain_features import TerrainFeatures,raster_from_spec
from sttw_control.playground_prediction import BodyEnvelope
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-root',type=Path,required=True);p.add_argument('--checkpoint',default='best.pt');p.add_argument('--evaluations',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);started=time.perf_counter();model=load_history_model(a.model_root,a.checkpoint);phase=int(json.loads((a.model_root/'contract.json').read_text()).get('phase',2));items=[];coverage=[];rasters=[];banks=[]
for map_id,source in enumerate(a.evaluations):
    cfg=json.loads((source/'resolved_config.json').read_text());schema=json.loads((source/'model_state_schema.json').read_text());bank=dict(np.load(cfg['ground_handoff']['reference_path']));banks.append(bank);rasters.append(raster_from_spec(cfg['ditch_terrain']))
    for row in json.loads((source/'status.json').read_text())['results']:
        if row['variant']['data_split']=='TRAIN' or (phase==0 and row['route']%3!=0):continue
        with np.load(source/row['case_dir']/'trajectory.npz') as d:
            pose,xi=states(d['qpos'][::4],d['qvel'][::4],schema);tr=d['trace'][::4];ids=np.flatnonzero((tr[:,-1]==phase)&(np.arange(len(tr))>=1))
            record=dict(evaluation=str(source),case=row['case_dir'],route=row.get('final_route',row['route']),root=row['variant']['root_variant'],physical_outcome=row['termination'])
            if len(ids)<3:coverage.append(dict(record,predicted=False,reason='No complete departure anchor'));continue
            anchor=int(ids[0]);stop=int(ids[-1]);h=stop-anchor;coverage.append(dict(record,predicted=True,anchor_s=float(tr[anchor,0]),last_observed_s=float(tr[stop,0]),steps=h))
            last_step=int(np.flatnonzero(d['trace'][:,-1]==phase)[-1]);items.append(dict(record,map_id=map_id,actual_xy=d['qpos'][anchor*4:last_step+1,:2].copy(),actual_terminal_s=float(d['trace'][last_step,0]),anchor_s=float(tr[anchor,0]),pose=pose[anchor:stop+1],xi=xi[anchor],previous=xi[anchor-1],controller=d['controller_state'][::4][anchor],request=float(tr[anchor,18]),steps=h))
# Padding is storage only. All errors, figures and counts stop at the real recorded endpoint.
n=len(items);steps=max(r['steps'] for r in items);actual=np.stack([np.pad(r['pose'],((0,steps-r['steps']),(0,0)),mode='edge') for r in items]);lengths=np.array([r['steps'] for r in items]);mask=np.arange(steps+1)[None]<=lengths[:,None]
tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
if model.terrain is not None:model.terrain=TerrainFeatures(np.stack([r['height'] for r in rasters]),np.stack([r['support'] for r in rasters]),rasters[0]['bounds'],float(rasters[0]['spacing']),device='cuda')
s=tensor(banks[0]['s']);xy=tensor(np.stack([banks[r['map_id']]['xy'][r['route']] for r in items]));heading=tensor(np.stack([banks[r['map_id']]['heading'][r['route']] for r in items]));curvature=tensor(np.stack([banks[r['map_id']]['curvature'][r['route']] for r in items]))
envelope=BodyEnvelope(mujoco.MjModel.from_binary_path(str(a.evaluations[0]/'terrain.mjb')),dimensions=3)
with torch.no_grad():
    predicted,_=model.rollout(tensor(actual[:,0]),tensor(np.stack([r['xi'] for r in items])),tensor(np.stack([r['previous'] for r in items])),tensor(np.stack([r['controller'] for r in items])),s,xy,heading,curvature,steps,speed_request=tensor([r['request'] for r in items]),terrain_ids=torch.tensor([r['map_id'] for r in items],device='cuda'))
    target=tensor(actual);valid=torch.as_tensor(mask,device='cuda');low,high=envelope(predicted);alo,ahi=envelope(target)
    support=torch.maximum((low-alo).abs(),(high-ahi).abs()).amax((2,3));xy_error=(predicted[:,:,:2]-target[:,:,:2]).norm(dim=-1)
    max_xy=xy_error.masked_fill(~valid,-torch.inf).amax(1).cpu().numpy();max_support=support.masked_fill(~valid,-torch.inf).amax(1).cpu().numpy();prediction=predicted.cpu().numpy()
rows=[]
for i,r in enumerate(items):rows.append({**{k:r[k] for k in ['evaluation','case','map_id','route','root','physical_outcome','anchor_s','actual_terminal_s','request','steps']},'observed_horizon_s':r['steps']*.02,'max_xy_m':float(max_xy[i]),'max_support_xyz_m':float(max_support[i])})
for map_id,source in enumerate(a.evaluations):
    selected=[i for i,r in enumerate(items) if r['map_id']==map_id]
    if not selected:continue
    fig,axes=plt.subplots((len(selected)+1)//2,2,figsize=(14,len(selected)),layout='constrained')
    for ax,i in zip(axes.flat,selected):
        r=items[i];end=r['steps']+1;ref=banks[map_id]['xy'][r['route']];use=(ref[:,0]>=actual[i,0,0]-.1)&(ref[:,0]<=actual[i,r['steps'],0]+.1)
        ax.plot(ref[use,0],ref[use,1],'k:',label='original reference');ax.plot(r['actual_xy'][:,0],r['actual_xy'][:,1],color='#0072b2',label='actual 5ms');ax.plot(prediction[i,:end,0],prediction[i,:end,1],'--',color='#d55e00',label='free prediction');ax.plot(*r['actual_xy'][-1],marker='o' if r['physical_outcome']=='goal' else 'x',color='black')
        ax.set(title=f"{r['case']} {r['request']}m/s {r['steps']*.02:.2f}s",xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7);fig.savefig(a.output/f'map{map_id:02d}_xy.png',dpi=130);plt.close(fig)
np.savez_compressed(a.output/'predictions.npz',actual=actual,predicted=prediction,valid=mask)
with (a.output/'cases.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
identity=dict(phase=phase,model_root=str(a.model_root),checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256((a.model_root/a.checkpoint).read_bytes()).hexdigest(),cases=rows,coverage=coverage,elapsed_s=time.perf_counter()-started,role=f'Existing DEV phase{phase}; first20ms-grid sample with a previous velocity sample, no future true history; >3s is horizon extrapolation. Padding excluded; no new physical success inferred.')
(a.output/'summary.json').write_text(json.dumps(identity,indent=2)+'\n')
lines=[f'# 地面阶段{phase}剩余实测时域预测','','从目标阶段首个具备前一步速度的20ms网格样本自由预测至该阶段最后真实20ms样本；阶段0起点为0.02秒，阶段2为首个可用回交样本。末采样与真实5ms事件终点最多相差15ms。XY实际轨迹画到该阶段真实5ms终点，整回合结局单独保存，不将阶段0末端称目标成功。每例时域不同，超过3秒属于训练时域外展开；存储补齐区不计误差、不画入轨迹，也不推断未知后续。不能以此计算新物理成功率。','']
for map_id in sorted({r['map_id'] for r in items}):lines+=['',f'![地图{map_id}全部实际/预测XY](map{map_id:02d}_xy.png)']
lines+=['','|来源/案例|物理结果|时域s|最大XY m|最大整车支持m|','|---|---|---:|---:|---:|']
for r in rows:lines.append(f"|{r['map_id']}/{r['case']}|{r['physical_outcome']}|{r['observed_horizon_s']:.2f}|{r['max_xy_m']:.4f}|{r['max_support_xyz_m']:.4f}|")
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(cases=n,max_xy_m=float(max_xy.max()),max_support_xyz_m=float(max_support.max()),elapsed_s=identity['elapsed_s'])))
