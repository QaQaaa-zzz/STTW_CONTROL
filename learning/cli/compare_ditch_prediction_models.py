"""Frozen whole-root offline completion comparison; never label this online benefit."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sttw_control.ditch_event_model import EventPlanner
from sttw_control.ditch_completion_model import CompletionPlanner
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--event-model',type=Path,required=True);p.add_argument('--direct-model',type=Path,required=True);p.add_argument('--test',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2);event=EventPlanner(a.event_model);direct=CompletionPlanner(a.direct_model)
train_base=Path(event.meta['collection']);sets=[('fit_collection',train_base),('frozen_test',a.test)];allrows=[];runtime=[];prefix_differences=[]
for label,base in sets:
 cfg=json.loads((base/'config.json').read_text());proto=json.loads((base/'protocol.json').read_text());summary=json.loads((base/'evaluation/summary.json').read_text())
 archive=np.load(base/'evaluation/trajectories.npz');root=archive['initial_root_xyzw'];obs=archive['initial_observation'];history=archive['controller_history'][0];road=archive['road_violation_history'][0];archive.close()
 # The publication state precedes the first segment transition. Prior road history
 # is audited separately; no post-transition observation enters the encoder.
 state=event.encode(root,obs,history,np.zeros(len(root),bool));params=np.asarray(event.meta['candidate_parameters'],np.float32)[cfg['ground_handoff']['reference_route_ids']]
 scores={}
 for name,model in [('event',event),('direct',direct)]:
  start=time.perf_counter();scores[name]=model.score(state,params,particles=32);runtime.append(dict(dataset=label,model=name,states=len(state),seconds=time.perf_counter()-start))
 for c in proto['cases']:
  lane=c['lane'];allrows.append(dict(dataset=label,root=c['root'],candidate_id=cfg['ground_handoff']['reference_route_ids'][lane],role=c['role'],success=bool(summary['cases'][lane]['success']),event=float(scores['event']['predicted_completion'][lane]),direct=float(scores['direct']['predicted_completion'][lane]),lane=lane))
 for rid in sorted({c['root'] for c in proto['cases']}):
  ids=[c['lane'] for c in proto['cases'] if c['root']==rid];xyz=root[ids,:3];prefix_differences.append(dict(dataset=label,root=rid,max_publication_position_delta_m=float(np.linalg.norm(xyz-xyz[0],axis=-1).max())))
train=[r for r in allrows if r['role']=='TRAIN'];rate=float(np.mean([r['success'] for r in train]));ids=sorted({r['candidate_id'] for r in train});fixed=max(ids,key=lambda i:(np.mean([r['success'] for r in train if r['candidate_id']==i]),-i))
metrics=[];selections=[]
for role in ('TRAIN','DEV','TEST'):
 rows=[r for r in allrows if r['role']==role];y=np.array([r['success'] for r in rows]);keys=sorted({(r['dataset'],r['root']) for r in rows})
 if not len(rows):continue
 for name in ('event','direct','constant'):
  prob=np.array([r[name] if name!='constant' else rate for r in rows]);metrics.append(dict(role=role,model=name,n=len(rows),roots=len(keys),brier=float(np.mean((prob-y)**2)),mean_probability=float(prob.mean()),actual_rate=float(y.mean())))
 for ds,rid in keys:
  group=[r for r in rows if r['dataset']==ds and r['root']==rid];by_id={r['candidate_id']:r for r in group}
  if len(by_id)!=len(ids):raise ValueError('Offline selection requires the complete declared candidate panel; retain incomplete roots separately')
  row=dict(role=role,dataset=ds,root=rid,oracle_coverage=any(r['success'] for r in group))
  for name in ('event','direct','fixed'):
   cid=fixed if name=='fixed' else max(ids,key=lambda i:(by_id[i][name],-i));row[name]=dict(candidate=cid,success=by_id[cid]['success'])
  selections.append(row)
result=dict(metrics=metrics,selections=selections,predictions=allrows,inference=runtime,fixed_train_best=fixed,train_rate=rate,publication_position_differences=prefix_differences,scope='Frozen same-map offline comparison on actual published-state segments. Candidate lanes have individual physics states; retrospective selection is not actual online execution or a common-state counterfactual certificate. TEST roots previously viewed for skill comparison, not used to fit either model.')
(a.output/'comparison.json').write_text(json.dumps(result,indent=2))
text='# 事件滚动与直接成败预测的冻结对照\n\n[采集实际／参考XY](../chapter3_offroad_event_training/RESULTS.md) · [测试实际／参考XY](../chapter3_offroad_new_source/RESULTS.md)\n\n相同状态/历史编码、高程、动作候选与整根划分，两模型600次更新、每100次仅用DEV选模。事件模型按32粒子滚动预测完整任务，直接模型预测整程成败；此表不把单事件条件预测当整程预测。TEST为既有4个未拟合的来车初态，曾用于技能比较，不作为完全未查看的确认性测试。\n\n|分组|模型|条数/独立初态|Brier|平均预测|实际完成率|\n|---|---|---:|---:|---:|---:|\n'
for m in metrics:text+=f"|{m['role']}|{m['model']}|{m['n']}/{m['roots']}|{m['brier']:.4f}|{m['mean_probability']:.4f}|{m['actual_rate']:.4f}|\n"
text+='\n|分组|初态|事件候选/结果|直接候选/结果|TRAIN固定候选/结果|事后可完成|\n|---|---:|---|---|---|---|\n'
for s in selections:text+=f"|{s['role']}|{s['root']}|{s['event']['candidate']}/{s['event']['success']}|{s['direct']['candidate']}/{s['direct']['success']}|{s['fixed']['candidate']}/{s['fixed']['success']}|{s['oracle_coverage']}|\n"
text+='\n候选编号与原协议一致。按完整任务概率选最大值，不用TEST标签选阈值、checkpoint或固定档。以上选档只查已保存结果，候选入口存在克隆物理差异，不能称真实在线增益；应另跑共同前缀的在线执行。\n'
(a.output/'RESULTS.md').write_text(text);print(text)
