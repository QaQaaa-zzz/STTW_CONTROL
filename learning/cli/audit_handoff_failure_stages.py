"""Audit recorded task stages without new simulation or retrospective capture labels."""
import argparse, collections, json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
sources={'source720':'chapter3_offroad_event_training/collection','source1440':'chapter3_offroad_distance_training/collection','source240':'chapter3_offroad_new_roots/source','rb400_240':'chapter3_offroad_new_roots/rb400'}
all_rows=[];groups=[]
for name,relative in sources.items():
 run=a.runs/relative;summary=json.loads((run/'evaluation/summary.json').read_text())['cases'];protocol=json.loads((run/'protocol.json').read_text())['cases'];data=np.load(run/'evaluation/trajectories.npz');events=data['event_time'];road=data['road_violation_history'];times=data['time'];valid=data['valid'];planning=json.loads((run/'evaluation/moving_planning.json').read_text())['records'];published={r['lane'] for r in planning if r['status']=='published'}
 rows=[]
 for i,(c,q) in enumerate(zip(summary,protocol)):
  hand=c['handoff_s'];entry=c['entry_s'];flight,clear,far,handoff,goal=[float(t) if np.isfinite(t) else None for t in events[i]]
  bad=np.flatnonzero(road[:,i]&valid[:,i]);first_bad=float(np.asarray(times).reshape(-1)[bad[0]]) if len(bad) else None
  end_time=float(np.asarray(times).reshape(-1)[np.flatnonzero(valid[:,i])[-1]])
  stage=('success' if c['success'] else 'ground_only_failure' if c['skill']=='ground_only' else 'approach_no_entry' if entry is None else 'entry_no_flight' if flight is None else 'flight_no_far_contact' if far is None else 'far_contact_no_handoff' if hand is None else 'after_handoff_failure')
  rows.append(dict(dataset=name,lane=i,root=q['root'],candidate=q['candidate'],route=q['candidate'].split('_v')[0],x=q['x'],published=i in published,stage=stage,success=c['success'],skill=c['skill'],entry_s=entry,flight_s=flight,far_contact_s=far,handoff_s=hand,end_s=end_time,post_handoff_observed_s=end_time-hand if hand is not None else None,road_first_s=first_bad,road_before_handoff=bool(hand is not None and data['event_road_violation'][i,3]),road_violation=c['road_violation'],end_code=c['end_code'],capture_label='unavailable: no frozen capture envelope in original task'))
 all_rows.extend(rows)
 for pub in [None,True,False]:
  subset=[r for r in rows if pub is None or r['published']==pub]
  for route in ['ALL']+sorted({r['route'] for r in subset}):
   group=[r for r in subset if route=='ALL' or r['route']==route];post=[r for r in group if r['stage']=='after_handoff_failure'];clean=[r for r in group if r['handoff_s'] is not None and not r['road_before_handoff']]
   groups.append(dict(dataset=name,published=pub,route=route,n=len(group),stages=dict(collections.Counter(r['stage'] for r in group)),handoffs=sum(r['handoff_s'] is not None for r in group),post_handoff_failures=len(post),post_fail_already_outside=sum(r['road_before_handoff'] for r in post),clean_handoffs=len(clean),clean_handoff_success=sum(r['success'] for r in clean),post_failure_codes=dict(collections.Counter(str(r['end_code']) for r in post))))
(a.output/'stages.json').write_text(json.dumps(dict(sources={k:str(a.runs/v) for k,v in sources.items()},groups=groups,cases=all_rows,scope='Retrospective event ordering, not causal attribution. No new physics or training. No capture success labels invented. 720/1440 source only; RB400 available only in separate paired240.'),indent=2)+'\n')
lines=['# 交接瓶颈：既有轨迹阶段审计','','先看原始完整XY：[720条](../chapter3_offroad_event_training/RESULTS.md)、[1440条各位置](../chapter3_offroad_distance_training/RESULTS.md)、[原技能与RB400各240条](../chapter3_offroad_new_roots/ANALYSIS.md)。不新增仿真、不改原始标签。','','|数据|发布口径|N|成功|未到入口|入口后未腾空|腾空未触岸|触岸未回交|回交后失败|纯地面失败|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in groups:
 if g['route']!='ALL':continue
 s=g['stages'];lines.append('|'+ '|'.join(map(str,[g['dataset'],str(g['published']),g['n']]+[s.get(k,0) for k in ['success','approach_no_entry','entry_no_flight','flight_no_far_contact','far_contact_no_handoff','after_handoff_failure','ground_only_failure']]))+'|')
lines+=['','发布口径None=全部请求，True=候选真正发布，False=未发布而继续原方案。失败阶段按最后已记录事件分类，不能把未发布轨迹归给请求候选。纯绕行不计跳跃阶段。成功优先列出；完整事件记录、每路线、每根/候选、终止码在JSON保留。','','## 交回后的失败是否已经在交回前越廊','','|数据（已发布）|交回数|交回后失败|其中交回前已越廊|未先越廊的交回数|其中最终成功|','|---|---:|---:|---:|---:|---:|']
for g in groups:
 if g['route']=='ALL' and g['published'] is True:lines.append('|'+ '|'.join(map(str,[g[k] for k in ['dataset','handoffs','post_handoff_failures','post_fail_already_outside','clean_handoffs','clean_handoff_success']]))+'|')
lines+=['','**标签边界：**“交回”是控制权切换，不是捕获或恢复成功；原记录没有按无障碍包络冻结的捕获定义，因此不能从这些文件区分“交回捕获失败”与“已经捕获后续行失败”。交回前已越廊的失败不能全部归因于H75接不住。首次越廊时刻由50Hz记录近似定位；是否在交回前已越廊采用交回事件保存的物理频率累积标志，不从近似时间反推。','','720和1440批次只有原技能；RB400仅在另一个同布局的240条配对中存在，不能虚构其在2160条上的结果。各批次初态、查询距离与分母不同，不合并为一个成功率。']
lines += ['', '[路线基线、规则重复性与初始化核查](ANALYSIS.md)。']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[-11:]))
