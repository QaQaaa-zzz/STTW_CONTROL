"""Paired skill-substitution outcomes with pre-jump and ground-only controls."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--candidates',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
runs=[a.source]+a.candidates;loaded=[]
for run in runs:
 protocol=json.loads((run/'protocol.json').read_text());summary=json.loads((run/'evaluation/summary.json').read_text());trace=np.load(run/'evaluation/trajectories.npz');config=json.loads((run/'config.json').read_text());contract=json.loads((run/'map_contract.json').read_text());per=len(contract['speeds_m_s'])*len(contract['alphas'])
 loaded.append((run,protocol,summary,trace,config,contract,per))
base=loaded[0];lookup={(c['root'],c['candidate_id']):c['lane'] for c in base[1]['cases']};rows=[];groups=[]
for run,protocol,summary,trace,config,contract,per in loaded:
 for c in protocol['cases']:
  lane=c['lane'];old_lane=lookup[c['root'],c['candidate_id']];result=summary['cases'][lane];old=base[2]['cases'][old_lane];valid=trace['valid'][:,lane];steps=int(valid.sum());old_steps=int(base[3]['valid'][:,old_lane].sum());times=trace['time']
  entries=[t for t in (result['entry_s'],old['entry_s']) if t is not None];prefix=min(steps,old_steps)
  if entries:prefix=min(prefix,int(np.searchsorted(times,min(entries),side='left')))
  delta=np.linalg.norm(trace['root_xyzw'][:prefix,lane,:3]-base[3]['root_xyzw'][:prefix,old_lane,:3],axis=-1)
  indices=np.flatnonzero(trace['road_violation_history'][:,lane]&valid);first=float(times[indices[0]]) if len(indices) else None
  entry=result['entry_s'];hand=result['handoff_s'];phase='none' if first is None else ('approach' if entry is None or first<entry else 'jump' if hand is None or first<hand else 'continuation')
  rows.append(dict(model=run.name,root=c['root'],candidate_id=c['candidate_id'],route=contract['routes'][c['candidate_id']//per],success=result['success'],gained=result['success'] and not old['success'],lost=old['success'] and not result['success'],end_code=result['end_code'],entry_s=entry,handoff_s=hand,violation_phase=phase,first_violation_s=first,prefix_steps=prefix,pre_jump_max_position_delta_m=float(delta.max()) if len(delta) else None))
 for route in contract['routes']:
  subset=[c for c in rows if c['model']==run.name and c['route']==route]
  groups.append(dict(model=run.name,route=route,n=len(subset),completed=sum(c['success'] for c in subset),gained=sum(c['gained'] for c in subset),lost=sum(c['lost'] for c in subset),entry=sum(c['entry_s'] is not None for c in subset),handoff=sum(c['handoff_s'] is not None for c in subset),prejump_violation=sum(c['violation_phase']=='approach' for c in subset),max_prefix_delta_m=max(c['pre_jump_max_position_delta_m'] or 0 for c in subset)))
(a.output/'comparison.json').write_text(json.dumps(dict(groups=groups,cases=rows),indent=2))
text='# 新越野任务中的RB技能替换对照\n\n固定同地图、路线、速度、α和来车初态；仅替换跳跃权重。原模型已完成数据复用，列出的候选均使用已完成物理轨迹；各组分母见表。是开发条件，不是独立测试。不同进程的物理重复差异单独列出，不能把所有得失归因于跳跃技能。\n\n'
for run,_,_,_,config,_,_ in loaded:
 report=run.name if run.name.startswith('chapter3_') else 'chapter3_offroad_'+run.name
 text+=f'- [{run.name}全部实际/参考XY与奖励](../{report}/RESULTS.md)，权重 `{config.get("evaluation_checkpoint",config["bundle"])}`\n'
text+='\n|模型|路线|完成|新增/丢失|进入跳跃|交回STTW|首次越廊在接近|接管前最大位置差/m|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for g in groups:text+=f"|{g['model']}|{g['route']}|{g['completed']}/{g['n']}|{g['gained']}/{g['lost']}|{g['entry']}|{g['handoff']}|{g['prejump_violation']}|{g['max_prefix_delta_m']:.6g}|\n"
text+='\n接管前差异比较相同root/candidate的共同有效前缀，截止双方较早跳跃接管时间之前；地面绕行比较共同完整有效窗口。此统计检测非跳跃阶段差异，不证明相同前缀以外的因果效应。首次违规由20ms记录的累积标志确定。目标、道路、物理终止均未放宽；进入跳跃不等于真实离地，交接不等于到达目标。\n'
(a.output/'RESULTS.md').write_text(text);print(text)
