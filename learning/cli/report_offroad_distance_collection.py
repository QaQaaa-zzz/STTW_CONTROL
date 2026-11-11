"""Distinguish requested candidates, actually published plans and fallback outcomes."""
import argparse,json,collections
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
proto=json.loads((a.run/'protocol.json').read_text());contract=json.loads((a.run/'map_contract.json').read_text());summary=json.loads((a.run/'evaluation/summary.json').read_text());planning=json.loads((a.run/'evaluation/moving_planning.json').read_text());records={r['lane']:r for r in planning['records']};per=len(contract['speeds_m_s'])*len(contract['alphas']);rows=[]
for c in proto['cases']:
 r=summary['cases'][c['lane']];d=records.get(c['lane']);published=d is not None and d['status']=='published';same=published and d['selected']==c['candidate_id']
 rows.append(dict(root=c['root'],role=c['role'],lane=c['lane'],x=c['x'],route=contract['routes'][c['candidate_id']//per],candidate_id=c['candidate_id'],status=d['status'] if d else 'unqueried',published=published,intended_published=same,actual_success=r['success'],intended_success=same and r['success'],entry_s=r['entry_s'],handoff_s=r['handoff_s'],end_code=r['end_code']))
groups=[]
for x in sorted({c['x'] for c in rows}):
 for route in contract['routes']:
  g=[c for c in rows if c['x']==x and c['route']==route]
  groups.append(dict(x=x,route=route,n=len(g),published=sum(c['intended_published'] for c in g),completed=sum(c['intended_success'] for c in g),fallback_completed=sum(c['actual_success'] and not c['intended_published'] for c in g),entry=sum(c['intended_published'] and c['entry_s'] is not None for c in g),handoff=sum(c['intended_published'] and c['handoff_s'] is not None for c in g),statuses=dict(collections.Counter(c['status'] for c in g))))
result=dict(groups=groups,cases=rows,elapsed_s=summary['elapsed_s'],physics_lane_steps=summary['physics_lane_steps'],scope='All declared roots/positions kept; unavailable candidates excluded from executed-plan training, retained as admission failures. Fallback goal is never credited to requested plan.')
(a.output/'coverage.json').write_text(json.dumps(result,indent=2))
text='# 准备距离与路线接入采集\n\n'
for x in sorted({c['x'] for c in rows}):text+=f'- [x={x}m：全部实际／参考XY、奖励和速度](x{x}/RESULTS.md)\n'
text+='\n原地图、任务和物理约束固定；同初态跨查询位置全部留在同一TRAIN/DEV组。表中“完成”必须是所请求候选实际发布并到达任务目标；接入被拒后沿原方案到达单独列出，不归功于所请求候选。\n\n|查询x/m|路线|请求|发布|候选完成|跳跃入口|STTW回交|未发布但原方案完成|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
for g in groups:text+=f"|{g['x']}|{g['route']}|{g['n']}|{g['published']}|{g['completed']}|{g['entry']}|{g['handoff']}|{g['fallback_completed']}|\n"
text+=f"\n总物理采集{summary['elapsed_s']:.2f}秒、{summary['physics_lane_steps']}环境步、0PPO。查询位置不是瞬时发布位置：计算延迟内车辆继续运动，发布再次核查道路距离。未发布与上游失败均见[逐条原始标签](coverage.json)。没有放松0.6m路廊，不把静态距离准入称为动力学可行性保证。\n"
(a.output/'RESULTS.md').write_text(text);print(text)
