"""Summarize every actual moving-decision lane, preserving logical root pairing."""
import argparse,collections,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
protocol=json.loads((a.run/'protocol.json').read_text());summary=json.loads((a.run/'evaluation/summary.json').read_text());planning=json.loads((a.run/'evaluation/moving_planning.json').read_text());records={r['lane']:r for r in planning['records']};groups=[];rows=[]
for case in protocol['cases']:
 lane=case['lane'];result=summary['cases'][lane];decision=records.get(lane)
 rows.append(dict(root=case['root'],lane=lane,x=case['x'],method=case['method'],success=result['success'],end_code=result['end_code'],road_violation=result['road_violation'],selected=decision['selected'] if decision and decision['status']=='published' else None,query_ms=decision['query_seconds']*1000 if decision else None,status=decision['status'] if decision else 'unqueried'))
for x,method in sorted(set((c['x'],c['method']) for c in rows)):
 group=[r for r in rows if r['x']==x and r['method']==method];queries=[r['query_ms'] for r in group if r['query_ms'] is not None]
 groups.append(dict(x=x,method=method,n=len(group),completed=sum(c['success'] for c in group),road_violations=sum(c['road_violation'] for c in group),choices=dict(collections.Counter(str(c['selected']) for c in group)),mean_query_ms=float(np.mean(queries)),max_query_ms=float(np.max(queries))))
paired={}
methods=sorted(set(c['method'] for c in rows));reference='state_rule' if 'state_rule' in methods else 'geometry_rule'
for method in methods:
 wins=losses=same_choice_different_result=0
 for r in [c for c in rows if c['method']==method]:
  baseline=next(c for c in rows if c['root']==r['root'] and c['x']==r['x'] and c['method']==reference)
  wins+=r['success'] and not baseline['success'];losses+=baseline['success'] and not r['success'];same_choice_different_result+=r['selected']==baseline['selected'] and r['success']!=baseline['success']
 paired[method]=dict(reference=reference,wins=int(wins),losses=int(losses),same_choice_different_result=int(same_choice_different_result),scope='Descriptive paired outcomes across repeated root conditions; not an identified causal gain')
result=dict(groups=groups,paired=paired,cases=rows,elapsed_s=summary['elapsed_s'],physics_lane_steps=summary['physics_lane_steps'],source=str(a.run),scope=summary['scope']);(a.run/'results.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n')
lines=['# 途中规划方法实际执行结果','','![地图与全部XY](plots/map_and_actual_xy.png)','','![全部条件XY](plots/xy_all_candidates.png)','','|查询x/m|方法|完成|越廊|绕行/跳跃|平均查询ms|','|---|---|---:|---:|---:|---:|']
for g in groups:lines.append(f"|{g['x']}|{g['method']}|{g['completed']}/{g['n']}|{g['road_violations']}/{g['n']}|{g['choices'].get('0',0)}/{g['choices'].get('1',0)}|{g['mean_query_ms']:.3f}|")
lines+=['',f"仿真墙钟{summary['elapsed_s']:.2f}秒，{summary['physics_lane_steps']}物理环境步；0策略训练。每个初态重复多个查询位置，不能视作独立初态样本。方法共享原道路和物理失败判据；克隆环境同方案结果差异仍存在。",'', '[逐初态XY、每步与累计奖励、速度](plots/README.md)；[完整分组与逐条结果](results.json)。',''];(a.run/'RESULTS.md').write_text('\n'.join(lines));print(json.dumps(dict(groups=groups,paired=paired),indent=2))
