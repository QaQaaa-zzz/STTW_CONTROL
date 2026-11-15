"""Compare disk restart against a fixed same-snapshot in-memory repeat baseline."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
status=json.loads((a.root/'evaluation/status.json').read_text());cfg=json.loads((a.root/'config.json').read_text());tick=round(cfg['snapshot_replay_s']/.005);rows=status['results'];lanes=sorted({(r['route'],r['variant']['root_variant']) for r in rows});summary=[];fig,axes=plt.subplots(len(lanes),1,figsize=(12,3*len(lanes)),squeeze=False,layout='constrained')
for ax,(route,root) in zip(axes[:,0],lanes):
 selected=[r for r in rows if (r['route'],r['variant']['root_variant'])==(route,root)];data=[np.load(a.root/'evaluation'/r['case_dir']/'trajectory.npz') for r in selected];base=data[0];disk=data[-1];mem=data[1:-1]
 metrics={}
 for key in ['qpos','qvel','applied_targets','controller_state','rb_state']:
  def difference(x,y):
   n=min(len(x[key]),len(y[key]));return float(np.abs(x[key][tick:n]-y[key][tick:n]).max())
  noise=max(difference(x,y) for x,y in itertools.combinations(mem,2));error=difference(base,disk)
  metrics[key]=dict(repeat_max=noise,disk_vs_continuous=error,within_repeat=error<=noise,at_capture=float(abs(base[key][tick]-disk[key][tick]).max()))
 events={}
 for key in ['rb_entry_s','first_far_contact_s','handoff_s','duration_s']:
  noise=max(abs(x[key]-y[key]) for x,y in itertools.combinations(selected[1:-1],2));error=abs(selected[0][key]-selected[-1][key]);events[key]=dict(repeat_max_s=noise,disk_vs_continuous_s=error,within_repeat=error<=noise)
 record=dict(route=route,root=root,metrics=metrics,events=events,terminations=[r['termination'] for r in selected],planning_failed=[r.get('planning_failed') for r in selected],passed_numerical=all(m['within_repeat'] for m in metrics.values()) and all(e['within_repeat'] for e in events.values()))
 summary.append(record)
 for d,r in zip(data,selected):ax.plot(d['qpos'][:,0],d['qpos'][:,1],alpha=.65,label=r['skill'])
 ax.plot(base['reference_xy'][:,0],base['reference_xy'][:,1],':',label='original reference');ax.set(xlim=(1,19),ylim=(-1,1),xlabel='World x (m), travel →',ylabel='World y (m)',title=f'route{route} root{root}, snapshot {tick*.005:.2f}s; frozen task');ax.set_aspect('equal');ax.legend(ncol=3,fontsize=7)
fig.savefig(a.output/'same_state_xy.png',dpi=160)
(a.output/'summary.json').write_text(json.dumps(dict(raw_root=str(a.root),memory_repeats=cfg['snapshot_memory_repeats'],cases=summary,scope='Fixed finite-repeat engineering comparison of fieldwise maxima; not exact determinism or a statistical safety bound'),indent=2)+'\n')
(a.output/'RESULTS.md').write_text('# 同快照内存/序列化恢复对照\n\n![全部真实与参考XY](same_state_xy.png)\n\n固定四次内存恢复，不随结果追加；同物理与目标。比较恢复点至共同真实终点的五类字段最大绝对差，以及起跳/触地/交回/目标时间，全部原始数值见summary.json。混合字段的最大值不解释为单一位置单位；不是统计置信界或严格确定性证明。\n\n'+ '\n'.join(f"- route{r['route']} root{r['root']}：数值与事件基准{'通过' if r['passed_numerical'] else '未通过'}；终止 {r['terminations']}；覆盖失败 {r['planning_failed']}。" for r in summary)+'\n')
print(json.dumps(summary))
