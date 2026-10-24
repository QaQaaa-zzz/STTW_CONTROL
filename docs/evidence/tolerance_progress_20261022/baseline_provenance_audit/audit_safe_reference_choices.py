"""Exploratory choice-space audit of old known-onset branches, zero physics.

Path distance is an offline evaluator over the full physical observation,
including task-ineligible fallback. It does not restore task eligibility.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_distance(xy, path):
    a, vector = path[:-1], np.diff(path, axis=0)
    squared = np.sum(vector**2, axis=1)
    valid = squared > 1e-12
    a, vector, squared = a[valid], vector[valid], squared[valid]
    if not len(vector):
        raise ValueError('empty geometric path')
    u = np.clip(np.sum((xy[:, None] - a) * vector, axis=2) / squared, 0, 1)
    foot = a + u[:, :, None] * vector
    d2 = np.min(np.sum((xy[:, None] - foot)**2, axis=2), axis=1)
    tangent = vector[-1] / np.sqrt(squared[-1])
    ray = path[-1] + np.maximum((xy - path[-1]) @ tangent, 0)[:, None] * tangent
    return np.sqrt(np.minimum(d2, np.sum((xy - ray)**2, axis=1)))


def audit(evaluation, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old = json.loads((evaluation/'analysis/analysis.json').read_text())
    manifest = json.loads((evaluation/'manifest.json').read_text())
    rows = json.loads((evaluation/'results.json').read_text())['cases']
    groups, raw, hashes = defaultdict(list), {}, {}
    for row in rows:
        path = evaluation/(row['name']+'.npz')
        assert digest(path) == old['source_raw_sha256'][str(path)]
        hashes[str(path)] = digest(path)
        with np.load(path) as z:
            data = {k:z[k] for k in z.files}
        raw[row['name']] = data
        with np.load(evaluation/(row['route']+'_path.npz')) as z:
            geometric_path = z['xy'].astype(float)
        complete = row['balance_label']=='NO_FLOOR_CONTACT' and row['full_observation']
        record = dict(route=row['route'],speed=row['speed'],alpha=row['alpha'],force_N=row['force_N'],
                      candidate=row['candidate'],name=row['name'],physical_label=row['balance_label'],
                      task_recovered=row['task_recovered'],domain_failure=row['domain_failure'],
                      complete_safe=complete,observed_seconds=len(data['time'])*.005,
                      snapshot_sha256=row['snapshot_sha256'])
        if complete:
            assert len(data['time'])==row['active_ticks']
            distance = path_distance(data['actual_xy'].astype(float), geometric_path)
            speed_error = data['actual_forward_speed'].astype(float)-row['speed']
            assert np.isfinite(distance).all() and np.isfinite(speed_error).all()
            record.update(offline_path_distance_rmse_m=float(np.sqrt(np.mean(distance**2))),
                          full_physical_speed_rmse_m_s=float(np.sqrt(np.mean(speed_error**2))),
                          max_offline_path_distance_m=float(distance.max()),
                          end_offline_path_distance_m=float(distance[-1]))
            raw[row['name']]['offline_distance']=distance
            raw[row['name']]['offline_speed_error']=speed_error
        groups[(row['route'],row['speed'],row['alpha'],row['force_N'])].append(record)
    summaries=[]
    for key, records in groups.items():
        assert len({r['snapshot_sha256'] for r in records})==1
        safe=[r for r in records if r['complete_safe']]
        if safe:
            assert len({r['observed_seconds'] for r in safe})==1
        windows=[]
        if safe:
            times=raw[safe[0]['name']]['time']
            n=len(times)
            exits={}
            for row in safe:
                data=raw[row['name']]
                np.testing.assert_array_equal(data['time'],times)
                hits=np.flatnonzero(data['domain_exit'])
                exits[row['candidate']]=int(hits[0]) if len(hits) else n
            boundaries=sorted({0,n,*exits.values()})
            for start,end in zip(boundaries[:-1],boundaries[1:]):
                measures=[]
                for row in safe:
                    data=raw[row['name']]
                    measures.append(dict(candidate=row['candidate'],
                                         task_domain_not_yet_exited=start<exits[row['candidate']],
                                         distance_squared_integral_m2_s=float(np.sum(data['offline_distance'][start:end]**2)*.005),
                                         speed_squared_integral_m2_per_s=float(np.sum(data['offline_speed_error'][start:end]**2)*.005)))
                windows.append(dict(start_task_s=float(times[start]),duration_s=(end-start)*.005,
                                    end_task_s=float(times[end-1])+.005,measurements=measures))
        metrics=('offline_path_distance_rmse_m','full_physical_speed_rmse_m_s')
        frontier=[a for a in safe if not any(all(b[k]<=a[k] for k in metrics) and any(b[k]<a[k] for k in metrics) for b in safe if b is not a)]
        summaries.append(dict(context=list(key),records=records,safe_candidates=[r['candidate'] for r in safe],common_clock_windows=windows,
                              diagnostic_nondominated=[r['candidate'] for r in frontier]))
    output.mkdir(exist_ok=False)
    metadata=[evaluation/'manifest.json',evaluation/'results.json',evaluation/'analysis/analysis.json',Path(__file__).resolve()]
    metadata += list(evaluation.glob('*_path.npz'))
    payload=dict(groups=summaries,raw_sha256=hashes,metadata_sha256={str(p):digest(p) for p in metadata},new_physical_ticks=0,fits=0,
                 source_role=manifest['information_role'],
                 scope='Posthoc known-onset DEV choice-space diagnostic. Finite safe candidate library, not optimality, current causal execution, full-task success, or alpha-conditioned selection evidence.')
    (output/'analysis.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    target=next(g for g in summaries if g['context']==['right90_R1p1',1.5,0.,-32.])
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    colors=dict(zip([r['candidate'] for r in target['records']],plt.cm.tab10.colors))
    all_xy=np.concatenate([raw[r['name']]['actual_xy'] for r in target['records']])
    low,high=all_xy.min(axis=0),all_xy.max(axis=0)
    padding=np.maximum((high-low)*.1,.5)
    with np.load(evaluation/'right90_R1p1_path.npz') as z:
        path=z['xy'];shown=path[:,0]<=max(raw[r['name']]['actual_xy'][:,0].max() for r in target['records'])+1
        axes[0].plot(*path[shown].T,'--',color='gray',label='original path')
    for row in target['records']:
        data=raw[row['name']];xy=data['actual_xy'];safe=row['complete_safe']
        color=colors[row['candidate']]
        axes[0].plot(*xy.T,label=row['candidate'],alpha=1 if safe else .4,color=color)
        axes[0].scatter(*xy[-1],marker='o' if safe else 'x',color=color)
        axes[1].plot(data['time'],data['actual_forward_speed'],label=row['candidate'],alpha=1 if safe else .4,color=color)
        if safe:
            axes[2].scatter(row['offline_path_distance_rmse_m'],row['full_physical_speed_rmse_m_s'],color=color)
            axes[2].annotate(row['candidate'],(row['offline_path_distance_rmse_m'],row['full_physical_speed_rmse_m_s']),xytext=(5,5),textcoords='offset points')
    axes[0].set(xlabel='x /m',ylabel='y /m',title='All five actual branches; x = contact')
    axes[0].set_xlim(low[0]-padding[0],high[0]+padding[0])
    axes[0].set_ylim(low[1]-padding[1],high[1]+padding[1])
    axes[0].set_aspect('equal',adjustable='box');axes[0].legend(fontsize=7)
    axes[1].axhline(1.5,ls='--',color='gray');axes[1].set(xlabel='task seconds',ylabel='speed /m/s',title='Unequal failed traces retained')
    axes[2].set(xlabel='offline path distance RMSE /m',ylabel='speed RMSE /m/s',title='Complete safe branches only; task not recovered')
    fig.suptitle('Old known-onset right-turn -32N branches; not the current causal gate')
    fig.savefig(output/'right32_all_xy_and_tradeoff.png');plt.close(fig)
    lines=['# 已知起扰分支中的安全参考选择机会', '', '[全部五支实际XY与代价图](right32_all_xy_and_tradeoff.png)', '',
           '目的：查明已测防摔候选是否存在不同速度/路径代价，而不是先假设容扰选择器有用途。读取45条旧分支、认证全部原始SHA，新增0积分/训练/拟合。结局和右弯两个候选已在探索中看到，属于事后分析。', '',
           '原始完整快照与已知起扰时刻均来自旧协议；这不是当前检测时刻锚定的因果控制结果。仅对完整无触地且观察时长相同的候选比较连续物理代价，触地候选仍在图上保留，不用短轨迹低成本参与排序。', '',
           '离线几何距离为实际XY到原始路径折线及末端前向切线的最短欧氏距离。它在任务离域后仍可计算，但不是原在线投影器、V3奖励重算或任务达标证据；fallback和任务失败不被抹去。', '',
           '|候选|完整观察s|无触地|原任务达标|离线路径距离RMSE m|速度RMSE m/s|', '|---|---:|---|---|---:|---:|']
    for row in target['records']:
        lines.append(f"|{row['candidate']}|{row['observed_seconds']:.3f}|{row['complete_safe']}|{row['task_recovered']}|{row.get('offline_path_distance_rmse_m','不比较')}|{row.get('full_physical_speed_rmse_m_s','不比较')}|")
    lines+=['', '## 相同物理时钟的离域前后分解', '',
            '按所有候选首次离域时刻的并集切分共同时间轴；每段双方持续时间相同，不能按各自不同长度前缀直接比较RMSE。积分不含任务失败尾罚。离域时刻由本tick日志标志定位，非控制模式重置。', '',
            '|共同窗起点s/长度s|候选|尚未离域|距离平方积分 m²s|速度误差平方积分 m²/s|', '|---|---|---|---:|---:|']
    for window in target['common_clock_windows']:
        for row in window['measurements']:
            lines.append(f"|{window['start_task_s']:.3f}/{window['duration_s']:.3f}|{row['candidate']}|{row['task_domain_not_yet_exited']}|{row['distance_squared_integral_m2_s']:.6f}|{row['speed_squared_integral_m2_per_s']:.6f}|")
    for row in target['records']:
        if row['complete_safe']:
            lines.append(f"\n{row['candidate']}最大/终点扩展几何距离：{row['max_offline_path_distance_m']:.6f}/{row['end_offline_path_distance_m']:.6f}m。")
    lines += ['', '分段裁决：两者都未离域及仅一方离域的共同窗口内，speed_plus两项积分均更低；主要取舍在双方离域后的hold尾段形成。故只保留应急退出代价差异，不据此称H75持续偏好选择或值得再开旧点转向候选物理。两个原任务都未恢复；1.25m平均偏离并不是路径恢复。停止该点选择提案，不追加候选或训练选择网络。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(output/'REPORT.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();audit(args.evaluation.resolve(),args.output.resolve())
