"""Classify recorded termination signals separately from task constraint violations."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation', type=Path, required=True)
p.add_argument('--config', type=Path, required=True)
a = p.parse_args()
spec = json.loads(a.config.read_text())
index = json.loads((a.evaluation/'index.json').read_text())
metrics = json.loads((a.evaluation/'gravel_task_metrics.json').read_text())
rows = []
for case in index:
    run = Path(case['run'])
    contract = json.loads((run/'contract.json').read_text())
    with np.load(run/'trajectory.npz') as d:
        group = [r for r in metrics if (r['terrain'], r['speed_m_s'], r['parameter_seed']) == (case['terrain'], case['user_speed_m_s'], case['seed'])]
        for r in group:
            lane = r['lane']
            trace = d['lower_trace'][:, :, lane].reshape(-1, 11)
            ids = np.flatnonzero(trace[:, 6])
            times = ids*.005 + trace[ids, 7]*contract['physics_dt']
            signals = []
            signal_times = {}
            if r['failed']:
                step = int(np.flatnonzero(d['failure'][:, lane])[0])
                window = np.arange(4*step, 4*(step+1))
                for name, mask in [('domain_exit', d['domain_exit'][window, lane]), ('policy_fault', d['policy_fault'][window, lane]), ('roll_threshold_exceeded', abs(d['measurement'][window, lane, 0]) > .7)]:
                    hits = window[np.flatnonzero(mask)]
                    if len(hits):
                        signals.append(name)
                        i = int(hits[0])
                        signal_times[name] = float(i*.005 + trace[i, 7]*contract['physics_dt'])
                if not signals:
                    signals.append('unresolved_from_recorded_flags')
            task_reasons = {}
            for name, task in spec['tasks'].items():
                exceed = abs(trace[ids, 3]) > task['W_m']
                reasons = []
                for flag, reason in [(r['failed'], 'early_termination'), (not r['goal_crossed'], 'goal_not_reached'), (not r['final_hold'], 'final_hold_not_met'), (bool(exceed.any()), 'peak_path_limit_exceeded'), (r['speed_rmse_m_s'] > task['speed_rmse_m_s'], 'speed_rmse_limit_exceeded')]:
                    if flag:
                        reasons.append(reason)
                assert bool(r['task_'+name]) == (not reasons)
                task_reasons[name] = dict(reasons=reasons, first_path_exceed_s=float(times[np.flatnonzero(exceed)[0]]) if exceed.any() else None, path_exceed_sampled_s=float(exceed.sum()*.005))
            rows.append(dict(**r, termination_signals=signals, first_signal_observed_s=signal_times, task_reasons=task_reasons))
counts = Counter(signal for r in rows for signal in r['termination_signals'])
summary = dict(conditions=len(rows), completed_horizon=sum(not r['failed'] for r in rows), early_terminations=sum(r['failed'] for r in rows), nonexclusive_terminal_signals=dict(counts), completed_horizon_no_ABC=sum(not r['failed'] and not any(r['task_'+k] for k in spec['tasks']) for r in rows))
output = dict(scope='Signals in first failed20ms policy interval; co-occurrence retained, not unique causal attribution. Roll threshold is0.7rad, not proof of ground impact. Unresolved may include contact/numerical failure. Task path duration uses5ms samples. Task nonachievement after early termination is not an independent physical cause.', summary=summary, conditions=rows)
(a.evaluation/'outcome_reasons.json').write_text(json.dumps(output, indent=2)+'\n')
lines = ['## 终止与任务超限分开统计', '', output['scope'], '', str(summary), '', '|地形|速度|seed|方法|观察秒数|终止信号|未到达|末保持未达|任务A/B/C不合格原因|', '|---|---:|---:|---|---:|---|---|---|---|']
for r in rows:
    reasons = '; '.join(k+':'+(','.join(v['reasons']) or 'qualified') for k,v in r['task_reasons'].items())
    lines.append(f"|{r['terrain']}|{r['speed_m_s']}|{r['parameter_seed']}|{r['method']}|{r['observed_seconds']:.4f}|{','.join(r['termination_signals']) or '完整观察'}|{not r['goal_crossed']}|{not r['final_hold']}|{reasons}|")
lines += ['', '[逐条原因、首次信号时刻与路径超限时长](outcome_reasons.json)。运行域domain_exit不是A/B/C路径阈值；未到达/未保持与提前终止可同时出现，不重复相加作为独立失败数。']
report = a.evaluation/'RESULTS.md'
report.write_text(report.read_text().split('\n\n## 终止与任务超限分开统计')[0]+'\n\n'+'\n'.join(lines)+'\n')
print(json.dumps(summary))
