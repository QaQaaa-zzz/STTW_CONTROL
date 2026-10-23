"""Analyze saved completed pulse panels; no simulation or model changes."""
import argparse
import json
from pathlib import Path
import numpy as np


def analyze(root, output):
    root, output = Path(root), Path(output)
    if json.loads((root/'status.json').read_text())['phase'] != 'completed':
        raise ValueError('evaluation incomplete')
    protocol = json.loads((root/'manifest.json').read_text())['protocol']
    rows = json.loads((root/'results.json').read_text())['cases']
    keys = ('policy', 'alpha', 'route', 'speed', 'onset_s', 'force_N')
    lookup = {tuple(x[k] for k in keys): x for x in rows}
    if len(lookup) != len(rows) or len(rows) != protocol['total_cases']:
        raise ValueError('duplicate or missing cases')
    dt, horizon = .005, protocol['episode_s']
    cases = []
    for row in rows:
        with np.load(root/(row['name']+'.npz')) as d:
            t = d['time']; tail = t >= horizon-1-1e-5
            complete = len(t) == round(horizon/dt) and tail.sum() == round(1/dt)
            errors = dict(path=d['path_cross_track'], heading=d['path_heading_error'],
                          speed=d['actual_forward_speed']-d['v_user'])
            peaks = {k: float(np.max(np.abs(v[tail]))) if tail.any() else None
                     for k,v in errors.items()}
            practical = bool(complete and row['goal'] and not row['physical_failure']
                and not row['domain_failure'] and row['peak_roll'] <= .33
                and peaks['path'] <= .15 and peaks['heading'] <= .05 and peaks['speed'] <= .1)
            item = {k: row[k] for k in keys}
            item.update(original_success=row['success'], practical_success=practical,
                        tail_complete=bool(complete), tail_max=peaks,
                        prefix_valid=row['prefix_valid'], peak_roll=row['peak_roll'])
            if row['force_N']:
                nominal_key = tuple(0 if k == 'force_N' else row[k] for k in keys)
                nominal = lookup[nominal_key]
                with np.load(root/(nominal['name']+'.npz')) as z:
                    n = min(len(t),len(z['time']))
                    if not np.allclose(t[:n],z['time'][:n],atol=1e-6):
                        raise ValueError('unmatched time grids')
                    mask = (t[:n]>=row['onset_s']-1e-5)&(t[:n]<row['onset_s']+2-1e-5)
                    item['window_complete'] = int(mask.sum()) == round(2/dt)
                    for field in ('phi','path_cross_track','actual_forward_speed'):
                        delta = d[field][:n][mask]-z[field][:n][mask]
                        item[field+'_delta_rms'] = float(np.sqrt(np.mean(delta**2))) if len(delta) else None
            cases.append(item)
    policies = protocol['policies']
    context = lambda x: tuple(x[k] for k in ('alpha','route','speed','onset_s','force_N'))
    complete_sets = [{context(x) for x in cases if x['policy']==p and x['force_N']
                     and x['window_complete'] and x['prefix_valid']} for p in policies]
    shared = set.intersection(*complete_sets)
    summary=[]
    for alpha in protocol['endpoints']:
        for policy in policies:
            group=[x for x in cases if x['alpha']==alpha and x['policy']==policy]
            response=[x for x in group if x['force_N'] and context(x) in shared]
            result=dict(alpha=alpha,policy=policy,n=len(group),
                original_success=sum(x['original_success'] for x in group),
                practical_success=sum(x['practical_success'] for x in group),
                incomplete_tail=sum(not x['tail_complete'] for x in group),
                invalid_prefix=sum(not x['prefix_valid'] for x in group),
                shared_response_n=len(response),
                peak_roll_p95=float(np.quantile([x['peak_roll'] for x in group],.95)))
            for field in ('phi','path_cross_track','actual_forward_speed'):
                result[field+'_delta_rms_mean']=float(np.mean([x[field+'_delta_rms'] for x in response])) if response else None
            summary.append(result)
    output.mkdir(parents=True,exist_ok=False)
    result=dict(protocol='Posthoc practical tail sensitivity: path .15m, heading .05rad, speed .1m/s, roll .33rad; complete goal and no physical/domain failure. Same-time first2s response minus own nominal, shared valid-prefix complete contexts across ALL compared policies. Includes progress differences, not identical-state causal comparison. Used development panel; original labels unchanged.',summary=summary,cases=cases)
    (output/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Practical recovery and own-nominal response','',result['protocol'],'',
           '| Alpha | Policy | Original | Practical | Shared response n | Roll delta RMS | Path delta RMS |',
           '|---|---|---:|---:|---:|---:|---:|']
    for x in summary:
        def fmt(key): return f"{x[key]:.6f}" if x[key] is not None else 'NA'
        lines.append(f"| {x['alpha']} | {x['policy']} | {x['original_success']}/{x['n']} | {x['practical_success']}/{x['n']} | {x['shared_response_n']} | {fmt('phi_delta_rms_mean')} | {fmt('path_cross_track_delta_rms_mean')} |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(analyze(a.run,a.output),indent=2))
