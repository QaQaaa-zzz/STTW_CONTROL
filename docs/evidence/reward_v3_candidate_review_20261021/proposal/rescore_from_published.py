"""Exact UNDISCOUNTED v3 rescoring using published endpoint component integrals.

Usage:
 python rescore_from_published.py --audit path/to/results.json --output output_dir

No original NPZ is needed for the integrated return of this deliberately affine
reward. DISCOUNTED v3 returns cannot be recovered from these integrals and are
reported as unknown, not substituted with undiscounted or old discounted values.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import math

ALPHAS=(0.,.25,.5,.75,1.)
def convert(audit: dict, eps: float=.1) -> list[dict]:
    if not 0 < eps <= 1: raise ValueError('eps out of range')
    c=audit['scoring_config']
    if c.get('reward_version') is not None:
        raise ValueError('Expected the pinned UNVERSIONED Coverage scoring formula.')
    if c.get('roll_weight')!=80 or c.get('working_roll_weight')!=240:
        raise ValueError('Unexpected safety weights')
    if c.get('failure_cost_rate_bound')!=8980.2:
        raise ValueError('Unexpected failure bound')
    grouped={}
    for r in audit['records']:
        key=(r['dataset'],r['case'],r['method'])
        bucket=grouped.setdefault(key,{})
        a=float(r['score_alpha'])
        if a in bucket: raise ValueError(f'Duplicate {key} alpha={a}')
        bucket[a]=r
    out=[]
    for key,g in sorted(grouped.items()):
        if 0. not in g or 1. not in g:raise ValueError(f'Missing endpoints: {key}')
        r0,r1=g[0.],g[1.]
        # Do not mix different physical trajectories or mismatched failure tails.
        for k in ('samples','failed','final_hold','behavior_alpha','physical'):
            if r0[k]!=r1[k]:raise ValueError(f'Physical mismatch: {key}, {k}')
        common=set(r0['components'])-{'path','speed','primary'}
        for k in common:
            if not math.isclose(r0['components'][k],r1['components'][k],rel_tol=1e-9,abs_tol=1e-7):
                raise ValueError(f'Common cost mismatch: {key}, {k}')
        for r in (r0,r1):
            if not math.isclose(sum(r['components'].values()),r['total_return'],abs_tol=2e-7,rel_tol=1e-8):
                raise ValueError(f'Component reconciliation failed: {key}')
        g0=r0['total_return']
        g1=r1['total_return']-r1['components']['path']+eps*r0['components']['path']
        for a in ALPHAS:
            out.append(dict(dataset=key[0],case=key[1],method=key[2],score_alpha=a,
                old_total_return=g.get(a,{}).get('total_return'),
                new_total_return=(1-a)*g0+a*g1,
                failed=r0['failed'],final_hold=r0['final_hold'],
                old_discounted_return=g.get(a,{}).get('discounted_return'),
                new_discounted_return=None,
                validation_scope='exact integrated-component transformation; no dynamics or PPO rerun'))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--eps',type=float,default=.1)
    a=p.parse_args();obj=json.loads(a.audit.read_text());rows=convert(obj,a.eps)
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'new_aggregate_returns.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lookup={(r['dataset'],r['case'],r['method'],r['score_alpha']):r for r in rows}
    pairs=[]
    for r in rows:
        if r['method']!='policy0':continue
        key=(r['dataset'],r['case'],'policy1',r['score_alpha'])
        if key not in lookup:continue
        s=lookup[key]
        pairs.append(dict(dataset=r['dataset'],case=r['case'],score_alpha=r['score_alpha'],
            new_delta_policy1_minus_policy0=s['new_total_return']-r['new_total_return'],
            joint_final_hold=r['final_hold'] and s['final_hold'],
            any_failure=r['failed'] or s['failed']))
    with (a.output/'new_pair_deltas.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=pairs[0]);w.writeheader();w.writerows(pairs)
    summary=dict(rescores=len(rows),physical_trajectories=len(rows)//len(ALPHAS),
        eps=a.eps,training_updates=0,simulator_steps=0,
        new_discounted_returns_verified=False,
        note='Do not interpret the old high-demand policy labels as successful intended endpoint behavior.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
