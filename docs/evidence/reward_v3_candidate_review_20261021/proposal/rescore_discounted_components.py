"""Complete v3 fixed-trajectory rescore from the user's saved 18 MiB NPZ.

No simulation/training. This file was not run on the user's unavailable NPZ here.
The NPZ is produced by the pinned repository audit analyze.py.
Use together with its results.json to check identities, completeness, and sums.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from rescore_from_published import convert,ALPHAS

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--components',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--eps',type=float,default=.1)
    a=p.parse_args();audit=json.loads(a.audit.read_text())
    if audit['discount']!={'gamma':.997,'interval_s':.02,'from_reset':True}:
        raise ValueError('Unexpected discount contract; inspect before adapting.')
    expected=convert(audit,a.eps)
    exp={(r['dataset'],r['case'],r['method'],r['score_alpha']):r for r in expected}
    native={(r['dataset'],r['case'],r['method'],float(r['score_alpha'])):r for r in audit['records']}
    triples=sorted({k[:3] for k in exp});rows=[]
    with np.load(a.components,allow_pickle=False) as z:
        for group,case,method in triples:
            prefix=f'{group}/{case}/{method}'
            def original(score):
                r=native[(group,case,method,float(score))]
                arrays={k:np.asarray(z[f'{prefix}/score{score:g}/{k}'],dtype=float) for k in r['components']}
                if any(v.ndim!=1 or len(v)!=r['samples'] or not np.isfinite(v).all() for v in arrays.values()):
                    raise ValueError(f'Invalid component arrays at {prefix}')
                discount=.997**(np.arange(r['samples'])//4)
                total=sum(arrays.values())
                if not np.isclose(total.sum(),r['total_return'],atol=.002,rtol=1e-6):
                    raise ValueError(f'Integral mismatch at {prefix} alpha={score}')
                if not np.isclose(total@discount,r['discounted_return'],atol=.002,rtol=1e-6):
                    raise ValueError(f'Discount mismatch at {prefix} alpha={score}')
                return total,arrays,discount
            old0,c0,discount=original(0.)
            old1,c1,_=original(1.)
            for k in set(c0)-{'speed','path','primary'}:
                if not np.allclose(c0[k],c1[k],atol=1e-7,rtol=1e-6):
                    raise ValueError(f'Common component changed with alpha: {prefix}/{k}')
            new0=old0
            new1=old1-c1['path']+a.eps*c0['path']
            for alpha in ALPHAS:
                result=(1-alpha)*new0+alpha*new1
                integrated=float(result.sum());e=exp[(group,case,method,alpha)]
                if not np.isclose(integrated,e['new_total_return'],atol=.003,rtol=2e-6):
                    raise ValueError('v3 integral does not reconcile with independent published transform')
                rows.append(dict(dataset=group,case=case,method=method,score_alpha=alpha,
                    new_total_return=integrated,new_discounted_return=float(result@discount),
                    failed=e['failed'],final_hold=e['final_hold']))
    bykey={(r['dataset'],r['case'],r['method'],r['score_alpha']):r for r in rows}
    pairs=[]
    for r in rows:
        if r['method']!='policy0':continue
        s=bykey[(r['dataset'],r['case'],'policy1',r['score_alpha'])]
        pairs.append(dict(dataset=r['dataset'],case=r['case'],score_alpha=r['score_alpha'],
            delta_return=s['new_total_return']-r['new_total_return'],
            delta_discounted_return=s['new_discounted_return']-r['new_discounted_return'],
            jointly_final_hold=bool(r['final_hold'] and s['final_hold']),
            any_failed=bool(r['failed'] or s['failed'])))
    a.output.mkdir(parents=True,exist_ok=True)
    for filename,data in [('new_returns.csv',rows),('new_pairs.csv',pairs)]:
        with (a.output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
    anchors=[p for p in pairs if p['dataset']=='coverage_dual_u200' and '90_' in p['case'] and p['score_alpha'] in (0,1)]
    passed=len(anchors)==8 and all((p['delta_discounted_return']<0 if p['score_alpha']==0 else p['delta_discounted_return']>0) for p in anchors)
    summary=dict(physical_trajectories=len(triples),rescores=len(rows),simulator_steps=0,training_updates=0,
        discounted_historical_endpoint_rankings_passed=passed,
        validation_scope='saved step-cost reweighting, not new trajectories; goal and recovery deadlines require original physical NPZ')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    if not passed:raise SystemExit('Discounted endpoint gate did NOT pass; do not start curriculum automatically.')
if __name__=='__main__':main()
