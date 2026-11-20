"""Compact outcome/funnel and paired, root-clustered comparisons for the fixed two maps."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(raw,out):
    data=pd.read_csv(raw/'online_cases.csv')
    pairs=pd.read_csv(raw/'matched_cases.csv')
    nroots=data.root.nunique()
    rng=np.random.default_rng(20261120)
    boot=rng.integers(0,nroots,(5000,nroots))
    counts=data.groupby(['layout','skill','planner']).agg(contracts=('success','size'),jump_and_lock=('verified','sum'),
        feasible=('oracle_feasible','sum'),complete=('success','sum'),wall=('collision','sum'))
    counts.to_csv(out/'p3_system.csv')
    stats=[]
    def estimate(label, a, b):
        for score in ('success','collision'):
            diff=a.groupby('root')[score].mean()-b.groupby('root')[score].mean()
            values=100*diff.to_numpy()
            lo,hi=np.percentile(values[boot].mean(1),[2.5,97.5])
            stats.append(dict(comparison=label,outcome=score,roots=nroots,delta_pp=values.mean(),lo_pp=lo,hi_pp=hi))
    for skill,part in data.groupby('skill'):
        for name,baseline in (('C-R','C'),('C-R','fixed'),('C','fixed')):
            estimate(skill+' '+name+' minus '+baseline,part[part.planner==name],part[part.planner==baseline])
    for name,part in data.groupby('planner'):
        estimate('Full RB minus Source '+name,part[part.skill=='rb_run1'],part[part.skill=='source'])
    pd.DataFrame(stats).to_csv(out/'p3_paired_ci.csv',index=False)
    valid=pairs[pairs.same_prefix].copy()
    valid['net']=valid.rescued.astype(int)-valid.lost.astype(int)
    valid.groupby(['layout','skill']).agg(pairs=('root','size'),changed=('changed','sum'),rescued=('rescued','sum'),
        lost=('lost','sum'),net=('net','sum')).to_csv(out/'p3_reselection.csv')
    online=data[(data.planner=='C-R')&(data.query_batch_ms>0)]
    print(counts.to_string())
    print('C-R query wall ms per queried lane (shared batched timing)',online.query_batch_ms.quantile([.5,.95]).to_dict())
    print('reselection',valid[['changed','rescued','lost','net']].sum().to_dict())
    print('mismatched prefixes',len(pairs)-len(valid))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();summarize(args.raw,args.out)
