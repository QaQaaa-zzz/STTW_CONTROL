"""Score independent TEST reversals and paired system effects, clustering by the 40 initial roots."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from report_platform_online_review import report
from run_platform_task_oracle import read


def summarize(out):
    report(out)
    plan=read(out/'test_freeze_manifest.json');root_ids=plan['roots']
    system=pd.read_csv(out/'online_cases.csv')
    oracle=pd.read_csv(out/'oracle_cases.csv').query('force==0')
    keys=['layout','skill','section','root']
    actions=oracle.pivot_table(index=keys+['approach'],columns='side',values='success',aggfunc='any').rename(columns={1:'left',-1:'right'})
    info=oracle.groupby(keys+['approach']).agg(verified=('verified','any'),prefixes=('prefix','nunique'))
    actions=actions.join(info)
    root=actions.assign(left_only=actions.left&~actions.right,right_only=actions.right&~actions.left,
                        universal=actions.left&actions.right).groupby(keys).agg(verified=('verified','sum'),
                        prefixes=('prefixes','max'),left_only=('left_only','any'),right_only=('right_only','any'),
                        universal=('universal','any'),can_left=('left','any'),can_right=('right','any'))
    root['eligible']=(root.verified>=2)&(root.prefixes==1)
    root['strict']=root.eligible&root.left_only&root.right_only&~root.universal
    root['oracle_goals']=root.can_left.astype(int)+root.can_right.astype(int)
    fixed=system.query("force==0 and planner=='fixed'").groupby(keys).success.sum()
    root['fixed_goals']=fixed;root.to_csv(out/'h1_roots.csv')
    actions.to_csv(out/'h1_approaches.csv')
    rng=np.random.default_rng(plan['root_bootstrap_seed']);draw=rng.integers(0,len(root_ids),(10000,len(root_ids)))
    rows=[]
    def interval(labels,index,num,den):
        a=pd.DataFrame({'root':np.asarray(index),'num':np.asarray(num,dtype=float),'den':np.asarray(den,dtype=float)}).groupby('root').sum().reindex(root_ids,fill_value=0)
        n,d=a.num.to_numpy(),a.den.to_numpy();sd=d[draw].sum(1);sn=n[draw].sum(1)
        ok=sd>0;ci=np.quantile(sn[ok]/sd[ok],[.025,.975]) if ok.any() else (np.nan,np.nan)
        rows.append(dict(**labels,numerator=n.sum(),denominator=d.sum(),rate=n.sum()/d.sum() if d.sum() else np.nan,
                         low=ci[0],high=ci[1]))
    def metric(g,name,value,den,**labels):
        interval(dict(metric=name,**labels),g.root,value,np.broadcast_to(den,len(g)))
    h=root.reset_index()
    for (skill,section),g in h.groupby(['skill','section']):
        for name in ('eligible','strict'):metric(g,name,g[name],1,skill=skill,section=section)
        metric(g,'oracle_minus_fixed',g.oracle_goals-g.fixed_goals,2,skill=skill,section=section)
    any_h=h.groupby(['skill','layout','root'])[['strict','eligible']].any().reset_index()
    for skill,g in any_h.groupby('skill'):
        metric(g,'strict_any_section',g.strict,1,skill=skill)
    for (skill,planner),group in system.groupby(['skill','planner']):
        for f,g in [('all',group),*group.groupby('force')]:
            ctx=dict(skill=skill,planner=planner,force=f)
            for name in ['success','collision']:metric(g,name,g[name],1,**ctx)
            admitted=g.query_feasible & (g.query_batch_ms>0);veto=~g.query_feasible & (g.query_batch_ms>0)
            for name,source in [('admitted_success',g.success),('admitted_wall',g.collision)]:
                metric(g,name,admitted&source,admitted,**ctx)
            applied=admitted & (g.departure_applied_tick>=0)
            metric(g,'admitted_and_applied_wall',applied&g.collision,applied,**ctx)
            metric(g,'admitted_expired_before_apply',admitted&~applied,admitted,**ctx)
            metric(g,'veto_oracle_unsolvable',veto&~g.oracle_feasible,veto,**ctx)
    paired=pd.read_csv(out/'matched_cases.csv')
    for (skill,force),g in paired.groupby(['skill','force']):
        if not g.same_prefix.all():raise ValueError('C / C-R did not share their pre-handoff trajectory')
        for name,value in [('rescued',g.rescued),('lost',g.lost),('wall_reduction',g.c_wall.astype(int)-g.cr_wall.astype(int))]:
            metric(g,name,value,1,skill=skill,force=force)
    sk=['layout','root','section','side','force','planner']
    rb=system[system.skill=='rb_run1'].merge(system[system.skill=='source'],on=sk,suffixes=('','_source'),validate='one_to_one')
    for (method,force),g in rb.groupby(['planner','force']):
        metric(g,'R9_minus_Source',g.success.astype(int)-g.success_source.astype(int),1,planner=method,force=force)
    pd.DataFrame(rows).to_csv(out/'blind_metrics.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    summarize(p.parse_args().out)
