"""Compare two frozen C-R approach libraries on paired left-gate assignments."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from platform_alpha_states import development_states
from prepare_blind_platform import project, sha
from report_platform_online_review import report
from run_blind_platform import execute
from run_platform_task_oracle import read
from run_playground_wide_gap_oracle import save

ACTIONS = [(0,0.),(1,0.),(2,0.),(1,.5),(1,1.)]
ARMS = ['alpha0','alpha_enabled']
EXITS = ['left','straight','right']


def assignments(table, bank, root, sections, margin):
    ids=np.arange(8,38)
    features=np.column_stack((bank['qpos'][:,1:2],bank['qpos'][:,3:7],bank['qvel'][:,:6]))
    distances=((features[ids]-features[root])/features[ids].std(0).clip(.005))**2
    near=ids[np.argsort(distances.sum(1))[:5]]
    choices=[]
    for section in sections:
        gate=table[section].between(.06,3.9)
        safe=table.clear>margin
        value=16*(safe&gate&table.reachable).astype(float)+4*safe+2*gate+table.reachable.astype(float)+.01*table.clear.clip(-1,1)
        grid=table.assign(value=value).pivot(index='root',columns=['candidate','departure'],values='value')
        grid=grid.reindex(index=near,columns=pd.MultiIndex.from_product([range(5),EXITS])).fillna(-4)
        score=grid.mean().to_numpy()
        for arm,size in zip(ARMS,(9,15)):
            i,d=divmod(int(score[:size].argmax()),3)
            a,alpha=ACTIONS[i]
            choices.append(('C-R',section,1,a,d,alpha,arm))
    choices.extend(('oracle','near',1,a,d,alpha,'oracle') for (a,alpha),d in product(ACTIONS,range(3)))
    return choices


def score(out, arms=ARMS):
    report(out)
    rows=pd.read_csv(out/'online_cases.csv')
    index=['layout','root','force','section']
    records=[]
    for skill,g in rows.groupby('skill'):
        for layout,part in [('all',g),*g.groupby('layout')]:
            pivot=part.pivot(index=index,columns='planner',values=['success','collision'])
            for metric in ['success','collision']:
                v=pivot[metric].astype(int)
                delta=v[arms[1]]-v[arms[0]]
                byroot=delta.groupby(level='root').mean().to_numpy()
                draws=np.random.default_rng(20261120).choice(byroot,(10000,len(byroot)))
                lo,hi=np.quantile(draws.mean(1),[.025,.975])
                records.append((skill,layout,metric,len(v),*v[arms].sum(),int(delta.sum()),lo,hi))
    pd.DataFrame(records,columns=['skill','layout','metric','contracts',*arms,'net','delta_low','delta_high']).to_csv(out/'paired_metrics.csv',index=False)


def main(base, transfer=False):
    library=base/'post_review/alpha_library'
    out=library.with_name('alpha_transfer') if transfer else library
    out.mkdir(exist_ok=True)
    latency=pd.read_csv(library/'timing/latency.csv').query("method=='phase2_relative' and candidates==15").iloc[0]
    if not (latency.n==1000 and latency.horizon_steps==255 and latency.p95_ms<=15 and latency.decision_mismatches==0 and latency[['max_pose_abs','max_velocity_abs','max_domain_abs']].max()<1e-5):
        raise ValueError('15-candidate timing or parity exceeded the frozen budget')
    parent=base/'post_review/realtime_15ms'
    plan=read(parent/'test_freeze_manifest.json')
    plan.update(role='T3 paired post-TEST alpha ablation',parent_manifest=str(parent/'test_freeze_manifest.json'),
                parent_sha256=sha(parent/'test_freeze_manifest.json'),pair=ARMS,lanes_per_assignment=21,
                assignment_batches=[list(product(plan['roots'],plan['forces_N']))],dev_roots=list(range(8,38)),
                timings_ms={**plan['timings_ms'],'15':float(latency.p95_ms)},scope='Left gates. Four 2520-lane batches; two C-R arms with the same 30-root 5NN and 15 physical oracle branches.')
    if transfer:
        walls=read(base/'post_review/wall_layouts/test_freeze_manifest.json')
        plan.update(maps=walls['maps'],roots=walls['roots'],skills=['rb_run1'],
                    assignment_batches=[list(product(walls['roots'],plan['forces_N']))],
                    role='Frozen T3 library on previously evaluated T2 geometries',
                    parent_manifest=str(library/'test_freeze_manifest.json'),
                    parent_sha256=sha(library/'test_freeze_manifest.json'),
                    scope='R9 left gates only; three batches x 1890 lanes, same 30 DEV calibration states and both T3 arms.')
    save(out/'test_freeze_manifest.json',plan)
    project(out,parent/'test_a',development_states(base/'phase4_alpha'))
    execute(out,assignments)
    score(out)
    if transfer:
        cases = pd.read_csv(out/'online_cases.csv')
        veto = (cases.query_batch_ms > 0) & ~cases.query_feasible
        cases.assign(admitted_done=cases.query_feasible & cases.success,
                     admitted_wall=cases.query_feasible & cases.collision,
                     veto=veto, veto_wall=veto & cases.collision).groupby(['layout', 'planner']).agg(
            contracts=('root', 'size'), completed=('success', 'sum'), collisions=('collision', 'sum'),
            admitted=('query_feasible', 'sum'), admitted_done=('admitted_done', 'sum'),
            admitted_wall=('admitted_wall', 'sum'), vetoed=('veto', 'sum'),
            wall_after_veto=('veto_wall', 'sum')).to_csv(out/'admission_funnel.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('--transfer',action='store_true')
    a=p.parse_args();main(a.base,a.transfer)
