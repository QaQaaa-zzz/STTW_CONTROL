"""Build a relock-state exit bank, replay it by held-out root, and test frozen arms online."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sttw_control.platform_fallback import VetoExitBank, handback_features

SECTIONS = ['near', 'mid', 'far']
ARMS = ['expanded', 'veto_bank']


def read(path):
    return json.loads(path.read_text())


def build_bank(oracle, folder):
    records = oracle[oracle.verified]
    by_prefix = records.drop_duplicates('prefix').set_index('prefix')
    q, v = [], []
    for row in by_prefix.itertuples():
        tick = round(row.handoff_s / .005) + 1
        with np.load(row.trajectory) as z:
            q.append(z['qpos'][tick])
            v.append(z['qvel'][tick])
    wall = records.drop_duplicates(['prefix','planned_dep']).pivot(index='prefix', columns='planned_dep',values='collision')
    done = records.pivot(index='prefix', columns=['section','planned_dep'], values='success')
    done = done.reindex(index=by_prefix.index, columns=pd.MultiIndex.from_product([SECTIONS,range(3)]))
    assert not done.isna().any().any()
    bank = dict(state=handback_features(torch.tensor(np.stack(q)),torch.tensor(np.stack(v))).float().numpy(),
                root=by_prefix.root.to_numpy(), route=by_prefix.approach.to_numpy(),
                wall=wall.loc[by_prefix.index,[0,1,2]].to_numpy(dtype=np.int64),
                done=done.to_numpy(dtype=np.int64).reshape(-1,3,3))
    np.savez(folder/'exit_bank.npz', **bank)
    return bank, pd.Series(np.arange(len(by_prefix)),index=by_prefix.index)


def audit(source, out, plan):
    oracle = pd.read_csv(source/'oracle_cases.csv').query('side==1')
    assigned = pd.read_csv(source/'online_cases.csv').query("planner=='alpha_enabled'").copy()
    assigned['replay_dep'] = assigned.chosen_dep
    assigned['veto'] = (assigned.query_batch_ms>0) & ~assigned.query_feasible
    assigned['avoidable_wall'] = False
    for layout in plan['maps']:
        folder = out/layout
        folder.mkdir(exist_ok=True)
        bank, ids = build_bank(oracle[oracle.layout==layout],folder)
        veto = assigned[assigned.veto & (assigned.layout==layout)]
        for root, part in veto.groupby('root'):
            selector = VetoExitBank(bank, 'cpu', bank['root']!=root)
            pos = ids.loc[part.prefix].to_numpy()
            section = torch.tensor([SECTIONS.index(s) for s in part.section])
            dep = selector.choose(torch.tensor(bank['state'][pos]),torch.tensor(bank['route'][pos]),
                                  section,torch.tensor(part.chosen_dep.to_numpy())).numpy()
            assigned.loc[part.index,'replay_dep'] = dep
            assigned.loc[part.index,'avoidable_wall'] = part.collision & ~bank['wall'][pos].all(1)
    keys = ['layout','root','force','section','approach','approach_alpha']
    branches = oracle[keys+['planned_dep','prefix','success','collision']]
    for choice in ['chosen_dep','replay_dep']:
        matched = assigned.merge(branches,left_on=keys+[choice],right_on=keys+['planned_dep'],
                                 validate='many_to_one',suffixes=('','_branch'))
        assert len(matched)==len(assigned) and matched.prefix.eq(matched.prefix_branch).all()
        if choice=='chosen_dep':
            assert matched.success.eq(matched.success_branch).all() and matched.collision.eq(matched.collision_branch).all()
        else:
            matched[keys+['chosen_dep','replay_dep','query_feasible','veto','avoidable_wall','success','success_branch',
                          'collision','collision_branch']].to_csv(out/'development_pairs.csv',index=False)
            matched.groupby('layout').agg(contracts=('root','size'),veto=('veto','sum'),done=('success','sum'),
                fallback_done=('success_branch','sum'),wall=('collision','sum'),fallback_wall=('collision_branch','sum'),
                avoidable_wall=('avoidable_wall','sum')).to_csv(out/'development_summary.csv')


def freeze(source,out):
    from prepare_blind_platform import sha
    plan = read(source/'test_freeze_manifest.json')
    audit(source,out,plan)
    roots = list(range(1071,1121))
    with np.load(plan['bank']) as z:
        states = np.column_stack((z['qpos'],z['qvel']))
    with np.load(Path(plan['wide'])/'initial_bank.npz') as z:
        old = np.column_stack((z['qpos'],z['qvel']))
    seen = {r.tobytes() for r in np.concatenate((old,states[1001:1071]))}
    test = {r.tobytes() for r in states[roots]}
    assert len(test)==len(roots) and not seen.intersection(test)
    for layout in plan['maps']:
        for name in ('rb_run1_prior.csv','rb_run1_initializer.npz','terrain.npz'):
            target = out/layout/name
            if not target.exists():
                target.symlink_to(source/layout/name)
    banks={name:str(out/name/'exit_bank.npz') for name in plan['maps']}
    plan.update(roots=roots, assignment_batches=[list(product(roots,plan['forces_N']))],pair=ARMS,
                role='New initial states on previously calibrated wall layouts',veto_banks=banks,
                veto_fit_roots=read(source/'test_freeze_manifest.json')['roots'],
                bank_hashes={k:sha(Path(v)) for k,v in banks.items()},
                parent_manifest=str(source/'test_freeze_manifest.json'),parent_sha256=sha(source/'test_freeze_manifest.json'),
                scope=(out/'protocol.md').read_text())
    (out/'test_freeze_manifest.json').write_text(json.dumps(plan,indent=2)+'\n')


def assignments(*args):
    from run_platform_alpha_library import assignments as expanded_actions
    choices=expanded_actions(*args)
    return [(*c[:-1],arm) for c in choices if c[-1]=='alpha_enabled' for arm in ARMS] + [c for c in choices if c[-1]=='oracle']


def run(out):
    from run_blind_platform import execute
    from run_platform_alpha_library import score
    plan = read(out/'test_freeze_manifest.json')
    torch.set_num_threads(1)
    with np.load(next(iter(plan['veto_banks'].values()))) as bank:
        selector=VetoExitBank(bank,'cuda')
        state,route=selector.state[:1],selector.route[:1]
        args=(state,route,torch.zeros_like(route),torch.zeros_like(route))
        for _ in range(50):
            selector.choose(*args)
        import time
        torch.cuda.synchronize()
        milliseconds=[]
        for _ in range(1000):
            t=time.perf_counter()
            selector.choose(*args)
            torch.cuda.synchronize()
            milliseconds.append((time.perf_counter()-t)*1000)
        (out/'fallback_latency.json').write_text(json.dumps(dict(calls=len(milliseconds),single_robot_lookup_p95_ms=float(np.percentile(milliseconds,95))))+'\n')
    execute(out,assignments)
    score(out,ARMS)


    cases=pd.read_csv(out/'online_cases.csv')
    keys=['layout','root','force','section']
    paired=cases[cases.planner==ARMS[0]].merge(cases[cases.planner==ARMS[1]], on=keys,
                                               suffixes=('_old','_new'),validate='one_to_one')
    same_prefix=paired.prefix_old.eq(paired.prefix_new)
    veto=(paired.query_batch_ms_old>0) & ~paired.query_feasible_old
    switched=paired.chosen_dep_old!=paired.chosen_dep_new
    same_votes=paired.query_feasible_old.eq(paired.query_feasible_new)
    metrics=pd.read_csv(out/'paired_metrics.csv').query("layout=='all'").set_index('metric')
    result=dict(contracts=len(paired),matched_prefixes=int(same_prefix.sum()),same_admissions=int(same_votes.sum()),
                switches=int(switched.sum()),switches_outside_veto=int((switched & ~veto).sum()),
                completion_gains=int((~paired.success_old & paired.success_new).sum()),
                completion_losses=int((paired.success_old & ~paired.success_new).sum()),
                avoided_contacts=int((paired.collision_old & ~paired.collision_new).sum()),
                new_contacts=int((~paired.collision_old & paired.collision_new).sum()))
    result['meets_paired_outcome_criterion']=bool(same_prefix.all() and same_votes.all() and not (switched & ~veto).any()
        and metrics.at['collision','delta_high']<0 and metrics.at['success','delta_low']>-.05)
    (out/'decision.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','run'])
    p.add_argument('out',type=Path)
    a=p.parse_args()
    if a.stage=='freeze':
        freeze(a.out.with_name('alpha_transfer'),a.out)
    else:
        run(a.out)
