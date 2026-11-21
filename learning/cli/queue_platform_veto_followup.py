"""Diagnose the first new-root panel and run the predeclared second panel."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from itertools import product

import numpy as np
import pandas as pd

from prepare_blind_platform import sha
from run_platform_veto_fallback import build_bank, read


def diagnose(folder):
    plan = read(folder/'test_freeze_manifest.json')
    assigned = pd.read_csv(folder/'online_cases.csv')
    oracle = pd.read_csv(folder/'oracle_cases.csv').query('side == 1')
    keys = ['layout','root','force','section','approach','approach_alpha']
    availability = oracle.groupby(keys).agg(every_exit_collides=('collision','all'),
                                             any_exit_completes=('success','any'))
    traced = assigned.join(availability, on=keys, validate='many_to_one')
    veto = (traced.query_batch_ms > 0) & ~traced.query_feasible
    traced['veto_contact'] = veto & traced.collision
    traced['avoidable_veto_contact'] = traced.veto_contact & ~traced.every_exit_collides
    traced['unavoidable_veto_contact'] = traced.veto_contact & traced.every_exit_collides
    traced['lost_possible_after_veto'] = veto & ~traced.success & traced.any_exit_completes
    counts = traced.groupby(['layout','planner']).agg(contracts=('root','size'), completed=('success','sum'),
        walls=('collision','sum'), veto_contacts=('veto_contact','sum'), avoidable=('avoidable_veto_contact','sum'),
        unavoidable=('unavoidable_veto_contact','sum'), missed_feasible=('lost_possible_after_veto','sum'))
    counts.to_csv(folder/'fallback_diagnosis.csv')
    result = read(folder/'decision.json')
    expected = len(plan['roots']) * len(plan['maps']) * len(plan['forces_N']) * len(plan['section_x'])
    if (len(traced) != 2*expected or traced.every_exit_collides.isna().any()
            or result['matched_prefixes'] != expected or result['same_admissions'] != expected
            or result['switches_outside_veto']):
        raise ValueError(f'Incomplete or unmatched paired execution in {folder}')
    return counts, result


def prepare(first, follow, branch):
    original = first/'test_freeze_manifest.json'
    plan = read(original)
    roots = list(range(1121,1171))
    with np.load(plan['bank']) as bank, np.load(Path(plan['wide'])/'initial_bank.npz') as dev:
        states = np.column_stack((bank['qpos'],bank['qvel']))
        old = np.column_stack((dev['qpos'],dev['qvel']))
        previous = {x.tobytes() for x in np.concatenate((old,states[1001:1121]))}
        new = {x.tobytes() for x in states[roots]}
        if len(new)!=len(roots) or previous & new:
            raise ValueError('Repeated physical initial state')
    if branch == 'expand_bank':
        archives = [first.with_name('alpha_transfer'),first]
        oracle = pd.concat([pd.read_csv(p/'oracle_cases.csv').query('side == 1') for p in archives])
        plan['veto_fit_roots'] = sorted(oracle.root.unique().tolist())
    for name in plan['maps']:
        target = follow/name
        target.mkdir(exist_ok=True)
        for file in ('rb_run1_prior.csv','rb_run1_initializer.npz','terrain.npz'):
            (target/file).symlink_to((first/name/file).resolve())
        if branch == 'expand_bank':
            build_bank(oracle[oracle.layout==name],target)
            plan['veto_banks'][name]=str(target/'exit_bank.npz')
            plan['bank_hashes'][name]=sha(target/'exit_bank.npz')
    plan.update(roots=roots, assignment_batches=[list(product(roots,plan['forces_N']))],
        role='Separate test of frozen follow-up candidate on fresh starts and known walls',
        scope=(follow/'protocol.md').read_text(), parent_manifest=str(original),
        parent_sha256=sha(original), followup_branch=branch)
    (follow/'test_freeze_manifest.json').write_text(json.dumps(plan,indent=2)+'\n')


def queue(first, follow, upstream_pid):
    status = follow/'queue_status.json'
    def stage(name, **details):
        status.write_text(json.dumps(dict(stage=name, **details), indent=2)+'\n')
    stage('waiting_first_panel', upstream_pid=upstream_pid, first=str(first))
    try:
        if not (first/'decision.json').exists():
            subprocess.run(['/usr/bin/python3','-c',
                'import os,select,sys; fd=os.pidfd_open(int(sys.argv[1])); select.select([fd],[],[])',
                str(upstream_pid)],check=True)
        stage('diagnosing_first_panel')
        first_counts, decision = diagnose(first)
        repairable = int(first_counts.xs('veto_bank',level='planner').avoidable.sum())
        branch = ('repeat_frozen' if decision['meets_paired_outcome_criterion'] else
                  'expand_bank' if repairable else 'exit_library_limit')
        panels = [('首批 50 个初态',first,first_counts,decision)]
        if branch != 'exit_library_limit':
            stage('preparing_followup', branch=branch)
            prepare(first,follow,branch)
            stage('queued_or_running_followup', branch=branch, live=str(follow/'live.json'))
            with (follow/'run.log').open('w') as log:
                subprocess.run(['bash','learning/scripts/gpu_queue.sh',sys.executable,'-u',
                    'learning/cli/run_platform_veto_fallback.py','run',str(follow)],stdout=log,
                    stderr=subprocess.STDOUT,check=True)
            counts, decision = diagnose(follow)
            panels.append(('第二批 50 个初态',follow,counts,decision))
        summary = ['# 三墙保底策略顺序复验','',
            '第二阶段分支为 '+branch+'。两批分开计数，各批 50 个初态按根配对。','']
        for label, path, counts, outcome in panels:
            summary += ['## '+label,'','|墙布局|策略|通关/任务|碰撞|否决后可避免 / 三个出口均碰撞|',
                        '|---|---|---:|---:|---:|']
            for (layout,arm),r in counts.iterrows():
                summary.append(f'|{layout}|{arm}|{r.completed}/{r.contracts}|{r.walls}|{r.avoidable} / {r.unavoidable}|')
            metrics = pd.read_csv(path/'paired_metrics.csv').query("layout=='all'")
            for r in metrics.itertuples():
                summary.append(f'\n{r.metric} 配对差 {100*r.net/r.contracts:+.2f} 个百分点，95% 区间 [{100*r.delta_low:.2f}, {100*r.delta_high:.2f}]。')
            summary += ['',f'达到预先声明的碰撞与通关判据：{outcome["meets_paired_outcome_criterion"]}。',
                        f'[配对实际与参考 XY]({os.path.relpath(path/"online_xy.png",follow)})','']
        (follow/'RESULTS.md').write_text('\n'.join(summary)+'\n')
        stage('complete',branch=branch,panels=len(panels),report=str(follow/'RESULTS.md'))
    except Exception as exc:
        stage('error',reason=str(exc),failed_stage=read(status)['stage'])
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('first',type=Path)
    p.add_argument('follow',type=Path)
    p.add_argument('pid',type=int)
    a=p.parse_args()
    queue(a.first,a.follow,a.pid)
