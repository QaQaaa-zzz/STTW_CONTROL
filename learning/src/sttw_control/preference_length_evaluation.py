"""Frozen-panel evaluation of the declared full-tail learner length comparison."""
from pathlib import Path
import json
from .preference_best import digest,atomic_json

KEYS=('route','speed','alpha','force_N','onset_s')

def paired_counts(base,candidate):
    a={tuple(r[k] for k in KEYS):r for r in base}
    b={tuple(r[k] for k in KEYS):r for r in candidate}
    if len(a)!=len(base) or len(b)!=len(candidate) or a.keys()!=b.keys():
        raise ValueError('duplicate or mismatched evaluation cases')
    return dict(n=len(a),baseline=sum(r['task_recovered'] for r in a.values()),
        candidate=sum(r['task_recovered'] for r in b.values()),
        gains=sum(not a[k]['task_recovered'] and b[k]['task_recovered'] for k in a),
        losses=sum(a[k]['task_recovered'] and not b[k]['task_recovered'] for k in a))

def fixed_records(train):
    import torch
    train=Path(train);status=json.loads((train/'uniform/status.json').read_text())
    if status['phase']!='completed' or status['completed_updates']!=77:
        raise ValueError('declared training incomplete')
    manifest=json.loads((train/'uniform/manifest.json').read_text())
    rows=[json.loads(x) for x in (train/'uniform/metrics.jsonl').read_text().splitlines()]
    if [r['update'] for r in rows]!=list(range(28,78)):
        raise ValueError('continuation update accounting incomplete')
    if status['executed_policy_transitions']!=50*512*128:
        raise ValueError('continuation interaction accounting mismatch')
    records={}
    for update,name in [(50,'ppo48'),(77,'ppo75')]:
        cp=train/'uniform/checkpoints'/f'update_{update:04d}.pt'
        actor=cp.with_name(f'actor_{update:04d}.pkl')
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        if saved['identity']!=manifest['identity'] or saved['update']!=update:
            raise ValueError('fixed learner identity mismatch')
        if update==77:
            receipt=json.loads((train/'uniform/last_completed.json').read_text())
            if receipt['checkpoint_sha256']!=digest(cp) or receipt['update']!=77:
                raise ValueError('final learner receipt mismatch')
        records[name]=dict(actor=str(actor),actor_sha256=digest(actor),checkpoint=str(cp),
            checkpoint_sha256=digest(cp),identity=saved['identity'],source_update=update)
    return records

def evaluate(plan_path,output,register):
    import os,time,subprocess,sys
    from .preference_tolerance_reporting import report
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text());out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    state=dict(phase='running',pid=os.getpid(),stage='waiting_for_training',completed_cases=0,
        declared_cases=940,active_lower_ticks=0,training_steps=0)
    def update(**kw):
        state.update(kw,checked_unix=time.time());atomic_json(out/'status.json',state)
    update()
    try:
        train=Path(plan['training'])
        if digest(train/'training_declaration.json')!=plan['training_declaration_sha256']:
            raise ValueError('training declaration changed')
        while True:
            s=json.loads((train/'status.json').read_text())
            if s['phase']=='completed':break
            if s['phase'] in ('error','failed'):raise RuntimeError('training failed; no evaluation')
            os.kill(s['pid'],0);time.sleep(60)
        records=fixed_records(train);atomic_json(out/'candidate_records.json',records)
        comparisons={}
        for name,panel in plan['panels'].items():
            source=Path(panel['source'])
            for filename,expected in panel['source_hashes'].items():
                if digest(source/filename)!=expected:raise ValueError('cached panel changed: '+filename)
            d=json.loads((source/'declaration.json').read_text())
            if d['policies']['full10']['checkpoint_sha256']!=plan['baseline_checkpoint_sha256']:
                raise ValueError('wrong baseline learner')
            base=json.loads((source/'evaluation/results.json').read_text())['cases']
            if len(base)!=panel['cases_per_policy']:raise ValueError('baseline panel incomplete')
            folder=out/name;folder.mkdir()
            d.update(policies=records,maximum_lower_ticks=2*len(base)*2000,
                experiment_id=out.name+'_'+name,data_role='fixed previously used DEV; final75 primary,48 descriptive',
                question='Effect of longer PPO training with all other settings held fixed',
                baseline_cached=str(source),training_campaign=str(train),
                training_declaration_sha256=plan['training_declaration_sha256'],
                stop_rule='evaluate fixed48/75 once; no candidate selection')
            atomic_json(folder/'declaration.json',d)
            register(folder/'ACTIVE_RUN.json','STTW_CONTROL 训练长度对照评测 '+name,folder/'evaluation/status.json')
            with (folder/'evaluation.log').open('x') as log:
                p=subprocess.Popen([sys.executable,'-u','learning/cli/screen_preference_tolerance.py',
                    '--declaration',str(folder/'declaration.json'),'--output',str(folder/'evaluation')],stdout=log,stderr=subprocess.STDOUT)
            update(stage='evaluating',panel=name,child_pid=p.pid)
            while p.poll() is None:time.sleep(60)
            if p.returncode:raise RuntimeError(name+' evaluation exited '+str(p.returncode))
            s=json.loads((folder/'evaluation/status.json').read_text())
            if s['phase']!='completed' or s['completed_cases']!=2*len(base):raise ValueError('panel incomplete')
            fresh=json.loads((folder/'evaluation/results.json').read_text())['cases']
            comparisons[name]={policy:paired_counts(base,[r for r in fresh if r['policy']==policy]) for policy in records}
            atomic_json(out/'comparisons.json',comparisons)
            update(completed_cases=state['completed_cases']+len(fresh),active_lower_ticks=state['active_lower_ticks']+s['active_lower_ticks'],stage='reporting')
            report(folder/'evaluation',folder/'analysis')
        if state['completed_cases']!=940:raise ValueError('final case accounting mismatch')
        update(phase='completed',stage='fixed_panel_results_ready')
    except BaseException as e:
        update(phase='error',error=repr(e));raise
