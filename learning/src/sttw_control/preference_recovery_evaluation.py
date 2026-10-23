"""Authenticate fixed-final conditional actors and evaluate the matched pilot."""
from pathlib import Path
import json


def candidate_record(campaign,arm,declaration):
    import pickle,numpy as np,torch
    from .preference_best import digest,config_hash
    from .path_command_policy import SCHEMA,validate_actor_parameters,MODULATION_SHAPES
    job=Path(campaign)/arm;status=json.loads((job/'status.json').read_text())
    updates=declaration['updates'];receipt=json.loads((job/'last_completed.json').read_text())
    if status['phase']!='completed' or status['completed_updates']!=updates or status['policy_transitions_total']!=updates*512*128:raise ValueError('conditional training incomplete')
    learner=Path(receipt['checkpoint']);actor=learner.with_name(f'actor_{updates:04d}.pkl')
    if receipt['update']!=updates or learner.name!=f'update_{updates:04d}.pt' or learner.parent.resolve()!=(job/'checkpoints').resolve():raise ValueError('wrong final learner boundary')
    if digest(learner)!=receipt['checkpoint_sha256']:raise ValueError('conditional learner checksum changed')
    manifest=json.loads((job/'manifest.json').read_text());identity=manifest['identity']
    saved=torch.load(learner,map_location='cpu',weights_only=False)
    if saved['schema']!=SCHEMA or saved['identity']!=identity or saved['update']!=updates:raise ValueError('conditional learner identity mismatch')
    if (identity['experiment']!='preference_recovery_v1' or identity['arm']!=arm or identity['alpha']!='conditional' or identity['seed']!=declaration['seed'] or identity['config_sha256']!=config_hash(manifest['config']) or identity['parent_checkpoint_sha256']!=declaration['parent_record']['checkpoint_sha256'] or identity['parent_source_update']!=declaration['parent_record']['source_update'] or identity['lower_actor_sha256']!=manifest['source']['candidate']['actor_sha256']):raise ValueError('wrong conditional learner lineage')
    rows=[json.loads(line) for line in (job/'metrics.jsonl').read_text().splitlines()]
    if [r['update'] for r in rows]!=list(range(1,updates+1)):raise ValueError('learner update accounting incomplete')
    if not all(r['optimizer'].get('actor_bitwise_unchanged') is True for r in rows[:declaration['critic_warmup_updates']]):raise ValueError('critic warmup changed actor')
    with actor.open('rb') as stream:params=pickle.load(stream)
    validate_actor_parameters(params,manifest['config']);p=params['params'];state=saved['policy']
    try:
        for i,layer in enumerate((0,2,4,6)):
            np.testing.assert_array_equal(p[f'Dense_{i}']['kernel'],state[f'actor.{layer}.weight'].numpy().T)
            np.testing.assert_array_equal(p[f'Dense_{i}']['bias'],state[f'actor.{layer}.bias'].numpy())
        for key in MODULATION_SHAPES:np.testing.assert_array_equal(p[key],state['actor.'+key].numpy())
    except AssertionError as exc:raise ValueError('conditional exported Actor differs from learner') from exc
    return dict(actor=str(actor),actor_sha256=digest(actor),checkpoint=str(learner),checkpoint_sha256=digest(learner),identity=identity,source_update=updates)


def paired_results(parent_rows,candidates,alphas):
    from .tolerance_comparison import paired_counts
    all_policies={'u325':parent_rows,**candidates};pairs=[('u325',p) for p in candidates]
    pairs +=[(a,b) for a,b in (('uniform','guided'),('uniform','shuffled'),('shuffled','guided')) if a in candidates and b in candidates]
    result=[]
    for baseline,candidate in pairs:
      for alpha in alphas:
       for metric in ('task_recovered','performance_success'):
        data=[dict(row,policy=name,success=row[metric]) for name in (baseline,candidate) for row in all_policies[name]]
        result.append(dict(comparison=baseline+'_vs_'+candidate,alpha=alpha,metric=metric,**paired_counts(data,baseline,candidate,alpha)))
    return result


def cached_baseline(run,protocol,parent):
    """Reuse only the exact completed parent/protocol panel."""
    from .preference_best import digest,config_hash
    run=Path(run);status=json.loads((run/'status.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    saved=manifest['declaration'];rows=json.loads((run/'results.json').read_text())['cases']
    if status['phase']!='completed' or status['completed_cases']!=200 or len(rows)!=200:raise ValueError('cached baseline incomplete')
    for key in ('config','source','routes','speeds','alphas','forces_N','onsets_s','duration_s'):
        if config_hash(saved[key])!=config_hash(protocol[key]):raise ValueError('cached baseline protocol differs: '+key)
    for key in ('actor','checkpoint'):
        if saved['policies']['u325'][key+'_sha256']!=parent[key+'_sha256'] or digest(parent[key])!=parent[key+'_sha256']:raise ValueError('cached baseline parent identity differs')
    from itertools import product
    keys=('route','speed','alpha','force_N','onset_s')
    expected=set(product([r['name'] for r in protocol['routes']],protocol['speeds'],protocol['alphas'],protocol['forces_N'],protocol['onsets_s']))
    if len({r['name'] for r in rows})!=200 or {tuple(r[k] for k in keys) for r in rows}!=expected or any(r['policy']!='u325' for r in rows):raise ValueError('cached baseline case identities differ')
    return rows


def finish(campaign,output,register_run):
    import os,time,subprocess,sys
    from .preference_best import atomic_json,digest
    from .preference_tolerance_reporting import report
    root=Path(campaign).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    declaration=json.loads((root/'training_declaration.json').read_text());started=time.monotonic()
    status=dict(state='running',phase='running',pid=os.getpid(),training_steps=0)
    def update(**kw):
        status.update(kw);status.update(checked_unix=time.time(),elapsed_s=time.monotonic()-started)
        atomic_json(out/'status.json',status)
    update(stage='waiting_for_training')
    try:
        campaign_info=json.loads((root/'campaign.json').read_text())
        if digest(root/'training_declaration.json')!=campaign_info['training_declaration_sha256']:raise ValueError('training declaration changed after launch')
        while True:
            train=json.loads((root/'status.json').read_text())
            if train['phase']=='completed':break
            if train['phase'] in ('error','failed'):raise RuntimeError('conditional training campaign failed; preserve all workers/artifacts')
            stat=Path(f"/proc/{train['pid']}/stat")
            if not stat.exists() or stat.read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):raise RuntimeError('training controller not live; evaluation not launched')
            update(stage='waiting_for_training',completed_jobs=train.get('completed_jobs',0),declared_jobs=3,policy_transitions_total=train.get('policy_transitions_total',0))
            time.sleep(10)
        records={arm:candidate_record(root,arm,declaration) for arm in declaration['arms']}
        atomic_json(out/'candidate_records.json',records)
        post_declaration=json.loads((root/'post_declaration.json').read_text())
        atomic_json(out/'manifest.json',dict(training_declaration_sha256=digest(root/'training_declaration.json'),post_declaration_sha256=digest(root/'post_declaration.json'),post_declaration=post_declaration,candidates=records))
        all_results={}
        for panel,base_name in (('moderate','preference_tolerance_screen_v1'),('strong','preference_tolerance_stress_v1'),('straight',None)):
          if base_name:
            base=root.parent/base_name/'evaluation';saved=json.loads((base/'manifest.json').read_text())['declaration']
            protocol=dict(saved,policies=records,maximum_lower_ticks=1200000,experiment_id='preference_recovery_post_v1_'+panel,
                training_campaign=str(root),training_declaration_sha256=digest(root/'training_declaration.json'),data_role='matched development only; fixed-final candidates, no checkpoint selection',baseline_cached=str(base))
            parent_rows=cached_baseline(base,protocol,declaration['parent_record'])
          else:
            saved=json.loads((root.parent/'preference_tolerance_screen_v1/declaration.json').read_text())
            protocol=dict(saved,policies={'u325':declaration['parent_record'],**records},routes=[dict(name='straight',builder='straight')],forces_N=[0.],onsets_s=[1.],maximum_lower_ticks=80000,experiment_id='preference_recovery_post_v1_straight',training_campaign=str(root),data_role='original two straight speeds and five-alpha nominal DEV, no new force cases')
          if protocol['maximum_lower_ticks']!=post_declaration['panels'][panel]['maximum_lower_ticks']:raise ValueError('post-evaluation budget differs from declaration')
          expected_cases=post_declaration['panels'][panel]['fresh_cases']
          folder=out/panel;folder.mkdir();atomic_json(folder/'declaration.json',protocol)
          register_run(root/f'ACTIVE_POST_{panel.upper()}.json',f'STTW_CONTROL｜单网容扰度训练后评测｜{panel}',folder/'evaluation/status.json')
          with (folder/'evaluation.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-u','learning/cli/screen_preference_tolerance.py','--declaration',str(folder/'declaration.json'),'--output',str(folder/'evaluation')],stdout=log,stderr=subprocess.STDOUT)
          while process.poll() is None:
                update(stage='evaluating',panel=panel,evaluator_pid=process.pid);time.sleep(10)
          if process.returncode:raise RuntimeError(f'{panel} conditional evaluation exited{process.returncode}')
          evaluation=json.loads((folder/'evaluation/status.json').read_text())
          if evaluation['phase']!='completed' or evaluation['completed_cases']!=expected_cases:raise ValueError('conditional evaluation incomplete')
          update(stage='reporting',panel=panel)
          report(folder/'evaluation',folder/'analysis')
          rows=json.loads((folder/'evaluation/results.json').read_text())['cases']
          if not base_name:parent_rows=[r for r in rows if r['policy']=='u325']
          per_policy={name:[r for r in rows if r['policy']==name] for name in records}
          comparisons=paired_results(parent_rows,per_policy,protocol['alphas'])
          atomic_json(folder/'paired_results.json',dict(comparisons=comparisons,baseline_reused_cases=200 if base_name else 0,candidate_cases=expected_cases,actual_lower_ticks=evaluation['active_lower_ticks']))
          all_results[panel]=comparisons
        atomic_json(out/'paired_results.json',all_results)
        lines=['# Conditional recovery matched pilot','',
            'Fixed final u27, one training seed, original V3/local300/10s task; not final TEST. Parent400 cases authenticated and reused; candidate1200 pulse-panel cases plus40 nominal straight cases freshly evaluated. The latter includes parent10 and candidate30, completing the original30 unique nominal cases per policy. Task and working-performance remain separate. Deterministic paired cases are correlated; totals are not independent training repeats.','',
            '| panel | comparison | α | metric | baseline | candidate | gained | lost |','|---|---|---|---|---|---|---|---|']
        for panel,comparisons in all_results.items():
            for r in comparisons:lines.append(f"| {panel} | {r['comparison']} | {r['alpha']:g} | {r['metric']} | {r['baseline_success']}/{r['n']} | {r['candidate_success']}/{r['n']} | {r['improved']} | {r['regressed']} |")
        lines+=['','[Moderate curves and continuous metrics](moderate/analysis/REPORT.md). [Strong curves and continuous metrics](strong/analysis/REPORT.md). [Nominal straight tracking](straight/analysis/REPORT.md). No automatic policy adoption; actual training exposures and nominal preference curves still require research review.']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(state='completed',phase='completed',stage='reports_ready',evaluation_status='complete',adoption_status='pending_research_review')
    except BaseException as exc:
        update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise
