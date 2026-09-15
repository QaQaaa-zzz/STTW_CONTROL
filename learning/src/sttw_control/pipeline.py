"""Bounded training and declared fixed-alpha evaluation using canonical CLIs."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]


def evaluation_alphas(task,panel):
    values=panel.get('priority_alphas')
    if task.get('priority') is None:
        if values is not None:raise ValueError('priority alphas require a conditioned task')
        return [None]
    if not isinstance(values,list) or not values:raise ValueError('conditioned panel requires explicit priority_alphas')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 0<=x<=1 for x in values) or len(set(values))!=len(values):
        raise ValueError('priority alphas must be finite, unique and within [0,1]')
    return list(map(float,values))


def select_endpoint(result,*,conditioned):
    return result['last_checkpoint'] if conditioned else result['best_checkpoint'] or result['last_checkpoint']


def source_identity():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('learning/src/sttw_control','learning/cli') for p in sorted((ROOT/folder).glob('*.py'))}


def run(task_path,training_path,panel_path,output,*,resume=False):
    from .env import load_config
    from .training import TrainingConfig
    from .disturbance import plot_panel
    task_path,training_path,panel_path=map(Path,(task_path,training_path,panel_path))
    if resume:
        frozen=Path(output).resolve()/'frozen'
        for name,path in [('task',task_path),('training',training_path),('panel',panel_path)]:
            if path.read_bytes()!=(frozen/f'{name}.json').read_bytes():raise ValueError('resume requires identical frozen inputs: '+name)
    raw=json.loads(task_path.read_text());panel=json.loads(panel_path.read_text())
    cfg=load_config(task_path);training=TrainingConfig(**json.loads(training_path.read_text()))
    alphas=evaluation_alphas(raw,panel);conditioned=cfg.priority is not None
    seeds=panel['evaluation_seeds']
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):raise ValueError('evaluation seeds must be distinct integers')
    if set(seeds)&set(training.validation_seeds) or training.seed in set(seeds)|set(training.validation_seeds):raise ValueError('training, development and standard seeds must be separate')
    cases=panel.get('cases',[])
    count=1+len(cases)+len(panel.get('steer_rate_pulses',[]))+len(panel.get('lateral_forces',[]))
    if cfg.reference_paths is not None:count*=len(cfg.reference_paths.cases)
    horizon=max(cfg.horizon_seconds,panel['start_seconds']+max([panel['duration_seconds']]+[c['duration'] for c in cases])+panel.get('minimum_post_event_seconds',0.))
    if cfg.tracking is not None and horizon>cfg.horizon_seconds+1e-7:
        raise ValueError('tracking panel cannot extend the fixed episode')
    output=Path(output).resolve()
    if not resume:
        output.mkdir(parents=True,exist_ok=False)
        frozen=output/'frozen';frozen.mkdir()
        for name,p in [('task',task_path),('training',training_path),('panel',panel_path)]:
            (frozen/f'{name}.json').write_bytes(p.read_bytes())
    sources=source_identity()
    declaration={'source_sha256':sources,'priority_alphas':alphas,
                 'endpoint':('best geometric development-gated candidate or explicitly unqualified last fallback' if cfg.tracking is not None else ('fixed final update; priority controllability development' if conditioned else 'legacy best eligible or last fallback')),
                 'training_budget':training.num_envs*training.rollout_steps*training.updates,
                 'standard_budget':len(alphas)*len(seeds)*count*2*math.ceil(horizon/cfg.controller.dt),
                 'media_selection':'first predeclared seed, all events, every declared alpha',
                 'role':'development; no holdout, energy saving or hardware claim'}
    attempt=0
    if resume:
        original=json.loads((output/'declaration.json').read_text())
        changed={k for k in set(sources)|set(original['source_sha256']) if sources.get(k)!=original['source_sha256'].get(k)}
        allowed={'learning/src/sttw_control/pipeline.py','learning/cli/recovery_pipeline.py','learning/src/sttw_control/panel_media.py','learning/src/sttw_control/media.py'}
        if changed-allowed:raise ValueError('resume cannot change simulation/training sources: '+str(sorted(changed-allowed)))
        for key in ('priority_alphas','endpoint','training_budget','standard_budget'):
            if declaration[key]!=original[key]:raise ValueError('resume declaration mismatch: '+key)
        record=output/'resumptions';record.mkdir(exist_ok=True)
        attempt=len(list(record.glob('*.json')))+1
        with (record/f'{attempt:04d}.json').open('x') as stream:
            json.dump({'previous_status':json.loads((output/'pipeline_status.json').read_text()),'changed_sources':sorted(changed),'source_sha256':sources,'scope':'resume post-training evaluation/media only; original declaration and failed logs retained'},stream,indent=2)
    else:(output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    def status(phase,**extra):
        (output/'pipeline_status.json').write_text(json.dumps({'phase':phase,**extra},indent=2)+'\n')
    def invoke(args,log,cpu=False):
        if source_identity()!=sources:raise RuntimeError('source changed since stage declaration; refusing mixed-source execution')
        env=os.environ.copy()
        if cpu:env['JAX_PLATFORMS']='cpu'
        if resume:log=log.with_name(log.stem+f'.resume_{attempt:04d}'+log.suffix)
        with log.open('x') as stream:
            subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    train=output/'training'
    try:
        if not resume:
            status('training',endpoint=declaration['endpoint'])
            invoke(['learning/cli/train.py','--task',frozen/'task.json','--config',frozen/'training.json','--output',train],output/'training.log')
        result=json.loads((train/'status.json').read_text())
        if not result['complete']:raise RuntimeError('training did not finish its declared budget')
        if resume and result.get('control_transitions')!=declaration['training_budget']:raise ValueError('completed training budget mismatch')
        checkpoint=(result['best_checkpoint'] or result['last_checkpoint']) if cfg.tracking is not None else select_endpoint(result,conditioned=conditioned)
        if cfg.tracking is not None:
            (output/'endpoint.json').write_text(json.dumps({'checkpoint':checkpoint,
                'development_eligible':result['best_checkpoint'] is not None,
                'scope':'best development-gated checkpoint; unqualified last fallback remains diagnostic only'},indent=2)+'\n')
        rows=[];media_index=['# Complete standard recovery media','',declaration['media_selection'],'']
        for i,alpha in enumerate(alphas):
            prefix=f'alpha_{i}/' if conditioned else ''
            for seed in seeds:
                status('standard_evaluation',checkpoint=checkpoint,alpha=alpha,seed=seed,endpoint=declaration['endpoint'])
                out=output/'evaluation'/prefix/f'seed_{seed}'
                args=['learning/cli/disturbance.py','--task',frozen/'task.json','--panel',frozen/'panel.json','--training-run',train,'--checkpoint',checkpoint,'--output',out,'--seed',seed]
                if conditioned:args+=['--priority-alpha',alpha]
                if resume and out.exists():
                    done=json.loads((out/'status.json').read_text());saved=json.loads((out/'declaration.json').read_text())
                    if not done.get('complete') or done.get('episodes')!=count*2:raise ValueError('existing evaluation incomplete; refusing overwrite')
                    if saved.get('priority_alpha_override')!=alpha or saved['panel']['seed']!=seed or str(saved['checkpoint'])!=str(checkpoint):raise ValueError('saved evaluation provenance mismatch')
                    sidecar=hashlib.sha256((Path(checkpoint)/'identity.json').read_bytes()).hexdigest()
                    if saved['policy'].get('checkpoint_sidecar_sha256')!=sidecar:raise ValueError('saved evaluation checkpoint hash mismatch')
                    cached=[json.loads(x) for x in (out/'results.jsonl').read_text().splitlines()]
                    expected={(name,label) for name in saved['scenarios'] for label in ('baseline','residual')}
                    if len(cached)!=count*2 or {(x['scenario'],x['policy']) for x in cached}!=expected:raise ValueError('saved evaluation rows incomplete')
                else:invoke(args,output/f'evaluation_alpha_{i}_seed_{seed}.log',True)
                for line in (out/'results.jsonl').read_text().splitlines():
                    rows.append({'seed':seed,'priority_alpha':alpha,**json.loads(line)})
            status('media',checkpoint=checkpoint,alpha=alpha)
            first=output/'evaluation'/prefix/f'seed_{seeds[0]}'
            if source_identity()!=sources:raise RuntimeError('source changed before media generation')
            destination=output/'complete_media'/prefix if conditioned else output/'complete_media'
            if resume and destination.exists():
                manifest=json.loads((destination/'manifest.json').read_text())
                if manifest.get('status')!='complete' or manifest['panel_declaration_sha256']!=hashlib.sha256((first/'declaration.json').read_bytes()).hexdigest():raise ValueError('existing media incomplete or stale; refusing overwrite')
                media_decl=json.loads((first/'declaration.json').read_text())
                if set(manifest['cases'])!={name for name in media_decl['scenarios'] if name!='nominal'}:raise ValueError('incomplete media case coverage')
                for name,entry in manifest['cases'].items():
                    pair=[first/name/label for label in ('baseline','residual')]
                    if entry['runs']!=list(map(str,pair)) or entry['trace_sha256']!=[hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest() for p in pair]:raise ValueError('paired media source trace mismatch')
                    if hashlib.sha256((destination/name/'comparison.mp4').read_bytes()).hexdigest()!=entry['video_sha256']:raise ValueError('paired video hash mismatch')
                    if not (destination/name/'comparison.png').is_file():raise ValueError('missing paired figure')
            else:
                plot_panel(first)
                invoke(['learning/cli/panel_media.py','--panel-run',first,'--output',destination],output/f'media_alpha_{i}.log',True)
            if conditioned:media_index.append(f'- alpha={alpha}: [all scenarios](alpha_{i}/INDEX.md)')
        if conditioned:(output/'complete_media'/'INDEX.md').write_text('\n'.join(media_index)+'\n')
        (output/'standard_results.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        if conditioned:
            status('reward_breakdown',checkpoint=checkpoint)
            if source_identity()!=sources:raise RuntimeError('source changed before reward diagnostics')
            from .reward_breakdown import generate
            generate(output)
        status('complete',checkpoint=checkpoint,episodes=len(rows),endpoint=declaration['endpoint'],
               legacy_eligible_candidate_exists=result['best_checkpoint'] is not None,
               **({'development_eligible':result['best_checkpoint'] is not None} if not conditioned else {}),
               eligibility_scope=('geometric development gates; not held-out safety evidence' if cfg.tracking is not None else 'legacy speed/nominal gate, not priority controllability evidence'))
    except Exception as exc:
        status('error',error=str(exc));raise
