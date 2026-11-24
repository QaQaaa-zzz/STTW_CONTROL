"""Bounded E1a/E1b CEM above frozen H75, with per-generation raw archives."""
import argparse
import csv
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from pcrp_replay_gate import REPO, LABELS, read, sha, write


LAYOUTS = ('flat_original','gravel010','position_s3.9','position_s4.2','position_s4.5','position_s4.275')
PILOT = ('flat_original','position_s4.2','position_s4.5')
BOUND = np.array([1.25,.40])
SIGMA_INIT = np.array([.40,.10])
SIGMA_MIN = np.array([.02,.005])
ALPHAS = (0., .25, .5, .75, 1.)
KNOT_TIMES = np.r_[0., .5, np.arange(1.,4.25,.25), np.arange(4.5,10.5,.5)]
LAB = '/home/qy/miniconda3/envs/env_isaaclab/bin/python'


def case_name(layout, group):
    suffix = '_layout0' if layout=='gravel010' else ''
    return f'{layout}_v2.6{suffix}_seed{group}'


def pure_cem_flat_passed(rows):
    """The flat pilot must supply its own passing candidate after the warm starts."""
    return any(
        row['case'] == case_name('flat_original', 401)
        and int(row['generation']) > 0 and row['method'].startswith('CEM_')
        and int(row['C']) == 1 and float(row['V']) <= 0.676
        for row in rows
    )


def archived_flat_passed(directory, status):
    if directory.name != 'E1a':
        return status.get('flat_control_passed') is True
    with (directory/'candidates.csv').open() as stream:
        return pure_cem_flat_passed(csv.DictReader(stream))


def case_records(root, stage):
    raw = read(root.parent/'pcrp_curvature_governor_20261121'/'source_cases.json')
    protocol = read(root/'step0/protocol.json')
    common = read(root.parent/'h75_contact_oat/position/flat_original_v2.6_seed401.json')
    cases = []
    for layout in LAYOUTS:
        for seed in range(401,405):
            pilot = layout in PILOT and seed==401
            if (stage=='E1a')!=pilot:
                continue
            name = case_name(layout,seed)
            source = Path(raw[name] if name in raw else raw[case_name('position_s4.2',seed)])
            archive = read(source/'contract.json')
            settings = archive['settings']
            if name in raw:
                terrain = Path(settings['terrains']['evaluation'])
            elif layout=='position_s4.275':
                terrain = root.parent/'pcrp_p1_20261121'/'terrain_4.275'
            else:
                raise ValueError(f'Missing original case {name}')
            identity = read(terrain/'identity.json')
            if (archive['identity']['files']['actor.npz']!=protocol['identities']['actor.npz']
                    or settings['lower_sha256']!=protocol['identities']['local300']
                    or identity.get('route_sha256',archive['identity']['files']['route.npz'])!=protocol['identities']['route.npz']
                    or settings['seed']!=seed or settings['user_speed_m_s']!=2.6
                    or settings['plant_options']!=common['plant_options']
                    or settings['initial_state']!=common['initial_state']
                    or (layout=='position_s4.275' and identity['center_s_m']!=4.275)):
                raise ValueError(f'Frozen case contract mismatch at {name}')
            cases.append(dict(case=name,layout=layout,group=seed,terrain=str(terrain),
                              terrain_sha256=identity['sha256'],source=str(source),
                              slot=len(cases),
                              randomization=settings['randomization']))
    return cases


def prepare(root, stage, times, seed):
    times = np.asarray(times,dtype=float)
    if not np.array_equal(times,KNOT_TIMES) or seed<0:
        raise ValueError('E1 requires the registered 27 knots on the 0–10 s task clock and a nonnegative seed')
    cases=case_records(root,stage)
    directory=root/'e1'/stage
    directory.mkdir(parents=True,exist_ok=True)
    protocol=dict(stage=stage,times_s=times.tolist(),dimensions=2*len(times),random_seed=seed,
                  rng='PCG64 SeedSequence([random_seed, index in declared six layouts, physical group])',
                  population=32,elites=6,generations=10,mean='0.7 * current elite mean + 0.3 * previous mean',
                  sigma='Population std of current six elites with per-channel floors',
                  initial_std=SIGMA_INIT.tolist(),std_floors=SIGMA_MIN.tolist(),
                  bounds=[(-BOUND).tolist(),BOUND.tolist()],sampler='Independent diagonal Gaussian then clip to bounds',
                  incumbent='minimum V over six warm-start and all search candidates; ties retain earliest',
                  alpha_seed='minimum V over five fixed H75; ties retain earliest',
                  budget=len(cases)*326,new_budget=len(cases)*326,
                  packing='One fixed physical slot per case. Six warm-start batches then 32 single-candidate batches per generation. Order, terrain inventory and native world origins stay fixed across all 326 batches.',
                  stage_batch_size=len(cases),warm_start_archive_reuse=0,candidates_per_generation=32,
                  reference_scope='Five fixed alphas and Zero measured in each candidate slot within this stage',
                  counted_budget=dict(step0=72,E2=1240,E1a=978,E1b=6846,E1c_reserved=96,
                                      total_reserved=9232,cap=9500,separate_replay672=672),
                  role='Privileged case-specific feasibility search on previously examined DEV groups',
                  times_sha256_f64le=hashlib.sha256(times.astype('<f8').tobytes()).hexdigest(),
                  reference_bias_timing='50 Hz before original 200 Hz shaper; shared across each four lower ticks',
                  cases=cases,step0_protocol_sha256=sha(root/'step0/protocol.json'),
                  replay672_summary_sha256=sha(root/'replay672/analysis/summary.json'))
    path=directory/'protocol.json'
    if path.exists() and read(path)!=protocol:
        raise ValueError('Registered E1 protocol differs; existing run must not be overwritten')
    write(path,protocol)
    return directory,protocol


def update_distribution(mean, population, costs):
    elite=np.argsort(costs,kind='stable')[:6]
    return .3*mean+.7*population[elite].mean(0), np.maximum(SIGMA_MIN,population[elite].std(0)), elite


def case_manifest(directory, cases, label, alphas):
    return [dict(case_labels=[label],user_speed_m_s=2.6,seed=c['group'],
                 randomization=c['randomization'],alphas=[alpha],zero_middle=[label=='Zero'],
                 zero_reference_channels=[[False,False]],terrain_path=c['terrain'],
                 output=str(directory/c['case'])) for c,alpha in zip(cases,alphas,strict=True)]


def save(path, value):
    """Publish a restartable index or journal without partially replacing its predecessor."""
    tmp=path.with_name(path.name+'.tmp')
    write(tmp,value)
    tmp.replace(path)


def candidate_inputs(root, directory, protocol, label, alphas, offsets, *, create):
    cases=case_manifest(directory,protocol['cases'],label,alphas)
    knots=directory/'knots.npz'
    if create:
        directory.mkdir(parents=True,exist_ok=True)
    if knots.exists():
        with np.load(knots) as z:
            if not (np.array_equal(z['times_s'],protocol['times_s']) and np.array_equal(z['offsets'],offsets)):
                raise ValueError(f'Regenerated CEM knots differ from recorded candidate {directory}')
    elif create:
        tmp=directory/'knots.tmp.npz'
        np.savez(tmp,times_s=protocol['times_s'],offsets=np.asarray(offsets))
        tmp.replace(knots)
    cfg=read(root.parent/'h75_contact_oat/position/flat_original_v2.6_seed401.json')
    cfg['evaluation_role']='PCRP E1 matched packing: one unchanged native physical slot per case'
    if knots.exists():
        cfg.update(oracle_knots=str(knots),oracle_knots_sha256=sha(knots))
    for name,content in [('cases.json',cases),('settings.json',cfg)]:
        path=directory/name
        if path.exists():
            if read(path)!=content:
                raise ValueError(f'Frozen candidate input differs at {path}')
        elif create:
            save(path,content)
    return dict(config=str(directory/'settings.json'),terrain=cases[0]['terrain_path'],
                case_manifest=str(directory/'cases.json'),output=str(directory/'batch'))


def score_batch(directory, protocol, label, alphas, score, *, readonly=False, wall=None, queued=False):
    cases=protocol['cases']
    batch_status=read(directory/'batch/status.json')
    if batch_status['policy_steps']!=500 or batch_status['stage']!='complete':
        raise ValueError(f'Incomplete E1 subbatch in {directory}')
    meta=read(directory/'batch/contract.json')
    manifests=read(directory/'cases.json')
    settings=meta['settings']
    if (meta['lab_physics']['num_envs']!=len(cases) or settings['alphas']!=list(alphas)
            or settings['zero_middle']!=[label=='Zero']*len(cases)
            or settings['oracle_knots_sha256']!=sha(directory/'knots.npz')
            or settings['oracle_knots']!=str(directory/'knots.npz')):
        raise ValueError(f'Incorrect physical slots, policy or bias schedule in {directory}')
    # The warm and CEM runs intentionally use different policy conditions. The
    # full packing, world transforms and all frozen physical inputs must agree.
    identity_keys=('backend','identity','terrain','bodies','origin','physics_dt','lower_dt','policy_dt',
                   'randomization_readback','delay_steps')
    same={k:meta[k] for k in identity_keys}
    same.update(plant=meta['settings']['plant_options'],terrain_ids=meta['settings']['lane_terrain_ids'],
                fixed_randomization_groups=meta['settings']['fixed_randomization_groups'],
                terrain_layout=meta['lab_physics']['terrain_layout'])
    reference=directory.parents[1]/'matched_runtime.json'
    if not reference.exists():
        if readonly:
            raise FileNotFoundError(reference)
        save(reference,same)
    elif same!=read(reference):
        raise ValueError(f'Physical batch or lane identity differs at {directory}')
    records=[]
    scoring_start=time.monotonic()
    for c,m in zip(cases,manifests,strict=True):
        folder=Path(m['output'])
        split=read(folder/'contract.json')
        if (split['batch_source']['lane_start']!=c['slot'] or split['batch_source']['lane_stop']!=c['slot']+1
                or split['batch_source']['output']!=str(directory/'batch') or split['settings']['case_labels']!=[label]
                or split['settings']['alphas']!=m['alphas'] or split['settings']['seed']!=c['group']
                or split['settings']['zero_middle']!=m['zero_middle'] or split['terrain'][0]['sha256']!=c['terrain_sha256']
                or read(folder/'status.json')['policy_steps']!=500):
            raise ValueError(f'Case assignment differs at {folder}')
        row, = score(folder,[label])
        records.append(row)
    scores_path=directory/'scores.json'
    if scores_path.exists() and read(scores_path)!=records:
        raise ValueError(f'Rescoring differs from the recorded candidate in {directory}')
    if not readonly:
        if not scores_path.exists():
            save(scores_path,records)
        if not (directory/'timing.json').exists():
            recovered=wall is None
            if recovered:
                # A previous writer may have exited after export, before recording wall time.
                wall=(directory/'batch/status.json').stat().st_mtime-(directory/'settings.json').stat().st_mtime
                wall=max(batch_status['elapsed_with_output_s'],wall)
            timing=dict(wall_s=wall,sampling_s=batch_status['elapsed_s'],
                        export_s=batch_status['elapsed_with_output_s']-batch_status['elapsed_s'],
                        scoring_and_archive_s=time.monotonic()-scoring_start,lanes=len(cases),
                        measured_wall_includes_queue=queued and not recovered,
                        recovered_wall_estimate=recovered)
            resident=directory/'batch/resident_timing.json'
            if resident.is_file():
                timing['resident']=read(resident)
            save(directory/'timing.json',timing)
    return records


def csv_write(path,rows):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


class ResidentGeneration:
    """Hold the GPU lease while sending one acknowledged, frozen job at a time."""
    def __init__(self, group, jobs, command, env):
        self.group=group
        self.jobs=jobs
        self.command=command
        self.env=env

    def __enter__(self):
        first=Path(self.jobs[0]['output']).parent.name
        manifest=self.group/f'resident_from_{first}.json'
        if manifest.exists() and read(manifest)!=self.jobs:
            raise ValueError(f'Resident manifest changed at {manifest}')
        save(manifest,self.jobs)
        self.started=time.monotonic()
        self.log=(self.group/f'resident_from_{first}.log').open('a',buffering=1)
        self.process=subprocess.Popen([*self.command,'--resident-jobs','-'],cwd=REPO,env=self.env,
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',bufsize=1)
        return self

    def evaluate(self, index, job):
        before=self.started if index==0 else time.monotonic()
        self.process.stdin.write(json.dumps(job)+'\n')
        self.process.stdin.flush()
        self.log.write(f'>>> {job["output"]}\n')
        with (Path(job['output']).parent/'simulation.log').open('a',buffering=1) as candidate_log:
            for line in self.process.stdout:
                self.log.write(line)
                candidate_log.write(line)
                if line.startswith('{"resident_job_complete"'):
                    receipt=json.loads(line)
                    if receipt.get('resident_job_complete')!=job['output'] or receipt.get('index')!=index:
                        raise ValueError(f'Resident completion does not match {job["output"]}: {receipt}')
                    return time.monotonic()-before
        raise RuntimeError(f'Resident evaluator exited {self.process.wait()} before exporting {job["output"]}')

    def __exit__(self, kind, exc, traceback):
        if self.process.poll() is None:
            self.process.stdin.close()
        for line in self.process.stdout:
            self.log.write(line)
        self.process.stdout.close()
        result=self.process.wait()
        self.log.close()
        if result and kind is None:
            raise subprocess.CalledProcessError(result,self.process.args)


def evaluate_group(root, directory, protocol, specs, status, score, resident, inspect, completed_floor=0):
    """Replay the completed prefix on CPU, then dispatch only untouched candidates."""
    finished=[]
    jobs=[]
    group=specs[0][0].parent
    count=len(protocol['cases'])
    stage_start=0 if group.name=='warm' else len(LABELS)+(int(group.name[1:])-1)*protocol['population']
    for index,(folder,label,alphas,bias) in enumerate(specs):
        job=candidate_inputs(root,folder,protocol,label,alphas,bias,create=False)
        physical=folder/'batch'
        if (physical/'status.json').is_file():
            if jobs:
                raise ValueError(f'Completed candidate follows a gap: {folder}')
            for name in ('settings.json','cases.json','knots.npz'):
                if not (folder/name).is_file():
                    raise FileNotFoundError(folder/name)
            finished.append(score_batch(folder,protocol,label,alphas,score,readonly=inspect))
        elif physical.exists() or any((folder/c['case']).exists() for c in protocol['cases']):
            message=f'Candidate {folder} has physical artifacts without a complete batch status; preserve it and finish its export before resuming.'
            if inspect:
                print(message)
                print(f'Complete prefix in {group.name}: {len(finished)}/{len(specs)}')
                return None
            raise RuntimeError(message)
        else:
            if completed_floor>(stage_start+index)*count:
                raise ValueError(f'Stage already counted {folder} as complete but its physical archive is missing')
            jobs.append(job)
    if not jobs:
        return finished
    if inspect:
        print(f'Complete prefix in {group.name}: {len(finished)}/{len(specs)}; next {jobs[0]["output"]}')
        return None
    first=len(finished)
    command=['/bin/bash',str(REPO/'learning/scripts/gpu_queue.sh'),LAB,'-u',
             str(REPO/'learning/cli/evaluate_isaac_gpu.py'),'--backend','lab','--bundle',
             read(root/'step0/protocol.json')['bundle'],'--seconds','10']
    env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    group.mkdir(parents=True,exist_ok=True)
    ctx=ResidentGeneration(group,jobs,command,env) if resident else nullcontext()
    with ctx as worker:
        for j,(folder,label,alphas,bias) in enumerate(specs[first:]):
            job=candidate_inputs(root,folder,protocol,label,alphas,bias,create=True)
            n=(stage_start+first+j)*count
            status.update(subbatch=str(folder.relative_to(directory)),submitted_new=n+count,completed_new=n)
            save(directory/'status.json',status)
            if resident:
                wall=worker.evaluate(j,job)
            else:
                start=time.monotonic()
                argv=[*command,'--config',job['config'],'--terrain',job['terrain'],
                      '--case-manifest',job['case_manifest'],'--output',job['output']]
                with (folder/'simulation.log').open('a') as log:
                    subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                wall=time.monotonic()-start
            finished.append(score_batch(folder,protocol,label,alphas,score,wall=wall,queued=not resident or j==0))
            status['completed_new']=n+count
            save(directory/'status.json',status)
    return finished


def execute(root,stage,*,resident=False,resume=False,inspect=False):
    authorization=read(root/'paired_authorization.json')
    replay=read(root/'replay672/analysis/summary.json')
    if authorization.get('decision')!='matched_packing' or not replay.get('passed') or replay['matched_lanes']!=672:
        raise SystemExit('E1 requires the registered matched-packing decision and completed 672-lane replay.')
    directory=root/'e1'/stage
    protocol=read(directory/'protocol.json')
    if stage=='E1b':
        pilot=read(root/'e1/E1a/status.json')
        pilot_protocol=read(root/'e1/E1a/protocol.json')
        if pilot.get('stage')!='complete' or not archived_flat_passed(root/'e1/E1a',pilot):
            raise SystemExit('E1b held: E1a flat control has not passed.')
        if any(pilot_protocol[k]!=protocol[k] for k in ('times_s','random_seed','packing','replay672_summary_sha256')):
            raise ValueError('E1 stages differ in search or packing rules')
    status_path=directory/'status.json'
    previous=read(status_path) if status_path.exists() else None
    if previous is not None and not (resume or inspect):
        raise FileExistsError('E1 stage already started; use --resume to reconcile its archived candidates.')
    if not previous and not (resume or inspect) and ((directory/'warm').exists() or (directory/'g01').exists()):
        raise FileExistsError('E1 candidate directories exist without the stage status; inspect them with --inspect-resume.')
    if previous and previous.get('stage')=='complete' and not inspect:
        if previous.get('completed_new')!=protocol['budget']:
            raise ValueError('Complete E1 stage has an inconsistent physical budget')
        return
    sys.path.insert(0,str(REPO/'docs/paper/english'))
    from score_decision_oracle import score_trajectories
    completed_floor=previous['completed_new'] if previous else 0
    cases=protocol['cases']
    rows=[]; pending={}; states={}; incumbents={}; history=[]
    status=dict(stage='running',generation=0,completed_new=0,submitted_new=0,completed_reused=0)
    zeros=np.zeros((len(cases),len(protocol['times_s']),2))
    warm_specs=[(directory/'warm'/f'w{i:02}',label,[(*ALPHAS,0.)[i]]*len(cases),zeros) for i,label in enumerate(LABELS)]
    warm_results=evaluate_group(root,directory,protocol,warm_specs,status,score_trajectories,resident,inspect,completed_floor)
    if warm_results is None:
        return
    for i,raw in enumerate(warm_results):
        for c,r in zip(cases,raw):
            pending.setdefault(c['case'],[]).append(dict(case=c['case'],**r,generation=0,candidate=i,
                                                        physical_slot=c['slot'],knots_npz='',knots_lane=''))
    for c in cases:
        name=c['case'];warm=pending[name]
        seed_lane=min(range(5),key=lambda i:float(warm[i]['V']))
        for row in warm:
            row['alpha_seed']=ALPHAS[seed_lane]
        rows.extend(warm)
        mean=np.zeros((len(protocol['times_s']),2))
        rng=np.random.default_rng(np.random.SeedSequence([protocol['random_seed'],LAYOUTS.index(c['layout']),c['group']]))
        states[name]=dict(mean=mean,sigma=np.broadcast_to(SIGMA_INIT,mean.shape),alpha=ALPHAS[seed_lane],rng=rng)
        incumbents[name]=min(warm,key=lambda r:float(r['V']))
    if not inspect and not (directory/'candidates.csv').exists():
        csv_write(directory/'candidates.csv',rows)
    recorded_history=read(directory/'history.json') if (directory/'history.json').exists() else []
    for gen in range(1,protocol['generations']+1):
        started=time.monotonic()
        generation_dir=directory/f'g{gen:02}'
        samples=[]
        alphas=[]
        for c in cases:
            state=states[c['case']]
            sample=state['rng'].normal(state['mean'],state['sigma'],(protocol['population'],len(protocol['times_s']),2))
            samples.append(np.clip(sample,-BOUND,BOUND))
            alphas.append(state['alpha'])
        samples=np.stack(samples)
        scored={c['case']:[] for c in cases}
        specs=[(generation_dir/f'c{i:02}',f'CEM_{gen:02}_{i:02}',alphas,samples[:,i])
               for i in range(protocol['population'])]
        physical_rows=evaluate_group(root,directory,protocol,specs,status,score_trajectories,resident,inspect,completed_floor)
        if physical_rows is None:
            return
        for i,physical in enumerate(physical_rows):
            folder=specs[i][0]
            for c,r,alpha in zip(cases,physical,alphas):
                row=dict(case=c['case'],**r,generation=gen,candidate=i,physical_slot=c['slot'],alpha_seed=alpha,
                         knots_npz=str(folder/'knots.npz'),knots_lane=c['slot'])
                scored[c['case']].append(row)
                rows.append(row)
        for j,c in enumerate(cases):
            name=c['case'];state=states[name];genrows=scored[name]
            mean,sigma,elite=update_distribution(state['mean'],samples[j],[r['V'] for r in genrows])
            state.update(mean=mean,sigma=sigma)
            incumbents[name]=min([incumbents[name],*genrows],key=lambda r:float(r['V']))
            history.append(dict(case=name,generation=gen,elites=elite.tolist(),elite_V=[genrows[i]['V'] for i in elite],
                                min_V=incumbents[name]['V'],incumbent_generation=incumbents[name]['generation'],
                                incumbent_candidate=incumbents[name]['candidate'],next_mean=mean.tolist(),next_std=sigma.tolist()))
        previous_rows=recorded_history[(gen-1)*len(cases):gen*len(cases)]
        if previous_rows and previous_rows!=history[-len(cases):]:
            raise ValueError(f'Recovered CEM elites, moments or incumbent differ in generation {gen}')
        count=len(rows)
        status.update(generation=gen,completed_new=count,submitted_new=count)
        if inspect:
            print(f'{stage} generation {gen} verified: {count} physical rows, original CEM history checked')
        elif not previous or gen>=previous.get('generation',0):
            csv_write(directory/'candidates.csv',rows)
            save(directory/'history.json',history)
            csv_write(directory/'best.csv',list(incumbents.values()))
            if not (generation_dir/'generation_time.json').exists():
                elapsed=time.monotonic()-started
                recovered=previous is not None and previous['submitted_new']>(len(LABELS)+(gen-1)*protocol['population'])*len(cases)
                if recovered:
                    elapsed=max(elapsed,time.time()-(specs[0][0]/'settings.json').stat().st_mtime)
                save(generation_dir/'generation_time.json',dict(full_generation_s=elapsed,recovered_start_from_first_settings=recovered))
            if not previous or count>=previous['completed_new']:
                save(status_path,status)
    flat=[r for k,r in incumbents.items() if k.startswith('flat_original')]
    passed=pure_cem_flat_passed(rows) if stage=='E1a' else all(int(r['C'])==1 for r in flat)
    status.update(stage='complete',flat_control_passed=passed,completed_total=len(rows))
    if status['submitted_new']!=protocol['new_budget'] or status['completed_new']!=len(rows) or len(rows)!=protocol['budget']:
        raise ValueError('Realized E1 count differs from the registered budget')
    if not inspect:
        save(status_path,status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=('E1a','E1b'),required=True)
    actions=p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare-only',action='store_true')
    actions.add_argument('--execute',action='store_true')
    actions.add_argument('--inspect-resume',action='store_true',help='CPU-only: reconcile frozen inputs, scores and CEM history through the archived prefix')
    p.add_argument('--resident',action='store_true',help='Use one queued Isaac application/scene per CEM generation or warm-start block')
    p.add_argument('--resume',action='store_true',help='Resume only from complete physical candidate archives or unstarted inputs')
    p.add_argument('--knot-times',type=Path,help='Optional check against the 27 registered knot times')
    p.add_argument('--seed',type=int,help='CEM search RNG seed, declared independently of physics groups')
    args=p.parse_args();root=args.run.resolve()
    if (args.resident or args.resume) and not args.execute:
        p.error('--resident and --resume require --execute')
    if args.prepare_only:
        prepare(root,args.stage,read(args.knot_times) if args.knot_times else KNOT_TIMES,
                401 if args.seed is None else args.seed)
    else:
        if args.knot_times is not None or args.seed is not None:
            p.error('Execution reads the stage protocol already on disk')
        execute(root,args.stage,resident=args.resident,resume=args.resume,inspect=args.inspect_resume)


if __name__=='__main__':
    main()
