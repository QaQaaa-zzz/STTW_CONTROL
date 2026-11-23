"""Bounded E1a/E1b CEM above frozen H75, with per-generation raw archives."""
import argparse
import csv
import hashlib
import os
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


def batch(root, directory, protocol, label, alphas, offsets, score):
    cases = protocol['cases']
    directory.mkdir(parents=True,exist_ok=False)
    cfg = read(root.parent/'h75_contact_oat/position/flat_original_v2.6_seed401.json')
    manifests = case_manifest(directory,cases,label,alphas)
    cfg['evaluation_role'] = 'PCRP E1 matched packing: one unchanged native physical slot per case'
    # Use the same oracle adapter, including exactly zero input on all warm starts.
    knots = directory/'knots.npz'
    np.savez(knots,times_s=protocol['times_s'],offsets=np.asarray(offsets))
    cfg.update(oracle_knots=str(knots), oracle_knots_sha256=sha(knots))
    write(directory/'cases.json',manifests)
    write(directory/'settings.json',cfg)
    gate_protocol=read(root/'step0/protocol.json')
    cmd=['/bin/bash',str(REPO/'learning/scripts/gpu_queue.sh'),LAB,'-u',
         str(REPO/'learning/cli/evaluate_isaac_gpu.py'),'--backend','lab','--bundle',gate_protocol['bundle'],
         '--config',str(directory/'settings.json'),'--terrain',manifests[0]['terrain_path'],
         '--case-manifest',str(directory/'cases.json'),'--output',str(directory/'batch'),'--seconds','10']
    before=time.monotonic()
    with (directory/'simulation.log').open('x') as log:
        subprocess.run(cmd,cwd=REPO,env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+
                       os.pathsep+os.environ.get('PYTHONPATH','')),stdout=log,stderr=subprocess.STDOUT,check=True)
    batch_status=read(directory/'batch/status.json')
    batch_wall=time.monotonic()-before
    if batch_status['policy_steps']!=500 or batch_status['stage']!='complete':
        raise ValueError(f'Incomplete E1 subbatch in {directory}')
    meta=read(directory/'batch/contract.json')
    if meta['lab_physics']['num_envs']!=len(cases):
        raise ValueError(f'Incorrect number of physical slots in {directory}')
    # The warm and CEM runs intentionally use different policy conditions. The
    # full packing, world transforms and all frozen physical inputs must agree.
    identity_keys=('backend','identity','terrain','bodies','origin','physics_dt','lower_dt','policy_dt',
                   'randomization_readback','delay_steps')
    same={k:meta[k] for k in identity_keys}
    same.update(plant=meta['settings']['plant_options'],terrain_ids=meta['settings']['lane_terrain_ids'],
                fixed_randomization_groups=meta['settings']['fixed_randomization_groups'],
                terrain_layout=meta['lab_physics']['terrain_layout'])
    reference=root/'e1'/protocol['stage']/'matched_runtime.json'
    if not reference.exists():
        write(reference,same)
    elif same!=read(reference):
        raise ValueError(f'Physical batch or lane identity differs at {directory}')
    records=[]
    for c,m in zip(cases,manifests):
        folder=Path(m['output'])
        split=read(folder/'contract.json')
        if (split['batch_source']['lane_start']!=c['slot'] or split['batch_source']['lane_stop']!=c['slot']+1
                or split['settings']['case_labels']!=[label] or split['terrain'][0]['sha256']!=c['terrain_sha256']):
            raise ValueError(f'Case assignment differs at {folder}')
        row, = score(folder,[label])
        records.append(row)
    timing=dict(wall_s=batch_wall,sampling_s=batch_status['elapsed_s'],
                export_s=batch_status['elapsed_with_output_s']-batch_status['elapsed_s'],
                scoring_and_archive_s=time.monotonic()-before-batch_wall,lanes=len(cases),
                measured_wall_includes_queue=True)
    write(directory/'timing.json',timing)
    return records


def csv_write(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def execute(root,stage):
    authorization=read(root/'paired_authorization.json')
    replay=read(root/'replay672/analysis/summary.json')
    if authorization.get('decision')!='matched_packing' or not replay.get('passed') or replay['matched_lanes']!=672:
        raise SystemExit('E1 requires the registered matched-packing decision and completed 672-lane replay.')
    directory=root/'e1'/stage
    protocol=read(directory/'protocol.json')
    if stage=='E1b':
        pilot=read(root/'e1/E1a/status.json')
        pilot_protocol=read(root/'e1/E1a/protocol.json')
        if pilot.get('stage')!='complete' or pilot.get('flat_control_passed') is not True:
            raise SystemExit('E1b held: E1a flat control has not passed.')
        if any(pilot_protocol[k]!=protocol[k] for k in ('times_s','random_seed','packing','replay672_summary_sha256')):
            raise ValueError('E1 stages differ in search or packing rules')
    status_path=directory/'status.json'
    if status_path.exists():
        raise FileExistsError('E1 stage already started; its recorded physical batches consume budget.')
    sys.path.insert(0,str(REPO/'docs/paper/english'))
    from score_decision_oracle import score_trajectories
    cases=protocol['cases']
    rows=[]; pending={}; states={}; incumbents={}; history=[]
    status=dict(stage='running',generation=0,completed_new=0,submitted_new=0,completed_reused=0)
    zeros=np.zeros((len(cases),len(protocol['times_s']),2))
    def evaluate(folder,label,alphas,bias):
        status.update(subbatch=str(folder.relative_to(directory)),submitted_new=status['submitted_new']+len(cases))
        write(status_path,status)
        scores=batch(root,folder,protocol,label,alphas,bias,score_trajectories)
        status['completed_new']+=len(cases)
        write(status_path,status)
        return scores
    for i,label in enumerate(LABELS):
        raw=evaluate(directory/'warm'/f'w{i:02}',label,[(*ALPHAS,0.)[i]]*len(cases),zeros)
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
    csv_write(directory/'candidates.csv',rows)
    for gen in range(1,11):
        started=time.monotonic()
        generation_dir=directory/f'g{gen:02}'
        samples=[]
        alphas=[]
        for c in cases:
            state=states[c['case']]
            sample=state['rng'].normal(state['mean'],state['sigma'],(32,len(protocol['times_s']),2))
            samples.append(np.clip(sample,-BOUND,BOUND))
            alphas.append(state['alpha'])
        samples=np.stack(samples)
        scored={c['case']:[] for c in cases}
        for i in range(32):
            folder=generation_dir/f'c{i:02}'
            physical=evaluate(folder,f'CEM_{gen:02}_{i:02}',alphas,samples[:,i])
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
        csv_write(directory/'candidates.csv',rows);write(directory/'history.json',history)
        csv_write(directory/'best.csv',list(incumbents.values()))
        write(generation_dir/'generation_time.json',dict(full_generation_s=time.monotonic()-started))
        status.update(generation=gen)
        write(status_path,status)
    flat=[r for k,r in incumbents.items() if k.startswith('flat_original')]
    status.update(stage='complete',flat_control_passed=all(int(r['C'])==1 for r in flat),completed_total=len(rows))
    if status['submitted_new']!=protocol['new_budget'] or status['completed_new']!=len(rows) or len(rows)!=protocol['budget']:
        raise ValueError('Realized E1 count differs from the registered budget')
    write(status_path,status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=('E1a','E1b'),required=True)
    actions=p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare-only',action='store_true')
    actions.add_argument('--execute',action='store_true')
    p.add_argument('--knot-times',type=Path,help='Optional check against the 27 registered knot times')
    p.add_argument('--seed',type=int,help='CEM search RNG seed, declared independently of physics groups')
    args=p.parse_args();root=args.run.resolve()
    if args.prepare_only:
        prepare(root,args.stage,read(args.knot_times) if args.knot_times else KNOT_TIMES,
                401 if args.seed is None else args.seed)
    else:
        if args.knot_times is not None or args.seed is not None:
            p.error('Execution reads the stage protocol already on disk')
        execute(root,args.stage)


if __name__=='__main__':
    main()
