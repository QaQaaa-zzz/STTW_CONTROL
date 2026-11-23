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
                              reuse_warm=name in raw,origins=archive['origin'][:6],
                              randomization=settings['randomization']))
    return cases


def prepare(root, stage, times, seed):
    times = np.asarray(times,dtype=float)
    if (times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or times[0]!=0 or times[-1]!=10.
            or np.any(np.diff(times)<=0) or seed<0):
        raise ValueError('Supply an explicit increasing knot grid spanning 0–10 s and nonnegative RNG seed')
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
                  budget=len(cases)*326,new_budget=len(cases)*320+sum(6 for c in cases if not c['reuse_warm']),
                  role='Privileged case-specific feasibility search on previously examined DEV groups',
                  times_sha256_f64le=hashlib.sha256(times.astype('<f8').tobytes()).hexdigest(),
                  reference_bias_timing='50 Hz before original 200 Hz shaper; shared across each four lower ticks',
                  cases=cases,step0_protocol_sha256=sha(root/'step0/protocol.json'))
    path=directory/'protocol.json'
    if path.exists() and read(path)!=protocol:
        raise ValueError('Registered E1 protocol differs; existing run must not be overwritten')
    write(path,protocol)
    return directory,protocol


def update_distribution(mean, population, costs):
    elite=np.argsort(costs,kind='stable')[:6]
    return .3*mean+.7*population[elite].mean(0), np.maximum(SIGMA_MIN,population[elite].std(0)), elite


def batch(root, directory, protocol, groups, generation, offsets=None):
    directory.mkdir(parents=True,exist_ok=True)
    cfg=read(root.parent/'h75_contact_oat/position/flat_original_v2.6_seed401.json')
    all_cases=[];origins=[]
    for case,alpha_idx in groups:
        if generation:
            labels=[f'CEM_{generation:02}_{i:02}' for i in range(32)]
            alphas=[(0.,.25,.5,.75,1.)[alpha_idx]]*32
            zero=[False]*32
            # Pair all privileged candidates to the exact world placement of their fixed H75 seed.
            origins.extend([case['origins'][alpha_idx]]*32)
        else:
            labels=LABELS
            alphas=[0.,.25,.5,.75,1.,0.]
            zero=[False]*5+[True]
            origins.extend(case['origins'])
        all_cases.append(dict(case_labels=labels,user_speed_m_s=2.6,seed=case['group'],
                              randomization=case['randomization'],alphas=alphas,zero_middle=zero,
                              zero_reference_channels=[[False,False]]*len(labels),terrain_path=case['terrain'],
                              output=str(directory/case['case'])))
    cfg['plant_options']['replay_clone_origins']=origins
    # Each collision group contains only its own vehicle and terrain pair, including co-located clones.
    cfg['evaluation_role']='PCRP privileged case-specific DEV feasibility search, frozen plant and H75/Local300'
    if generation:
        knots=directory/'knots.npz'
        np.savez(knots,times_s=np.array(protocol['times_s']),offsets=np.concatenate(offsets))
        cfg['oracle_knots']=str(knots)
        cfg['oracle_knots_sha256']=sha(knots)
    write(directory/'cases.json',all_cases);write(directory/'settings.json',cfg)
    gate_protocol=read(root/'step0/protocol.json')
    cmd=[LAB,'-u',str(REPO/'learning/cli/evaluate_isaac_gpu.py'),'--backend','lab','--bundle',gate_protocol['bundle'],
         '--config',str(directory/'settings.json'),'--terrain',all_cases[0]['terrain_path'],
         '--case-manifest',str(directory/'cases.json'),'--output',str(directory/'batch'),'--seconds','10']
    before=time.monotonic()
    with (directory/'simulation.log').open('x') as log:
        subprocess.run(cmd,cwd=REPO,env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+
                       os.pathsep+os.environ.get('PYTHONPATH','')),stdout=log,stderr=subprocess.STDOUT,check=True)
    batch_status=read(directory/'batch/status.json')
    if batch_status['policy_steps']!=500 or batch_status['stage']!='complete':
        raise ValueError(f'Incomplete E1 generation in {directory}')
    write(directory/'timing.json',dict(wall_s=time.monotonic()-before,sampling_s=batch_status['elapsed_s'],
          export_s=batch_status['elapsed_with_output_s']-batch_status['elapsed_s'],lanes=len(origins)))
    return all_cases


def csv_write(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def execute(root,stage):
    gate_path=root/'step0/status.json'
    gate=read(gate_path) if gate_path.exists() else {}
    if gate.get('stage')!='complete' or gate.get('passed') is not True:
        raise SystemExit('E1 held: step 0 replay gate has not passed.')
    directory=root/'e1'/stage;protocol=read(directory/'protocol.json')
    if stage=='E1b':
        pilot=read(root/'e1/E1a/status.json')
        pilot_protocol=read(root/'e1/E1a/protocol.json')
        if pilot.get('stage')!='complete' or pilot.get('flat_control_passed') is not True:
            raise SystemExit('E1b held: E1a flat control has not passed.')
        if any(pilot_protocol[k]!=protocol[k] for k in ('times_s','random_seed','step0_protocol_sha256')):
            raise ValueError('E1a and E1b must use identical knots, search seed and protocol')
    if (directory/'status.json').exists():
        raise FileExistsError('E1 stage already started. Review its immutable generation outputs before a recovery.')
    sys.path.insert(0,str(REPO/'docs/paper/english'))
    from score_decision_oracle import score_trajectories
    cases=protocol['cases']
    missing=[(c,None) for c in cases if not c['reuse_warm']]
    status=dict(stage='running',generation=0,completed_new=0,completed_reused=0)
    write(directory/'status.json',status)
    if missing:
        batch(root,directory/'warm',protocol,missing,0)
    with (root/'archive_baseline/scores.csv').open() as f:
        archived=list(csv.DictReader(f))
    rows=[]; states={}; incumbents={};history=[]
    for c in cases:
        name=c['case']
        if c['reuse_warm']:
            records=[r for r in archived if r['case']==name]
            if len(records)!=6 or [r['method'] for r in records]!=LABELS:
                raise ValueError(f'Incomplete archived scoring at {name}')
            status['completed_reused']+=6
        else:
            records=score_trajectories(directory/'warm'/name,LABELS)
            status['completed_new']+=6
        seed_lane=min(range(5),key=lambda i:float(records[i]['V']))
        alpha=(0.,.25,.5,.75,1.)[seed_lane]
        warm=[]
        for i,r in enumerate(records):
            row={k:r[k] for k in ('method','raw','lane','V','A','B','C','full','hold','peak_path_m',
                                   'path_rmse_m','speed_rmse_m_s','time_s','terminal','hold_ratio')}
            warm.append(dict(case=name,**row,generation=0,candidate=i,alpha_seed=alpha,knots_npz='',knots_lane=''))
        rows.extend(warm)
        mean=np.zeros((len(protocol['times_s']),2))
        sequence=[protocol['random_seed'],LAYOUTS.index(c['layout']),c['group']]
        states[name]=dict(mean=mean,sigma=np.broadcast_to(SIGMA_INIT,mean.shape),lane=seed_lane,
                          rng=np.random.default_rng(np.random.SeedSequence(sequence)))
        incumbents[name]=min(warm,key=lambda r:float(r['V']))
    csv_write(directory/'candidates.csv',rows)
    for gen in range(1,11):
        started=time.monotonic()
        directory_gen=directory/f'g{gen:02}'
        population=[]
        for c in cases:
            state=states[c['case']]
            shape=(32,len(protocol['times_s']),2)
            sample=np.clip(state['rng'].normal(state['mean'],state['sigma'],shape),-BOUND,BOUND)
            population.append(sample)
        groups=[(c,states[c['case']]['lane']) for c in cases]
        manifests=batch(root,directory_gen,protocol,groups,gen,population)
        sim_done=time.monotonic()
        for j,(c,manifest,sample) in enumerate(zip(cases,manifests,population)):
            name=c['case']
            scored=score_trajectories(Path(manifest['output']),manifest['case_labels'])
            state=states[name]
            mean,sigma,elite=update_distribution(state['mean'],sample,[r['V'] for r in scored])
            state.update(mean=mean,sigma=sigma)
            genrows=[]
            for i,r in enumerate(scored):
                row={k:r[k] for k in ('method','raw','lane','V','A','B','C','full','hold','peak_path_m',
                                       'path_rmse_m','speed_rmse_m_s','time_s','terminal','hold_ratio')}
                genrows.append(dict(case=name,**row,generation=gen,candidate=i,alpha_seed=manifest['alphas'][i],
                                    knots_npz=str(directory_gen/'knots.npz'),knots_lane=j*32+i))
            rows.extend(genrows)
            incumbents[name]=min([incumbents[name],*genrows],key=lambda r:float(r['V']))
            history.append(dict(case=name,generation=gen,elites=elite.tolist(),elite_V=[scored[i]['V'] for i in elite],
                                min_V=incumbents[name]['V'],incumbent_generation=incumbents[name]['generation'],
                                incumbent_candidate=incumbents[name]['candidate'],next_mean=mean.tolist(),next_std=sigma.tolist()))
        csv_write(directory/'candidates.csv',rows);write(directory/'history.json',history)
        csv_write(directory/'best.csv',list(incumbents.values()))
        t=read(directory_gen/'timing.json')
        t.update(scoring_and_archive_s=time.monotonic()-sim_done,full_generation_s=time.monotonic()-started)
        write(directory_gen/'timing.json',t)
        status.update(generation=gen,completed_new=status['completed_new']+32*len(cases))
        write(directory/'status.json',status)
    flat=[r for k,r in incumbents.items() if k.startswith('flat_original')]
    status.update(stage='complete',flat_control_passed=all(int(r['C'])==1 for r in flat),
                  completed_total=len(rows))
    if status['completed_new']!=protocol['new_budget'] or len(rows)!=protocol['budget']:
        raise ValueError('E1 realized run count differs from the registered budget')
    write(directory/'status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=('E1a','E1b'),required=True)
    actions=p.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare-only',action='store_true')
    actions.add_argument('--execute',action='store_true')
    p.add_argument('--knot-times',type=Path,help='Explicit JSON array, including both 0 and 10 s')
    p.add_argument('--seed',type=int,help='CEM search RNG seed, declared independently of physics groups')
    args=p.parse_args();root=args.run.resolve()
    if args.prepare_only:
        if args.knot_times is None or args.seed is None:
            p.error('Registering E1 requires --knot-times and --seed')
        prepare(root,args.stage,read(args.knot_times),args.seed)
    else:
        if args.knot_times is not None or args.seed is not None:
            p.error('Execution reads the stage protocol already on disk')
        execute(root,args.stage)


if __name__=='__main__':
    main()
