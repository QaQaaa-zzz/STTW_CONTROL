"""A1 short-window, candidate-parallel CEM. 'prepare' is CPU-only; 'execute' is invoked under the shared GPU lease."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
W=Path('/home/qy/STTW_CONTROL/runs/worktrees/pcrp-decision-delivery')
sys.path[:0]=[str(W/'learning/cli'),str(W/'learning/src'),str(W/'docs/paper/english')]
from pcrp_oracle_search import BOUND,SIGMA_MIN,LAB,ResidentGeneration,save,read,sha,update_distribution
from score_decision_oracle import score_trajectories
from sttw_control.isaac_evaluation_batch import batch_settings

LAYOUTS=['flat_original','position_s4.2','position_s4.5','position_s3.9']
TIMES=np.array([0.,1.,1.2,1.6,2.,2.4,2.8,3.2,3.6,3.8,4.,10.])
FREE=slice(2,10)
SIGMA=np.array([.20,.05])
N=32
GENERATIONS=8
BASE=ROOT.parent/'h75_contact_oat/position/flat_original_v2.6_seed401.json'
SOURCE=[W/'learning/cli/evaluate_isaac_gpu.py',W/'learning/src/sttw_control/isaac_resident_evaluation.py',
        W/'learning/src/sttw_control/isaaclab_vector_env.py',W/'learning/src/sttw_control/pcrp_oracle.py',
        W/'learning/src/sttw_control/isaac_evaluation_batch.py',W/'learning/cli/pcrp_oracle_search.py',
        W/'docs/paper/english/score_decision_oracle.py',W/'docs/paper/english/analyze_matched_controls.py']

def prepare():
    step=read(ROOT/'step0/protocol.json');past=pd.read_csv(ROOT/'e1/E1a/candidates.csv')
    more=pd.read_csv(ROOT/'archive_baseline/scores.csv')
    src=read(ROOT.parent/'pcrp_curvature_governor_20261121/source_cases.json')
    shifts=read(ROOT/'e2/geometry.json')
    cases=[]
    for layout in LAYOUTS:
        name=f'{layout}_v2.6_seed401';folder=Path(src[name]);raw=read(folder/'contract.json')
        s=raw['settings'];terrain=Path(s['terrains']['evaluation'])
        hist=past[past.case.eq(name)&past.generation.eq(0)] if layout!='position_s3.9' else more[more.case.eq(name)]
        warm=hist[hist.method.str.startswith('fixed')].sort_values('V',kind='stable').iloc[0]
        case=dict(case=name,layout=layout,seed=401,alpha=float(warm.method.removeprefix('fixed')),
            alpha_source=str(ROOT/('e1/E1a/candidates.csv' if layout!='position_s3.9' else 'archive_baseline/scores.csv')),
            reference_V=float(warm.V),terrain=str(terrain),terrain_sha256=read(terrain/'identity.json')['sha256'],
            randomization=s['randomization'],initial_state=s['initial_state'],offsets={})
        assert s['initial_state']==read(BASE)['initial_state'] and s['plant_options']==read(BASE)['plant_options']
        assert s['seed']==401 and s['user_speed_m_s']==2.6 and s['lower_sha256']==step['identities']['local300']
        if layout!='flat_original':
            center=float(layout.split('_s')[1])
            for row in shifts:
                d=row['ds_cm']
                if row['center_m']==center and abs(d)==1:
                    target=Path(row['terrain']);case['offsets'][str(d)]=dict(terrain=str(target),sha256=read(target/'identity.json')['sha256'])
            assert set(case['offsets'])=={'-1','1'}
        cases.append(case)
    reg=dict(role='archived DEV oracle development on four previously examined PCRP DEV cases; not an independent test',
        cases=cases,bundle=step['bundle'],identity=step['identities'],code_commit=subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(f):sha(f) for f in [*SOURCE, Path(__file__)]},base_config=str(BASE),base_sha256=sha(BASE),
        times_s=TIMES.tolist(),free_indices=list(range(2,10)),outside='zero through t=1.0 and from t=4.0; frozen linear interpolation on native 50Hz register',
        initial_std=SIGMA.tolist(),floor=SIGMA_MIN.tolist(),bounds=BOUND.tolist(),seed=401,seed_mapping='PCG64 SeedSequence([401, case_index]); one stream per declared case',
        update='six lowest V, ties by candidate index; 0.7 elite mean + 0.3 previous mean, six elite std with floors. Carry those six unchanged, then draw 26 new candidates on each later round.',
        population_per_case=32,real_physics_envs_per_search_step=128,zero_reference_physical=128,search_max=1024,
        single_nominal=4,shifted_geometry=6,scope_max=1162,global_cap=1200,
        clock='Target around 30 minutes for search. Bark if waiting beyond 30 minutes. Run eight 128-lane CEM generations and the registered verification, bounded by physical count.',
        single='One genuine one-environment replay per selected case, same winning physical world origin, terrain/seed/alpha/offsets; raw bitwise fields and finite errors reported.',
        robust='Two batches of three tilted-location assets; each case keeps winning world origin, full bias, initial condition and alpha. Report nominal Task C and count of two full 10s survivors, and C at both offsets separately.',
        decision='Pure nonzero CEM flat V<=0.676, check in single slot. Route B if >=2/3 ridges have nominal C and >=1/2 shifted full-window survival. Exactly one qualifying ridge is partial (no expansion), no robust ridge with confirmed flat positive follows route A. If flat or physical verification incomplete, report searcher failure/incomplete. Expansion requires exact nominal packed versus single-slot replay at flat and at each qualifying ridge. Report nominal mismatches explicitly.',
        historical_reference_roles='generation-zero, same-slot zeros and all archived baselines excluded from pure search positives')
    path=OUT/'registration.json'
    if path.exists():
        if read(path)!=reg:raise ValueError('Existing A1 declaration differs')
    else:save(path,reg)
    return reg

def sample(states,g):
    pop=[]
    for state in states:
        new=state['rng'].normal(state['mu'],state['sigma'],size=(N if g==1 else N-6,8,2)).clip(-BOUND,BOUND)
        free=new if g==1 else np.concatenate([state['elite'],new])
        bias=np.zeros((N,len(TIMES),2));bias[:,FREE]=free
        assert np.any(free!=0,axis=(1,2)).all()
        pop.append(bias)
    return np.stack(pop)

def job(folder,cases,offsets,labels,origins=None):
    folder.mkdir(parents=True,exist_ok=True)
    spec=[];allbias=[]
    for c,z,names in zip(cases,offsets,labels,strict=True):
        spec.append(dict(case_labels=names,user_speed_m_s=2.6,seed=401,randomization=c['randomization'],
            alphas=[c['alpha']]*len(z),zero_middle=[False]*len(z),zero_reference_channels=[[False,False]]*len(z),
            terrain_path=c['terrain'],output=str(folder/c['case'])))
        allbias.extend(z)
    knot=folder/'knots.npz';np.savez(knot,times_s=TIMES,offsets=np.asarray(allbias))
    cfg=read(BASE);cfg.update(oracle_knots=str(knot),oracle_knots_sha256=sha(knot),evaluation_role='PCRP A1 archived DEV bounded CEM')
    if origins is not None:cfg['plant_options']['replay_clone_origins']=origins
    assert batch_settings(cfg,json.loads(json.dumps(spec)))['num_envs']==len(allbias)
    save(folder/'cases.json',spec);save(folder/'settings.json',cfg)
    return dict(config=str(folder/'settings.json'),terrain=spec[0]['terrain_path'],case_manifest=str(folder/'cases.json'),output=str(folder/'batch'))

def physical(folder,cases,offsets,labels,reg,stats,resident=None,idx=0,origins=None):
    n=sum(map(len,offsets));stats['submitted']+=n
    assert stats['submitted']<=reg['scope_max']
    j=job(folder,cases,offsets,labels,origins)
    save(OUT/'status.json',dict(stage=folder.name,**stats))
    if resident:
        resident.evaluate(idx,j)
    else:
        command=physics_command(reg)+['--case-manifest',j['case_manifest'],'--config',j['config'],'--output',j['output'],'--terrain',j['terrain']]
        with (folder/'simulation.log').open('w') as log:subprocess.run(command,cwd=W,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    batch=read(folder/'batch/contract.json');state=read(folder/'batch/status.json')
    assert batch['lab_physics']['num_envs']==n and state['policy_steps']==500 and state['stage']=='complete'
    assert batch['physics_dt']==.0002 and batch['lower_dt']==.005 and batch['policy_dt']==.02
    scores=[]
    for c,names in zip(cases,labels,strict=True):
        f=folder/c['case'];v=score_trajectories(f,names);split=read(f/'contract.json')
        assert split['settings']['alphas']==[c['alpha']]*len(v) and split['settings']['case_labels']==names
        assert all(r['sha256']==c['terrain_sha256'] for r in split['terrain'])
        for row in v:row.update(case=c['case'],alpha=c['alpha'],origin=split['origin'][row['lane']],knots=str(folder/'knots.npz'))
        scores.append(v)
    stats['finished']+=n
    save(folder/'score.json',scores);save(OUT/'status.json',dict(stage=folder.name+'_complete',**stats))
    return scores

def physics_command(reg):
    return [LAB,'-u',str(W/'learning/cli/evaluate_isaac_gpu.py'),'--backend','lab','--bundle',reg['bundle'],'--seconds','10']
def environment():
    return dict(os.environ,PYTHONPATH=str(W/'learning/src')+os.pathsep+os.environ.get('PYTHONPATH',''),PYTHONUNBUFFERED='1')

def equality(packed,isolated):
    axis2={'lower_trace','wheel_contact_points','wheel_contact_forces','executed_commands','body_poses'}
    with np.load(Path(packed['raw'])/'trajectory.npz') as a,np.load(Path(isolated['raw'])/'trajectory.npz') as b:
        out={}
        for key in a.files:
            axis=2 if key in axis2 else 1
            old=a[key].take(packed['lane'],axis=axis);new=b[key].take(0,axis=axis)
            equal=old.shape==new.shape and old.dtype==new.dtype and old.tobytes()==new.tobytes()
            out[key]=dict(bitwise=equal,max_abs=float(np.nanmax(np.abs(old.astype(float)-new.astype(float)))) if old.shape==new.shape and np.isfinite(old.astype(float)-new.astype(float)).any() else None)
    return out

def execute():
    reg=read(OUT/'registration.json')
    assert all(sha(Path(p))==v for p,v in reg['source_files'].items()) and sha(BASE)==reg['base_sha256']
    assert not (OUT/'status.json').exists(), 'A1 already has submitted physics; preserve its budget'
    begin=time.monotonic()
    cases=reg['cases']
    stats={'submitted':0,'finished':0,'complete_search_generations':0}
    zeros=np.zeros((4,N,len(TIMES),2))
    lists=lambda prefix:[[f'{prefix}{j:02}' for j in range(N)] for _ in cases]
    work=OUT/'physics';work.mkdir(exist_ok=True)
    zjob=job(work/'zero',cases,zeros,lists('zero'))
    states=[dict(mu=np.zeros((8,2)),sigma=np.broadcast_to(SIGMA,(8,2)).copy(),elite=None,
                 rng=np.random.default_rng(np.random.SeedSequence([401,i]))) for i in range(4)]
    winners=[None]*4
    # Native resident session keeps exactly 4*32 physical slots and their geometry across all search rounds.
    with ResidentGeneration(work,[zjob],physics_command(reg),environment()) as sess:
        start=time.monotonic();z=physical(work/'zero',cases,zeros,lists('zero'),reg,stats,sess,0)
        times=[]
        for g in range(1,GENERATIONS+1):
            block=sample(states,g);tick=time.monotonic()
            results=physical(work/f'g{g:02}',cases,block,lists('cem'),reg,stats,sess,g)
            times.append(time.monotonic()-tick)
            for i,(st,rows) in enumerate(zip(states,results,strict=True)):
                costs=np.array([r['V'] for r in rows])
                st['mu'],st['sigma'],ids=update_distribution(st['mu'],block[i,:,FREE,:],costs)
                st['elite']=block[i,ids,FREE,:].copy()
                ix=int(np.argmin(costs));r=rows[ix]
                if winners[i] is None or r['V']<winners[i]['score']['V']:
                    winners[i]=dict(score=r,generation=g,physical_lane=i*N+ix,zero_V_same_search_slot=z[i][ix]['V'],
                        offsets=block[i,ix].tolist())
            stats['complete_search_generations']=g
            save(OUT/'winners.json',winners)
    if not all(winners):
        save(OUT/'status.json',dict(stage='search_incomplete',**stats));return
    canonical=[];replays=[];robust=[]
    for i,(case,winner) in enumerate(zip(cases,winners,strict=True)):
        pick=winner['score'];offset=np.array(winner['offsets'])
        near=physical(work/f'single_{i}',[case],[[offset]],[['cem_single']],reg,stats,origins=[pick['origin']])[0][0]
        assert near['origin']==pick['origin']
        canonical.append(near)
        comparison=equality(pick,near)
        replays.append(dict(case=case['case'],packed_V=pick['V'],single_V=near['V'],packed_C=pick['C'],single_C=near['C'],
            all_recorded_arrays_bitwise=all(v['bitwise'] for v in comparison.values()),arrays=comparison))
        save(OUT/'single_comparisons.json',replays)
    for shift in (-1,1):
        displaced=[]
        for c in cases[1:]:
            m=c['offsets'][str(shift)]
            displaced.append(dict(c,terrain=m['terrain'],terrain_sha256=m['sha256']))
        origins=[v['score']['origin'] for v in winners[1:]]
        shifted=physical(work/f'shift_{shift:+d}',displaced,[[v['offsets']] for v in winners[1:]],[['cem_shift'] for _ in displaced],reg,stats,origins=origins)
        for (row,),center,c in zip(shifted,origins,displaced,strict=True):
            assert row['origin']==center
            robust.append(dict(case=c['case'],shift_cm=shift,full=row['full'],C=row['C'],V=row['V'],raw=row['raw']))
        save(OUT/'shift_results.json',robust)
    positive=winners[0]['score']['V']<=.676 and canonical[0]['V']<=.676
    per_case=[]
    for case,pick,single in zip(cases[1:],winners[1:],canonical[1:],strict=True):
        q=[r for r in robust if r['case']==case['case']]
        full=sum(r['full'] for r in q)
        per_case.append(dict(case=case['case'],nominal_C=single['C'],search_C=pick['score']['C'],shift_survived=full,shift_C=sum(r['C'] for r in q),
            qualifies=bool(single['C'] and pick['score']['C'] and full>=1)))
    n=sum(r['qualifies'] for r in per_case)
    verdict='flat_control_failed' if not positive else 'route_B_eligible' if n>=2 else 'route_A_no_robust_solution' if n==0 else 'one_ridge_partial'
    if verdict=='route_B_eligible' and (not replays[0]['all_recorded_arrays_bitwise'] or sum(v['qualifies'] and r['all_recorded_arrays_bitwise'] for v,r in zip(per_case,replays[1:]))<2):
        verdict='nominal_replay_disagreement_no_expansion'
    save(OUT/'decision.json',dict(verdict=verdict,flat_passed=bool(positive),ridge_cases=per_case,
        reproducible_bitwise=[r['all_recorded_arrays_bitwise'] for r in replays],wall_s=time.monotonic()-begin,**stats))
    save(OUT/'status.json',dict(stage='complete',decision=verdict,**stats))

def notified_result(code):
    # Only the new A1 service uses this command, after its own bounded subprocess has returned.
    prev=read(OUT/'status.json') if (OUT/'status.json').is_file() else dict(submitted=0,finished=0)
    if code:
        save(OUT/'status.json',dict(prev,stage='wall_clock_limit' if code in (124,137) else 'error',returncode=code))
    print(json.dumps(read(OUT/'status.json'),indent=2))

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('stage',choices=['prepare','execute','record-exit']);cli.add_argument('code',nargs='?',type=int)
    a=cli.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='execute':execute()
    else:notified_result(a.code)
