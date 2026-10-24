"""One real mid-turn capacity measurement, using existing frozen predictors."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BASE=Path(__file__).resolve().parent
WORKTREE=BASE.parents[3]


def prepare(root):
    from sttw_control.preference_best import atomic_json,digest
    root.mkdir(parents=True,exist_ok=False)
    nominal=WORKTREE/'runs/low_tight_preference_nominal_v1_attempt2/R1p1'
    dual=WORKTREE/'runs/low_tight_dual_nominal_review_v1'
    receipt=json.loads((dual/'analysis_status.json').read_text())
    if receipt['phase']!='completed' or receipt['completed_cases']!=48:
        raise ValueError('historical-model review must complete first')
    source=nominal/'declaration.json';d=json.loads(source.read_text())
    prefix_cli=BASE.parent/'alpha_physical_boundary/collect_prefixes.py'
    files=[Path(__file__),prefix_cli,BASE/'prepare_middle_state_capability.py',
           BASE/'analyze_middle_state_capability.py',source,
           WORKTREE/'learning/src/sttw_control/geometric_capture.py',
           WORKTREE/'learning/cli/probe_balance_boundary.py']
    files += [WORKTREE/'learning/src/sttw_control'/(name+'.py') for name in
              ['path_command_policy','path_command_env','path_command_training','path_reference',
               'balance_probe','conditional_tolerance','frozen_local_lower','controller','closed_loop_kernel','teleop_env']]
    files += [Path(d['policies']['H75'][key]) for key in ('actor','checkpoint')]
    sources={str(p):digest(p) for p in files}
    archive=root/'source_at_launch';archive.mkdir()
    for i,p in enumerate(files):(archive/f'{i:02d}_{p.name}').write_bytes(p.read_bytes())
    routes=[]
    for route in d['routes']:
        name='left' if route['name'].startswith('left') else 'right';child=root/name;child.mkdir()
        p=dict(project='STTW_CONTROL',model=d['policies']['H75'],policy_name='H75',
            source_declaration=str(source),path_evidence_directory=str(nominal/'evaluation'),
            route=route['name'],builder=route['builder'],speeds=d['speeds'],alphas=[0.,1.],
            phase_fractions=[.5],forces_N=[0.],pulse_duration_s=.1,observation_s=4.,
            declared_prefixes=4,declared_cases=4,maximum_prepare_lower_ticks=8000,
            maximum_scan_lower_ticks=0,source_sha256=sources,training_steps=0,
            purpose='Capture only; generated middle-continuation scan is never executed',
            stop_rule='One true first crossing; invalid prefix retained as unavailable, no phase or state replacement')
        atomic_json(child/'prepare_plan.json',p);routes.append(name)
    plan=dict(experiment='tight_midturn_capability_v1',source_declaration=str(source),
        nominal_source=str(nominal),dual_review=str(dual),routes=routes,
        source_sha256=sources,force_grid_N=[0.,-16.,16.,-32.,32.,-48.,48.,-64.,64.],
        prefixes=8,declared_cases=72,maximum_prefix_lower_ticks=16000,
        maximum_suffix_lower_ticks=57600,maximum_lower_ticks=73600,
        baseline_reuse=str(WORKTREE/'runs/low_middle_state_capability_v1_attempt3/standardized_candidates.json'),
        inherited_fits=2,new_classifier_fits=0,policy_updates=0,
        hypothesis='Path-first deceleration and altered steering history may change signed no-fall capacity; compare actual task cost separately',
        scope='Own-policy state plus captured executed reference plus frozen local300; middle inactive during suffix',
        stop_rule='All eight states including nominal task failures; one force grid, no refit or adaptive refinement',
        capture_optimization='vmap four contexts, scan50, freeze each complete state at first20ms crossing',
        counters='prefix and suffix active lower ticks separately; no nominal/dual replay counted as new')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=72,completed_cases=0,
        completed_prefixes=0,active_lower_ticks=0,policy_updates=0,new_classifier_fits=0))
    print(json.dumps(dict(root=str(root),prefixes=8,cases=72,maximum_lower_ticks=73600)))


def execute(root):
    from sttw_control.preference_best import atomic_json,digest
    plan=json.loads((root/'plan.json').read_text())
    state=dict(phase='running',pid=os.getpid(),stage='starting',completed_cases=0,declared_cases=72,
        completed_prefixes=0,prefix_lower_ticks=0,suffix_lower_ticks=0,active_lower_ticks=0,
        new_classifier_fits=0,policy_updates=0)
    def update(**kw):
        state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
    def call(name,command,cpu=False):
        with (root/(name+'.log')).open('x') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,**({'JAX_PLATFORMS':'cpu'} if cpu else {})})
            update(stage=name,child_pid=process.pid)
            if process.wait():raise RuntimeError(name+' failed; retain source and partial costs')
    update()
    try:
        for file,sha in plan['source_sha256'].items():
            if digest(file)!=sha:raise ValueError('execution source changed '+file)
        rows=[];identity=None;nominal=json.loads((Path(plan['nominal_source'])/'evaluation/results.json').read_text())['cases']
        for name in plan['routes']:
            child=root/name
            call('capture_'+name,[sys.executable,'-u',str(BASE.parent/'alpha_physical_boundary/collect_prefixes.py'),'--run',str(child)])
            receipt=json.loads((child/'prepare_status.json').read_text())
            if receipt['phase']!='completed' or receipt['completed_cases']!=4:raise ValueError('incomplete geometric prefixes')
            bundle=json.loads((child/'prefix_results.json').read_text())
            if identity is not None and identity!=bundle['physical_identity']:raise ValueError('prefix physical identities differ')
            identity=bundle['physical_identity']
            for row in bundle['cases']:
                old=next(r for r in nominal if r['policy']=='H75' and r['route']==row['route'] and r['speed']==row['speed'] and r['alpha']==row['alpha'])
                row['nominal_task_recovered']=old['task_recovered'];rows.append(row)
            (child/'scan_plan.json').rename(child/'generated_middle_scan_NOT_EXECUTED.json')
            state['prefix_lower_ticks']+=receipt['active_lower_ticks'];state['completed_prefixes']+=4
            state['active_lower_ticks']=state['prefix_lower_ticks'];update()
        d=json.loads(Path(plan['source_declaration']).read_text())
        bundle=root/'prefix_bundle.json'
        atomic_json(bundle,dict(cases=rows,physical_identity=identity,declaration=d,source_declaration=plan['source_declaration']))
        scan=root/'scan'
        call('freeze_forecasts',[sys.executable,'-u',str(BASE/'prepare_middle_state_capability.py'),
             '--output',str(scan),'--prefix-bundle',str(bundle),'--reuse-baselines',plan['baseline_reuse'],
             '--forces-N',*map(str,plan['force_grid_N'])],cpu=True)
        prediction=json.loads((scan/'prepare_status.json').read_text())
        if prediction['phase']!='completed' or prediction['classifier_fits']!=0 or prediction['cases']!=72:
            raise ValueError('forecasts missing or unexpected refit')
        atomic_json(scan/'scan_launch_receipt.json',dict(started_unix=time.time(),parent_run=str(root),policy_updates=0))
        call('force_scan',[sys.executable,'-u','learning/cli/probe_balance_boundary.py','--plan',str(scan/'scan_plan.json')])
        receipt=json.loads((scan/'status.json').read_text())
        if receipt['phase']!='completed' or receipt['completed_cases']!=72:raise ValueError('force scan incomplete')
        state['completed_cases']=72;state['suffix_lower_ticks']=receipt['active_lower_ticks']
        state['active_lower_ticks']=state['prefix_lower_ticks']+state['suffix_lower_ticks']
        if state['active_lower_ticks']>plan['maximum_lower_ticks']:raise ValueError('active physical budget exceeded')
        update()
        call('analyze',[sys.executable,'-u',str(BASE/'analyze_middle_state_capability.py'),'--run',str(scan)],cpu=True)
        update(phase='completed',stage='all_geometric_capacity_results_ready',child_pid=None)
    except BaseException as exc:
        # Charge completed child physics even when that child failed.
        prefix=0
        for name in plan['routes']:
            path=root/name/'prepare_status.json'
            if path.exists():prefix+=json.loads(path.read_text()).get('active_lower_ticks',0)
        path=root/'scan/status.json';suffix=json.loads(path.read_text()).get('active_lower_ticks',0) if path.exists() else 0
        update(phase='error',error=repr(exc),prefix_lower_ticks=prefix,suffix_lower_ticks=suffix,active_lower_ticks=prefix+suffix)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    (execute if args.execute else prepare)(args.run.resolve())
