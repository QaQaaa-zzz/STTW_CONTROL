#!/usr/bin/env python3
"""Frozen teleop protocol with persistent, non-bypassable phase prerequisites."""
import argparse,json,pickle
from pathlib import Path
from sttw_control.teleop_commands import load_teleop_config,random_schedule
from sttw_control.teleop_budget import Budget,BudgetExhausted
from sttw_control.teleop_metrics import core_gate,governor_nominal_metrics,random_gate

def run(a,c):
    from sttw_control.teleop_env import TeleopEnv
    from sttw_control.teleop_review import Review
    from sttw_control.teleop_contract_checks import physical_contracts
    from sttw_control.teleop_provenance import provenance
    from sttw_control.teleop_reporting import report
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);budget=Budget(root,a.predictor_budget)
    review=None
    try:
        if (root/'status.json').exists():
            old=json.loads((root/'status.json').read_text())
            if old['status'].startswith(('stopped','budget_exhausted','numerical_failure','implementation_error')) and not (a.resume_contracts and old['status'].startswith('stopped_contracts')):
                raise ValueError('run stopped; no automatic continuation or replacement of failed evidence')
        import jax
        if a.float64:jax.config.update('jax_enable_x64',True)
        env=TeleopEnv(c);provenance(root,a.config,env)
        frozen=root/'frozen_config.json'
        if frozen.exists() and json.loads(frozen.read_text())!=c:raise ValueError('frozen configuration mismatch')
        if not frozen.exists():frozen.write_text(json.dumps(c,indent=2)+'\n')
        contracts=root/'contracts';contracts.mkdir(exist_ok=True)
        if not (contracts/'preparation.json').exists():
            from teleop_governor_check import prepare
            prepared,prep=prepare(env,budget,contracts)
        else:
            prep=json.loads((contracts/'preparation.json').read_text())
            with (contracts/'prepared_snapshot.pkl').open('rb') as f:prepared=pickle.load(f)
        pointer=root/'contracts/current_contracts.json'
        current=json.loads(pointer.read_text()) if pointer.exists() else None
        if current:
            if current['float64']!=a.float64:raise ValueError('validated precision differs from requested execution')
            with (root/current['snapshot']).open('rb') as f:prepared=pickle.load(f)
        elif a.float64:
            from sttw_control.teleop_precision import promote_snapshot
            prepared=promote_snapshot(prepared)
        review=Review(env,prepared,budget,root,c)
        if (root/'metrics.json').exists():review.restore()
        if a.resume_contracts and (root/'status.json').exists():
            history=root/'status_history';history.mkdir(exist_ok=True)
            import time,shutil
            shutil.copy2(root/'status.json',history/(str(time.time_ns())+'.json'))
        if not prep['passed']:
            review.status.update(status='stopped_preparation',classification='baseline_not_ready')
            review.status['gates']['interfaces']='preparation_failed';review.status['gates']['prediction_replay']='not_run';review.save();return
        if current:checks=json.loads((root/current['result']).read_text())
        elif not (contracts/'physical_contracts.json').exists():checks=physical_contracts(env,prepared,budget,contracts)
        else:checks=json.loads((contracts/'physical_contracts.json').read_text())
        if not checks['passed']:
            review.status.update(status='stopped_contracts',classification='timing_or_state_mismatch')
            review.status['gates']['prediction_replay']='failed';review.save();return
        if a.phase=='contracts':review.status['status']='contracts_phase_finished';review.save();return
        methods=a.methods or ['B0','G0','G1']
        phases=['baseline','core','random','ablation'] if a.phase=='all' else [a.phase]
        for phase in phases:
            review.status.update(status='in_progress',phase=phase);review.save()
            if phase=='baseline':
                cases=a.cases or [f'N{i}' for i in range(6)]
                if any(not case.startswith('N') for case in cases):raise ValueError('baseline cases must be N0--N5')
                missing=[case for case in cases if (case,'B0') not in review.traces]
                if missing and 'B0' in methods:review.baseline_batch(missing)
                required=[f'N{i}' for i in range(6)]
                if not all((case,'B0') in review.traces for case in required):
                    review.status['status']='partial_baseline';review.save();return
                ok=all(review.metrics[case]['B0']['passed'] for case in required)
                review.status['gates']['A_B0']='passed' if ok else 'failed'
                if review.status['gates']['cpu_mjx_audit']=='not_run':review.cpu_audit()
                if not ok:
                    review.status.update(status='stopped_gate_A',classification='baseline_not_ready');review.save()
                    failed_case=next(case for case in required if not review.metrics[case]['B0']['passed'])
                    review.failure_diagnosis(failed_case);review.save();return
                for case in cases:
                    for method in [m for m in methods if m!='B0']:
                        if (case,method) not in review.traces:review.episode(case,method,c['fixed_cases'][case]['target_rows_time_speed_steer'])
                        gate=governor_nominal_metrics(review.traces[(case,method)])
                        review.metrics[case][method]['nominal_gate']=gate
                        if not gate['passed']:
                            review.status.update(status='stopped_gate_A',classification='nominal_governor_failed');review.status['gates']['A_governor']='failed';review.save();return
                all_done=all((case,m) in review.traces for case in required for m in ['G0','G1'])
                review.status['gates']['A_governor']='passed' if all_done else 'incomplete'
                if not all_done:review.status['status']='partial_baseline';review.save();return
            elif phase=='core':
                if review.status['gates']['A_governor']!='passed':raise ValueError('core requires full Gate A')
                cases=a.cases or ['C25L','C25R','C30L','C30R']
                for case in cases:
                    if case not in ['C25L','C25R','C30L','C30R']:raise ValueError('invalid core case')
                    if case.startswith('C30') and not all(review.metrics.get(k,{}).get('gate',{}).get('passed',False) for k in ['C25L','C25R']):raise ValueError('C30 requires both C25 gates')
                    for method in methods:
                        if (case,method) not in review.traces:review.episode(case,method,c['fixed_cases'][case]['target_rows_time_speed_steer'])
                    if not all((case,m) in review.traces for m in ['B0','G0','G1']):continue
                    result=core_gate(*(review.traces[(case,m)] for m in ['B0','G0','G1']),any(review.metrics[case][m]['raw_conflict_seen'] for m in ['G0','G1']))
                    review.metrics[case]['gate']=result
                    if not result['passed']:
                        review.status.update(status='stopped_gate_B',classification=result['classification']);review.status['gates']['B_preference']='failed';review.status['gates']['B_heading']='see per-method metrics';review.save();return
                all_done=all(review.metrics.get(k,{}).get('gate',{}).get('passed',False) for k in ['C25L','C25R','C30L','C30R'])
                review.status['gates']['B_preference']=review.status['gates']['B_heading']='passed' if all_done else 'incomplete'
                if not all_done:review.status['status']='partial_core';review.save();return
            elif phase=='random':
                if review.status['gates']['B_preference']!='passed':raise ValueError('random requires full Gate B')
                cases=a.cases or [f'R{s}' for s in c['random_panel']['seeds']]
                for case in cases:
                    if not case.startswith('R') or int(case[1:]) not in c['random_panel']['seeds']:raise ValueError('invalid random case')
                    rows=random_schedule(int(case[1:]))
                    schedule=root/f'{case}_schedule.json'
                    if not schedule.exists():schedule.write_text(json.dumps(rows.tolist(),indent=2)+'\n')
                    for method in methods:
                        if (case,method) not in review.traces:review.episode(case,method,rows)
                        if method=='B0':continue
                        gate=random_gate(review.traces[(case,method)]);review.metrics[case][method]['random_gate']=gate
                        if not gate['passed']:
                            review.status.update(status='stopped_gate_C',classification='random_panel_failed');review.status['gates']['C_random']='failed';review.save();return
                all_done=all((f'R{s}',m) in review.traces for s in c['random_panel']['seeds'] for m in ['B0','G0','G1'])
                review.status['gates']['C_random']='passed' if all_done else 'incomplete'
                if not all_done:review.status['status']='partial_random';review.save();return
            elif phase=='ablation':
                if review.status['gates']['B_preference']!='passed':raise ValueError('ablation requires Gate B')
                cases=a.cases or ['C25L','C25R']
                for case in cases:
                    if case not in ['C25L','C25R']:raise ValueError('ablation only C25L/R')
                    for method in [m for m in methods if m!='B0']:
                        label=case+'_no_recovery'
                        if (label,method) not in review.traces:review.episode(label,method,c['fixed_cases'][case]['target_rows_time_speed_steer'],ablation=True)
                review.status['gates']['ablation']='recorded' if all((case+'_no_recovery',m) in review.traces for case in ['C25L','C25R'] for m in ['G0','G1']) else 'incomplete'
            review.status['status']=phase+'_phase_finished';review.save()
        review.status['status']='declared_panels_finished' if a.phase=='all' else a.phase+'_phase_finished';review.save()
    except (BudgetExhausted,FloatingPointError) as exc:
        status='budget_exhausted' if isinstance(exc,BudgetExhausted) else 'numerical_failure'
        if review is not None:review.status.update(status=status,classification='runtime_or_budget' if status=='budget_exhausted' else status);review.save()
        else:report(root,dict(status=status,classification=str(exc),gates={'contracts':'incomplete'}),{}, {})
    except Exception:
        # Input/precondition errors must not overwrite prior evidence or a failed gate.
        if review is not None and review.status['status']=='in_progress':
            review.status.update(status='implementation_error',classification='implementation_error');review.save()
        raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['contracts','baseline','core','random','ablation','all'],default='all')
    p.add_argument('--float64',action='store_true');p.add_argument('--resume-contracts',action='store_true');p.add_argument('--predictor-budget',type=int,default=80000000);p.add_argument('--cases',nargs='*');p.add_argument('--methods',nargs='*');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();c=load_teleop_config(a.config)
    if a.predictor_budget<0:raise ValueError('negative budget')
    if a.methods and set(a.methods)-{'B0','G0','G1'}:raise ValueError('unknown method')
    if a.phase=='all' and (a.cases or a.methods):raise ValueError('all requires full coverage; use individual phases for subsets')
    if a.dry_run:print(json.dumps(dict(valid=True,phase=a.phase,predictor_budget=min(a.predictor_budget,80000000),physical_ticks=0)));return
    run(a,c)

if __name__=='__main__':main()
