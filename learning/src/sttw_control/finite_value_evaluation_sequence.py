"""One original-process exit event, then the already declared fixed300 protocol.

No training restart, optimization, repeated endpoint evaluation, repair or model
adoption. The controller is a finite task, not a new persistent watcher.
"""
import json
import os
import select
import subprocess
import sys
import time
import traceback
from pathlib import Path

from .preference_best import atomic_json,digest
from .survival_endpoint_sequence import open_process_event,process_identity,require_fixed_terminal


def validate_training_identity(identity, training, workdir):
    cwd=Path(identity['cwd']).resolve();argv=identity['argv']
    script=Path(workdir).resolve()/'learning/cli/train_finite_value.py'
    try:
        matches=any((cwd/a).resolve()==script for a in argv)
        output=(cwd/argv[argv.index('--output')+1]).resolve()
        mode=argv[argv.index('--phase')+1]
    except (ValueError,IndexError):
        raise ValueError('original training execution identity missing') from None
    if not matches or cwd!=Path(workdir).resolve() or output!=Path(training).resolve() or mode!='execute':
        raise ValueError('PID is not the declared actual training execution')


def validate_stage(stage, state):
    if state.get('optimizer_updates')!=0:raise ValueError('evaluation cannot perform optimization')
    if stage=='prepare':
        if state.get('phase')!='prepared' or state.get('declared_cases')!=300:raise ValueError('fixed300 preparation incomplete')
        return
    if state.get('phase')!='completed':raise ValueError('stage incomplete: '+stage)
    if stage=='shape':
        if state.get('stage')!='shape_complete':raise ValueError('shape receipt mismatch')
    elif stage=='execute':
        if state.get('completed_cases')!=300 or state.get('stage')!='physical_complete_analysis_pending':
            raise ValueError('fixed300 actual evaluation incomplete')
        actual=state.get('active_lower_ticks',-1);allocated=state.get('allocated_lower_slots',-1)
        if not 0<=actual<=allocated<=600000:raise ValueError('fixed evaluation cost mismatch')
    elif stage=='analyze':
        if (state.get('completed_cases')!=300 or state.get('physics_steps')!=0
                or state.get('stage')!='numeric_and_figures_complete_review_pending'):
            raise ValueError('read-only complete300 analysis required')
    else:raise ValueError('unknown evaluation stage')


def execute_sequence(training, evaluation, output):
    training,evaluation,output=[Path(p).resolve() for p in (training,evaluation,output)]
    workdir=Path(__file__).resolve().parents[3]
    monitor=json.loads((training/'monitoring.json').read_text());pid=monitor['pid']
    before=process_identity(pid);validate_training_identity(before,training,workdir)
    fd=open_process_event(pid)
    try:
        if process_identity(pid)!=before:raise ValueError('source PID identity changed while binding event')
        if evaluation.exists():raise ValueError('fresh fixed evaluation required; no implicit retry')
        # One owner for this source/fixed evaluation even if a second controller
        # is accidentally launched with a different output directory.
        with (training/'FIXED64_EVALUATION_OWNER.json').open('x') as stream:
            json.dump(dict(controller=str(output),evaluation=str(evaluation),pid=os.getpid()),stream)
        output.mkdir(parents=True,exist_ok=False)
        status=dict(phase='running',stage='waiting_training_exit',pid=os.getpid(),training_pid=pid,
                    training_status_reads=0,completed_stages=[],maximum_evaluation_lower_ticks=600000,
                    physics_steps_by_controller=0,optimizer_updates=0,automatic_retry=False,
                    visual_review_pending=True,model_adopted=False)
        def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(output/'status.json',status)
        update()
        sys.path.insert(0,str(workdir/'learning/cli'))
        from prepare_short_recovery_campaign import register
        register(output/'ACTIVE_RUN.json','STTW_CONTROL 三臂训练后固定300评价接续',output/'status.json')
        try:
            plan=json.loads((training/'plan.json').read_text())
            if digest(training/'plan.json')!=json.loads((training/'prelaunch_receipt.json').read_text())['plan_sha256']:
                raise ValueError('training plan changed')
            script=workdir/'learning/cli/evaluate_finite_value.py'
            extra=[workdir/'learning/src/sttw_control'/name for name in [
                'finite_value_evaluation.py','finite_value_evaluation_run.py',
                'finite_value_evaluation_report.py','finite_value_evaluation_sequence.py']]
            extra += [script,Path(__file__),workdir/'learning/cli/finish_finite_value.py',
                      workdir/'docs/evidence/tolerance_progress_20261022/finite_value_online/EVALUATION_PROTOCOL.md']
            sources=dict(plan['source_sha256']);sources.update({str(p.resolve()):digest(p) for p in extra})
            sealed_inputs={str(training/'plan.json'):digest(training/'plan.json')}
            sealed_inputs.update({str(training/name):sha for name,sha in plan['files_sha256'].items()})
            def check_sources():
                for file,sha in {**sources,**sealed_inputs}.items():
                    if digest(file)!=sha:raise ValueError('sealed handoff source/input changed: '+file)
            commands=[]
            for stage in ['prepare','shape','execute','analyze']:
                argv=[sys.executable,str(script),'--output',str(evaluation),'--phase',stage]
                if stage=='prepare':argv+=['--training',str(training)]
                receipt=(evaluation/'shape/status.json' if stage=='shape' else
                         evaluation/'analysis/status.json' if stage=='analyze' else evaluation/'status.json')
                commands.append(dict(stage=stage,argv=argv,status=str(receipt),
                                     jax_platforms='cpu' if stage in ['prepare','analyze'] else 'cuda,cpu'))
            check_sources()
            atomic_json(output/'plan.json',dict(project='STTW_CONTROL',original_process=before,
                code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workdir,text=True).strip(),
                training=str(training),evaluation=str(evaluation),source_sha256=sources,sealed_inputs=sealed_inputs,
                commands=commands,maximum_new_lower_ticks=600000,optimizer_updates=0,
                trigger='Original authenticated OS process exit plus completed64 receipt, then independent endpoint authentication',
                stop='Any source/terminal/child/receipt error stops without retry, repair, extra sampling or training',
                visual_review='Numeric/figure generation only; no automatic adoption or paper-completion claim'))
            snapshot=output/'source_snapshot';snapshot.mkdir()
            for i,file in enumerate(sorted(sources)):
                (snapshot/f'{i:03d}_{Path(file).name}').write_bytes(Path(file).read_bytes())
            # Kernel event wait, not a sleep/status polling loop. Can be
            # interrupted by the process exiting; never restarts that process.
            update(wait_mechanism='pidfd kernel event; no repeated training status reads')
            select.select([fd],[],[])
            terminal=json.loads((training/'status.json').read_text());update(training_status_reads=1)
            atomic_json(output/'training_terminal_receipt.json',terminal)
            require_fixed_terminal(terminal,pid);check_sources()
            for command in commands:
                check_sources()
                env=dict(os.environ,PYTHONPATH=str(workdir/'learning/src')+':'+str(workdir/'learning/cli'),
                    JAX_PLATFORMS=command['jax_platforms'],XLA_PYTHON_CLIENT_PREALLOCATE='false',
                    OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (output/(command['stage']+'.log')).open('xb') as stream:
                    child=subprocess.Popen(command['argv'],cwd=workdir,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    update(stage=command['stage'],child_pid=child.pid,child_status=command['status'])
                    code=child.wait()
                if code:raise RuntimeError(command['stage']+f' exited{code}; original log preserved, no retry')
                receipt=json.loads(Path(command['status']).read_text());validate_stage(command['stage'],receipt)
                status['completed_stages'].append(command['stage'])
                atomic_json(output/(command['stage']+'_receipt.json'),receipt);update()
            check_sources()
            update(phase='completed',stage='fixed300_and_figures_complete_visual_review_pending',child_pid=None)
        except BaseException as exc:
            (output/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
    finally:os.close(fd)
