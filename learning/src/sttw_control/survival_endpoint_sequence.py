"""Finite event-driven continuation of the already declared fixed64 evaluation.

No new training, retries, repair agents, model selection, adoption or push.
"""
import os
from pathlib import Path


def open_process_event(pid):
    """Use the Linux pidfd API even in this project's older-glibc Python build.

    The x86_64 fallback syscall number is from the installed Linux UAPI header
    asm/unistd_64.h. Unsupported hosts fail explicitly, never busy-poll a PID.
    """
    if hasattr(os,'pidfd_open'):return os.pidfd_open(pid)
    import ctypes,platform
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise OSError('pidfd unavailable in this Python build/platform')
    libc=ctypes.CDLL(None,use_errno=True)
    fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
    if fd<0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    os.set_inheritable(fd,False)
    return int(fd)


def process_identity(pid):
    root=Path('/proc')/str(pid)
    rest=(root/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=int(pid),start_ticks=int(rest[19]),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        cwd=str((root/'cwd').resolve()),
        argv=[s.decode() for s in (root/'cmdline').read_bytes().split(b'\0') if s])


def require_fixed_terminal(status,pid):
    if (status.get('phase')!='completed' or status.get('pid')!=pid
            or status.get('completed_updates')!=64 or status.get('declared_updates')!=64):
        raise ValueError('original process did not leave a completed64 source: '+str(status))


def require_stage_receipt(stage,status):
    if status.get('phase')!='completed':raise ValueError('stage has no completed receipt: '+stage)
    if stage=='collection' and (status.get('completed_cases')!=488 or not status.get('endpoint_authenticated')):
        raise ValueError('fixed488 authenticated collection incomplete')
    if stage=='task_analysis' and status.get('completed_cases')!=488:raise ValueError('fixed488 task analysis incomplete')
    if stage=='figures' and status.get('stage')!='generated_pending_visual_review':
        raise ValueError('figure generation receipt mismatch')


def execute_sequence(training,evaluation,output):
    import json,select,time,subprocess,sys
    from .preference_best import atomic_json,digest
    training,evaluation,output=[Path(p).resolve() for p in (training,evaluation,output)]
    workdir=Path(__file__).resolve().parents[3]
    state=json.loads((training/'status.json').read_text());pid=state['pid']
    before=process_identity(pid);fd=open_process_event(pid)
    try:
        if process_identity(pid)!=before:raise ValueError('original process identity changed while binding pidfd')
        argv=before['argv']
        if (not any(Path(a).name=='train_survival_online.py' for a in argv) or '--run' not in argv or '--execute' not in argv
                or (Path(before['cwd'])/argv[argv.index('--run')+1]).resolve()!=training):
            raise ValueError('PID is not the declared actual training execution')
        if json.loads((evaluation/'status.json').read_text())['phase']!='prepared':
            raise ValueError('unused prepared fixed evaluation required; no implicit continuation/retry')
        # One plan-bound owner per evaluation. Does not replace any global watcher.
        marker=evaluation/'SEQUENCE_OWNER.json'
        with marker.open('x') as stream:json.dump(dict(sequence=str(output),pid=os.getpid()),stream)
        output.mkdir(parents=True,exist_ok=False)
        status=dict(phase='running',pid=os.getpid(),stage='waiting_training_exit',training_pid=pid,
            completed_stages=[],training_status_reads=1,heartbeat_s=300,physics_steps_by_sequence_itself=0,
            maximum_evaluation_lower_ticks=976000,automatic_retry=False,visual_review_pending=True)
        def update(**kw):
            status.update(kw,checked_unix=time.time());atomic_json(output/'status.json',status)
        update()
        from importlib import import_module
        sys.path.insert(0,str(workdir/'learning/cli'))
        register=import_module('prepare_short_recovery_campaign').register
        register(output/'ACTIVE_RUN.json','STTW_CONTROL 原64轮训练后固定终点评测接续',output/'status.json')
        try:
            plan=json.loads((training/'plan.json').read_text());prepared=json.loads((evaluation/'plan.json').read_text())
            if prepared['protocol_plan']['training_run']!=str(training):raise ValueError('evaluation linked to another training')
            scripts=[workdir/'learning/cli'/name for name in
                ['evaluate_survival_endpoint.py','analyze_survival_endpoint.py','render_survival_endpoint.py','continue_survival_endpoint.py']]
            extra=[Path(__file__),workdir/'learning/src/sttw_control/survival_endpoint_task_metrics.py']
            sources=dict(plan['source_sha256']);sources.update(prepared['source_sha256'])
            sources.update({str(p.resolve()):digest(p) for p in scripts+extra})
            def check_sources():
                for path,sha in sources.items():
                    if digest(path)!=sha:raise ValueError('declared runtime source changed: '+path)
            check_sources()
            commands=[dict(stage='collection',argv=[sys.executable,str(scripts[0]),'--output',str(evaluation),'--execute'],
                           status=str(evaluation/'status.json'),jax_platforms='cuda,cpu'),
                      dict(stage='task_analysis',argv=[sys.executable,str(scripts[1]),'--run',str(evaluation)],
                           status=str(evaluation/'task_analysis/status.json'),jax_platforms='cpu'),
                      dict(stage='figures',argv=[sys.executable,str(scripts[2]),'--run',str(evaluation)],
                           status=str(evaluation/'figures_status.json'),jax_platforms='cpu')]
            atomic_json(output/'plan.json',dict(project='STTW_CONTROL',training=str(training),evaluation=str(evaluation),
                original_process=before,training_plan_sha256=digest(training/'plan.json'),
                evaluation_plan_sha256=digest(evaluation/'plan.json'),source_sha256=sources,commands=commands,
                maximum_evaluation_lower_ticks=976000,episodes=488,optimizer_steps=0,
                trigger='Captured original OS process exits, completed64 receipt, then existing native endpoint authentication.',
                stop='Any status/identity/source/child error stops without retry, repair, extra samples or training.',
                preparation_note='Earlier case-matrix documents did not launch evaluation. This explicitly launched sequence now continues that declared evaluation after64 only.',
                visual_review='Generated figures still need visual review; no model adoption or RA-L completion claim.'))
            archive=output/'source_snapshot';archive.mkdir()
            for index,path in enumerate(sorted(sources)):(archive/f'{index:03d}_{Path(path).name}').write_bytes(Path(path).read_bytes())
            poll=select.poll();poll.register(fd,select.POLLIN|select.POLLERR|select.POLLHUP)
            heartbeat=time.monotonic()
            while not poll.poll(60000):
                if time.monotonic()-heartbeat>=300:
                    update(stage='waiting_training_exit',wait_mechanism='pidfd OS event; no repeated training status reads')
                    heartbeat=time.monotonic()
            terminal=json.loads((training/'status.json').read_text());update(training_status_reads=2)
            atomic_json(output/'training_terminal_receipt.json',terminal)
            require_fixed_terminal(terminal,pid);check_sources()
            for command in commands:
                check_sources()
                env=dict(os.environ,PYTHONPATH=str(workdir/'learning/src'),JAX_PLATFORMS=command['jax_platforms'],
                    XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONUNBUFFERED='1')
                with (output/(command['stage']+'.log')).open('xb') as stream:
                    child=subprocess.Popen(command['argv'],cwd=workdir,env=env,stdout=stream,stderr=subprocess.STDOUT)
                    update(stage=command['stage'],child_pid=child.pid,child_status=command['status'])
                    while True:
                        try:rc=child.wait(timeout=60);break
                        except subprocess.TimeoutExpired:update()
                if rc:raise RuntimeError(f'{command["stage"]} exited{rc}; see {output/(command["stage"]+".log")}')
                receipt=json.loads(Path(command['status']).read_text());require_stage_receipt(command['stage'],receipt)
                status['completed_stages'].append(command['stage'])
                atomic_json(output/(command['stage']+'_receipt.json'),receipt);update()
            check_sources()
            update(phase='completed',stage='evaluation_and_figures_generated_visual_review_pending',child_pid=None,
                visual_review_pending=True,model_adopted=False,extra_training_steps=0)
        except BaseException as exc:
            update(phase='error',error=repr(exc));raise
    finally:os.close(fd)
