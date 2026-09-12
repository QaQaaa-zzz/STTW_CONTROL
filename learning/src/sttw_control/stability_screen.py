"""At most two predeclared PPO engineering runs; no unbounded retraining."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .deferred_training import verify_files


def wait_stage(child,timeout,warning_seconds,notify):
    """Optional legacy hard limit; otherwise warn once and respect step budget."""
    if timeout is None:
        try:return child.wait(timeout=warning_seconds)
        except subprocess.TimeoutExpired:
            notify('Wall-clock advisory exceeded; continuing the declared step budget')
            return child.wait()
    try:return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:child.wait(timeout=15)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        raise TimeoutError('bounded stage timeout; no further stages')


def assess(training,criteria,constrained):
    status=json.loads((training/'status.json').read_text())
    rows=[json.loads(x) for x in (training/'metrics.jsonl').read_text().splitlines()]
    if not status['complete']:raise ValueError('training incomplete')
    v=rows[-1]['validation'];ids=[i for i,x in enumerate(v['event_present']) if x]
    failed=sum(bool(v['failed'][i]) for i in ids)
    recovered=sum(bool(v['post_event_hold_complete'][i]) for i in ids)
    nominal_failed=sum(v.get('nominal_failed',[]))
    disturbed=sum(r.get('steer_disturbed_transitions',0)+r.get('force_disturbed_transitions',0) for r in rows)
    coverage=disturbed/rows[-1]['control_transitions']
    accepted=sum(r.get('optimizer_audit',{}).get('accepted_minibatches',0) for r in rows if not r.get('optimizer_audit',{}).get('full_update_rolled_back',False))
    kl_ok=all(r.get('optimizer_audit',{}).get('final_exact_kl',float('inf'))<=criteria['target_kl']+1e-6 for r in rows) if constrained else True
    passed=bool(ids) and failed==0 and nominal_failed==0 and recovered==len(ids)
    if constrained:passed=passed and kl_ok and coverage>=criteria['minimum_disturbed_fraction'] and accepted>=criteria['minimum_accepted_minibatches']
    return dict(passed=passed,failed=failed,recovered=recovered,cases=len(ids),nominal_failed=nominal_failed,disturbed_fraction=coverage,accepted_minibatches=accepted,kl_ok=kl_ok,scope='Short development engineering screen, not proof of long-run stability or improvement')


def run(plan_path):
    plan=json.loads(Path(plan_path).read_text());root=Path(plan['output']);root.mkdir(exist_ok=False)
    def status(phase,**kw):(root/'status.json').write_text(json.dumps({'phase':phase,**kw},indent=2)+'\n')
    env=os.environ.copy();env.pop('JAX_PLATFORMS',None)
    env.update(PYTHONPATH=str(Path(plan['repository'])/'learning/src'),XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',PYTHONUNBUFFERED='1')
    def invoke(args,log,cpu=False):
        verify_files(plan['input_sha256']);e=dict(env)
        if cpu:e['JAX_PLATFORMS']='cpu'
        with log.open('x') as f:
            child=subprocess.Popen([sys.executable,*map(str,args)],cwd=plan['repository'],env=e,stdout=f,stderr=subprocess.STDOUT)
            def warning(message):
                (root/'wall_clock_warning.json').write_text(json.dumps({'message':message,'log':str(log)},indent=2)+'\n')
                print(message,flush=True)
            code=wait_stage(child,plan.get('stage_timeout_seconds'),plan.get('stage_warning_seconds',2700),warning)
        if code:raise RuntimeError('stage subprocess failed: '+str(code))
    results=[]
    try:
        for index,stage in enumerate(plan['stages']):
            out=root/stage['name'];out.mkdir();frozen=out/'frozen';frozen.mkdir()
            for name in ['task','training','panel']:(frozen/(name+'.json')).write_bytes(Path(stage[name]).read_bytes())
            (out/'declaration.json').write_text(json.dumps({'priority_alphas':plan['alphas'],'role':'bounded engineering comparison','stage':stage,'source_sha256':plan['input_sha256']},indent=2)+'\n')
            status('training',stage=stage['name'],completed=results)
            invoke(['learning/cli/train.py','--task',frozen/'task.json','--config',frozen/'training.json','--output',out/'training'],out/'training.log')
            result=assess(out/'training',plan['criteria'],index==0);results.append({'stage':stage['name'],**result})
            (out/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
            if result['passed']:break
        # Only the last tested candidate receives the small recorded CPU panel.
        checkpoint=json.loads((out/'training/status.json').read_text())['last_checkpoint']
        panel=json.loads((frozen/'panel.json').read_text());seed=panel['evaluation_seeds'][0]
        for i,a in enumerate(plan['alphas']):
            status('recorded_evaluation',stage=stage['name'],alpha=a,completed=results)
            invoke(['learning/cli/disturbance.py','--task',frozen/'task.json','--panel',frozen/'panel.json','--training-run',out/'training','--checkpoint',checkpoint,'--output',out/'evaluation'/f'alpha_{i}'/f'seed_{seed}','--seed',seed,'--priority-alpha',a],out/f'evaluation_alpha_{i}.log',True)
        status('reward_diagnostics',completed=results)
        invoke(['learning/cli/reward_breakdown.py','--run',out],out/'reward_diagnostics.log',True)
        status('complete',completed=results,second_stage_skipped=len(results)==1,recorded_panel=str(out/'analysis/reward_breakdown/INDEX.md'))
    except Exception as exc:status('error',error=str(exc),completed=results);raise
