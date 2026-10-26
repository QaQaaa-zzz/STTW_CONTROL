"""Execute a sealed eight-case native panel, then read-only paired analysis."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo


def main(root):
    from sttw_control.preference_best import digest,atomic_json
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text());receipt=json.loads((root/'prelaunch_receipt.json').read_text())
    for filename,key in [('plan.json','plan_sha256'),('declaration.json','declaration_sha256'),('cache_receipt.json','cache_sha256')]:
        if digest(root/filename)!=receipt[key]:raise ValueError('prelaunch input changed')
    for file,sha in plan['native_runtime_files'].items():
        if digest(file)!=sha:raise ValueError('archived numerical source changed')
    if (root/'dispatch_status.json').exists():raise ValueError('no implicit restart of a previous dispatch')
    state=dict(phase='running',controller_pid=os.getpid(),children=[])
    def save():atomic_json(root/'dispatch_status.json',state)
    def notify(title,body):
        sys.path.insert(0,'/home/qy/DVGC/runs/worktrees/jit-email-alerts/JIT/src')
        from jit_dvgc.bark_notifications import load_config,send_bark
        r=dict(title=title,accepted=False)
        try:
            send_bark(load_config(Path.home()/'.config/jit-email/bark.json'),title,datetime.datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()+' '+body);r['accepted']=True
        except Exception as e:r['error_type']=type(e).__name__
        with (root/'bark_receipts.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    save()
    try:
        cli=root/'native_runtime/learning/cli/screen_preference_tolerance.py'
        stages=[('execute',[str(cli),'--declaration',str(root/'declaration.json'),'--output',str(root/'evaluation')]),
            ('analyze',['learning/cli/analyze_current_native_preference.py','--run',str(root)])]
        for stage,args in stages:
            env=dict(os.environ);env.update(JAX_PLATFORMS='cuda,cpu' if stage=='execute' else 'cpu',
                XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4' if stage=='execute' else '1',
                PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(root/'native_runtime/learning/src') if stage=='execute' else 'learning/src:learning/cli')
            with (root/(stage+'_stdout.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT,env=env)
                state.update(stage=stage,child_pid=child.pid);state['children'].append(dict(stage=stage,pid=child.pid));save();print(stage,child.pid,flush=True)
                code=child.wait()
            if code:raise RuntimeError(f'{stage} exited{code}; evidence retained, no automatic retry')
            if stage=='execute':notify('STTW当前模型原生八例完成','8条固定无扰原生轨迹已结束，随后复用H75八例分析偏好保持；未训练或添加辅助控制。')
        state.update(phase='completed',stage='analysis_complete_visual_review_pending',child_pid=None);save()
        notify('STTW原生偏好对照分析完成','当前8条与历史H75八条的XY和原用户效用已分析，等待审阅；原生终止与目标判据分别保留。')
    except BaseException as e:
        state.update(phase='error',error=repr(e),child_pid=None);save();(root/'dispatch_traceback.txt').write_text(traceback.format_exc());notify('STTW原生偏好对照报错',str(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
