"""One sealed16-case real wiring check; never starts the production searches."""
import argparse
import copy
import json
import os
import traceback
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def notify(root,title,body):
    import sys
    from datetime import datetime
    from zoneinfo import ZoneInfo
    sys.path.insert(0,'/home/qy/DVGC/runs/worktrees/jit-email-alerts/JIT/src')
    try:
        from jit_dvgc.bark_notifications import load_config,send_bark
        result=send_bark(load_config(Path.home()/'.config/jit-email/bark.json'),title,datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()+' '+body)
    except Exception as exc:result=dict(accepted=False,error=repr(exc))
    with (root/'bark_receipts.jsonl').open('a') as f:f.write(json.dumps(dict(title=title,receipt=result),default=str,ensure_ascii=False)+'\n')


def prepare(root):
    root=root.resolve();out=root/'engineering';out.mkdir(exist_ok=False)
    events=json.loads((root/'evaluation_events.json').read_text());plan=json.loads((root/'plan.json').read_text())
    cases=[];vectors=[]
    for context in plan['contexts']:
        for alpha in [0.,1.]:
            base=next(c for c in events if c['context']==context and c['alpha']==alpha and c['force_N']==0)
            for kind,vector in [('zero_bias',[0.]*6),('probe_bias',[.5,-.5,-.5,.5,.25,-.25])]:
                c=copy.deepcopy(base);c.update(case_id=base['case_id'].rsplit('/',1)[0]+'/'+kind,role=kind)
                cases.append(c);vectors.append(vector)
    if len(cases)!=16 or any(c['force_N']!=0 for c in cases):raise ValueError('complete original four-context nominal16 required')
    files=list(Path('learning/src/sttw_control').glob('*.py'))+[Path(__file__)]
    sources={str(f.resolve()):digest(f) for f in files}
    archive=out/'source';archive.mkdir()
    for i,f in enumerate(files): (archive/f'{i:03d}_{f.name}').write_bytes(f.read_bytes())
    design=dict(cases=cases,vectors=vectors,source_plan_sha256=digest(root/'plan.json'),source_sha256=sources,
        maximum_active_lower_ticks=32000,maximum_allocated_lower_slots=32000,optimizer_updates=0,
        question='Does zero bias retain original full-middle wiring, and does one fixed nonzero bias actually affect legal references under unchanged limits? Not an improvement/optimization test.',
        stop='One16-case batch; physical or analysis error stops with all evidence. No automatic retry/parameter scan/optimization/production launch.',
        scope='All four selected contexts, alpha endpoints only, zero force; alpha.5 was shape-checked previously, not included in these real16. Known correlated DEV.')
    atomic_json(out/'design.json',design);atomic_json(out/'seal.json',dict(design_sha256=digest(out/'design.json')))
    atomic_json(out/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_ENGINEERING.json','STTW参考偏置16条真实接线检查',out/'status.json')


def execute(root):
    root=root.resolve();out=root/'engineering'
    if json.loads((out/'status.json').read_text())['phase']!='prepared':raise ValueError('single prepared run only, no implicit retry')
    if digest(out/'design.json')!=json.loads((out/'seal.json').read_text())['design_sha256']:raise ValueError('design changed')
    design=json.loads((out/'design.json').read_text())
    if digest(root/'plan.json')!=design['source_plan_sha256']:raise ValueError('source plan changed')
    for f,sha in design['source_sha256'].items():
        if digest(f)!=sha:raise ValueError('sealed engineering source changed '+f)
    atomic_json(out/'status.json',dict(phase='running',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0))
    try:
        from sttw_control.reference_bias_runtime import BiasRuntime
        runtime=BiasRuntime(root);rows=runtime.rollout(design['cases'],design['vectors'],out/'physical')
        cost=json.loads((out/'physical/status.json').read_text())
        if cost['active_lower_ticks']>32000 or cost['allocated_lower_slots']>32000:raise ValueError('fixed wiring budget exceeded')
        atomic_json(out/'status.json',dict(cost,phase='completed',pid=os.getpid(),stage='numeric_audit_complete_XY_review_pending',production_ready=False))
        notify(out,'STTW参考偏置接线数值检查完成',f'16条完整轨迹及偏置/原目标/原奖励审计完成，实际{cost["active_lower_ticks"]}个底层步。XY与独立分析待审，未启动优化或训练。')
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());p=out/'physical/status.json'
        cost=json.loads(p.read_text()) if p.exists() else {}
        atomic_json(out/'status.json',dict(phase='error',pid=os.getpid(),error=repr(exc),physical_progress=cost))
        notify(out,'STTW参考偏置接线检查报错','原始输出和错误已保存，任务停止，不自动重跑：'+repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',choices=['prepare','execute'],required=True);a=p.parse_args()
    (prepare if a.stage=='prepare' else execute)(a.root)
