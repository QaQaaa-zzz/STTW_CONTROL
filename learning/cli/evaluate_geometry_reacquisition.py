"""Prepare/trace/run/analyze six bounded complete-state recovery suffixes."""
import argparse,os,time,traceback
from pathlib import Path
from sttw_control.geometric_reacquisition_run import prepare,execute
from sttw_control.preference_best import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inventory',type=Path,default=Path('runs/geometric_reacquisition_inventory_v1/inventory.json'))
    p.add_argument('--phase',choices=['prepare','shape','execute','analyze'],required=True);a=p.parse_args();root=a.output.resolve()
    if a.phase=='prepare':prepare(a.inventory,root)
    elif a.phase in ['shape','execute']:execute(root,shape_only=a.phase=='shape')
    else:
        from prepare_short_recovery_campaign import register
        from sttw_control.anticipatory_analysis import analyze
        target=root/'extension/analysis_status.json'
        if target.exists():raise FileExistsError('preserve existing analysis attempt')
        state=dict(phase='running',stage='original22s_task_analysis',pid=os.getpid(),physics_steps=0,optimizer_updates=0,checked_unix=time.time());atomic_json(target,state)
        register(root/'extension/ANALYSIS_ACTIVE_RUN.json','STTW_CONTROL 几何再捕获六后缀22秒原任务分析',target)
        try:
            print(analyze(root,'extension'));state.update(phase='completed',checked_unix=time.time());atomic_json(target,state)
        except BaseException as exc:
            (root/'extension/analysis_traceback.txt').write_text(traceback.format_exc());state.update(phase='error',error=repr(exc),checked_unix=time.time());atomic_json(target,state);raise
