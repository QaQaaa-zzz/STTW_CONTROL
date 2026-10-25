"""Analyze one authenticated completed stage without executing physics."""
import argparse
import os
import time
import traceback
from pathlib import Path
from sttw_control.anticipatory_analysis import analyze
from sttw_control.preference_best import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=['nominal','disturbed','endpoints','expert_relay','sequence_nominal'],required=True);a=p.parse_args()
    root=a.output.resolve();target=root/a.stage/'analysis_status.json'
    if target.exists():raise FileExistsError('preserve existing analysis status '+str(target))
    state=dict(phase='running',stage='offline_original_alpha_analysis',pid=os.getpid(),
        physics_steps=0,optimizer_updates=0,checked_unix=time.time())
    atomic_json(target,state)
    from prepare_short_recovery_campaign import register
    label={'endpoints':'历史中层端点回看','expert_relay':'旧端点因果短时借用'}.get(a.stage,'提前参考'+a.stage)
    register(root/a.stage/'ANALYSIS_ACTIVE_RUN.json','STTW_CONTROL '+label+'原偏好与轨迹分析',target)
    try:
        print(analyze(root,a.stage));state.update(phase='completed',checked_unix=time.time())
        atomic_json(target,state)
    except BaseException as exc:
        (root/a.stage/'analysis_traceback.txt').write_text(traceback.format_exc())
        state.update(phase='error',error=repr(exc),checked_unix=time.time());atomic_json(target,state);raise
