"""Render every factual endpoint panel without new physics or policy updates."""
import argparse,json,os,time
from pathlib import Path
from sttw_control.preference_best import atomic_json
from sttw_control.survival_endpoint_figures import render_endpoint_run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    root=p.parse_args().run.resolve()
    target=root/'figures_status.json'
    if target.exists():raise ValueError('render status exists; preserve partial outputs and inspect before retry')
    status=dict(phase='running',pid=os.getpid(),stage='read_only_figures',physics_steps=0,optimizer_steps=0,checked_unix=time.time())
    atomic_json(target,status)
    from prepare_short_recovery_campaign import register
    register(root/'FIGURES_ACTIVE_RUN.json','STTW_CONTROL 固定终点评测全覆盖绘图',target)
    try:
        render_endpoint_run(root)
        status.update(phase='completed',stage='generated_pending_visual_review',checked_unix=time.time())
        atomic_json(target,status)
    except BaseException as exc:
        status.update(phase='error',error=repr(exc),checked_unix=time.time());atomic_json(target,status);raise
