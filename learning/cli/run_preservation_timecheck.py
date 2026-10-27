"""Bind/run one fixed new-time comparison; terminal notifications in Chinese."""
import argparse
import traceback
from pathlib import Path
from sttw_control.preference_best import atomic_json
from sttw_control.reference_preservation_confirmation import bind,execute
from engineer_reference_bias import notify

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('--stage',choices=['bind','execute'],required=True);a=p.parse_args();out=a.parent/'preservation_timecheck_v1'
    try:
        if a.stage=='bind':bind(a.parent)
        else:
            atomic_json(out/'dispatch_status.json',dict(phase='running',stage='numeric'))
            execute(a.parent)
            notify(out,'STTW新起扰时刻数值评价完成','固定240实际/384逻辑评价完成，随后复核全部XY和得失，不自动采用新控制器。')
            atomic_json(out/'dispatch_status.json',dict(phase='running',stage='analysis'))
            from analyze_preservation_timecheck import analyze
            analyze(a.parent)
            from diagnose_reference_bias_campaign import diagnose
            diagnose(a.parent,confirmation=True)
            atomic_json(out/'dispatch_status.json',dict(phase='completed',stage='figures_generated_visual_review_pending'))
            notify(out,'STTW新时刻确认报告已生成','四组固定选择全部评价完成，全XY已生成待目视，结论待审。')
    except BaseException as exc:
        (out/'dispatch_traceback.txt').write_text(traceback.format_exc())
        atomic_json(out/'dispatch_status.json',dict(phase='error',requested_stage=a.stage,error=repr(exc)))
        notify(out,'STTW新时刻确认报错','保留已执行成本和原数据，不自动重跑：'+repr(exc));raise
