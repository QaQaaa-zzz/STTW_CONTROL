"""Prepare or run the fixed reference-bias pilot, with Chinese terminal Bark."""
import argparse
import traceback
from pathlib import Path
from engineer_reference_bias import notify
from sttw_control.reference_bias_campaign import prepare,execute
from sttw_control.preference_best import atomic_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--stage',choices=['prepare','execute'],required=True);a=p.parse_args()
    if a.stage=='prepare':
        try:prepare(a.root)
        except BaseException as exc:
            notify(a.root,'STTW参考优化准备报错','未启动搜索，错误需核查：'+repr(exc));raise
    else:
        try:
            atomic_json(a.root/'production/dispatch_status.json',dict(phase='running',stage='search_and_evaluation'))
            execute(a.root)
        except BaseException as exc:
            atomic_json(a.root/'production/dispatch_status.json',dict(phase='error',stage='search_and_evaluation',error=repr(exc)))
            notify(a.root/'production','STTW参考优化对照报错','原数据保留，停止且不自动重跑：'+repr(exc));raise
        notify(a.root/'production','STTW参考优化对照数值完成','24名义搜索和120冻结参考事件评价完成，随后进行全XY及强对照分析，不自动采用控制器。')
        try:
            from analyze_reference_bias_campaign import analyze
            atomic_json(a.root/'production/dispatch_status.json',dict(phase='running',stage='analysis'))
            analyze(a.root)
        except BaseException as exc:
            (a.root/'production/analysis_traceback.txt').write_text(traceback.format_exc())
            atomic_json(a.root/'production/dispatch_status.json',dict(phase='error',stage='analysis',error=repr(exc)))
            notify(a.root/'production','STTW参考优化分析报错','原数值评价已完成，分析错误保留，不重跑物理：'+repr(exc));raise
        atomic_json(a.root/'production/dispatch_status.json',dict(phase='completed',stage='numeric_and_figures_complete_visual_review_pending'))
        notify(a.root/'production','STTW参考优化报告已生成','全部120新评价和240冻结基线已汇总，XY图已生成待目视；研究结论待审，不自动采用。')
