"""Complete fixed-panel score tables and all XY, no new physics."""
import argparse
from pathlib import Path
from sttw_control.goal_endpoint_analysis import analyze
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--h75-run',type=Path,help='separately completed48-query H75 supplement; primary128 unchanged');a=p.parse_args()
    from engineer_reference_bias import notify
    try:analyze(a.run,a.output,h75_root=a.h75_run)
    except BaseException as exc:
        notify(a.run,'STTW固定末端分析报错',repr(exc)[:300]+'；保留已有图和证据，不重跑物理。');raise
    else:notify(a.run,'STTW固定末端分析完成','封存面板与全例XY已生成；含H75时为176物理轨迹/240相关评分视图。待人工/独立复核，不自动采用。')
