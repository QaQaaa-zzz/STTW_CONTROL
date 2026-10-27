"""Explicit preparation, real-model shape, or fixed48 replanning comparison."""
import argparse
from pathlib import Path
from sttw_control.goal_replanning_run import prepare,prepare_h75,run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','prepare-h75','shape','execute']);p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path,default=Path('runs/same_state_reference_pause_v1'));a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.source,a.root)
        elif a.stage=='prepare-h75':prepare_h75(a.source,Path('runs/h75_disturbed_common_interface_v1'),a.root)
        else:run(a.root,shape_only=a.stage=='shape')
    except BaseException:
        if a.stage=='execute':
            from engineer_reference_bias import notify
            notify(a.root,'STTW目标重规划对照报错','固定重规划对照已停止，原始错误和实际计费保留，不自动重试。')
        raise
    else:
        if a.stage=='execute':
            from engineer_reference_bias import notify
            notify(a.root,'STTW目标重规划对照完成','封存的20秒固定比较完成；随后分析10/20秒到达和触地，旧完整结果不重跑。')
