"""Explicit preparation, real-model shape, or fixed48 replanning comparison."""
import argparse
from pathlib import Path
from sttw_control.goal_replanning_run import prepare,run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','shape','execute']);p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path,default=Path('runs/same_state_reference_pause_v1'));a=p.parse_args()
    try:
        if a.stage=='prepare':prepare(a.source,a.root)
        else:run(a.root,shape_only=a.stage=='shape')
    except BaseException:
        if a.stage=='execute':
            from engineer_reference_bias import notify
            notify(a.root,'STTW目标重规划对照报错','48条固定对照已停止，原始错误和实际计费保留，不自动重试。')
        raise
    else:
        if a.stage=='execute':
            from engineer_reference_bias import notify
            notify(a.root,'STTW目标重规划对照完成','原路径与重规划、完整与零中层的20秒固定比较完成；随后分析10/20秒到达和触地。')
