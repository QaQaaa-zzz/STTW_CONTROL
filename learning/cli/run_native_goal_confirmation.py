"""Explicit prepare/shape/execute for one fixed native-permission80 panel."""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode=True

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','shape','execute']);parser.add_argument('root',type=Path)
    parser.add_argument('--source',type=Path,default=Path('runs/native_zero_preference_v1'))
    args=parser.parse_args()
    file=Path(__file__).resolve().parents[1]/'src/sttw_control/native_goal_run.py'
    spec=importlib.util.spec_from_file_location('_sttw_native_goal_runner',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        if args.stage=='prepare':module.prepare(args.source,args.root)
        else:module.run(args.root,shape_only=args.stage=='shape')
    except BaseException:
        if args.stage=='execute':
            from engineer_reference_bias import notify
            notify(args.root,'STTW新情境80条比较报错','已停止并保存partial原始数据和计费，不自动重试或扩力。')
        raise
    else:
        if args.stage=='execute':
            from engineer_reference_bias import notify
            notify(args.root,'STTW新情境80条比较完成','固定20秒比较完成，无训练；随后分析原目标到达、实际触地、偏好代价及完整XY。')
