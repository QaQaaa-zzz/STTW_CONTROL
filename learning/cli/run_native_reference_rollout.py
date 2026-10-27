"""Prepare, shape or execute one sealed native cadence comparison."""
import argparse
import importlib.util
from pathlib import Path
import traceback


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','shape','execute'])
    parser.add_argument('root',type=Path)
    parser.add_argument('--engineering',type=Path,default=Path('runs/native_reference_cost_engineering_v1'))
    args=parser.parse_args();root=args.root.resolve()
    file=Path(__file__).resolve().parents[1]/'src/sttw_control/native_reference_rollout.py'
    spec=importlib.util.spec_from_file_location('native_cadence_comparison',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        if args.stage=='prepare':module.prepare(root,args.engineering)
        else:module.run(root,shape_only=args.stage=='shape')
    except BaseException:
        if root.exists():
            (root/(args.stage+'_traceback.txt')).write_text(traceback.format_exc())
            from engineer_reference_bias import notify
            notify(root,'STTW：原生动态参考对照报错','已停止，保留原始证据和未知费用，不自动重试。阶段：'+args.stage)
        raise
    else:
        if args.stage=='execute':
            from engineer_reference_bias import notify
            notify(root,'STTW：原生动态参考16条结束','固定无扰对照已结束；预测、实际执行和缓存分别计费，随后分析原偏好、目标和失败。尚不宣称控制增益。')
