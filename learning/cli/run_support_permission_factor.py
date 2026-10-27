"""Explicit stages for the one fixed support-permission DEV experiment."""
import argparse
from pathlib import Path
from sttw_control.support_permission_experiment import prepare,shape,execute


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--stage',required=True,choices=['prepare','shape','execute'])
    args=parser.parse_args()
    try:
        if args.stage=='prepare':
            if args.parent is None:parser.error('--parent required for prepare')
            prepare(args.root,args.parent)
        else:globals()[args.stage](args.root)
    except BaseException:
        if args.root.exists():
            from engineer_reference_bias import notify
            notify(args.root,'STTW支持权限实验报错',f'{args.stage}阶段失败；保留现有记录，不自动重试。')
        raise
    if args.stage=='execute':
        from engineer_reference_bias import notify
        notify(args.root,'STTW支持权限实验数值完成','固定108条评测已数值完成；尚待完整轨迹和成对结果分析。')
