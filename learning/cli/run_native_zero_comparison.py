"""Prepare/shape/execute missing native zero controls without patching archives."""
import argparse
import importlib.util
from pathlib import Path
import sys

sys.dont_write_bytecode=True

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','shape','execute']);p.add_argument('root',type=Path)
    p.add_argument('--source',type=Path,default=Path('runs/current_native_preference_v1_attempt2'))
    a=p.parse_args()
    file=Path(__file__).resolve().parents[1]/'src/sttw_control/native_zero_comparison.py'
    spec=importlib.util.spec_from_file_location('native_zero_runner',file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        if a.stage=='prepare':module.prepare(a.source,a.root)
        else:module.run(a.root,shape_only=a.stage=='shape')
    except BaseException:
        if a.stage=='execute':
            from engineer_reference_bias import notify
            notify(a.root,'STTW原生零中层对照报错','固定8条对照已停止，保存原始错误，不自动重试。')
        raise
    else:
        if a.stage=='execute':
            from engineer_reference_bias import notify
            notify(a.root,'STTW原生零中层8条完成','原生接口零神经参考修正已评测完，随后与缓存16条网络结果比较偏好用途；没有训练或重规划。')
