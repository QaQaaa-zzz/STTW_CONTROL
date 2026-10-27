"""Explicit preparation/shape/execution around an isolated legacy runtime."""
import argparse
import importlib.util
from pathlib import Path
import sys
sys.dont_write_bytecode=True

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare-execution','shape','execute'])
    p.add_argument('root',type=Path);a=p.parse_args()
    f=Path(__file__).resolve().parents[1]/'src/sttw_control/legacy_goal_run.py'
    spec=importlib.util.spec_from_file_location('_sttw_legacy_goal_host',f)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if a.stage=='prepare-execution':module.prepare_execution(a.root)
    else:
        try:module.run(a.root,shape_only=a.stage=='shape')
        except BaseException:
            native=Path(__import__('json').loads((a.root/'plan.json').read_text())['runtime_root'])
            sys.path.insert(0,str(native))
            from engineer_reference_bias import notify
            notify(a.root,'STTW旧紧弯接续检查报错','已保存错误与实际成本；不自动重试、扩力或修改旧数据。阶段：'+a.stage)
            raise
        else:
            if a.stage=='execute':
                from engineer_reference_bias import notify
                notify(a.root,'STTW旧紧弯90条接续完成','固定90条20秒共同规划比较已保存，随后核对原目标首达、触地和逐例增失；没有训练。')
