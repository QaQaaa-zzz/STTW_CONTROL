"""Prepare, shape or execute the scientifically reviewed frozen native check."""
import argparse
import importlib.util
from pathlib import Path
import traceback


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'shape', 'execute'])
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    folder = Path(__file__).resolve().parents[1]
    file = folder / 'src/sttw_control/native_frozen_validation.py'
    spec = importlib.util.spec_from_file_location('native_frozen_validation_runner', file)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    root = args.root.resolve()
    try:
        if args.stage == 'prepare':
            review = Path('runs/nominal_program_feedback_v1/route_reframe')
            runner.prepare(root, review / 'native_validation_protocol_candidate.json',
                           review / 'native_validation_protocol_review.md', Path('runs/native_reference_rollout_v1'))
        else:
            runner.run(root, shape_only=args.stage == 'shape')
    except BaseException:
        if root.exists():
            (root / (args.stage + '_traceback.txt')).write_text(traceback.format_exc())
            from engineer_reference_bias import notify
            notify(root, 'STTW：冻结原生功能验证报错', '阶段：' + args.stage + '；保留原始错误与计费，不自动重试，不调整条件。')
        raise
    else:
        if args.stage == 'execute':
            from engineer_reference_bias import notify
            notify(root, 'STTW：冻结原生52条验证结束', '仿真执行结束，尚待分析全部正反结果；0训练，不宣称新抗扰机制。')
