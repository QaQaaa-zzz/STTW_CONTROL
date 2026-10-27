"""Fixed goal-collector real-plant engineering; not policy training."""
import argparse
from pathlib import Path
from sttw_control.goal_engineering_run import prepare, execute

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=['prepare', 'shape', 'execute'], required=True)
    a = p.parse_args()
    from engineer_reference_bias import notify
    try:
        if a.phase == 'prepare': prepare(a.output)
        else: execute(a.output, shape_only=a.phase == 'shape')
    except BaseException as exc:
        notify(a.output if a.output.exists() else Path('runs/goal_credit_cpu_v1'),
            'STTW目标采集工程报错', a.phase + '：' + repr(exc)[:300] + '；证据保留，不自动重启。')
        raise
    else:
        if a.phase != 'prepare':
            notify(a.output, 'STTW目标采集工程阶段完成',
                a.phase + '已结束；仅接口工程，未训练，不代表抗扰效果。')
