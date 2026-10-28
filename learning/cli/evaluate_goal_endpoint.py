"""Fixed ordinary-goal endpoint evaluation, no training or best-model selection."""
import argparse
from pathlib import Path
from sttw_control.goal_endpoint_run import prepare,execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--training',type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute and a.training is None:p.error('--training required for preparation')
    from engineer_reference_bias import notify
    try:
        if a.execute:execute(a.run)
        else:prepare(a.training,a.run)
    except BaseException as exc:
        notify(a.run if a.run.exists() else Path('runs/goal_credit_cpu_v1'),'STTW固定末端评价报错',repr(exc)[:300]+'；保留原始资料，不自动重试。')
        raise
    else:
        if a.execute:notify(a.run,'STTW固定末端评价物理结束','固定128轨迹完成；全例分析尚待，不能据完成声称收益。')
