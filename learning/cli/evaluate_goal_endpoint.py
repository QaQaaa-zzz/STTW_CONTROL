"""Fixed ordinary-goal endpoint evaluation, no training or best-model selection."""
import argparse
from pathlib import Path
from sttw_control.goal_endpoint_run import prepare,execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path);p.add_argument('--training',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--h75-supplement',action='store_true',help='prepare separate48 fixed H75 controls; never changes primary128')
    p.add_argument('--replan',action='store_true',help='prepare opt-in1s causal replanning for all methods; unchanged original goal/H10')
    a=p.parse_args()
    if not a.execute and a.training is None:p.error('--training required for preparation')
    from engineer_reference_bias import notify
    try:
        if a.execute:execute(a.run)
        else:prepare(a.training,a.run,h75_supplement=a.h75_supplement,replan=a.replan)
    except BaseException as exc:
        notify(a.run if a.run.exists() else Path('runs/goal_credit_cpu_v1'),'STTW固定末端评价报错',repr(exc)[:300]+'；保留原始资料，不自动重试。')
        raise
    else:
        if a.execute:notify(a.run,'STTW固定末端评价物理结束','已完成该封存面板的物理执行；全例分析尚待，不能据完成声称收益。')
