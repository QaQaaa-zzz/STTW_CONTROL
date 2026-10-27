"""Fixed ordinary goal baseline; preparation never launches physics."""
import argparse
from pathlib import Path
from sttw_control.goal_training_run import prepare,execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    from engineer_reference_bias import notify
    try:
        if not a.execute:prepare(a.run)
        else:execute(a.run)
    except BaseException as exc:
        notify(a.run if a.run.exists() else Path('runs/goal_credit_cpu_v1'),'STTW普通目标学习报错',repr(exc)[:300]+'；原始资料保留，不自动重启。')
        raise
    else:
        if a.execute:notify(a.run,'STTW普通目标学习完成','固定64次采集学习完成；评价尚待，不代表抗扰或投稿证据成立。')
