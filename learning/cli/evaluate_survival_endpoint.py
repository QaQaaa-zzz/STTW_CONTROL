"""Prepare or explicitly execute fixed64 terminal evaluation; never select an interim model."""
import argparse
from pathlib import Path
from sttw_control.survival_endpoint_run import prepare_run,execute_run

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.execute:
        from prepare_short_recovery_campaign import register
        register(a.output.resolve()/'ACTIVE_RUN.json','STTW_CONTROL 固定64轮终点评测采样',a.output.resolve()/'status.json')
        execute_run(a.output)
    elif a.protocol:prepare_run(a.protocol,a.output)
    else:p.error('--protocol required to prepare; --execute uses an existing prepared run')
