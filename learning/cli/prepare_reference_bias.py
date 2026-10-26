"""Prepare only; this command cannot launch reference optimization physics."""
import argparse
from pathlib import Path
from sttw_control.reference_bias_protocol import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,default=Path('runs/entry_history_confirmation_v1'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.output,a.source)
