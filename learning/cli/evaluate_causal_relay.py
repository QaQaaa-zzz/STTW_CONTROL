"""Finite measured-sensor recovery relay, no training or event oracle."""
import argparse
from pathlib import Path
from sttw_control.causal_relay_run import prepare,execute

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['prepare','shape','execute','analyze'],required=True);a=p.parse_args()
    if a.phase=='prepare':prepare(a.output)
    elif a.phase=='analyze':
        from sttw_control.causal_relay_analysis import analyze
        analyze(a.output)
    else:execute(a.output,shape_only=a.phase=='shape')
