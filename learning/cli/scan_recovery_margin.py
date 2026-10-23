"""Collect a predeclared conditional recovery margin dataset; no training."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.recovery_margin_scan import collect

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declaration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();collect(a.declaration,a.output)
