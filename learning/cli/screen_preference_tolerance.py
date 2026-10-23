"""Frozen preference policy pulse screen; no training."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_tolerance_screen import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declaration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.declaration,a.output)
