#!/usr/bin/env python3
"""Run one bounded diagnostic preflight; no optimization or scientific gate."""
import os,sys,argparse
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.path_command_evaluation import run_conflict_preflight
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 run_conflict_preflight(a.config,a.output)
