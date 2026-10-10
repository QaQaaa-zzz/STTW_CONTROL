#!/usr/bin/env python3
"""Explicit bounded geometric upper training; default is no-physics preparation."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.path_command_training import prepare,train

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='learning/configs/path_feedback_phase_C_run.json');p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--check-interface',action='store_true');p.add_argument('--resume');a=p.parse_args()
 if a.execute:train(a.config,a.output,execute=True,resume=a.resume)
 else:
  if a.resume:raise ValueError('resume is execution-only')
  r=prepare(a.config,a.output,a.check_interface);print('prepared_not_run; training transitions='+str(r['training_policy_transitions_total'])+'; additional physics=0')
if __name__=='__main__':main()
