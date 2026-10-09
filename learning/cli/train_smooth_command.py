#!/usr/bin/env python3
import os,sys,argparse
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--fresh',action='store_true');p.add_argument('--train-wall',type=int,default=1800);p.add_argument('--total-wall',type=int,default=5400);p.add_argument('--resume-parent');p.add_argument('--target-updates',type=int);p.add_argument('--unlimited-wall',action='store_true');p.add_argument('--preference-v5',action='store_true');p.add_argument('--preference-v51',action='store_true');p.add_argument('--preference-stage2-parent');p.add_argument('--recover-preference-v51-evaluation',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args()
from sttw_control.smooth_command_training import run,run_preference,run_preference_stage2,run_preference_v51,recover_preference_v51_evaluation
if a.recover_preference_v51_evaluation:
    recover_preference_v51_evaluation(a.output)
    raise SystemExit(0)
if a.preference_v5 and a.preference_v51:p.error('select only one preference mode')
if a.preference_stage2_parent and not a.preference_v5:p.error('--preference-stage2-parent requires --preference-v5')
if a.preference_v51:
    run_preference_v51(a.output,smoke=a.smoke)
    raise SystemExit(0)
if a.preference_v5:
    if a.preference_stage2_parent:run_preference_stage2(a.output,a.preference_stage2_parent)
    else:run_preference(a.output,smoke=a.smoke)
    raise SystemExit(0)
run(a.output,fresh=a.fresh,train_wall=a.train_wall,total_wall=a.total_wall,resume_parent=a.resume_parent,target_updates=a.target_updates,unlimited_wall=a.unlimited_wall)
