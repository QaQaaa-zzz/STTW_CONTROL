#!/usr/bin/env python3
"""Freeze completed Coverage endpoint parents. Does not launch physics/training."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.path_command_fork import freeze_reward_parents
from sttw_control.preference_best import atomic_json
p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output)
if out.exists():raise FileExistsError('immutable parent identity receipt exists')
atomic_json(out,{'parent_run':str(Path(a.parent).resolve()),'expected_parents':freeze_reward_parents(a.parent)})
print(out)
