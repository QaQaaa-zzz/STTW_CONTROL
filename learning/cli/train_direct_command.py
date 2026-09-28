#!/usr/bin/env python3
"""Bounded direct command V3 campaign, one shared fresh alpha-conditioned policy."""
import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2')
from sttw_control.direct_command_training import load_spec,run,validate_run_limits
p=argparse.ArgumentParser()
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
p.add_argument('--updates',type=int,default=None);p.add_argument('--compute-wall-budget',type=float,default=None)
p.add_argument('--dry-run',action='store_true');p.add_argument('--internal-worker',action='store_true',help=argparse.SUPPRESS)
a=p.parse_args();spec=load_spec(a.config)
try:a.updates,a.compute_wall_budget=validate_run_limits(spec,a.updates,a.compute_wall_budget)
except ValueError as exc:p.error(str(exc))
if (Path(a.output)/'manifest.json').exists():p.error('output identity already exists; no overwrite or automatic continuation')
if not a.internal_worker and not a.dry_run:
    from sttw_control.direct_command_supervisor import supervise
    spec['budget']['additional_compute_wall_seconds']=min(spec['budget']['additional_compute_wall_seconds'],a.compute_wall_budget)
    raise SystemExit(supervise([sys.executable,__file__,*sys.argv[1:],'--internal-worker'],a.output,spec))
run(a.config,a.output,a.updates,a.compute_wall_budget,a.dry_run)
