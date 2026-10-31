"""Prepare isolated new-physics source/assets, or explicitly execute fresh bank preparation."""
import argparse
import importlib.util
from pathlib import Path
import sys
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','execute','bind-domain']);p.add_argument('root',type=Path);p.add_argument('--source',type=Path);p.add_argument('--config',type=Path,default=Path('learning/configs/native_physics_transition.json'));a=p.parse_args()
f=Path(__file__).resolve().parents[1]/'src/sttw_control/native_physics_transition.py';s=importlib.util.spec_from_file_location('_native_physics_transition',f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
if a.stage in ('prepare','bind-domain'):
    if a.source is None:p.error('prepare requires --source')
    (m.prepare_transition if a.stage=='prepare' else m.bind_domain_bank)(a.source,a.root,a.config)
else:m.execute_preparation(a.root)
