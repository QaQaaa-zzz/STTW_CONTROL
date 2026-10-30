"""Prepare complete physical/controller states under their own frozen domain."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.lower_domains import prepare_domain_states
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
prepare_domain_states(a.config,a.output)
