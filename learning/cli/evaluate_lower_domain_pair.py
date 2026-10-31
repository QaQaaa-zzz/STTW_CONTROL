"""Paired frozen lower-controller test on held-out physical domains."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.lower_domains import evaluate_domain_pair
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
a=p.parse_args();evaluate_domain_pair(a.config,a.output)
