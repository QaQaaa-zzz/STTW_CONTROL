"""Collect declared authentic single-alpha TRAIN prefixes and parent anchors."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.recovery_bank import collect_preference

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declaration',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();collect_preference(a.declaration,a.output)
