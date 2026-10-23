"""Analyze a completed frozen single-alpha screen, without simulation."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_tolerance_reporting import report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(report(a.run,a.output),indent=2))
