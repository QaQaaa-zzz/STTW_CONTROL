"""Thin entrypoint for real TRAIN prefix collection, no optimization."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.recovery_bank import collect
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--alpha',type=int,choices=[0,1],required=True);p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();collect(a.campaign,a.alpha,a.output,smoke=a.smoke)
