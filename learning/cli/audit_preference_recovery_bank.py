"""CPU-only real-bank identity, clocks and conditional sampler audit."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.margin_state_sampling import audit_bank

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('bank');p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(audit_bank(a.bank,a.output),indent=2))
