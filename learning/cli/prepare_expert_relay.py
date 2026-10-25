"""Seal10 new causal teacher relays and10 inherited matched baseline traces."""
import argparse
from pathlib import Path
from sttw_control.expert_relay_protocol import prepare_relay

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('runs/anticipatory_reference_pilot_v2'))
    p.add_argument('--history',type=Path,default=Path('runs/middle_lineage_endpoint_review_v1'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prepare_relay(a.source,a.history,a.output)
