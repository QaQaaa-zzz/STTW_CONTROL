"""Read sealed complete fixed64 DEV evaluation, without physics."""
import argparse
from pathlib import Path
from sttw_control.live_endpoint_report import analyze

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True)
    analyze(p.parse_args().evaluation)
