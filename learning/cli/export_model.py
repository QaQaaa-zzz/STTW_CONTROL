#!/usr/bin/env python3
"""Export plugin-free XML for ROS on a host without the bundled ARM plugin."""
import argparse
import json
from pathlib import Path
from sttw_control.model import export_model

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(export_model(args.output),indent=2))
