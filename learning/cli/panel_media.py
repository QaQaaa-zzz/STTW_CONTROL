#!/usr/bin/env python3
"""Export every standard event as a synchronized baseline/residual comparison."""
import argparse
from pathlib import Path
from sttw_control.panel_media import export_panel

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2,choices=range(1,5))
    a=p.parse_args();export_panel(a.panel_run,a.output,workers=a.workers)
