#!/usr/bin/env python3
"""Generate complete command reward diagnostics, optionally after a running pipeline."""
import argparse,json,time
from pathlib import Path
from sttw_control.command_diagnostics import generate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--wait',action='store_true')
    a=p.parse_args()
    if a.wait:
        deadline=time.monotonic()+24*3600
        while True:
            try:phase=json.loads((a.run/'status.json').read_text()).get('phase')
            except (FileNotFoundError,json.JSONDecodeError):phase=None
            if phase=='complete':break
            if phase in ('error','failed','timeout'):raise RuntimeError(f'Pipeline ended with {phase}')
            if time.monotonic()>deadline:raise TimeoutError('Diagnostics wait exceeded 24h; no training was changed')
            time.sleep(30)
    print(json.dumps({'evaluations':len(generate(a.run))}))
