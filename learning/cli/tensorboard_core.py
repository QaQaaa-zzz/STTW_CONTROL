#!/usr/bin/env python3
"""Live compact projection from explicit JSONL lineages; originals stay intact."""
import argparse
import json
import time
from pathlib import Path
from sttw_control.tensorboard_logging import TrainingEvents


def project(manifest, output, watch=False, interval=15, resume=False):
    runs=json.loads(Path(manifest).read_text())['runs']
    output=Path(output)
    if output.exists():
        if not resume:raise FileExistsError('Use --resume only for this existing compact projection')
        previous=json.loads((output/'projection.json').read_text())
        if previous['runs'] != runs:raise ValueError('Projection lineage changed')
    writers={name:TrainingEvents(output/name,profile='core',register=False) for name in runs}
    if resume:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for name,writer in writers.items():
            events=EventAccumulator(str(output/name)).Reload()
            steps=[e.step for tag in events.Tags()['scalars'] for e in events.Scalars(tag)]
            writer.last_step=max(steps) if steps else None
    try:
        while True:
            for name,paths in runs.items():
                records={}
                for source in paths:
                    path=Path(source)
                    if not path.exists():continue
                    for line in path.read_text().splitlines():
                        try:r=json.loads(line)
                        except json.JSONDecodeError:continue  # Writer may be appending the last line.
                        update=int(r['update'])
                        if update in records:raise ValueError(f'Overlapping lineage update: {name} {update}')
                        records[update]=r
                writer=writers[name]
                for step in sorted(records):
                    if writer.last_step is None or step>writer.last_step:writer.write(records[step])
            output.mkdir(parents=True,exist_ok=True)
            (output/'projection.json').write_text(json.dumps({'runs':runs,'profile':'core','last_update':{k:v.last_step for k,v in writers.items()}},indent=2)+'\n')
            if not watch:break
            time.sleep(interval)
    finally:
        for writer in writers.values():writer.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=float,default=15)
    a=p.parse_args();project(a.manifest,a.output,a.watch,a.interval,a.resume)
