"""Run the Phase 4 ground-approach alpha and speed contrasts on paired development roots."""
import argparse
from itertools import product
from pathlib import Path

from run_platform_task_oracle import read
from run_playground_wide_gap_oracle import run, save


def evaluate(protocol):
    plan=read(protocol);out=protocol.parent
    for part in range(2):
        folder=out/f'part_{part}';folder.mkdir(exist_ok=True)
        for skill in plan['skills']:
            case=folder/skill
            if (case/'status.json').exists():continue
            cfg=read(Path(plan['base'])/(skill+'.json'))
            hand=cfg['ground_handoff'];template=hand['handoff_variants'][0]
            variants=[];lanes=[];directions=[]
            for root,alpha,speed,approach,direction in product(plan['roots'][15*part:15*(part+1)],plan['alphas'],
                                                               plan['speeds'],range(3),range(3)):
                if (alpha,speed)==(0.,2.):continue
                lanes.append([approach,'h75',len(variants)])
                variants.append(dict(template,bank_row=root,root_variant=root,approach_alpha=alpha,
                                     approach_speed_m_s=speed,departure=['left','straight','right'][direction]))
                directions.append(direction)
            cfg.update(evaluation_lanes=lanes,ccd_capacity=8*len(lanes),role='Phase 4 DEV measured entry-speed pairs')
            hand['handoff_variants']=variants
            cfg['paired_departures']['direction_ids']=directions
            run(cfg,case,skill,10,out)
    save(out/'complete.json',dict(batches=4,worlds_per_batch=1080,simulated=4320,reused=540))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('protocol',type=Path)
    evaluate(p.parse_args().protocol)
