"""Execute the frozen TEST in twelve GPU batches; every online method shares its physical oracle batch."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from run_platform_online_review import planned_actions
from run_platform_task_oracle import read
from run_playground_wide_gap_oracle import run, save


def execute(out):
    plan=read(out/'test_freeze_manifest.json');wide=Path(plan['wide']);base=wide/'primary/roots_0008'
    bank=dict(np.load(plan['bank']));sections=dict(zip(('near','mid','far'),plan['section_x']))
    save(out/'plan.json',dict(blind=True,section_x=plan['section_x']))
    for layout,map_entry in plan['maps'].items():
        folder=out/layout
        for part,force in enumerate(plan['forces_N']):
            group=folder/f'part_{part}';group.mkdir(exist_ok=True)
            for skill in ('source','rb_run1'):
                target=group/skill
                if (target/'status.json').exists():continue
                eps=plan['margin'][skill];tab=pd.read_csv(folder/(skill+'_prior.csv'))
                cfg=read(base/(skill+'.json'));hand=cfg['ground_handoff'];template=hand['handoff_variants'][0]
                options=[];lanes=[];directions=[]
                for root in plan['roots']:
                    planned=planned_actions(tab,bank,root,sections,eps)
                    choices=[(name,section,side,a,d) for (section,side),methods in planned.items() for name,(a,d) in methods.items()]
                    choices.extend(('oracle','near',1,a,d) for a,d in product(range(3),repeat=2))
                    for name,section,side,a,d in choices:
                        lanes.append([a,'h75',len(options)]);directions.append(d)
                        options.append(dict(template,bank_row=root,root_variant=root,planner=name,bin=section,side=side,
                            data_split='TEST',section_x=sections[section],force_y_N=force,planned_departure=d,
                            departure=['left','straight','right'][d],approach_alpha=plan['alpha'],approach_speed_m_s=plan['speed']))
                hand['handoff_variants']=options;cfg['paired_departures']['direction_ids']=directions
                cfg.update(world_map=map_entry['path'],approach_initial_bank=plan['bank'],evaluation_lanes=lanes,
                    ccd_capacity=8*len(lanes),role='Frozen independent TEST',platform_online=dict(ground_model=plan['ground_model'],
                    margin_m=eps,committed_vote=True,latency_ticks=plan['latency_ticks'],relative=dict(height=.16,
                    raster=str(folder/'terrain.npz'),initializer=str(folder/(skill+'_initializer.npz')))))
                run(cfg,target,skill,10,out)
    save(out/'complete.json',dict(physical_trajectories=15840,worlds_per_batch=1320,batches=12))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    execute(p.parse_args().out)
