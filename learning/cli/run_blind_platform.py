"""Execute the frozen TEST in paired GPU batches; every online method shares its physical oracle batch."""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from run_platform_online_review import planned_actions
from run_platform_task_oracle import read
from run_playground_wide_gap_oracle import run, save


def execute(out, alpha_choices=None):
    plan=read(out/'test_freeze_manifest.json');wide=Path(plan['wide']);base=wide/'primary/roots_0008'
    bank=dict(np.load(plan['bank']));sections=dict(zip(('near','mid','far'),plan['section_x']))
    save(out/'plan.json',dict(blind=True,section_x=plan['section_x'],pair=plan.get('pair',['C','C-R'])))
    batches = plan.get('assignment_batches') or [[(root,force) for root in plan['roots']] for force in plan['forces_N']]
    packing_file = out/'execution_packing.json'
    packing = read(packing_file)['by_layout_skill'] if packing_file.exists() else {}
    sizes = []
    for layout,map_entry in plan['maps'].items():
        folder=out/layout
        by_skill = packing.get(layout, {skill:batches for skill in plan.get('skills',('source','rb_run1'))})
        for part in range(max(map(len,by_skill.values()))):
            for skill,parts in by_skill.items():
                if part >= len(parts):
                    continue
                assignments = parts[part]
                sizes.append(len(assignments)*plan.get('lanes_per_assignment',33))
                group=folder/f'part_{part}';group.mkdir(exist_ok=True)
                target=group/skill
                if (target/'status.json').exists():continue
                eps=plan['margin'][skill];tab=pd.read_csv(folder/(skill+'_prior.csv'))
                cfg=read(base/(skill+'.json'));hand=cfg['ground_handoff'];template=hand['handoff_variants'][0]
                options=[];lanes=[];directions=[]
                for root,force in assignments:
                    if alpha_choices:
                        choices=alpha_choices(tab,bank,root,sections,eps)
                    else:
                        planned=planned_actions(tab,bank,root,sections,eps)
                        choices=[(name,section,side,a,d,plan['alpha'],name) for (section,side),methods in planned.items() for name,(a,d) in methods.items()]
                        choices.extend(('oracle','near',1,a,d,plan['alpha'],'oracle') for a,d in product(range(3),repeat=2))
                    for name,section,side,a,d,alpha,arm in choices:
                        lanes.append([a,'h75',len(options)]);directions.append(d)
                        options.append(dict(template,bank_row=root,root_variant=root,planner=name,bin=section,side=side,
                            data_split='TEST',section_x=sections[section],force_y_N=force,planned_departure=d,
                            departure=['left','straight','right'][d],approach_alpha=alpha,arm=arm,approach_speed_m_s=plan['speed']))
                hand['handoff_variants']=options;cfg['paired_departures']['direction_ids']=directions
                cfg.update(world_map=map_entry['path'],approach_initial_bank=plan['bank'],evaluation_lanes=lanes,
                    ccd_capacity=8*len(lanes),role=plan['role'],platform_online=dict(ground_model=plan['ground_model'],
                    margin_m=eps,committed_vote=True,latency_ticks=plan['latency_ticks'],relative=dict(height=.16,
                    raster=str(folder/'terrain.npz'),initializer=str(folder/(skill+'_initializer.npz')))))
                if 'veto_banks' in plan:
                    cfg['platform_online']['veto_bank'] = plan['veto_banks'][layout]
                run(cfg,target,skill,10,out)
    save(out/'complete.json',dict(physical_trajectories=sum(sizes),worlds_per_batch=sizes[0] if len(set(sizes))==1 else sizes,
                                  batches=len(sizes)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    execute(p.parse_args().out)
