"""Branch the frozen ground routes only from actual qualified disturbed handbacks."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from run_playground_task_reversal import write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    root=parser.parse_args().run.resolve()
    plan=json.loads((root/'plan.json').read_text())
    pulses=Path(plan['input'])
    with (pulses/'paired_pulses.csv').open() as stream:
        physical=list(csv.DictReader(stream))
    assert len(physical)==json.loads((pulses/'terminal.json').read_text())['scored']
    contracts=json.loads(Path(plan['contracts']).read_text())['contracts']
    outcomes=[]
    for row in physical:
        accepted=row['qualified']=='True'
        case=root/f"{row['window']}_{row['skill']}"
        if accepted:
            if not (case/'plan.json').exists():
                case.mkdir()
                inputs=case/'input'
                inputs.mkdir()
                (inputs/'screen').symlink_to(pulses/case.name,target_is_directory=True)
                (inputs/'screen.json').symlink_to(pulses/(case.name+'.json'))
                (case/'gpu.lock').symlink_to((pulses/'gpu.lock').resolve())
                cfg=json.loads((inputs/'screen.json').read_text())
                lane=int(row['lane'])
                offset=cfg['ground_handoff']['handoff_variants'][lane]['activation_offset_m']
                write_rows(inputs/'prefixes.csv',[dict(lane=lane,trigger_offset_m=offset)])
                snap=(round(float(row['handoff_s'])/.005)+1)*.005
                protocol=dict(input=str(inputs),skill=row['skill'],lanes=[lane],snapshots_s=[snap],
                              horizon_s=10,contracts=contracts)
                (case/'plan.json').write_text(json.dumps(protocol,indent=2)+'\n')
            if not (case/'complete.json').exists():
                flags=[] if (case/'repeat/jump_validation.json').exists() else ['--capture']
                subprocess.run([sys.executable,'learning/cli/run_playground_task_reversal.py',str(case),
                                plan['departure_bank'],*flags],check=True)
            subprocess.run([sys.executable,'learning/cli/report_task_reversal_matrix.py',str(case),str(case/'report')],check=True)
            with (case/'oracle_task_reversal.csv').open() as stream:
                scores=list(csv.DictReader(stream))
        for label in ('near','mid','far'):
            for side in ('left','right'):
                options=sorted(r['departure'] for r in scores if r['bin']==label and r['task']==side and r['success']=='True') if accepted else []
                outcomes.append(dict(window=row['window'],skill=row['skill'],qualified=accepted,bin=label,task=side,
                                     evaluated_departures=3 if accepted else 0,
                                     feasible_departures=' '.join(options) if accepted else 'NA',
                                     legal_suffix_exists=bool(options) if accepted else 'NA',online_selected='NA'))
        write_rows(root/'contract_opportunities.csv',outcomes)
    (root/'terminal.json').write_text(json.dumps(dict(physical_prefixes=len(physical),
        handbacks=sum(r['qualified']=='True' for r in physical),task_rows=len(outcomes)))+'\n')


if __name__=='__main__':
    main()
