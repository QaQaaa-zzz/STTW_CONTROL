"""Cross-fit measured wide-gap support margins by root and compare deadline errors."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from run_playground_task_reversal import write_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('deadline_run','model','calibration','output'):
        parser.add_argument(name,type=Path)
    a=parser.parse_args()
    out=a.output
    out.mkdir(parents=True,exist_ok=True)
    groups=json.loads((a.deadline_run/'plan.json').read_text())['groups']
    edges=next(r['edges'] for r in json.loads(a.calibration.read_text()) if r['model']=='M1')
    fits=[]
    roots={}
    def evaluate(group,cal,dst,*extra):
        matrix=Path(groups[group])/'oracle_task_reversal.csv'
        subprocess.run([sys.executable,'learning/cli/evaluate_task_reversal_predictions.py',str(matrix),
                        str(a.model),str(cal),str(dst),*extra],check=True)
    for group,folder in groups.items():
        path=out/group/'legacy10'
        evaluate(group,a.calibration,path,'--fit-low-speed')
        fits.extend(json.loads((path/'support_fit.json').read_text()))
        with (Path(folder)/'oracle_task_reversal.csv').open() as f:
            roots[group]=next(csv.DictReader(f))['root']
    counts=[]
    for group,root in roots.items():
        fit=[r for r in fits if r['root']!=root]
        margin=np.zeros((len(edges)-1,3)); samples=np.zeros(len(margin),int)
        for r in fit:
            region=r['region']
            margin[region]=np.maximum(margin[region],r['peak_m'])
            samples[region]+=r['samples']
        assert samples.any()
        margin[samples==0]=margin.max(0)
        cal=out/group/'low_speed_calibration.json'
        cal.write_text(json.dumps([dict(model='M1',edges=edges,axis_margin_m=(margin+.02).tolist(),
                                    samples=samples.tolist(),fit_roots=sorted({r['root'] for r in fit}),
                                    excluded_root=root,velocity='world_vx in [1.1,1.5] m/s')],indent=2))
        for deadline in (10,12):
            extra=[] if deadline==10 else ['--deadline','12','--extended-truth',str(a.deadline_run/'deadlines.csv'),'--group',group]
            for label,source in [('legacy',a.calibration),('low_speed',cal)]:
                dst=out/group/f'{label}{deadline}'
                tag='shared_dev' if label=='legacy' else 'low_speed_shared'
                if (label,deadline)!=('legacy',10):
                    evaluate(group,source,dst,'--margin-name',tag,*extra)
                with (dst/'offline_false_accepts.csv').open() as f:
                    rows=[r for r in csv.DictReader(f) if r['margin']==tag]
                for method in ('kinematic','phase2'):
                    cases=[r for r in rows if r['method']==method]
                    accepts=[r for r in cases if r['false_pass']=='True']
                    physical={(r['lane'],r['departure']):r for r in cases}.values()
                    counts.append(dict(group=group,root=root,deadline_s=deadline,margin=label,method=method,
                          actual_positive=sum(r['actual_pass']=='True' for r in cases),contract_rows=len(cases),
                          false_reject=sum(r['false_reject']=='True' for r in cases),false_accept=len(accepts),
                          false_accept_collision=sum(r['actual_collision']=='True' for r in accepts),
                          false_accept_timeout=sum(r['actual_termination']=='timeout' for r in accepts),
                          false_accept_other=sum(r['actual_termination'] not in ('timeout','forbidden_obstacle_contact') for r in accepts),
                          wall_hits_predicted_clear=sum(r['actual_collision']=='True' and r['predicted_collision_free']=='True' for r in physical),
                          actual_wall_hits=sum(r['actual_collision']=='True' for r in physical),physical_suffixes=len(physical)))
        write_rows(out/'comparison.csv',counts)


if __name__=='__main__':
    main()
