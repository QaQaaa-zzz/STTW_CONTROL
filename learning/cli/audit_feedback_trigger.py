"""Calibrate one DEV feedback trigger from recorded nominal roll-rate increments."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def main(source,output):
    cases=json.loads((source/'evaluation/case_index.json').read_text());signals=[];calibration=[]
    for c in cases:
        path=Path(c['raw_file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==c['raw_sha256']
        with np.load(path) as z:d=dict(z)
        idx=np.arange(4,int(d['active_tick'].sum()),4)
        delta=np.abs(d['coord_pre_roll_rate'][idx]-d['coord_pre_roll_rate'][idx-4])
        signals.append((c,d,idx,delta))
        if c['force_N']==0:calibration.append(dict(case_id=c['case_id'],raw_sha256=c['raw_sha256'],maximum_increment_rad_s=float(delta.max())))
    threshold=2*max(c['maximum_increment_rad_s'] for c in calibration)
    records=[]
    for c,d,idx,delta in signals:
        hits=np.flatnonzero((delta>threshold)&d['coord_held'][idx]&d['coord_permission'][idx])
        first=None if not len(hits) else int(idx[hits[0]])
        records.append(dict(case_id=c['case_id'],force_N=c['force_N'],first_trigger_tick=first,first_trigger_s=None if first is None else first/200,increment_rad_s=None if first is None else float(delta[hits[0]]),pre_roll_rate_rad_s=None if first is None else float(d['coord_pre_roll_rate'][first])))
    result=dict(source=str(source.resolve()),source_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),rule='abs(pre_roll_rate[t]-pre_roll_rate[t-0.02]) > threshold AND held AND publication_permitted',calibration=calibration,multiplier=2.,threshold_rad_s=threshold,records=records,scope='Exploratory DEV design after observing failure; nominal-only scale, engineering multiplier, no noise bound. Trigger does not choose alpha. No physical execution or online implementation.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(threshold)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();main(a.source,a.output)
