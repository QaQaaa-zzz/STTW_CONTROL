"""Recompute S3 continuous margins with the frozen DEV measurement definitions."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from build_mechanism_forecast import collect
from sttw_control.margin_theory import tracking_domain_margins
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--model',type=Path,required=True);a=p.parse_args();model=json.loads(a.model.read_text());frozen=json.loads((a.run/'frozen_predictions.json').read_text());out=a.run/'prediction_analysis';allrows=[]
for family in ['ridge','gravel']:
    rows=collect(a.run,family,model['controller']);(out/f'{family}_measured_states.json').write_text(json.dumps(rows,indent=2));allrows+=rows
pairs=[]
for row in allrows:
    context=next(x for x in frozen['contexts'] if (x['family'],x['height_m'],x['speed_m_s'])==(row['family'],row['height_m'],row['requested_speed_m_s']));pr=next(x for x in context['predictions'] if x['alpha']==row['alpha']);dm,hm=tracking_domain_margins(*row['domain_state'])
    values=dict(recovery_margin_rad=row['min_recovery_margin_rad'],domain_margin_m=float(dm),heading_margin_rad=float(hm),path_peak_m=row['path_peak_m'],speed_rmse_m_s=row['speed_rmse_m_s'])
    for key,value in values.items():
        forecast=float(np.mean([m[key] for m in pr['members']]))
        pairs.append(dict(case=row['case'],family=row['family'],group=row['group'],alpha=row['alpha'],metric=key,forecast=forecast,actual=value,error=forecast-value,cause=row['observed_cause']))
with (out/'continuous_forecast_errors.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
summary=[]
for family in ['ridge','gravel']:
    for metric in values:
        cells=[x for x in pairs if x['family']==family and x['metric']==metric];err=np.array([x['error'] for x in cells]);summary.append(dict(family=family,metric=metric,n=len(cells),mae=float(np.mean(abs(err))),median_absolute_error=float(np.median(abs(err))),bias=float(err.mean())))
(out/'continuous_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
