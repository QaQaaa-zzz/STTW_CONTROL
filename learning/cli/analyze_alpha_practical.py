"""Post-hoc practical tolerance sensitivity on existing alpha trajectories."""
import argparse,json
from pathlib import Path
import numpy as np


def analyze(panel,output):
 panel=Path(panel);out=Path(output);out.mkdir(parents=True,exist_ok=True)
 metrics=json.loads((panel/'metrics.json').read_text());rows=[]
 for row in metrics['cases']:
  label=f"{row['alpha']:g}".replace('.','p');path=panel/('alpha_'+label)/(row['case']+'.npz')
  with np.load(path) as d:
   time=d['time'];tail=time>=9.-1e-5
   if len(time)!=2000 or tail.sum()!=200:raise ValueError('expected complete 10s/2000tick trace')
   item=dict(row,tail_max_path_m=float(np.max(np.abs(d['path_cross_track'][tail]))),tail_max_heading_rad=float(np.max(np.abs(d['path_heading_error'][tail]))),tail_max_speed_m_s=float(np.max(np.abs(d['actual_forward_speed'][tail]-d['v_user'][tail]))),source=str(path))
  rows.append(item)
 alphas=sorted({x['alpha'] for x in rows});sensitivity=[]
 for path_limit in (.1,.15,.2):
  for roll_limit in (.302,.33):
   for alpha in alphas:
    group=[x for x in rows if x['alpha']==alpha]
    passed=lambda r:not(r['physical'] or r['domain']) and r['goal'] and r['peak_roll']<=roll_limit and r['tail_max_path_m']<=path_limit and r['tail_max_heading_rad']<=.05 and r['tail_max_speed_m_s']<=.1
    sensitivity.append(dict(alpha=alpha,path_limit_m=path_limit,roll_limit_rad=roll_limit,passed=sum(passed(r) for r in group),n=len(group)))
 pairs=[]
 for case in sorted({r['case'] for r in rows}):
  a,b=[next(r for r in rows if r['case']==case and r['alpha']==alpha) for alpha in (0,1)]
  pairs.append(dict(case=case,alpha1_speed_rmse_advantage=a['speed_rmse']-b['speed_rmse'],alpha0_path_rmse_advantage=b['path_rmse']-a['path_rmse']))
 result=dict(source=str(panel),classification='posthoc development sensitivity; no original labels changed',cases=rows,endpoint_comparison=pairs,sensitivity=sensitivity)
 (out/'analysis.json').write_text(json.dumps(result,indent=2))
 lines=['# Single-alpha practical sensitivity','',f'Source: {panel}', '', 'Post-hoc analysis of saved 10s trajectories; original pass/fail labels preserved. Limits below are diagnostic options, not a proven safety envelope. Training is not modified.','', '| Alpha | Full-trace speed RMSE | Full-trace path RMSE | Largest final path error |','|---|---:|---:|---:|']
 for a in alphas:
  g=[r for r in rows if r['alpha']==a];lines.append(f"| {a:g} | {np.mean([r['speed_rmse'] for r in g]):.4f} m/s | {np.mean([r['path_rmse'] for r in g]):.4f} m | {max(r['tail_max_path_m'] for r in g):.4f} m |")
 lines+=['','## Endpoints','','| Case | Alpha1 speed advantage (m/s) | Alpha0 path advantage (m) |','|---|---:|---:|']
 for p in pairs:lines.append(f"| {p['case']} | {p['alpha1_speed_rmse_advantage']:.4f} | {p['alpha0_path_rmse_advantage']:.4f} |")
 lines+=['','## Sensitivity','','| Path limit (m) | Roll limit (rad) | Alpha | Pass / total |','|---:|---:|---:|---:|']
 for r in sensitivity:lines.append(f"| {r['path_limit_m']} | {r['roll_limit_rad']} | {r['alpha']:g} | {r['passed']}/{r['n']} |")
 lines+=['',f"[Existing matched-alpha XY figure]({panel/'XY.png'})",'', 'Interpretation: check turn trade-offs, straight-line bias, and final recovery separately. Do not force alpha1 to worsen its path merely to create separation. A relaxed threshold does not remove measured residual bias.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(cases=len(rows),relaxed_pass=sum(r['passed'] for r in sensitivity if r['path_limit_m']==.2 and r['roll_limit_rad']==.33))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.panel,a.output)
