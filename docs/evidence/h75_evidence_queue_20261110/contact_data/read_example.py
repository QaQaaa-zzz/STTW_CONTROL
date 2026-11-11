import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--case',default='single10_v2.3_seed101');p.add_argument('--lane',type=int,default=6);p.add_argument('--contacts',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parent
cases=json.loads((r/'cases.json').read_text());case=next(c for c in cases if c['case']==a.case);lane=a.lane
with np.load(r/'cases'/a.case/'trajectory.npz') as z:
 trace=z['lower_trace'].reshape(-1,len(case['labels']),11);m=z['measurement'];valid=trace[:,lane,6]>0
 t=np.arange(len(trace))*.005+trace[:,lane,7]*.0002
 table=np.column_stack((t[valid],trace[valid,lane,2],m[valid,lane,:4]))
 np.savetxt(r/(a.case+'_'+str(lane)+'_signals.csv'),table,delimiter=',',header='time_s,v_m_s,phi_rad,negative_imu_x_rad_s,delta_rad,delta_dot_rad_s',comments='')
 print(case['labels'][lane],table.shape)
if a.contacts:
 idx=json.loads((r/'physics_contacts/index.json').read_text());j=case['lane_start']+lane
 with np.load(r/'physics_contacts'/idx['chunks'][0]['file']) as z:
  wheel='frontwheel';tick=0;start=int(z[wheel+'_normal_start'][tick,j,0]);count=int(z[wheel+'_normal_count'][tick,j,0]);sl=slice(start,start+count)
  force=(z[wheel+'_normal_force'][tick,sl]*z[wheel+'_normal_direction'][tick,sl]).sum(0)
  print('first recorded time_s',z['physics_step'][tick]*idx['physics_dt_s'],'front normal force world N',force,'alive',z['alive_before'][tick,j])
