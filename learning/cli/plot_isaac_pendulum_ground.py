"""Plot saved single-lane interface evidence without rerunning physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
d=np.load(a.run/'trajectory.npz');tr=d['trace'];names=list(d['columns']);v=lambda n:tr[:,names.index(n)]
f,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
ax=axes[0,0];ax.plot(v('x'),v('y'),color='#D55E00',label='ECBC + fixed pendulum');ax.plot([v('x')[0],v('x')[0]+2*v('time')[-1]],[v('y')[0],v('y')[0]],':',color='gray',label='2 m/s straight reference');ax.plot(v('x')[-1],v('y')[-1],'x',color='black',label='actual endpoint');ax.set(xlabel='World x (m)',ylabel='World y (m)',aspect='equal',ylim=(-1,1));ax.legend(loc='lower left',fontsize=8)
axes[0,1].plot(v('time'),v('speed'),color='#D55E00');axes[0,1].axhline(2,color='gray',ls=':');axes[0,1].set(xlabel='Time (s)',ylabel='Forward speed (m/s)');axes[0,1].annotate('initial read may retain reset cache',xy=(v('time')[0],v('speed')[0]),xytext=(.5,2.22),arrowprops={'arrowstyle':'->'},fontsize=8)
axes[1,0].plot(v('time'),np.rad2deg(v('roll')),label='roll');axes[1,0].plot(v('time'),np.rad2deg(v('steer')),'--',label='steer');axes[1,0].set(xlabel='Time (s)',ylabel='Angle (deg)');axes[1,0].legend()
for name,style in [('hip','-'),('knee','--')]:axes[1,1].plot(v('time'),v(name),style,label=name)
axes[1,1].set(xlabel='Time (s)',ylabel='Pendulum angle (rad)');axes[1,1].legend()
f.suptitle('Common STTW chassis + pendulum / Isaac Lab GPU\nNominal interface diagnostic; original ECBC, no learned policy')
f.savefig(a.run/'xy_response.png',dpi=160);plt.close(f)
r=json.loads((a.run/'results.json').read_text())
(a.run/'RESULTS.md').write_text('# Isaac Lab pendulum ground diagnostic\n\n![Actual XY and response](xy_response.png)\n\n'+json.dumps(r,indent=2)+'\n\nSingle nominal initialization, maximum 3 s, original ECBC gains, fixed hip/knee targets. This is neither an H75 evaluation nor a jump handoff. No training or learned-policy reward. Source protocol and actual state samples are protocol.json and trajectory.npz.\n')
