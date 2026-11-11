"""Compare archived alpha operating states, contact windows, and failure mechanisms."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[]
 for path in sorted(a.source.glob('*_seed*.npz')):
  with np.load(path) as z:
   for lane,label in enumerate(z['labels']):
    valid=z['valid'][:,lane];t=z['time'][valid,lane];phi=z['phi'][valid,lane];w=z['euler_roll_rate'][valid,lane];v=z['speed'][valid,lane];delta=z['delta'][valid,lane];cmd=z['command'][valid,lane];contact=z['contact_mean'][valid,lane];fields=list(z['contact_fields'])
    firsts=[]
    for wheel in ['front','rear']:
     hit=np.flatnonzero(contact[:,fields.index(wheel+'_raised')]>0)
     firsts.append(int(hit[0]) if len(hit) else None)
    front,rear=firsts;ref=front if front is not None else 0
    windows={'pre_contact_100ms':(max(t[0],t[ref]-.1),t[ref]),'after_front_250ms':(t[ref],t[ref]+.25),'last_100ms':(t[-1]-.1,t[-1]+.001)}
    for name,(lo,hi) in windows.items():
     ids=np.flatnonzero((t>=lo)&(t<hi))
     if not len(ids):continue
     ii,jj=ids[0],ids[-1];moment=contact[:,[fields.index(k) for k in ['front_normal_roll_Nm','front_friction_roll_Nm','rear_normal_roll_Nm','rear_friction_roll_Nm']]].sum(1)
     rows.append(dict(case=path.stem,label=str(label),window=name,start_s=float(t[ii]),end_s=float(t[jj]),observation_end_s=float(t[-1]),first_front_raised_s=None if front is None else float(t[front]),first_rear_raised_s=None if rear is None else float(t[rear]),actual_first_wheel_gap_s=None if front is None or rear is None else float(t[rear]-t[front]),speed_mean_m_s=float(np.mean(v[ids])),phi_start_deg=float(np.degrees(phi[ii])),phi_end_deg=float(np.degrees(phi[jj])),omega_start_deg_s=float(np.degrees(w[ii])),omega_end_deg_s=float(np.degrees(w[jj])),omega_change_deg_s=float(np.degrees(w[jj]-w[ii])),peak_abs_roll_deg=float(np.degrees(abs(phi[ids]).max())),peak_abs_omega_deg_s=float(np.degrees(abs(w[ids]).max())),peak_abs_delta_rad=float(abs(delta[ids]).max()),rate_command_saturated_fraction=float(np.mean(abs(cmd[ids])>=2.999)),total_contact_roll_impulse_Nms=float(np.sum(moment[ids])*.005),rear_contact_roll_impulse_Nms=float(np.sum(contact[ids,fields.index('rear_normal_roll_Nm')]+contact[ids,fields.index('rear_friction_roll_Nm')])*.005),front_unloaded_fraction=float(np.mean(contact[ids,fields.index('front_normal_N')]<.1)),rear_unloaded_fraction=float(np.mean(contact[ids,fields.index('rear_normal_N')]<.1)),path_peak_m=float(abs(z['path_error'][valid,lane][ids]).max()),source=str(path)))
 with (a.output/'contact_windows.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# S2：全216条轨迹的接触与运行状态','','[原始全工况XY与时序](/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_channel_scan/roll_contact_analysis/INDEX.md)','','窗口固定为首次前轮高地接触前100ms、其后250ms、真实观察终点前100ms。接触量来自原5kHz记录的5ms均值；总力矩包含承重及摩擦，单位Nm·s，不能直接当成侧倾速度冲量。末窗未满100ms时按实际观察记。','','## 单脊2.6m/s：三档末100ms比较','','|物理组|α|终点s|φ起→终°|ω起→终°/s|命令限速占比|后轮力矩积分Nm·s|','|---|---|---:|---|---|---:|---:|']
 for r in rows:
  if r['case'].startswith('single10_v2.6') and r['label'] in ['full_0','full_0.5','full_1'] and r['window']=='last_100ms':
   lines.append(f'|{r["case"].split("seed")[-1]}|{r["label"]}|{r["observation_end_s"]:.3f}|{r["phi_start_deg"]:.1f}→{r["phi_end_deg"]:.1f}|{r["omega_start_deg_s"]:.1f}→{r["omega_end_deg_s"]:.1f}|{r["rate_command_saturated_fraction"]:.2f}|{r["rear_contact_roll_impulse_Nms"]:.4f}|')
 lines+=['','全工况/全通道/五档/Zero与重复数据见contact_windows.csv。报告按真实终点排列；完整运行的末窗是任务结束附近，失败回合的末窗是失败前，不将二者当作同一事件的因果比较。下一步使用共同的接触事件窗口对照各档，并结合名义包络检查α=0与1的不同失效路径。']
 (a.output/'CONTACT.md').write_text('\n'.join(lines)+'\n');print('rows',len(rows),'trajectories',len(set((r['case'],r['label']) for r in rows)));print('\n'.join(lines[-11:]))
if __name__=='__main__':main()
