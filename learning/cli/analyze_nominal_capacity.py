"""Source-parameter impulse response, rate capacity, and two-impulse cascade."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import ss2tf
from scipy.sparse.linalg import expm_multiply
from sttw_control.torch_controller import TorchController

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 c=json.loads(a.config.read_text());rows=[];t=np.linspace(0,3,6001)
 for v in [1.,1.5,1.8,2.,2.25,2.3,2.567,2.6,3.]:
  a1,a2,a4,q,k=TorchController(c,1,dtype=torch.float64)._system(torch.tensor([v],dtype=torch.float64))
  a1,a2,a4,q=[x.item() for x in [a1,a2,a4,q]];kd,kphi,kp=k[0].numpy();gain=np.array([kphi,kp,kd])
  A=np.array([[0,1,0],[a4-a1*kphi,-a1*kp,a2-a1*kd],[-kphi,-kp,-kd]])
  response=expm_multiply(A,np.array([0.,1.,0.]),start=0,stop=3,num=len(t))
  u=-response@gain;phi=response[:,0];delay=c['wheelbase']/v
  shifted=np.interp(t-delay,t,phi,left=0);combined=phi+shifted
  gamma=kd+kphi/q;B=np.array([[0],[a1*gamma],[gamma]])
  numerator,den=ss2tf(A,B,np.array([[0.,0.,1.]]),np.zeros((1,1)))
  np.testing.assert_allclose(den,[1,15,75,125],atol=1e-9)
  np.testing.assert_allclose(numerator[0],[0,gamma,0,-gamma*a4],atol=1e-9)
  rows.append(dict(v_m_s=v,kdelta=kd,peak_roll_per_unit_impulse=float(abs(phi).max()),peak_time_s=float(t[np.argmax(abs(phi))]),steering_rate_peak_per_unit_impulse=float(abs(u).max()),linear_unsaturated_impulse_limit_rad_s=float(3/abs(u).max()),wheel_delay_s=delay,roll_at_delay=float(np.interp(delay,t,phi)),roll_rate_residual_fraction=float(np.interp(delay,t,response[:,1])),roll_at_delay_over_peak=float(np.interp(delay,t,phi)/abs(phi).max()),two_equal_same_sign_impulses_peak_ratio=float(abs(combined).max()/abs(phi).max()),reference_to_steer_positive_zero=float(np.sqrt(a4))))
 (a.output/'nominal_capacity.json').write_text(json.dumps(dict(config=str(a.config),rows=rows,unit_impulse='initial Euler roll-rate increment 1 rad/s',rate_limit_rad_s=3,assumptions='fixed speed; linear ECBC; no ESO, Local300, filters, travel bounds or contact; equal same-sign impulses for cascade example'),indent=2))
 lines=['# S0/S2：名义冲量容量与接触间隔计算','','源码增益下的转向参考→转角传递函数分子验证为γ(s²−a₄)，正零点为√a₄；分母为(s+5)³。单位冲量响应由状态矩阵指数计算。','','|v m/s|单冲量侧倾峰值/单位冲量|L/v s|L/v时刻响应/峰值|双同向等冲量峰值倍率|不触发±3rad/s的冲量上限 rad/s|','|---|---:|---:|---:|---:|---:|']
 for r in rows:lines.append(f'|{r["v_m_s"]}|{r["peak_roll_per_unit_impulse"]:.5f}|{r["wheel_delay_s"]:.4f}|{r["roll_at_delay_over_peak"]:.4f}|{r["two_equal_same_sign_impulses_peak_ratio"]:.4f}|{r["linear_unsaturated_impulse_limit_rad_s"]:.4f}|')
 lines+=['','这里的容量是线性反馈不触发转向速度限幅的冲量上限，不是极限恢复能力。级联表固定两个同号等幅冲量；实际前后轮的符号、幅值和时间须由接触记录输入。将L/v时刻响应除以单次峰值，明确定义残余比例，避免把不同归一化混成同一个p₁。']
 (a.output/'NOMINAL.md').write_text('\n'.join(lines)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
