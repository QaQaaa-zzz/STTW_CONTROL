"""Cross-score frozen trajectories at complete reward endpoints, retaining termination."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_controller import TorchController
from sttw_control.torch_path_reward import costs

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data=json.loads(a.source.read_text());a.output.mkdir(parents=True,exist_ok=True);panels=[];closure=0.;steps=0
 for case in data['cases']:
  samples=[]
  for seed in case['per_seed']:
   run=Path(seed['run']);contract=json.loads((run/'contract.json').read_text());config=json.loads((Path(contract['bundle'])/'config.json').read_text())
   cc=json.loads(Path(contract['settings']['controller_config']).read_text());speed=case['speed_m_s'];c=config['path_reward']
   with np.load(run/'trajectory.npz') as z,np.load(Path(contract['bundle'])/'route.npz') as route:
    tr=z['lower_trace'].reshape(-1,6,11);n=len(tr);shape=(n,6)
    ratio=TorchController(cc,1)._system(torch.tensor([speed],dtype=torch.float32))[3].item()
    curvature=np.interp(z['progress'],route['s'],route['curvature'])
    chi=np.clip((abs(np.arctan(cc['wheelbase']*curvature/np.cos(cc['caster']))/ratio)-.18)/.12,0,1)
    args=dict(chi=torch.tensor(chi),ev=torch.tensor(tr[:,:,2]-speed),ey=torch.tensor(tr[:,:,3]),eh=torch.zeros(shape),peak_roll=torch.tensor(tr[:,:,4]),roll_rate=torch.tensor(z['measurement'][:,:,1]),speed=torch.tensor(tr[:,:,2]),offsets=torch.tensor(z['offsets']),offset_rate=torch.zeros(n,6,2),config=config)
    endpoints=[costs(alpha=torch.full(shape,float(alpha)),**args)['effective'] for alpha in [0,1]]
    alpha=z['alpha'];pathbase=z['effective_path']/(1-(1-c['path_secondary_ratio'])*alpha)
    common=sum(z[k] for k in z.files if k.startswith('effective_') and k not in ['effective_speed','effective_path','effective_primary','effective_offset_rate'])
    offsets=z['offsets'].reshape(-1,4,6,2)[:,-1];prev=np.concatenate([np.zeros_like(offsets[:1]),offsets[:-1]])
    rate=np.minimum(c['offset_rate_weight']*(((offsets-prev)/.02/np.asarray(c['offset_rate_scales']))**2).sum(-1),c['component_caps']['offset_rate'])
    endpoint_steps=[]
    for alpha0,terms in zip([0,1],endpoints):
     value=common+terms['speed'].numpy()+terms['primary'].numpy()+pathbase*(1-(1-c['path_secondary_ratio'])*alpha0)
     reward=-c['scale']*.005*value
     reward=np.where(tr[:,:,6].astype(bool),reward,0).reshape(-1,4,6).sum(1)-c['scale']*.02*rate
     reward=np.where(z['failure'],z['reward'],reward)
     reward=np.where(z['valid'],reward,0);endpoint_steps.append(reward)
    # Actual-alpha reward must be recovered before any endpoint comparison.
    aa=alpha.reshape(-1,4,6)[:,0]
    rebuilt=(1-aa)*endpoint_steps[0]+aa*endpoint_steps[1]
    err=float(abs(rebuilt-z['reward']).max());closure=max(closure,err);steps+=z['reward'].size
    np.testing.assert_allclose(rebuilt,z['reward'],rtol=3e-4,atol=3e-4)
    gamma=config['training_future_phase_C']['gamma'];discount=gamma**np.arange(len(rebuilt))
    fs=np.array([-np.sum(reward*discount[:,None],axis=0) for reward in endpoint_steps])
    samples.append(dict(seed=seed['seed'],F0=fs[0,:5].tolist(),F1=fs[1,:5].tolist(),zero_F0=float(fs[0,5]),zero_F1=float(fs[1,5]),gamma=gamma,closure_max=err,failures=z['failure'].any(0).tolist(),run=str(run)))
  f0=np.mean([r['F0'] for r in samples],axis=0);f1=np.mean([r['F1'] for r in samples],axis=0)
  mono=(np.diff(f0)>=-1e-5)&(np.diff(f1)<=1e-5)
  alpha=np.array([0,.25,.5,.75,1]);cross=(1-alpha[:,None])*f0+alpha[:,None]*f1
  panels.append(dict(terrain=case['terrain'],speed_m_s=speed,F0=f0.tolist(),F1=f1.tolist(),monotone_edges=mono.tolist(),best_policy_at_evaluation_alpha=alpha[cross.argmin(1)].tolist(),self_preference_regret=(np.diag(cross)-cross.min(1)).tolist(),per_seed=samples))
 result=dict(panels=panels,reward_steps_checked=steps,max_reward_reconstruction_error=closure,monotone_edges=sum(sum(r['monotone_edges']) for r in panels),all_monotone_panels=sum(all(r['monotone_edges']) for r in panels),new_simulations=0)
 (a.output/'objectives.json').write_text(json.dumps(result,indent=2))
 lines=['# S1：完整训练目标重评分','','[全部实际/参考轨迹](xy_all.png) · [原始误差前沿](README.md)','','固定物理轨迹重评分α=0与1，包含姿态/低速/超速/动作幅值共同代价、50Hz动作变化代价、实际终止替换和原gamma折扣。失败窗口按源码整体替换，终止后置零。路径/航向端点由已记录有效分项反解；速度与主任务带外项按原始状态及χ门控重算。','',f'共核对{steps}个策略步奖励，最大逐步差{closure:.6g}。完整目标有{result["monotone_edges"]}/80条相邻边满足F0不减、F1不增；{result["all_monotone_panels"]}/20个面板全部满足。','','|地形|速度|单调边数/4|评价α=0/.25/.5/.75/1时的最佳实测策略α|','|---|---:|---:|---|']
 for r in panels:lines.append(f'|{r["terrain"]}|{r["speed_m_s"]}|{sum(r["monotone_edges"])}|{r["best_policy_at_evaluation_alpha"]}|')
 lines+=['','该结果检验冻结策略族是否实现标量化目标的预期排序。最佳档和regret是在同一批已见工况中的重评分；下一阶段预测应保留这些偏离，不能把每档等同精确最优策略。各物理组、失败、Zero配对代价均保存在objectives.json。']
 (a.output/'OBJECTIVES.md').write_text('\n'.join(lines)+'\n');print({k:v for k,v in result.items() if k!='panels'})
if __name__=='__main__':main()
