"""Saved-data-only physical comparisons and independent reward reconstruction."""
from pathlib import Path
import json,importlib.util
import numpy as np
from .smooth_command_training import OLD
from .smooth_command_config import resolve
from .direct_command_training import write,clean_numbers

ROOT=Path(__file__).resolve().parents[3]
def helper(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'docs/smooth_v4/attachment/{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def physical(d,end=10.):
    t=d['time'];ev=d['actual_forward_speed']-d['limited_command'][:,0];ed=d['actual_delta']-d['limited_command'][:,1];eh=d['e_psi_unwrapped']
    valid=t<end-1e-6;last=(t>=end-.5-1e-6)&valid
    failure=bool(np.any(d['physical_failure'][valid]));complete=int(valid.sum())==round(end/.005) and not failure
    hold=[bool(complete and last.sum()==100 and np.all(np.abs(x[last])<=lim)) for x,lim in [(ev,.1),(ed,.04),(eh,.05)]]
    result=dict(observed_ticks=int(valid.sum()),physical_failure=failure,final_hold_speed_steer_heading=hold,joint_final_hold=all(hold),peak_roll=float(np.max(np.abs(d['phi'][valid]))),working_roll_violation_s=float(np.sum(np.abs(d['phi'][valid])>.3)*.005))
    for name,start,stop in [('command',1,6),('stable_fast',2,4),('final',end-.5,end)]:
        mask=(t>=start-1e-6)&(t<stop-1e-6);mask&=valid
        result[name]={key:float(np.sqrt(np.mean(x[mask]**2))) if mask.any() else None for key,x in [('speed_rmse',ev),('steer_rmse',ed),('heading_rmse',eh)]}
    change=np.flatnonzero(np.any(np.abs(np.diff(d['limited_command'],axis=0))>1e-7,axis=1))
    tail_start=float(t[change[-1]+1]) if len(change) else 0.
    good=(abs(ev)<=.1)&(abs(ed)<=.04)&(abs(eh)<=.05)&(abs(d['phi'])<=.3)&~d['physical_failure']&(t>=tail_start)
    recovery=None
    for ids in helper('audit_saved_commands').contiguous_runs(good&valid):
        if len(ids)>=100:recovery=float(t[ids[99]]+.005-tail_start);break
    result['first_joint_safe_hold_delay_after_last_raw_slew_s']=recovery
    result['mean_abs_offsets']=np.mean(np.abs(d['governed'][valid]-d['limited_command'][valid]),axis=0).tolist()
    # First 100ms actual steering within tolerance, while the first turn target is fully issued.
    target=d['target'];plateau=(np.abs(d['limited_command'][:,1]-target[:,1])<1e-6)&(abs(target[:,1])>.01)&(t>=1)&(t<3 if np.any(t>=3) and np.any(target[:,1]<0) and np.any(target[:,1]>0) else t<4)
    ids0=np.flatnonzero(plateau)
    result['first_turn_tracking_delay_from_issued_plateau_s']=None
    if len(ids0):
        for ids in helper('audit_saved_commands').contiguous_runs(plateau&(abs(ed)<=.04)):
            if len(ids)>=20:result['first_turn_tracking_delay_from_issued_plateau_s']=float(t[ids[19]]+.005-t[ids0[0]]);break
    return result

def smooth(d,quiet):
    h=helper('audit_saved_commands');idx=np.arange(3,min(len(quiet),len(d['time'])),4);mask=quiet[idx];z=np.tanh(d['latent_z']);proposal=np.column_stack([z[:,0]*np.where(z[:,0]>=0,.25,1.),.2*z[:,1]])
    offset=d['governed']-d['limited_command'];out={}
    for name,x in [('proposal_speed',proposal[:,0]),('proposal_steer',proposal[:,1]),('offset_speed',offset[:,0]),('offset_steer',offset[:,1]),('actual_steer',d['actual_delta'])]:out[name]=h.stats(x[idx],.02,mask)
    return out

def quiet_mask(old):
    raw=old['limited_command'];rates=np.vstack([np.zeros(2),np.diff(raw,axis=0)/.005]);q=(abs(rates[:,0])<=.10001)&(abs(rates[:,1])<=.02001);mask=np.zeros(len(raw),bool)
    for ids in helper('audit_saved_commands').contiguous_runs(q):
        if len(ids)>80:mask[ids[80:]]=True
    return mask

def reward_audit(d,alpha):
    ref=helper('reward_reference');spec=resolve(alpha);caps=spec['reward']['independent_component_caps'];maxerr=0.;maxrew=0.;prior=np.zeros(2);rate=np.zeros(2);valid=False
    for start in range(0,len(d['time']),4):
        n=min(4,len(d['time'])-start);part=slice(start,start+n)
        if abs(float(d['time'][start]))<1e-6:prior=np.zeros(2);rate=np.zeros(2);valid=False
        total=0.
        for i in range(start,start+n):
            x=ref.state_cost(alpha=alpha,chi=float(d['chi'][i]),recovery_weight=float(d['g'][i]),speed=float(d['actual_forward_speed'][i]),steer=float(d['actual_delta'][i]),raw_speed=float(d['limited_command'][i,0]),raw_steer=float(d['limited_command'][i,1]),heading_debt=float(d['e_psi_unwrapped'][i]),yaw_rate=float(d['yaw_rate'][i]),roll=float(d['phi'][i]),roll_rate=float(d['phi_dot'][i]),offsets=d['offsets'][i],wheelbase=spec['reference']['wheelbase_m_expected'],caster_rad=np.deg2rad(spec['reference']['caster_deg_expected']))
            for k,v in x['raw'].items():maxerr=max(maxerr,abs(v-float(d['raw_cost_'+k][i])))
            total-=.1*.005*x['cost']
        motion=ref.upper_motion_cost(d['offsets'][start+n-1],prior,rate,acceleration_valid=valid)
        for k,v in motion['raw'].items():maxerr=max(maxerr,abs(v-float(d['raw_cost_'+k][start])))
        total-=.1*.02*motion['cost']
        if np.any(d['physical_failure'][part]):total=ref.failure_reward(800-round(float(d['time'][start])/.02))
        maxrew=max(maxrew,abs(total-float(d['scored_tick_reward'][part].sum())))
        prior=d['offsets'][start+n-1];rate=motion['rate'];valid=True
    return dict(max_component_abs_error=maxerr,max_policy_reward_abs_error=maxrew,passed=maxerr<.003 and maxrew<.003,component_caps={k:float(np.mean(d['raw_cost_'+k]>v)) for k,v in caps.items()})

def report(root,stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/f'evaluation{stage}';oldroot=OLD/'R196_comparison/review';metrics={};audits={};lines=[f'# R196 SmoothV4 — stage {stage}', '', '黑实线是原始发布指令 limited_command；灰虚线 target 是 slew 前目标。网络 proposal=limited_command+tanh映射修正；governed 是保护后发给底层的参考。actual 是真实运动。所有误差对 limited_command 评价。', '', 'B0复用同R196、同准备状态/初态/指令的历史轨迹；旧alpha0@250/alpha1@143。初始化以运行manifest为准；本次用户修订为Actor/Critic/Adam/std全新初始化，不加载旧模型。旧新reward不可直接排名；无独立泛化或实车结论。', '', '|场景|方法|速度RMSE[1,6)|转角RMSE[1,6)|末端航向RMSE|速度/转角/航向保持|峰值侧倾|', '|---|---|---:|---:|---:|---|---:|']
    colors={'B0':'.5','old0':'#88b9e0','new0':'#0066b3','old1':'#e5b381','new1':'#db6400'}
    for casepath in sorted(out.iterdir()):
        if not casepath.is_dir() or not (casepath/'alpha0.npz').exists():continue
        case=casepath.name;traces={k:dict(np.load(oldroot/case/(m+'.npz'))) for k,m in [('B0','B0'),('old0','pi_alpha0'),('old1','pi_alpha1')]}
        traces.update({f'new{a}':dict(np.load(casepath/f'alpha{a}.npz')) for a in [0,1]})
        quiet=quiet_mask(traces['B0']);metrics[case]={};audits[case]={}
        for m,d in traces.items():
            val=physical(d);val['smoothness']=smooth(d,quiet);metrics[case][m]=val
            if m.startswith('new'):
                audits[case][m]=reward_audit(d,int(m[-1]))
                if len(d['time'])>=3200:val['extension16']=physical(d,16.)
            fmt=lambda x:f'{x:.4f}' if x is not None else 'missing'
            lines.append(f"|{case}|{m}|{fmt(val['command']['speed_rmse'])}|{fmt(val['command']['steer_rmse'])}|{fmt(val['final']['heading_rmse'])}|{val['final_hold_speed_steer_heading']}|{val['peak_roll']:.4f}|")
        for a in [0,1]:
            old=metrics[case][f'old{a}'];new=metrics[case][f'new{a}'];key='steer_rmse' if a==0 else 'speed_rmse'
            ratio=lambda x,y:float(x/y) if y>1e-10 else None
            sm0=old['smoothness']['offset_steer'];sm1=new['smoothness']['offset_steer'];rev='direction_reversals_adjacent_rate_above_0p05'
            new['old_relative']=dict(primary_rmse_ratio=ratio(new['command'][key],old['command'][key]),offset_rate_ratio=ratio(sm1['rate_rms'],sm0['rate_rms']),offset_acceleration_ratio=ratio(sm1['acceleration_rms'],sm0['acceleration_rms']),offset_reversal_ratio=ratio(sm1[rev],sm0[rev]))
        fig,axes=plt.subplots(6,1,figsize=(14,17),sharex=True)
        b=traces['B0']
        for j in range(2):
            axes[j].plot(b['time'],b['target'][:,j],color='.6',ls='--',label='target before slew')
            axes[j].plot(b['time'],b['limited_command'][:,j],'k',lw=2,label='original issued command')
            axes[j+2].plot(b['time'],b['limited_command'][:,j],'k',lw=2,label='original issued command')
        for m,d in traces.items():
            c=colors[m];ls='--' if m.startswith('old') else '-';t=d['time'];lab=m+(f" update{int(d['checkpoint_update'])}" if m.startswith('new') else '')
            if m!='B0':
                for j in range(2):axes[j].plot(t,d['governed'][:,j],color=c,ls=ls,label=lab+' governed')
            for ax,key in zip(axes[2:],['actual_forward_speed','actual_delta','phi','e_psi_unwrapped']):ax.plot(t,d[key],color=c,ls=ls,label=lab)
            if np.any(d['physical_failure']):
                for ax in axes:ax.axvline(t[-1],color=c,ls=':')
        axes[4].axhline(.3,color='r',ls=':');axes[4].axhline(-.3,color='r',ls=':');axes[5].axhline(.05,color='r',ls=':');axes[5].axhline(-.05,color='r',ls=':')
        for ax,label in zip(axes,['Speed command (m/s)','Steer command (rad)','Actual speed (m/s)','Actual wheel angle (rad)','Roll (rad)','Heading debt (rad)']):
            ax.set_ylabel(label);ax.grid(alpha=.25);ax.axvspan(9.5,10,color='green',alpha=.08);ax.legend(ncol=3,fontsize=8)
        axes[-1].set_xlabel('Time (s)');fig.suptitle(f'{case} | SmoothV4 stage{stage} | lower R196 alpha1 | seed77001');fig.tight_layout();fig.savefig(out/f'{case}_motion.png',dpi=125);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
        for m,d in traces.items():
            if m=='B0':continue
            z=np.tanh(d['latent_z']);proposal=np.column_stack([z[:,0]*np.where(z[:,0]>=0,.25,1),.2*z[:,1]])
            for a,ax in enumerate(axes):
                ax.plot(d['time'],proposal[:,a],color=colors[m],ls=':',alpha=.45,label=m+' proposal offset')
                ax.plot(d['time'],d['offsets'][:,a],color=colors[m],ls='--' if m.startswith('old') else '-',label=m+' executed offset')
        axes[0].set_ylabel('Speed correction (m/s)');axes[1].set_ylabel('Steer correction (rad)');axes[1].set_xlabel('Time (s)')
        for ax in axes:ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.25)
        fig.suptitle(f'{case} | before / after guards | quiet-window metrics in metrics.json');fig.tight_layout();fig.savefig(out/f'{case}_offsets.png',dpi=125);plt.close(fig)
        fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
        for m,d in traces.items():
            for j,ax in enumerate(axes):ax.plot(d['time'],d['applied_residual'][:,j],color=colors[m],ls='--' if m.startswith('old') else '-',label=m)
        for j,ax in enumerate(axes):
            bound=[1.5,10][j];ax.axhline(bound,color='r',ls=':');ax.axhline(-bound,color='r',ls=':');ax.set_ylabel(['Front residual (rad/s)','Rear residual (rad/s)'][j]);ax.legend(ncol=5);ax.grid(alpha=.25)
        axes[1].set_xlabel('Time (s)');fig.suptitle(f'{case} | actual bounded residual at frozen lower interface');fig.tight_layout();fig.savefig(out/f'{case}_residual.png',dpi=125);plt.close(fig)
        lines+=['',f'![{case} physical comparison]({case}_motion.png)',f'[保护前后修正]({case}_offsets.png) · [实际受限残差]({case}_residual.png)','']
    write(out/'metrics.json',clean_numbers(metrics));write(out/'reward_audit.json',clean_numbers(audits));(out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    if not all(v['passed'] for d in audits.values() for v in d.values()):raise AssertionError('independent SmoothV4 reward reconstruction failed')
    select_physical_checkpoint(root)
    return metrics


def select_physical_checkpoint(root):
    """Full six-case candidates only; no selection by training reward."""
    candidates={}
    for stage in [60,150]:
        path=root/f'evaluation{stage}/metrics.json'
        if path.exists():
            data=json.loads(path.read_text())
            if len(data)==6:candidates[stage]=data
    if not candidates:
        # Stage20 only covers three diagnostic cases; preserve checkpoint identity
        # without presenting it as a full six-case best.
        return
    for a in [0,1]:
        ranked=[]
        for stage,data in candidates.items():
            rows=[v[f'new{a}'] for v in data.values()]
            failed=sum(r['physical_failure'] for r in rows);unsafe=sum(r['working_roll_violation_s']>0 for r in rows);hold=sum(r['joint_final_hold'] for r in rows)
            primary=np.mean([r['command']['steer_rmse' if a==0 else 'speed_rmse'] for r in rows]);secondary=np.mean([r['command']['speed_rmse' if a==0 else 'steer_rmse'] for r in rows]);rough=np.mean([r['smoothness']['offset_steer']['acceleration_rms'] for r in rows])
            ranked.append((failed,unsafe,-hold,primary,secondary,rough,stage))
        rank=min(ranked);stage=rank[-1];trace=np.load(root/f'evaluation{stage}/straight_hold/alpha{a}.npz');update=int(trace['checkpoint_update'])
        link=root/f'alpha{a}/bestmodel.pt';temp=link.with_suffix('.tmp-link')
        if temp.is_symlink():temp.unlink()
        temp.symlink_to(f'checkpoints/update_{update:04d}.pt');temp.replace(link)
        write(root/f'alpha{a}/best_model.json',dict(bestmodel=str(link.resolve()),selection='fixed_six_physical_lexicographic_failure_working_range_hold_primary_secondary_roughness',checkpoint=str((root/f'alpha{a}/checkpoints/update_{update:04d}.pt').resolve()),update=update,evaluation_stage=stage,candidate_stages=list(candidates),ranking=rank,qualified=rank[0]==0 and rank[1]==0 and rank[2]==-6,stage20='three-case diagnostic; not a complete six-case candidate',reward_used=False))

    lines=['# Selected bestmodels — fixed six-case evidence','',
           '按完整六场景候选的物理指标选择，各alpha独立；训练reward不参与。所选模型的同协议实测已保存，以下直接复用其评测轨迹，不将last冒充best。第20轮仅三场景诊断，未完成六场景，单独保留。','']
    for a in [0,1]:
        d=json.loads((root/f'alpha{a}/best_model.json').read_text());stage=d['evaluation_stage']
        lines += [f"- alpha{a}: update{d['update']}, qualified={d['qualified']}; [bestmodel](alpha{a}/bestmodel.pt), [selection](alpha{a}/best_model.json), [同模型实测](evaluation{stage}/RESULTS_ZH.md)."]
    (root/'BESTMODELS.md').write_text('\n'.join(lines)+'\n')
