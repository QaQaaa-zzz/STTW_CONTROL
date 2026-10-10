"""Fixed-path final comparison: actual geometry first, all commands unfiltered."""
from pathlib import Path
import json
import numpy as np

def reward_series(d,alpha,config,horizon=None):
    """Same-alpha reweighting of fixed physical traces; terminal tail retained."""
    c=config['path_reward'];dt=config['timing']['lower_dt_s'];chi=np.asarray(d['chi'],float)
    ev=np.asarray(d['actual_forward_speed'],float)-d['v_user'];ey=np.asarray(d['path_cross_track'],float);eh=np.asarray(d['path_heading_error'],float)
    huber=lambda x:np.where(abs(x)<=1,x*x,2*abs(x)-1)
    effective={k:np.asarray(d['effective_cost_'+k],float).copy() for k in c['component_caps']}
    raw=dict(speed=(10-9*chi*(1-alpha))*huber(ev/c['speed_scale_m_s']),
             path=(10-9*chi*alpha)*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad'])),
             primary=c['primary_weight']*chi*((1-alpha)*huber(np.maximum(abs(ey)-c['primary_path_band_m'],0)/c['primary_path_scale_m'])+alpha*huber(np.maximum(abs(ev)-c['primary_speed_band_m_s'],0)/c['primary_speed_scale_m_s'])))
    effective.update({k:np.minimum(v,c['component_caps'][k]) for k,v in raw.items()})
    components={k:-c['scale']*dt*v for k,v in effective.items()};components['failure_tail']=np.zeros(len(ev))
    if np.any(d['physical_failure']):
        last=len(ev)-1;interval_start=(last//4)*4
        for v in components.values():v[interval_start:]=0
        h=config['training_future_phase_C']['episode_s'] if horizon is None else horizon
        remaining=round(h/config['timing']['upper_dt_s'])-last//4;gamma=config['training_future_phase_C']['gamma']
        components['failure_tail'][-1]=-c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining)/(1-gamma)
    return sum(components.values()),components

def report_final_panel(root,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path_command_selection import cases
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True);lines=['# Geometric upper fixed-path comparison','', 'Same immutable path and prepared state; nominal PP reference may differ by actual pose. No post-hoc smoothing. Best is fixed-DEV ranking, not independent validation or proof of alpha preference.','']
    for name,route,speed in cases(config):
        data={'zero_upper':dict(np.load(root/'zero_upper'/f'{name}.npz'))}
        for alpha in (0,1):data[f'alpha{alpha}']=dict(np.load(root/f'alpha{alpha}'/'final_best'/f'{name}.npz'))
        path=dict(np.load(root/'zero_upper'/f'{name}_path.npz'));fig,axes=plt.subplots(5,2,figsize=(15,17));ax=axes[0,0]
        maxs=max(float(d['path_progress'].max()) for d in data.values());mask=path['s']<=maxs+2
        ax.plot(*path['xy'][mask].T,'k--',label='fixed original path')
        for method,d in data.items():
            t=d['time']+.005;ax.plot(*d['actual_xy'].T,label=method);ax.plot(*d['actual_xy'][-1],marker='x' if d['physical_failure'].any() else 'o')
            axes[0,1].plot(t,d['path_heading_error'],label=method+' heading rad')
            axes[1,0].plot(t,d['actual_forward_speed']-speed,label=method+' speed error m/s');axes[1,1].plot(t,d['path_cross_track'],label=method+' path error m')
            for ch in (0,1):
                for key,style in [('nominal','--'),('governed',':')]:axes[2,ch].plot(t,d[key][:,ch],style,label=method+' '+key,lw=.8)
                axes[2,ch].plot(t,d['actual_forward_speed'] if ch==0 else d['actual_delta'],label=method+' actual',lw=.8)
                axes[3,ch].plot(t,d['target_offset'][:,ch],'--',label=method+' target',lw=.5);axes[3,ch].plot(t,d['filtered_offset'][:,ch],':',label=method+' filtered',lw=.5);axes[3,ch].plot(t,d['offsets'][:,ch],label=method+' correction');axes[3,ch].plot(t,d['lower_error'][:,ch],':',label=method+' lower error')
        for alpha in (0,1):
            for method,style in [(f'alpha{alpha}','-'),('zero_upper','--')]:
                d=data[method];reward,components=reward_series(d,alpha,config)
                if method!='zero_upper':np.testing.assert_allclose(reward,d['scored_tick_reward'],rtol=2e-4,atol=2e-6)
                step=np.arange(1,len(reward)+1);label=method+f' scored alpha{alpha}'
                axes[4,0].plot(step,reward,style,label=label);axes[4,1].plot(step,np.cumsum(reward),style,label=label)
                np.savez_compressed(out/f'{name}_{method}_scored_alpha{alpha}.npz',reward=reward,**components)
        cf,ca=plt.subplots(2,1,figsize=(12,8))
        for alpha in (0,1):
            for method,style in [(f'alpha{alpha}','-'),('zero_upper','--')]:
                _,components=reward_series(data[method],alpha,config)
                for component,values in components.items():
                    ca[0].plot(values,style,label=f'{method} a{alpha} {component}',lw=.6)
                    ca[1].plot(np.cumsum(values),style,lw=.6)
        ca[0].set_ylabel('Signed per-step reward components');ca[1].set_ylabel('Cumulative components');ca[1].set_xlabel('Actual lower steps');ca[0].legend(fontsize=5,ncol=4)
        cf.tight_layout();cf.savefig(out/f'{name}_reward_components.png',dpi=130);plt.close(cf)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
        for row in range(5):
            for col in range(2):
                a=axes[row,col];a.legend(fontsize=6);a.grid(alpha=.2)
                if (row,col)!=(0,0):a.set_xlabel('Time (s)' if row<4 else 'Actual lower steps')
        axes[2,0].set_ylabel('m/s');axes[2,1].set_ylabel('rad');axes[3,0].set_ylabel('m/s');axes[3,1].set_ylabel('rad')
        fig.suptitle(name+' | local300 | endpoint fixed-path best | no external disturbance\nReward weights differ by alpha: do not rank physical quality by cross-alpha return');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/f'{name}.png',dpi=130);plt.close(fig)
        # A separate equal-aspect XY artifact is always available first.
        f,a=plt.subplots(figsize=(8,6));a.plot(*path['xy'][mask].T,'k--',label='fixed path')
        for method,d in data.items():a.plot(*d['actual_xy'].T,label=method);a.plot(*d['actual_xy'][-1],marker='x' if d['physical_failure'].any() else 'o')
        a.set_aspect('equal',adjustable='datalim');a.set_xlabel('World X (m)');a.set_ylabel('World Y (m)');a.legend();a.set_title(name);f.tight_layout();f.savefig(out/f'{name}_XY.png',dpi=130);plt.close(f)
        lines.extend([f'## {name}',f'![XY]({name}_XY.png)',f'[Full commands/errors/reward]({name}.png)',f'[Signed and cumulative components]({name}_reward_components.png)',''])
    identities={f'alpha{alpha}':json.loads((root/f'alpha{alpha}'/'best_model.json').read_text()) for alpha in (0,1)}
    (out/'identities.json').write_text(json.dumps(identities,indent=2));lines.append('[Exact checkpoint identities and fixed-validation metrics](identities.json)');(out/'INDEX.md').write_text('\n'.join(lines)+'\n')
