"""One fixed four-case reference-prefilter comparator; no learning/force."""
import argparse,json,os,pickle,subprocess,sys,time
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from prepare_predictive_confirmation import register


def render(root,plan,traces,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    prior=Path(plan['paired_nominal_root']);old=json.loads((prior/'analysis/analysis.json').read_text());out=root/'figures';out.mkdir()
    def read(file):
        with np.load(file) as z:return {k:z[k] for k in z.files}
    lines=['# 固定参考低通：四条完整名义对照','','同一参考接口的200Hz常规预处理；H75推理50Hz，非同频决策消融。0训练/外力、τ固定原.08s；实际新成本见status.json。','',
           '|路线/速度|终点保持|实际时长s|逆映射/slew/幅限clip ticks|低通递推最大误差rad|','|---|---|---:|---|---:|']
    for row in rows:
        route,speed=row['route'],row['speed'];base=traces[row['name']]
        members=[('reference_filter',base,'#CC79A7'),('zero_middle',read(prior/'zero/evaluation'/(row['name']+'.npz')),'black')]
        for alpha,color in zip([0.,.25,.5,.75,1.],['#0072B2','#009E73','#56B4E9','#E69F00','#D55E00']):
            record=next(x for x in old['comparisons'] if x['route']==route and x['speed']==speed and x['alpha']==alpha)
            members.append((f'H75_a{alpha:g}',read(prior/'H75'/(record['H75_labels']['name']+'.npz')),color))
        path=read(root/'evaluation'/(row['name']+'_path.npz'));fig,axes=plt.subplots(4,1,figsize=(10,11),layout='constrained')
        for label,trace,color in members:
            axes[0].plot(*trace['actual_xy'].T,color=color,label=label,ls='--' if label=='zero_middle' else '-')
            for ax,key in zip(axes[1:],['actual_forward_speed','path_cross_track','phi']):ax.plot(trace['time'],trace[key],color=color,label=label)
        axes[0].plot(*path['xy'][path['s']<=speed*10+1].T,'k:',label='original route');axes[0].set_aspect('equal');axes[1].axhline(speed,color='gray',ls=':')
        for ax,ylabel in zip(axes,['world Y [m]','speed [m/s]','signed path [m]','roll [rad]']):ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=6)
        tag=f'{route}_v{speed:g}';fig.suptitle(tag+' | fixed reference prefilter versus factual policies');fig.savefig(out/(tag+'_overview.png'),dpi=100);plt.close(fig)
        lines.append(f"|{route}/{speed}|{row['labels']['task_recovered']}|{len(base['time'])*.005:g}|{row['inverse_clip_ticks']}/{row['slew_clip_ticks']}/{row['reference_clip_ticks']}|{row['lowpass_max_error_rad']:g}|")
        filter_curve=read(root/'utility/curves'/(tag+'_reference_filter.npz'))
        for i,alpha in enumerate([0.,.25,.5,.75,1.]):
            label=f'H75_a{alpha:g}';target=next(x for x in members if x[0]==label);hc=read(root/'utility/curves'/(tag+'_'+label+'.npz'))
            fig,axes=plt.subplots(9,1,figsize=(11,17),layout='constrained')
            for name,trace,color,curve in [('reference_filter',base,'#CC79A7',filter_curve),(*target,hc)]:
                axes[0].plot(*trace['actual_xy'].T,color=color,label=name);axes[0].scatter(*trace['actual_xy'][-1],color=color,marker='x' if trace['physical_failure'].any() else 'o')
                for ax,key in zip(axes[1:6],['actual_forward_speed','path_cross_track','path_heading_error','phi','actual_delta']):ax.plot(trace['time'],trace[key],color=color,label=name)
                ix=np.arange(0,len(trace['time']),4);axes[6].plot(trace['time'][ix],np.add.reduceat(curve['reward'][i],ix),color=color,label=name)
                names=curve['component_names'].tolist();signed=np.add.reduceat(curve['signed_components'][i],ix,axis=0)
                for ax,key,style in [(axes[7],'speed','-'),(axes[7],'path','--'),(axes[7],'primary',':'),(axes[8],'roll','-'),(axes[8],'working_roll','--')]:ax.plot(trace['time'][ix],signed[:,names.index(key)],color=color,ls=style,label=name+' '+key)
            axes[0].set_aspect('equal');axes[1].axhline(speed,color='gray',ls=':')
            for ax,ylabel in zip(axes,['world Y [m]','speed [m/s]','signed path [m]','heading [rad]','roll [rad]','steer [rad]','reward /50Hz step','signed task components','signed roll components']):ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=6)
            fig.suptitle(tag+f' | user alpha={alpha:g}, same-alpha scoring');fig.savefig(out/(tag+f'_a{alpha:g}.png'),dpi=85);plt.close(fig)
    lines+=['','[全部140评分与固定行为对照](utility/REPORT.md)。figures含四组全策略概览与二十组同用户α实际轨迹/逐50Hz奖励/有符号分量；提前终止显示真实终点，不补齐曲线。','若有截断，低通误差仅为实际受限行为诊断，不宣称严格低通等价。无论胜负均不扫τ/频率，不推广到所有常规控制。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);prior=Path('runs/nominal_preference_transfer_v1').resolve()
    if json.loads((prior/'status.json').read_text())['phase']!='completed':raise ValueError('Prior nominal study incomplete')
    old=prior/'zero/plan.json';plan=json.loads(old.read_text());check=Path('runs/reference_filter_check_v2/check.json').resolve()
    verified=json.loads(check.read_text())
    if verified['independent_recurrence_error_rad']>2e-7 or max(x['actual_vs_ideal200Hz_max_rad'] for x in verified['cases'])>2e-7:raise ValueError('Reference mathematics unverified')
    if digest(plan['constant_actor'])!=plan['constant_actor_sha256']:raise ValueError('Constant carrier changed')
    scripts=Path(__file__).resolve().parent
    files=[Path(__file__).resolve(),scripts/'simple_reference_filter.py',scripts/'REFERENCE_FILTER_DESIGN.md',scripts/'audit_preference_utility.py',scripts/'check_reference_filter.py',old,check,Path(plan['constant_actor']),Path(plan['source_declaration'])]
    files+=[Path('learning/src/sttw_control')/(n+'.py') for n in ['path_command_evaluation','path_command_policy','path_command_training','path_command_env','path_command_reward','path_reference','controller','closed_loop_kernel','teleop_env','contact_only_environment']]
    plan.update(purpose='Fixed200Hz nominal reference prefilter; conventional comparator, not learned policy',maximum_lower_ticks=8000,training_steps=0,paired_nominal_root=str(prior),math_check=str(check),source_sha256={str(f.resolve()):digest(f) for f in files})
    atomic_json(root/'plan.json',plan);atomic_json(root/'status.json',dict(phase='prepared',completed_cases=0,declared_cases=4,active_lower_ticks=0,training_steps=0))
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 固定参考平滑四例与140效用分析',root/'status.json');print('prepared4 cases;maximum8000 lower ticks;0 learning/force')


def execute(root):
    import jax,jax.numpy as j
    from simple_reference_filter import load_environment
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.recovery_margin_scan import outcome_labels
    from sttw_control.runtime import configure_compilation_cache
    plan=json.loads((root/'plan.json').read_text());cfg=plan['config'];s=dict(phase='running',state='running',pid=os.getpid(),stage='loading',completed_cases=0,declared_cases=4,active_lower_ticks=0,training_steps=0)
    def update(**kw):s.update(kw,checked_unix=time.time());atomic_json(root/'status.json',s)
    def account(ticks):
        s['active_lower_ticks']+=ticks
        if s['active_lower_ticks']>8000:raise ValueError('Fixed physical budget exceeded')
        update()
    update()
    try:
        for f,sha in plan['source_sha256'].items():
            if digest(f)!=sha:raise ValueError('Declared source changed '+f)
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,initial=load_environment(cfg,plan['source'],'terminal_contract_v2')
        with Path(plan['constant_actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
        atomic_json(root/'manifest.json',dict(plan=plan,physical_identity=env.physics.bundle.identity,semantics='Fixed200Hz nominal-reference preprocessor through original correction interface; zero Actor is carrier, not executed controller'))
        traces=BatchedPanelEvaluator(env,initial,cfg,status=update,account=account).evaluate(params,0.,root/'evaluation');rows=[]
        decay=np.exp(-.005/cfg['residual_interface']['filter_tau_s'][1])
        for case in plan['cases']:
            trace=traces[case['name']]
            with np.load(root/'evaluation'/(case['name']+'_path.npz')) as p:goal=float(p['goal'])
            labels=outcome_labels(trace,speed=case['actual_requested_speed_m_s'],goal_s=goal,working_roll=cfg['stage_B_baseline']['working_roll_rad'],horizon=10.,hold_s=cfg['stage_B_baseline']['hold_s'])
            np.testing.assert_allclose(trace['governed'][:,0],trace['nominal'][:,0],atol=1e-6,rtol=0)
            inverse=int(trace['baseline_inverse_map_clipped'].sum());slew=int(trace['correction_rate_clipped'].sum());clip=int(trace['reference_clip_channels'].any(axis=1).sum())
            y=trace['nominal'][0,1]-.005*trace['nominal_rates'][0,1];expected=[]
            for x in trace['nominal'][:,1]:y=decay*y+(1-decay)*x;expected.append(y)
            error=float(np.max(np.abs(trace['governed'][:,1]-expected)))
            if not (inverse or slew or clip) and error>2e-6:raise ValueError('Actual unbounded reference stream differs from lowpass')
            rows.append(dict(**case,route=plan['route_name_map'][case['builder']],speed=case['actual_requested_speed_m_s'],labels=labels,inverse_clip_ticks=inverse,slew_clip_ticks=slew,reference_clip_ticks=clip,lowpass_max_error_rad=error))
        atomic_json(root/'results.json',dict(cases=rows,training_steps=0,new_lower_ticks=s['active_lower_ticks']))
        update(stage='utility_analysis',completed_cases=4)
        scripts=Path(__file__).resolve().parent;prior=Path(plan['paired_nominal_root'])
        with (root/'utility.log').open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',str(scripts/'audit_preference_utility.py'),'--source-analysis',str(prior/'analysis/analysis.json'),'--zero-run',str(prior/'zero'),'--extra-reference-run',str(root),'--save-curves','--output',str(root/'utility')],stdout=log,stderr=subprocess.STDOUT)
            update(child_pid=child.pid);code=child.wait()
        if code:raise RuntimeError('Utility analysis failed; preserve original log')
        update(stage='rendering_all24_figures')
        render(root,plan,traces,rows)
        update(phase='completed',state='completed',stage='four_real_traces_140_scores_and24_figures_ready',report=str(root/'REPORT.md'))
    except BaseException as exc:update(phase='error',state='error',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();(execute if a.execute else prepare)(a.run.resolve())
