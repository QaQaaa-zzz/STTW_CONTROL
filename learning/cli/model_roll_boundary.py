"""Parameter-derived nominal braking bound, fixed-state model tests, and phase figures."""
import argparse,csv,json,itertools
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_roll_contacts import nominal_coeff
from analyze_roll_braking_boundary import margin


def save_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fixed_tests(config,dt):
    result=[]
    for v in [1.8,2.3,2.6]:
        states=np.array(list(itertools.product(np.radians([-30,-15,0,15,30]),np.radians([-400,-200,-100,-50,50,100,200,400]),[-.4,0,.4])))
        initial=states.copy();side=np.sign(states[:,1]);alive=np.ones(len(states),bool);elapsed=np.zeros(len(states));outcome=np.full(len(states),'timeout',dtype=object)
        a1,a2,a4=nominal_coeff(np.array(v),config);A=np.array([[0,1,0,0],[a4,0,a2,a1],[0,0,0,1],[0,0,0,0]],float);E=expm(A*dt)
        for k in range(round(1/dt)):
            ids=np.flatnonzero(alive)
            if not len(ids):break
            old=states[ids].copy();command=3*side[ids];command=np.clip(command,(-.8-old[:,2])/dt,(.8-old[:,2])/dt)
            new=np.column_stack([old,command])@E.T;states[ids]=new[:,:3];elapsed[ids]=(k+1)*dt
            stopped=states[ids,1]*side[ids]<=0;crossed=abs(states[ids,0])>.7
            outcome[ids[crossed]]='angle_limit';outcome[ids[stopped&~crossed]]='stopped';alive[ids[crossed|stopped]]=False
        for i,x in enumerate(initial):
            travel=side[i]*(states[i,0]-x[0]);beq=np.degrees(x[1]**2/(2*travel)) if outcome[i]=='stopped' and travel>0 else None
            result.append(dict(speed=v,phi0_deg=float(np.degrees(x[0])),omega0_deg_s=float(np.degrees(x[1])),delta0_rad=float(x[2]),outcome=outcome[i],end_s=elapsed[i],phi_end_deg=float(np.degrees(states[i,0])),B_stop_equiv_deg_s2=beq))
    return result


def main(run,out,config_path):
    out.mkdir(exist_ok=True,parents=True);config=json.loads(config_path.read_text());speeds=np.linspace(.5,3,501);a1,a2,a4=nominal_coeff(speeds,config);B_rad=float(np.max(abs(a1)*3+abs(a2)*.8+a4*.7));B=np.degrees(B_rad);limit=np.degrees(.7)
    protocol=dict(config=config,config_path=str(config_path),velocity_domain_m_s=[.5,3],steer_angle_limit_rad=.8,ideal_actual_steer_rate_limit_rad_s=3,roll_limit_rad=.7,B_model_deg_s2=float(B),formula='max_v(|a1(v)|*3 + |a2(v)|*0.8 + a4*0.7)',assumptions='Nominal roll-steer model, no external/contact residual, ideal rate tracking; fixed-test v constant; not PhysX full-vehicle tests',test_matrix='3 speeds x5 roll angles x8 roll rates x3 steer angles =360; max directional steering until angle limit;1s horizon;dt1ms and0.5ms',selection='For every context with a taskC full-alpha positive, choose first in [.5,.75,1,0,.25]; compare full1 unless selected1 then full.5; report all five full-alpha results')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));test=fixed_tests(config,.001);fine=fixed_tests(config,.0005);save_csv(out/'fixed_model_tests.csv',test);save_csv(out/'fixed_model_tests_half_dt.csv',fine)
    check=dict(cases=len(test),outcome_disagreement=sum(a['outcome']!=b['outcome'] for a,b in zip(test,fine)),stopped=sum(r['outcome']=='stopped' for r in test),angle_limit=sum(r['outcome']=='angle_limit' for r in test),timeout=sum(r['outcome']=='timeout' for r in test));(out/'model_test_check.json').write_text(json.dumps(check,indent=2))
    parameters=[]
    for v in [1.8,2.3,2.6,3]:
        aa,bb,gg=nominal_coeff(np.array(v),config);parameters.append(dict(speed=v,a1=float(aa),a2=float(bb),a4=float(gg),B_deg_s2=float(np.degrees(abs(aa)*3+abs(bb)*.8+gg*.7))))
    save_csv(out/'model_parameters.csv',parameters)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK JP'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    src=run/'roll_contact_analysis';rows=json.loads((run/'analysis_summary.json').read_text())['rows'];labels=['full_0','full_0.25','full_0.5','full_0.75','full_1'];allrows=[];chosen=[];curves={}
    pp=np.linspace(-limit,limit,500);upper=np.sqrt(2*B*(limit-pp));lower=-np.sqrt(2*B*(limit+pp))
    for path in sorted(src.glob('*_seed*.npz')):
        z=np.load(path);case=path.stem;info={r['label']:r for r in rows if r['case']==case}
        for lab in labels:
            lane=list(z['labels']).index(lab);valid=z['valid'][:,lane];p=np.degrees(z['phi'][valid,lane]);w=np.degrees(z['euler_roll_rate'][valid,lane]);t=z['time'][valid,lane];H=margin(p,w,limit,B)[2];pre=np.flatnonzero((H<0)&(abs(p)<limit));r=info[lab]
            allrows.append(dict(case=case,label=lab,task_C=r['tasks']['C'],final_hold=r['final_hold'],cause=r['cause'],H_min_deg=float(H.min()),first_H_negative_inside_s=float(t[pre[0]]) if len(pre) else None,actual_rate_over3_fraction=float(np.mean(abs(z['delta_dot'][valid,lane])>3.0001)),speed_outside_domain_fraction=float(np.mean((z['speed'][valid,lane]<.5)|(z['speed'][valid,lane]>3)))))
        candidates=[lab for lab in ['full_0.5','full_0.75','full_1','full_0','full_0.25'] if info[lab]['tasks']['C']]
        if not candidates:continue
        positive=candidates[0];compare='full_1' if positive!='full_1' else 'full_0.5';chosen.append(dict(case=case,positive=positive,comparison=compare))
        fig,(ax,bx)=plt.subplots(1,2,figsize=(13.8,6.4));fig.subplots_adjust(left=.075,right=.98,bottom=.22,top=.77,wspace=.22)
        terrain='10 cm单脊' if case.startswith('single') else '10 mm碎石';speed=case.split('_v')[1].split('_')[0];seed=case.rsplit('seed',1)[1]
        fig.suptitle(f'{terrain} · {speed} m/s · 物理组{seed}',fontsize=20,y=.97)
        fig.text(.5,.89,'双侧制动余量：相轨迹与时间演化',ha='center',fontsize=14)
        fig.text(.5,.835,f'模型参数预先确定：φc = {limit:.2f}°，B = {B:.0f}°/s²；两条轨迹使用同一条线',ha='center',color='#4B5563',fontsize=11)
        ymax=max(1050,float(max(abs(np.degrees(z['euler_roll_rate'][z['valid'][:,list(z['labels']).index(lab)],list(z['labels']).index(lab)])).max() for lab in [positive,compare]))*1.15)
        ax.set_facecolor('#FBEDEE');ax.fill_between(pp,lower,upper,color='#EAF4F6');ax.plot(pp,upper,color='#343A40',lw=1.8);ax.plot(pp,lower,color='#343A40',lw=1.8,label='名义模型 H=0')
        for lab,color,style in [(positive,'#008B72','-'),(compare,'#D6453D','--')]:
            lane=list(z['labels']).index(lab);valid=z['valid'][:,lane];t=z['time'][valid,lane];p=np.degrees(z['phi'][valid,lane]);w=np.degrees(z['euler_roll_rate'][valid,lane]);H=margin(p,w,limit,B)[2];win=(t>=1.15)&(t<=3.2);r=info[lab];outcome='任务C通过' if r['tasks']['C'] else {'roll_limit':'侧倾超限','domain_exit':'运行域越界','full_window':'满窗，C未通过'}.get(r['cause'],r['cause']);legend=f"α={lab.removeprefix('full_')}：{outcome}"
            ax.plot(p[win],w[win],color=color,ls=style,lw=2,label=legend);bx.plot(t[win],H[win],color=color,ls=style,lw=2,label=legend)
            ids=np.flatnonzero(win);ax.scatter(p[ids[0]],w[ids[0]],s=27,color=color,zorder=5)
            for k in [int(len(ids)*.35),int(len(ids)*.7)]:
                i,j=ids[max(0,k-2)],ids[min(k+2,len(ids)-1)];ax.annotate('',xy=(p[j],w[j]),xytext=(p[i],w[i]),arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
            if t[-1]<=3.2:
                ax.scatter(p[-1],w[-1],marker='X',s=75,color=color,zorder=6);bx.scatter(t[-1],H[-1],marker='X',s=75,color=color,zorder=6);bx.annotate(f'终止 {t[-1]:.3f}s',xy=(t[-1],H[-1]),xytext=(12,15),textcoords='offset points',fontsize=10,color=color)
            neg=ids[H[ids]<0]
            if len(neg):
                j=neg[0];bx.scatter(t[j],H[j],marker='s',s=30,color=color);bx.annotate(f'首次 H<0：{t[j]:.3f}s',xy=(t[j],H[j]),xytext=(22,-38),textcoords='offset points',arrowprops=dict(arrowstyle='->',color=color),fontsize=9,color=color)
            for key,val in [('time',t),('phi_deg',p),('omega_deg_s',w),('H_deg',H),('xy',z['xy'][valid,lane])]:curves[case+'__'+lab+'__'+key]=val
        ax.set(xlim=(-45,45),ylim=(-ymax,ymax),xlabel='有符号侧倾角 φ（°）',ylabel='欧拉侧倾角速度 ω（°/s）',title='真实相轨迹与模型边界');bx.set(xlabel='时间（s）',ylabel='双侧余量 H（°）',title='相同参数下的 H(t)');bx.axhline(0,color='#343A40',lw=1);ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6)
        for a in [ax,bx]:a.grid(alpha=.16);a.legend(fontsize=9,loc='best')
        fig.text(.075,.11,'H± = φc ∓ φ − max(±ω, 0)²/(2B)；H = min(H+, H−)',fontsize=12)
        fig.text(.075,.065,'浅蓝：H≥0；浅红：H<0。黑线适用于无接触扰动、理想转向的名义模型；真实轨迹仍含轮地冲击。',fontsize=10,color='#4B5563')
        fig.text(.075,.025,'圆点：显示起点；箭头：时间方向；叉号：实际终止。成功轨迹显示至3.2s，完整任务结果按10s判定。',fontsize=10,color='#4B5563')
        fig.savefig(out/(case+'.png'),dpi=180);plt.close(fig)
    save_csv(out/'all_full_alpha.csv',allrows);(out/'selected.json').write_text(json.dumps(chosen,indent=2));np.savez_compressed(out/'figure_data.npz',**curves)
    # An XY overview precedes the phase figures, with original fixed route.
    route=np.load('runs/h75_selected_local300_flat/bundle/route.npz')['xy'];n=len(chosen);fig,axes=plt.subplots((n+2)//3,3,figsize=(13,3.1*((n+2)//3)),layout='constrained')
    for ax,item in zip(axes.flat,chosen):
        coords=[]
        for lab,color in [(item['positive'],'#008B72'),(item['comparison'],'#D6453D')]:
            xy=curves[item['case']+'__'+lab+'__xy'];coords.append(xy);ax.plot(xy[:,0],xy[:,1],color=color,lw=1.5,label=lab)
        points=np.concatenate(coords);lo=points.min(0)-.5;hi=points.max(0)+.5;ax.plot(route[:,0],route[:,1],color='gray',ls=':',label='原参考');ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='X（m）',ylabel='Y（m）',title=item['case']);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    fig.savefig(out/'XY_OVERVIEW.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(B=B,tests=check,positive_panels=n,all_full_alpha=len(allrows)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('runs/isaac_signals_parity/dynamic.json'));a=p.parse_args();main(a.run,a.output,a.config)
