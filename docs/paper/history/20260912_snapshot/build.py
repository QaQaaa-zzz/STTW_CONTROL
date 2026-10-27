"""Build publication-style figures and offline HTML/PDF from the audited draft.
Run at repository root using the project Python; markdown/weasyprint are export-only dependencies.
No training, physics stepping or mutation of original runs.
"""
from pathlib import Path
import json,hashlib,re,io,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white','mathtext.fontset':'dejavusans'})
BLUE='#35658A';TEAL='#278C82';ORANGE='#D78C39';RED='#B45659';GRAY='#647180';INK='#243548'

def save(fig,name):
    for ext in ('svg','pdf','png'):
        path = FIG / f'{name}.{ext}'
        fig.savefig(path, dpi=240, bbox_inches='tight', pad_inches=.13)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def control_diagram():
    fig,ax=plt.subplots(figsize=(16.5,8.3));ax.set(xlim=(0,17),ylim=(-.7,8.4));ax.axis('off')
    def box(x,y,w,h,text,color,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.07',ec=color,fc=fill,lw=1.4))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',color=INK,fontsize=9.6,linespacing=1.55)
    def arrow(points,color=INK,dashed=False):
        for p,q in zip(points[:-2],points[1:-1]):ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=1.25,ls='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=13,lw=1.25,color=color,linestyle='--' if dashed else '-'))
    ax.text(.1,8.05,'Preference-conditioned residual recovery: implemented control architecture',fontsize=16,fontweight='bold',color=INK)
    box(.2,5.1,2.35,1.15,'Reference path\nSpeed command $v_{ref}$',BLUE,'#EAF1F7')
    box(3.1,5.1,2.35,1.15,'Geometric path follower\n$\\delta_{ref}$, $v_{ref}$',BLUE,'#EAF1F7')
    box(6.05,5.1,3.0,1.15,'ECBC + ESO / drive reference\nDynamic $\\theta^{B}_{ref}$; 200 Hz\n$u_0=[\\dot\\delta_0,\\omega_{r,0}]$',BLUE,'#EAF1F7')
    ax.add_patch(Circle((10.05,5.67),.23,ec=INK,fc='white',lw=1.3));ax.text(10.05,5.67,'+',ha='center',va='center',fontsize=18)
    box(10.85,5.1,2.5,1.15,'Command bounds\nVelocity servos\nTorque limits',GRAY,'#F0F2F4')
    box(14.05,5.1,2.6,1.15,'Single-track robot\nMuJoCo / MJX\nGround contact',BLUE,'#EAF1F7')
    arrow([(2.61,5.67),(3.04,5.67)]);arrow([(5.51,5.67),(5.99,5.67)]);arrow([(9.11,5.67),(9.81,5.67)]);arrow([(10.29,5.67),(10.79,5.67)]);arrow([(13.41,5.67),(13.99,5.67)])
    box(8.8,7.05,2.6,.6,'Steering-rate bias $d_\\delta(t)$',RED,'#FAEEEE');arrow([(10.05,6.99),(10.05,5.92)],RED)
    box(14.05,7.05,2.6,.6,'Lateral force at COM',RED,'#FAEEEE');arrow([(15.35,6.99),(15.35,6.31)],RED)
    box(13.5,2.15,3.15,1.0,'Observable state / commands\nIMU, wheel rates, steering, pose\nESO disturbance estimate',BLUE,'#EAF1F7')
    arrow([(15.35,5.04),(15.35,3.21)])
    arrow([(13.44,2.65),(12.9,2.65),(12.9,4.5),(7.55,4.5),(7.55,5.04)],BLUE)
    arrow([(14.05,2.0),(14.05,1.75),(4.27,1.75),(4.27,5.04)],BLUE)
    box(8.75,2.15,3.2,1.0,'19 fields + valid masks\n10 frames: 200-D input\nIncludes past commands and $\\alpha$',TEAL,'#EAF5F2')
    box(5.05,2.15,2.9,1.0,'Conditional residual actor\nMLP 256 → 128, LeakyReLU\n2 outputs + tanh',TEAL,'#EAF5F2')
    box(5.05,3.6,2.9,.58,'Scale: $\\Delta\\dot\\delta$, $\\Delta\\omega_r$',TEAL,'#EAF5F2')
    arrow([(13.44,2.65),(12.01,2.65)],TEAL);arrow([(8.69,2.65),(8.01,2.65)],TEAL)
    arrow([(6.5,3.21),(6.5,3.54)],TEAL);arrow([(8.01,3.89),(10.05,3.89),(10.05,5.42)],TEAL)
    box(.2,2.15,2.95,1.0,'External preference $\\alpha$\nTrain: U[0, 1] per episode\nEvaluate: 0 / 0.5 / 1',ORANGE,'#FBF2E6')
    arrow([(3.21,2.65),(3.65,2.65),(3.65,3.35),(10.35,3.35),(10.35,3.21)],ORANGE)
    box(.2,.15,3.7,.85,'Learning roll reference $\\theta^{L}_{ref}$\nCurrent circle test: fixed 6.84°\nDoes not replace ECBC reference',ORANGE,'#FBF2E6')
    arrow([(2.0,.09),(2.0,-.45),(10.35,-.45),(10.35,2.09)],ORANGE)
    box(5.05,-.01,3.65,.95,'Training only: reward + critic + PPO\nSurvival, roll, speed, path, actions\nSimulator truth is used in reward',GRAY,'#F0F2F4')
    arrow([(6.5,1.0),(6.5,2.09)],GRAY,True)
    ax.text(11.05,.2,'Solid: online signals   Dashed: training update\nBlue: baseline / sensing   Teal: learned residual\nOrange: declared targets   Red: unknown disturbances',fontsize=9,color=GRAY)
    save(fig,'control_architecture')


def evidence_figures():
    names=['priority_history_20260911','priority_survival_20260911','priority_fixed_roll_20260912']
    summaries=[json.loads((ROOT/'runs'/n/'analysis/results.json').read_text()) for n in names]
    alphas=[0,.5,1];fig,axs=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,key,title in zip(axs,['recovered','failed'],['Joint recovery count / 12','Physical failure count / 12']):
        z=np.array([[next(g[key] for g in s['summary'] if g['policy']=='residual' and g['alpha']==a) for a in alphas] for s in summaries])
        ax.imshow(z,vmin=0,vmax=12,cmap='Blues' if key=='recovered' else 'Reds',aspect='auto')
        for (i,j),val in np.ndenumerate(z):ax.text(j,i,f'{val}/12',ha='center',va='center',color='white' if val>=7 else INK,fontsize=13)
        ax.set_xticks(range(3),['0','0.5','1']);ax.set_xlabel('Preference alpha');ax.set_yticks(range(3),['Dynamic target / original costs','Dynamic target / reduced costs','Fixed learning target / restored costs']);ax.set_title(title,loc='left',fontweight='bold')
    fig.suptitle('Development evidence: shared classic cases and evaluation seeds; baseline 12/12 recovery, 0 failures',fontsize=10)
    save(fig,'recovery_evidence')
    root=ROOT/'runs'/names[-1];fig,axs=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    from sttw_control.media import state_series
    scenario_names=['steer_negative_3s','steer_positive_3s','force_negative_3s','force_positive_3s']
    for r,case in enumerate(scenario_names):
        for i,a in enumerate(alphas):
            for policy in (['baseline','residual'] if i==0 else ['residual']):
                path=root/'evaluation'/f'alpha_{i}'/'seed_47001'/case/policy
                tr=dict(np.load(path/'trace.npz'));c=json.loads((path/'declaration.json').read_text())['config'];t=tr['time'];xy=tr['pose'][:,:2]-[c['circle']['center_x'],c['circle']['center_y']]
                ey=np.linalg.norm(xy,axis=1)-c['circle']['radius'];ev=state_series(tr,c)['speed']-c['speed_reference'];color=GRAY if policy=='baseline' else [BLUE,TEAL,ORANGE][i];label='ECBC + ESO' if policy=='baseline' else f'alpha = {a:g}'
                for col,y in enumerate([ey,ev]):
                    axs[r,col].plot(t,y,color=color,lw=1.2,label=label,ls='--' if policy=='baseline' else '-')
                    if tr['terminated'][-1]:axs[r,col].plot(t[-1],y[-1],'x',color=color,ms=7,mew=1.5)
        for col,ax in enumerate(axs[r]):
            ax.axvspan(6,9,color=GRAY,alpha=.14);ax.axhline(0,color=GRAY,lw=.5);ax.set_xlim(0,30);ax.grid(alpha=.16);ax.set_xlabel('Time [s]');ax.set_ylabel('Radial error [m]' if col==0 else 'True speed error [m/s]');ax.set_title(case.replace('_',' '),loc='left',fontsize=10)
    axs[0,0].legend(ncol=2,fontsize=9);fig.suptitle('Fixed learning target: all four classic disturbances, seed 47001 | crosses: real termination',fontsize=12)
    save(fig,'fixed_roll_errors')
    files=[ROOT/'runs'/n/'analysis/results.json' for n in names]+[ROOT/'runs'/names[-1]/'standard_results.json',ROOT/'runs/response_comparison_20260910/analysis/summary.json']
    for n in names:
        files.extend(sorted((ROOT/'runs'/n/'frozen').glob('*.json')))
        files.append(ROOT/'runs'/n/'training/checkpoints/update_0032/identity.json')
    manifest={'scope':'development only; common alphas shown, no independent training-seed uncertainty','inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'figures':['control_architecture','recovery_evidence','fixed_roll_errors']}
    (HERE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def export_draft():
    import markdown
    from weasyprint import HTML
    source=(HERE/'MANUSCRIPT.md').read_text()
    # Render math as embedded vector images so exported HTML/PDF work offline.
    def formula(match):
        expr=match.group(1).strip();number=re.search(r'\\tag\{(\d+)\}',expr);expr=re.sub(r'\\tag\{\d+\}', '',expr);buf=io.BytesIO()
        from matplotlib.mathtext import math_to_image
        with plt.rc_context({'svg.fonttype':'path'}):
            math_to_image('$'+expr+'$',buf,format='svg',dpi=180,color=INK)
        data=base64.b64encode(buf.getvalue()).decode()
        return '\n<div class="equation"><img alt="formula" src="data:image/svg+xml;base64,'+data+'">'+(' <span>('+number.group(1)+')</span>' if number else '')+'</div>\n'
    source=re.sub(r'\$\$(.*?)\$\$',formula,source,flags=re.S)
    body=markdown.markdown(source,extensions=['tables','fenced_code','toc'])
    css='''@page {size:A4;margin:18mm 17mm 18mm;@bottom-center{content:counter(page);font-size:9pt;color:#647180}} body{font-family:"Noto Serif CJK SC",serif;color:#243548;font-size:10pt;line-height:1.7} h1{font-size:21pt;color:#244b6b} h2{font-size:14pt;color:#35658a;border-bottom:1px solid #d5dfe5;margin-top:1.4em} h3{font-size:11pt;color:#278c82} h1,h2,h3{break-after:avoid} table{border-collapse:collapse;width:100%;font-size:8.5pt;margin:12px 0;line-height:1.5} th{background:#eaf1f7} th,td{border-bottom:1px solid #d8e0e4;padding:5px} tr{break-inside:avoid} img{max-width:100%;height:auto} .equation{margin:12px 0;text-align:center;break-inside:avoid}.equation img{max-width:100%;max-height:120px} blockquote{border-left:3px solid #d78c39;padding:5px 12px;background:#fbf2e6} a{color:#35658a;text-decoration:none} code{font-size:8.5pt;overflow-wrap:anywhere} pre{white-space:pre-wrap;font-size:8pt} p{orphans:3;widows:3}'''
    html='<!doctype html><html lang="zh"><meta charset="utf-8"><title>STTW paper draft</title><style>'+css+'</style><body>'+body+'</body></html>'
    (HERE/'MANUSCRIPT.html').write_text(html)
    HTML(string=html,base_url=str(HERE)).write_pdf(HERE/'MANUSCRIPT.pdf')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--figures-only',action='store_true');p.add_argument('--export-only',action='store_true');a=p.parse_args()
    if not a.export_only:control_diagram();evidence_figures()
    if not a.figures_only:export_draft()
