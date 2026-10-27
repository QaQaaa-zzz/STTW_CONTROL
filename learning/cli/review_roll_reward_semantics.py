"""Model-reference arithmetic only: working lean costs are not fall margins."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import q_ratio
from sttw_control.path_command_reward import costs


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    planfile=a.source/'plan.json';plan=json.loads(planfile.read_text());cfg=plan['config'];cc=ControllerConfig(**plan['initial_source']['controller'])
    a.output.mkdir(exist_ok=False)
    rows=[]
    for name,speed,radius in [('native26',2.6,1.8),('native28',2.8,1.8),('low_transfer',1.5,1.1)]:
        delta=float(np.arctan(cc.wheelbase/radius/np.cos(cc.caster)))
        q=float(q_ratio(speed,cc));phi=delta/q;chi=float(np.clip((abs(phi)-.18)/.12,0,1))
        value=costs(alpha=0.,chi=chi,ev=0.,ey=0.,eh=0.,peak_roll=abs(phi),roll_rate=0.,speed=speed,offsets=np.zeros(2),offset_rate=np.zeros(2),config=cfg)
        # Inverse exists on this positive-speed, inactive-m2-floor branch only.
        branch=[]
        for budget in [cfg['path_reward']['roll_start_rad'],cfg['path_reward']['working_roll_start_rad']]:
            v2=cc.cg_forward*cc.trail*cc.gravity/cc.cg_height+cc.gravity*cc.wheelbase*budget/(np.cos(cc.caster)*abs(delta))
            v=float(np.sqrt(v2));raw_m2=-(cc.mass*v*v*cc.cg_height-cc.mass*cc.cg_forward*cc.trail*cc.gravity)*np.cos(cc.caster)/cc.wheelbase
            assert v>=cc.minimum_speed and abs(raw_m2)>.1
            residual=abs(delta/float(q_ratio(v,cc)))-budget;assert abs(residual)<1e-6
            branch.append(dict(model_budget_rad=budget,model_speed_ceiling_m_s=v,q_inverse_residual_rad=residual))
        rows.append(dict(case=name,requested_speed_m_s=speed,radius_parameter_m=radius,
            model_feedforward_steer_rad=delta,model_q=q,model_nominal_roll_reference_rad=phi,
            chi=chi,hypothetical_zero_tracking_error_effective_roll_rate_cost=float(value['effective']['roll']),
            hypothetical_zero_tracking_error_effective_working_roll_rate_cost=float(value['effective']['working_roll']),
            inverse_model_budget=branch))
    speeds=np.linspace(1.,3.2,221);curves={}
    for radius in [1.1,1.8]:
        delta=np.arctan(cc.wheelbase/radius/np.cos(cc.caster))
        curves[str(radius)]=[float(delta/float(q_ratio(v,cc))) for v in speeds]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for radius,color,style in [(1.1,'#0072B2','-'),(1.8,'#D55E00','--')]:
        curve=np.asarray(curves[str(radius)]);axes[0].plot(speeds,curve,color=color,ls=style,label=f'R={radius:g}m model')
        axes[1].plot(speeds,np.clip((abs(curve)-.18)/.12,0,1),color=color,ls=style,label=f'R={radius:g}m model')
    for budget in [.26,.30]:axes[0].axhline(budget,color='.45',ls=':',label=f'working-cost onset {budget:.2f}rad')
    for row in rows:
        for ax,y in zip(axes,[row['model_nominal_roll_reference_rad'],row['chi']]):
            ax.scatter(row['requested_speed_m_s'],y,marker='o' if row['case']=='low_transfer' else 's',c='black',s=26)
    axes[0].set(xlabel='Requested speed (m/s)',ylabel='Model feedforward roll reference (rad)',ylim=(0,.9))
    axes[1].set(xlabel='Requested speed (m/s)',ylabel='Existing task conflict gate chi',ylim=(-.03,1.03))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Existing ECBC reference / V3 reward arithmetic\nNo measured lean, survival probability, disturbance-force limit or new controller')
    fig.savefig(a.output/'model_reference_and_chi.png',dpi=160);plt.close(fig)
    files=[planfile,Path(__file__),Path('learning/src/sttw_control/controller.py'),Path('learning/src/sttw_control/direct_command_policy.py'),Path('learning/src/sttw_control/path_command_reward.py'),Path('learning/src/sttw_control/path_command_env.py'),Path('learning/src/sttw_control/native_equilibrium_rule.py')]
    result=dict(rows=rows,controller_config=vars(cc),formula=dict(feedforward='atan(L*kappa/cos(caster))',nominal_model_roll='delta_ff / q(v_user)',
        q_unfloored_positive_branch='g*L / [cos(caster)*(v^2-cg_forward*trail*g/cg_height)]',
        inverse='v^2=cg_forward*trail*g/cg_height+g*L*budget/(cos(caster)*abs(delta_ff))'),
        semantics=['Zero task tracking error can carry a working-roll cost in the existing local reference model.',
            'This is an explicit working-envelope preference, not automatically a reward bug.',
            'The same q/budget idea already exists in the completed native_equilibrium_rule; not a new algorithm.',
            'No actual steady equilibrium, ESO shift, local300 behavior or physical contact threshold identified.',
            'Low model lean / chi is not evidence of high physical disturbance tolerance.'],
        source_sha256={str(f.resolve()):digest(f) for f in files},new_physics_ticks=0,new_training_steps=0)
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
