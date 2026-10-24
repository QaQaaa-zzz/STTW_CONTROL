"""Prepare, then run, one fixed-parameter prospective margin confirmation."""
import argparse,copy,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def register(path,name,status):
    sys.path.insert(0,str(Path('learning/cli').resolve()))
    from prepare_short_recovery_campaign import register as shared_register
    shared_register(path,name,status)


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),stage='authenticate_and_build_paths',active_lower_ticks=0,training_steps=0)
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(root/'prepare_status.json',status)
    update();register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 固定公式前瞻确认准备',root/'prepare_status.json')
    try:
        from sttw_control.feature_comparison_evaluation import authenticated_parent_record
        from sttw_control.path_command_training import load_environment
        from sttw_control.geometric_path import build_path
        old=Path('runs/reserve_feature_expansion_v1/post_fixed_final_v1')
        original=old/'high_nominal_retention/H75/declaration.json'
        template=json.loads(original.read_text())
        model=json.loads((old/'models.json').read_text())['records']['H75'];authenticated_parent_record(model)
        cfg=copy.deepcopy(template['config']);cfg['path'].update(turn_angle_rad=math.radians(110),turn_radius_m=2.6)
        cfg['stage_C_evaluation'].update(route_ids=['left90_R2','right90_R2'],low_speed_m_s=2.35,conflict_speed_m_s=2.75)
        routes=[dict(name='left110_R2p6',builder='left90_R2'),dict(name='right110_R2p6',builder='right90_R2')]
        declaration=dict(project='STTW_CONTROL',experiment_id='fixed_formula_prospective_confirmation_v1',
                         config=cfg,source=template['source'],routes=routes,
                         data_role='Prospective DEV confirmation; parameters frozen before new force outcomes',
                         original_task_horizon_s=10.,physical_observation_s=4.)
        declaration_file=root/'declaration.json';atomic_json(declaration_file,declaration)
        env,initial=load_environment(cfg,template['source'],'terminal_contract_v2')
        pose=np.asarray(env.physics.helpers.pose(initial.data));paths=root/'paths';paths.mkdir()
        for row in routes:
            path=build_path(row['builder'],pose,cfg)
            np.savez_compressed(paths/(row['name']+'_path.npz'),xy=np.asarray(path.xy),s=np.asarray(path.s),
                                heading=np.asarray(path.heading),curvature=np.asarray(path.curvature),goal=np.asarray(path.goal),turn_end=np.asarray(path.turn_end))
        scripts=Path(__file__).resolve().parent;boundary=scripts.parent/'alpha_physical_boundary'
        formula=Path('runs/reserve_feature_expansion_v1/physics_config.json').resolve()
        files=[Path(__file__).resolve(),scripts/'forecast_prefixes.py',scripts/'PREDICTIVE_CONFIRMATION_DESIGN.md',
               boundary/'execute.py',boundary/'collect_prefixes.py',boundary/'analyze.py',
               Path('learning/cli/probe_sensor_gate_domain_hold.py'),declaration_file,original.resolve(),formula,
               Path(model['actor']),Path(model['checkpoint'])]
        files += [Path('learning/src/sttw_control')/(name+'.py') for name in ['path_command_policy','path_command_training','path_command_env','reference_reserve_gate','reserve_feature_inputs','balance_probe','controller','closed_loop_kernel','teleop_env']]
        files += list(paths.glob('*.npz'))
        sources={str(file.resolve()):digest(file) for file in files}
        for side,route in zip(['left','right'],routes):
            child=root/side;child.mkdir()
            plan=dict(project='STTW_CONTROL',owner='Codex tolerance-calibration',model=model,policy_name='H75',
                      source_declaration=str(declaration_file),path_evidence_directory=str(paths),
                      route=route['name'],builder=route['builder'],speeds=[2.35,2.75],alphas=[0.,1.],phase_fractions=[.2,.7],
                      forces_N=[0.,-80.,80.],pulse_duration_s=.1,observation_s=4.,original_task_horizon_s=10.,
                      declared_prefixes=8,declared_cases=24,maximum_prepare_lower_ticks=8000,
                      maximum_scan_lower_ticks=19200,maximum_total_lower_ticks=27200,training_steps=0,
                      source_sha256=sources,source_formula_config=str(formula),
                      pre_scan_forecast_script=str(scripts/'forecast_prefixes.py'),
                      design=str(scripts/'PREDICTIVE_CONFIRMATION_DESIGN.md'),
                      stop_rule='Fixed48 cases across two routes once; no saturation extension, fitting or retry')
            atomic_json(child/'prepare_plan.json',plan)
            for name,count in [('prepare_status.json',8),('forecast_status.json',8),('status.json',24),('campaign_status.json',24)]:
                atomic_json(child/name,dict(phase='prepared',completed_cases=0,declared_cases=count,active_lower_ticks=0,training_steps=0))
                register(child/('ACTIVE_'+name.replace('.json','').upper()+'.json'),'STTW_CONTROL 固定公式前瞻确认 '+side+' '+name,child/name)
        atomic_json(root/'plan.json',dict(project='STTW_CONTROL',declared_cases=48,declared_prefixes=16,
                    maximum_lower_ticks=54400,training_steps=0,source_sha256=sources,
                    code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    physical_identity=env.physics.bundle.identity,route_order=['left','right'],
                    prediction_order='genuine prefix -> saved history-only forecasts -> force scan',
                    stop_rule='One prospective confirmation; no parameter/time/force sweep'))
        atomic_json(root/'campaign_status.json',dict(phase='prepared',declared_cases=48,completed_cases=0,active_lower_ticks=0,training_steps=0))
        register(root/'ACTIVE_CAMPAIGN.json','STTW_CONTROL 固定公式前瞻确认总控48分支',root/'campaign_status.json')
        update(phase='completed',stage='frozen_geometry_formula_and_runtime_plans_ready',declared_cases=48)
        print(root,'prepared48 physical branches,16 real prefixes;maximum54400 lower ticks;0 optimizer updates')
    except BaseException as exc:update(phase='error',error=repr(exc));raise


def execute(root):
    plan=json.loads((root/'plan.json').read_text());status=dict(phase='running',pid=os.getpid(),stage='starting',declared_cases=48,completed_cases=0,active_lower_ticks=0,training_steps=0)
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(root/'campaign_status.json',status)
    update()
    try:
        for side in plan['route_order']:
            child=root/side
            with (child/'controller.log').open('x') as log:
                process=subprocess.Popen([sys.executable,'-u','docs/evidence/tolerance_progress_20261022/alpha_physical_boundary/execute.py','--run',str(child)],stdout=log,stderr=subprocess.STDOUT)
                update(stage=side,child_pid=process.pid);code=process.wait()
            state=json.loads((child/'campaign_status.json').read_text())
            if code or state['phase']!='completed':raise RuntimeError(side+' failed; no retry or following route')
            status['active_lower_ticks']+=state['active_lower_ticks'];status['completed_cases']+=state['completed_cases']
            if status['active_lower_ticks']>plan['maximum_lower_ticks']:raise ValueError('Confirmation physics budget exceeded')
            update()
        update(phase='completed',stage='all48_forecasted_force_outcomes_ready')
    except BaseException as exc:update(phase='error',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--execute',action='store_true');a=parser.parse_args()
    (execute if a.execute else prepare)(a.output.resolve())
