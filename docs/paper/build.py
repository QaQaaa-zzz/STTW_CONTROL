"""Build the current local STTW research draft, never historical experiments.

The prior20260912 builder and every original artifact are SHA-archived under
history/. This builder draws only the current architecture and reuses existing
source-backed XY figures. It does not train, integrate physics, or select a model.
"""
import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import re
import subprocess
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIG=HERE/'figures'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def control_diagram():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(18,7));ax.set(xlim=(0,21),ylim=(.4,9));ax.axis('off')
    colors=dict(base='#35658A',learned='#278C82',rule='#647180',external='#B45659')
    def box(x,y,w,h,text,kind='base'):
        color=colors[kind]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05',ec=color,fc='white',lw=1.5))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,color='#243548',linespacing=1.35)
    def arrow(points,kind='base'):
        for p,q in zip(points[:-2],points[1:-1]):ax.plot([p[0],q[0]],[p[1],q[1]],c=colors[kind],lw=1.4)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=15,lw=1.4,color=colors[kind]))
    boxes=[(.2,6,2.3,1.1,'Original path\nand speed request'),(3,6,2.6,1.1,'Geometric follower\nNominal references'),
        (6.2,6,2.7,1.1,'NN + fixed rules\nMode / permission'),(9.5,6,3.1,1.1,'Original reference\nfilter / slew / bounds'),
        (13.2,6,3.5,1.1,'ECBC + ESO + local300\nActuator constraints\nFrozen lower /200 Hz'),
        (17.3,6,3.1,1.1,'Single-track plant\n25 physics substeps\nper lower tick')]
    for args in boxes:box(*args)
    for a,b in zip(boxes[:-1],boxes[1:]):arrow([(a[0]+a[2],6.55),(b[0],6.55)])
    box(6.2,3,2.7,1.4,'Frozen middle NN\n351-D input /50 Hz\nSpeed + steer offsets','learned')
    arrow([(7.55,4.4),(7.55,6)],'learned')
    box(9.6,3,3.0,1.4,'Fixed geometry feedback\nExisting speed support\nProtocol held constant','rule')
    arrow([(11.1,4.4),(11.1,5.2),(8.6,5.2),(8.6,6)],'rule')
    box(3,3,2.5,1.4,'User preference alpha\n0: path priority\n1: speed priority')
    arrow([(5.5,3.7),(6.2,3.7)])
    box(13.2,2.9,3.5,1.5,'Observed + controller state\nIMU / wheel / steer / pose\nHistory / observer / filters')
    arrow([(18.85,6),(18.85,3.65),(16.7,3.65)])
    arrow([(14.95,4.4),(14.95,6)])
    arrow([(15.5,2.9),(15.5,1.45),(2.7,1.45),(2.7,5.35),(4.3,5.35),(4.3,6)])
    arrow([(13.2,3.1),(13.0,2.1),(7.55,2.1),(7.55,3)],'learned')
    arrow([(11.1,2.1),(11.1,3)],'rule')
    box(17.3,7.8,3.1,.7,'World-frame COM pulse','external')
    arrow([(18.85,7.8),(18.85,7.1)],'external')
    ax.text(.2,8.45,'Recent expanded evaluation stack: middle references are not motor actions',fontsize=16,color='#243548')
    ax.text(.2,.8,'Zero middle masks only neural channels; fixed rules and the frozen lower remain.\nNo force/onset query enters the controller. Opt-in permission factor is a separate, non-adopted comparison.',fontsize=11,color='#647180')
    FIG.mkdir(exist_ok=True)
    for ext in ['svg','png','pdf']:
        fig.savefig(FIG/f'control_architecture.{ext}',dpi=160,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    svg=FIG/'control_architecture.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def validate_evidence_references(manuscript, sources):
    """Resolve bracketed evidence IDs, including inclusive citation ranges."""
    referenced=set()
    for bracket in re.findall(r'\[[^\]\n]*\]',manuscript):
        for match in re.finditer(r'\bE(\d{3})(?:\s*[-–—]\s*E?(\d{3}))?',bracket):
            start=int(match[1]);end=int(match[2] or match[1])
            if end<start:raise ValueError('descending manuscript evidence range '+match[0])
            referenced.update(f'E{number:03d}' for number in range(start,end+1))
    missing=sorted(referenced-sources.keys())
    if missing:raise ValueError('unresolved manuscript evidence references: '+', '.join(missing))
    return sorted(referenced)


def evidence_manifest():
    source=ROOT/'runs/support_permission_factor_v1/analysis/paper_claim_audit/claim_registry.json'
    data=json.loads(source.read_text());sources=copy.deepcopy(data['sources'])
    for record in sources.values():
        if digest(record['path'])!=record['sha256']:raise ValueError('changed manuscript evidence '+record['path'])
    archive=Path('/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025')
    extra={
        'E008':ROOT/'runs/support_permission_factor_v1/common_environment_config.json',
        'E009':ROOT/'runs/reference_bias_opportunity_v1/production/analysis/summary.json',
        'E010':ROOT/'runs/support_permission_factor_v1/analysis/existing_witness_scope_review.md',
        'E011':ROOT/'learning/src/sttw_control/contact_only_environment.py',
        'E012':ROOT/'learning/configs/frozen_middle_registry.json',
        'E013':ROOT/'learning/src/sttw_control/actuator.py',
        'E014':ROOT/'runs/finite_value_online_v1_recovery55/plan.json',
        'E015':ROOT/'runs/current_native_preference_v1_attempt2/plan.json',
        'E016':ROOT/'runs/online_preference_survival_v1_attempt3/plan.json',
        'E017':ROOT/'learning/src/sttw_control/path_command_policy.py',
        'E018':ROOT/'learning/src/sttw_control/path_command_env.py',
        'E019':ROOT/'learning/src/sttw_control/reference_reserve_gate.py',
        'E020':ROOT/'learning/src/sttw_control/reference_reentry.py',
        'E021':ROOT/'runs/goal_replanning_github_v1/plan.json',
        'E022':ROOT/'runs/goal_replanning_github_v1/analysis/summary.json',
        'E023':ROOT/'runs/goal_replanning_h75_v1/combined_analysis/summary.json',
        'E024':ROOT/'runs/native_zero_preference_v1/analysis/review.json',
        'E025':ROOT/'runs/native_zero_preference_v1/task_scope/review.json',
        'E026':ROOT/'runs/native_equilibrium_rule_v1/analysis/review.json',
        'E027':ROOT/'runs/native_equilibrium_rule_v1/task_scope/review.json',
        'E028':ROOT/'runs/native_goal_confirmation_v1/analysis/summary.json',
        'E029':ROOT/'runs/native_goal_confirmation_v1/analysis/initial_carry_review.json',
        'E030':ROOT/'runs/capability_goal_joint_1m_v1/joint_review.json',
        'E031':ROOT/'runs/capability_goal_joint_1m_v1/independent_review.json',
        'E032':ROOT/'runs/capability_preference_utility_v1/review.json',
        'E033':ROOT/'runs/capability_preference_utility_v1/independent_review.json',
        'E034':ROOT/'runs/replanning_headroom_review_v1/review.json',
        'E035':ROOT/'docs/evidence/tolerance_progress_20261022/finite_value_online/PRIOR_ART_AND_CONTRIBUTION_BOUNDARIES_20261027.md',
        'E036':ROOT/'runs/preference_capacity_order_v1/analysis/review.json',
        'E037':ROOT/'runs/preference_capacity_order_v1/independent_review.json',
        'E038':ROOT/'runs/joint_endpoint_tradeoff_review_v1/analysis/review.json',
        'E039':ROOT/'runs/joint_endpoint_tradeoff_review_v1/independent_review.json',
        'E040':ROOT/'runs/joint_endpoint_tradeoff_review_v1/arrival_mode_audit.json',
        'E041':ROOT/'runs/joint_endpoint_tradeoff_review_v1/model_identity_receipt.json',
        'E042':ROOT/'runs/legacy_credit_goal_continuation_v1/analysis/review.json',
        'E043':ROOT/'runs/legacy_credit_goal_continuation_v1/independent_review.json',
        'E044':ROOT/'runs/legacy_credit_goal_continuation_v1/execution_plan.json',
        'E045':ROOT/'runs/legacy_credit_goal_continuation_v1/diagnostics/receipt.json',
        'E046':ROOT/'runs/legacy_credit_goal_continuation_v1/evaluation/status.json',
        'E047':ROOT/'runs/native_transition_mechanism_v1/analysis/review.json',
        'E048':ROOT/'runs/native_transition_mechanism_v1/independent_review.json',
        'E049':ROOT/'docs/evidence/tolerance_progress_20261022/finite_value_online/DYNAMIC_REFERENCE_GOVERNOR_METHOD_REVIEW.md',
        'E050':ROOT/'docs/evidence/tolerance_progress_20261022/finite_value_online/DUAL_NETWORK_COVERAGE_REVIEW.md',
        'E051':ROOT/'runs/preference_history_factorial_v1/analysis_verified/review.json',
        'E052':ROOT/'runs/preference_history_factorial_v1/independent_outcome_review.json',
        'E053':ROOT/'runs/preference_history_baselines_v1/analysis/review.json',
        'E054':ROOT/'runs/preference_history_baselines_v1/independent_outcome_review.json',
        'E055':ROOT/'runs/preference_history_task_review_v1/review.json',
        'E056':ROOT/'runs/preference_history_task_review_v1/verification_receipt.json',
        'E057':ROOT/'runs/nominal_program_feedback_v1/analysis/review.json',
        'E058':ROOT/'runs/nominal_program_feedback_v1/independent_outcome_review.json',
        'E059':ROOT/'runs/nominal_program_feedback_v1/training_support_review/audit.json',
        'E060':ROOT/'runs/nominal_program_feedback_v1/training_support_review/independent_review.md',
        'E061':ROOT/'runs/nominal_program_feedback_v1/route_reframe/public_source_review.json',
        'E062':ROOT/'runs/native_fixed_midpoint_review_v1/analysis/review.json',
        'E063':ROOT/'runs/native_fixed_midpoint_review_v1/independent_review.json',
        'E064':ROOT/'runs/native_fixed_midpoint_review_v1/disturbed/analysis/review.json',
        'E065':ROOT/'runs/native_fixed_midpoint_review_v1/disturbed/independent_review.json',
        'E066':ROOT/'runs/native_reference_rollout_v1/analysis/review.json',
        'E067':ROOT/'runs/native_reference_rollout_v1/independent_final_review.json',
        'E068':ROOT/'runs/native_fixed_midpoint_review_v1/utility_envelope/review.json',
        'E069':ROOT/'runs/native_fixed_midpoint_review_v1/utility_envelope/independent_review.json',
        'E070':ROOT/'runs/capability_zero_common3_review_v1/review.json',
        'E071':ROOT/'runs/capability_zero_common3_review_v1/independent_review.json',
        'E072':ROOT/'runs/preference_direction_fullpanel_review_v1/review.json',
        'E073':ROOT/'runs/preference_direction_fullpanel_review_v1/independent_review.json',
        'E074':ROOT/'runs/preference_direction_fullpanel_review_v1/exposure_review/review.json',
        'E075':ROOT/'runs/preference_direction_fullpanel_review_v1/exposure_review/independent_exposure.json',
        'E076':ROOT/'runs/preference_direction_fullpanel_review_v1/arrival_phase_review/review.json',
        'E077':ROOT/'runs/preference_direction_fullpanel_review_v1/arrival_phase_review/independent_review.json',
        'E078':ROOT/'runs/native_frozen_geometry_validation_v1/analysis/review.json',
        'E079':ROOT/'runs/native_frozen_geometry_validation_v1/independent_final_review.json',
        'E080':ROOT/'runs/ecbc_modal_authority_math_v1/analysis.json',
        'E081':ROOT/'runs/ecbc_modal_authority_math_v1/independent_review.md',
        'E082':ROOT/'runs/ordinary_goal_learning_v1/plan.json',
        'E083':ROOT/'runs/ordinary_goal_learning_v1/first_actual_training_audit.json',
        'E084':ROOT/'learning/src/sttw_control/goal_endpoint_run.py',
        'E085':ROOT/'runs/goal_credit_cpu_v1/pulse_timing_first_real_collection_v2.json',
        'E086':ROOT/'learning/src/sttw_control/goal_networks.py',
        'E087':archive/'ordinary_goal_learning_v1/training_review/summary.json',
        'E088':archive/'ordinary_goal_endpoint_v1/analysis_with_h75/results.json',
        'E089':archive/'ordinary_goal_endpoint_replan_v1/paired_replanning_review/summary.json',
        'E090':archive/'ordinary_goal_endpoint_replan_v1/paired_replanning_review/raw_verification.json',
        'E091':archive/'h75_native_s_capability_v1/analysis/review.json',
        'E092':archive/'h75_nominal_preference_schedule_v1/analysis/review.json',
        'E093':archive/'h75_nominal_preference_schedule_v1/analysis/independent_review.json',
        'E094':archive/'h75_nominal_preference_schedule_v1/plan.json',
        'E095':archive/'ordinary_goal_endpoint_replan_v1/plan.json',
        'E096':archive/'h75_native_s_capability_v1/plan.json',
        'E097':archive/'ordinary_goal_learning_v1/training_review/REVIEW.md',
        'E098':archive/'h75_channel_factorial_v1/analysis/RESULTS.md',
        'E099':archive/'h75_channel_response_v1_attempt2/analysis/README.md',
        'E100':archive/'h75_switching_continuation_v1_attempt2/analysis/results.json',
        'E101':archive/'h75_switching_continuation_v1_attempt2/analysis/receipt.json',
        'E102':archive/'h75_switching_continuation_v1_attempt2/plan.json',
        'E103':archive/'h75_switching_continuation_v1_attempt2/root_outcome_check.json',
        'E104':archive/'h75_online_scalar_v1/analysis/results.json',
        'E105':archive/'h75_online_scalar_v1/analysis/receipt.json',
        'E106':archive/'h75_online_scalar_v1/plan.json',
        'E107':archive/'h75_online_scalar_v1/root_outcome_check.json',
        'E108':archive/'h75_online_confirmation_v1/analysis/results.json',
        'E109':archive/'h75_online_confirmation_v1/analysis/receipt.json',
        'E110':archive/'h75_online_confirmation_v1/plan.json',
        'E111':archive/'h75_online_confirmation_v1/root_outcome_check.json',
        'E112':archive/'h75_samestate_tolerance_v1/analysis/results.json',
        'E113':archive/'h75_samestate_tolerance_v1/analysis/receipt.json',
        'E114':archive/'h75_samestate_tolerance_v1/plan.json',
        'E115':archive/'h75_samestate_tolerance_v1/root_primary_label_check.json',
        'E116':archive/'h75_samestate_bank144_v1/analysis_final/summary.json',
        'E117':archive/'h75_samestate_bank144_v1/analysis_final/receipt.json',
        'E118':ROOT/'learning/configs/h75_tolerance_predictor_pilot.json',
        'E119':archive/'h75_tolerance_predictor_pilot_v1/selection_manifest.json',
        'E120':archive/'h75_tolerance_predictor_pilot_v1/heldout_metrics.json',
        'E121':archive/'h75_tolerance_predictor_pilot_v1/training_metrics.json',
        'E122':archive/'h75_tolerance_predictor_pilot_v1/root_training_review.json',
        'E123':archive/'h75_tolerance_predictor_ranking_v1/resolved_config.json',
        'E124':archive/'h75_tolerance_predictor_ranking_v1/selection_manifest.json',
        'E125':archive/'h75_tolerance_predictor_ranking_v1/development_metrics.json',
        'E126':archive/'h75_tolerance_predictor_ranking_v1/training_metrics.json',
        'E127':archive/'h75_samestate_confirmation144_v1/analysis_confirmation/results.json',
        'E128':archive/'h75_samestate_confirmation144_v1/analysis_confirmation/receipt.json',
        'E129':archive/'h75_samestate_confirmation144_v1/root_primary_label_check.json',
        'E130':archive/'h75_samestate_confirmation144_v1/predictor_evaluation/metrics.json',
        'E131':archive/'h75_samestate_confirmation144_v1/predictor_evaluation/data_receipt.json',
        'E132':archive/'h75_tolerance_online120_v1/plan.json',
        'E133':archive/'h75_tolerance_online120_v1/analysis/results.json',
        'E134':archive/'h75_tolerance_online120_v1/analysis/receipt.json',
        'E135':archive/'h75_tolerance_online120_v1/root_endpoint_check.json',
        'E136':archive/'h75_tolerance_online120_v1/analysis/original_task_rescoring/results.json',
        'E137':archive/'h75_tolerance_online120_v1/analysis/original_task_rescoring/method_summary.json',
        'E138':archive/'h75_tolerance_online120_v1/q_diagnostics/diagnostic.json',
        'E139':archive/'h75_late_state_tolerance48_v1/plan.json',
        'E140':archive/'h75_late_state_tolerance48_v1/evaluation/status.json',
        'E141':archive/'h75_late_state_tolerance48_v1/analysis_late/results.json',
        'E142':archive/'h75_late_state_tolerance48_v1/predictor_evaluation/metrics.json',
        'E143':archive/'h75_late_state_tolerance48_v1/predictor_evaluation/user_rejection.json',
        'E144':archive/'h75_tolerance_predictor_temporal_mix_v1/training/resolved_config.json',
        'E145':archive/'h75_tolerance_predictor_temporal_mix_v1/training/selection_manifest.json',
        'E146':archive/'h75_tolerance_predictor_temporal_mix_v1/training/development_temporal_metrics.json',
        'E147':archive/'h75_tolerance_predictor_temporal_mix_v1/training/timings.json',
        'E148':archive/'h75_tolerance_temporal_online40_v1/plan.json',
        'E149':archive/'h75_tolerance_temporal_online40_v1/status.json',
        'E150':archive/'h75_tolerance_temporal_online40_v1/analysis/root_method_summary.json',
        'E151':archive/'h75_tolerance_temporal_online40_v1/analysis/original_task_rescoring/results.json',
        'E152':archive/'h75_tolerance_temporal_online40_v1/analysis/root_prearrival_alpha.json',
        'E153':archive/'h75_tolerance_broaden120_v1/plan.json',
        'E154':archive/'h75_tolerance_broaden120_v1/status.json',
        'E155':archive/'h75_tolerance_broaden120_v1/analysis/root_method_summary.json',
        'E156':archive/'h75_tolerance_broaden120_v1/root_endpoint_strata.json',
        'E157':archive/'h75_tolerance_broaden120_v1/analysis/original_task_rescoring/results.json',
        'E158':archive/'h75_tolerance_broaden120_v1/analysis/original_task_rescoring/numerical_review.json',
        'E159':archive/'h75_user_preference120_v1/plan.json',
        'E160':archive/'h75_user_preference120_v1/status.json',
        'E161':archive/'h75_user_preference120_v1/analysis/root_user_summary.json',
        'E162':archive/'h75_user_preference120_v1/root_user_rejection.json',
        'E163':archive/'h75_user_preference120_v1/analysis/original_task_rescoring/results.json',
        'E164':archive/'h75_user_late_state192_v1/plan.json',
        'E165':archive/'h75_user_late_state192_v1/status.json',
        'E166':archive/'h75_user_late_state192_v1/predictor_evaluation/metrics.json',
        'E167':archive/'h75_user_late_state192_v1/predictor_evaluation/user_rejection.json',
        'E168':archive/'h75_tolerance_predictor_user_mix_v1/training/resolved_config.json',
        'E169':archive/'h75_tolerance_predictor_user_mix_v1/prepared/receipt.json',
        'E170':archive/'h75_tolerance_predictor_user_mix_v1/training/selection_manifest.json',
        'E171':archive/'h75_tolerance_predictor_user_mix_v1/training/development_temporal_metrics.json',
        'E172':archive/'h75_tolerance_predictor_user_mix_v1/training/bce_rank/seed0/history.json',
        'E173':archive/'h75_tolerance_predictor_user_mix_prior_v1/training/resolved_config.json',
        'E174':archive/'h75_tolerance_predictor_user_mix_prior_v1/training/selection_manifest.json',
        'E175':archive/'h75_tolerance_predictor_user_mix_prior_v1/training/development_temporal_metrics.json',
        'E176':archive/'h75_tolerance_predictor_user_mix_prior_v1/training/bce_rank/seed0/history.json',
        'E177':archive/'h75_tolerance_predictor_user_mix_prior_v1/prepared/receipt.json',
        'E178':archive/'h75_tolerance_predictor_user_mix_v1/training/status.json',
        'E179':archive/'h75_tolerance_predictor_user_mix_prior_v1/training/status.json',
        'E180':archive/'h75_tolerance_predictor_user_mix_prior_v1/training/bce_rank/seed1/history.json',
        'E181':archive/'h75_tolerance_predictor_user_mix_prior_v1/training/bce_rank/seed2/history.json',
        'E182':archive/'h75_user_prior_online72_v1/plan.json',
        'E183':archive/'h75_user_prior_online72_v1/status.json',
        'E184':archive/'h75_user_prior_online72_v1/analysis/summary.json',
        'E185':archive/'h75_user_prior_online72_v1/root_endpoint_check.json',
        'E186':archive/'h75_user_prior_online72_v1/root_failure_audit.json',
        'E187':archive/'h75_user_prior_online72_v1/analysis/receipt.json',
        'E188':archive/'h75_user_prior_online72_v1/analysis/original_task_rescoring/receipt.json',
        'E189':archive/'h75_user_prior_online72_v1/root_common_cost_summary.json',
        'E190':archive/'h75_user_prior_online72_v1/analysis/original_task_rescoring/results.json',
        'E191':archive/'h75_task_aware_fallback72_v1/plan.json',
        'E192':archive/'h75_task_aware_fallback72_v1/status.json',
        'E193':archive/'h75_task_aware_fallback72_v1/analysis/summary.json',
        'E194':archive/'h75_task_aware_fallback72_v1/root_endpoint_check.json',
        'E195':archive/'h75_task_aware_fallback72_v1/analysis/receipt.json',
        'E196':archive/'h75_task_aware_fallback72_v1/analysis/results.json',
        'E197':archive/'h75_task_aware_fallback72_v1/analysis/original_task_rescoring/receipt.json',
        'E198':archive/'h75_task_aware_fallback72_v1/analysis/original_task_rescoring/results.json',
        'E199':archive/'h75_task_aware_fallback72_v1/root_common_cost_summary.json',
        'E200':archive/'h75_frozen_confirmation200_v1/plan.json',
        'E201':archive/'h75_frozen_confirmation200_v1/status.json',
        'E202':archive/'h75_frozen_confirmation200_v1/root_endpoint_check.json',
        'E203':archive/'h75_frozen_confirmation200_v1/analysis/receipt.json',
        'E204':archive/'h75_frozen_confirmation200_v1/root_outcome_pairs.json',
        'E205':archive/'h75_frozen_confirmation200_v1/root_guard_behavior_comparison.json',
        'E206':archive/'h75_frozen_confirmation200_v1/root_common_cost_summary.json',
        'E207':archive/'h75_frozen_confirmation200_v1/analysis/original_task_rescoring/receipt.json',
        'E208':archive/'h75_frozen_confirmation200_v1/analysis/original_task_rescoring/results.json'}
    sources.update({key:dict(path=str(path.resolve()),sha256=digest(path),verification='AI artifact_checked',human_verification='not_recorded') for key,path in extra.items()})
    referenced=validate_evidence_references((HERE/'MANUSCRIPT.md').read_text(),sources)
    manifest=dict(scope='Current local research manuscript; previous20260912 direct-motor draft separately archived',
        manuscript_sha256=digest(HERE/'MANUSCRIPT.md'),builder_sha256=digest(__file__),
        evidence_cutoff='2026-10-28',submission_ready=False,human_approval='not_recorded',
        sources=sources,manuscript_evidence_references=referenced,claims_from_readback=data['claims'],
        detailed_evidence=dict(path=str(HERE/'EVIDENCE_DETAILS.md'),
            sha256=digest(HERE/'EVIDENCE_DETAILS.md'),
            role='Full historical evidence narrative; current research decisions remain in the shared ledger'),
        image_role='Architecture is a structural diagram; source-backed XY figures show saved actual trajectories, including nominal S-curve comparisons; rendering adds no experiments',
        figures={str(p.relative_to(HERE)):digest(p) for p in FIG.glob('control_architecture.*')},
        new_physics_steps=0,optimizer_updates=0)
    (HERE/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


def export_draft(pdf=False):
    import markdown
    source=(HERE/'MANUSCRIPT.md').read_text()
    image_inputs={}
    def embed(match):
        alt,relative=match.groups();path=(HERE/relative).resolve()
        if not path.is_file():raise ValueError('missing actual manuscript image '+str(path))
        image_inputs[str(path)]=digest(path)
        mime=mimetypes.guess_type(path)[0] or 'application/octet-stream'
        return f'![{alt}](data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()})'
    source=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',embed,source)
    body=markdown.markdown(source,extensions=['tables','fenced_code'])
    css='''@page{size:A4;margin:16mm}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:10pt;line-height:1.65;color:#243548;max-width:1100px;margin:24px auto}h1{font-size:19pt}h2{font-size:14pt;margin-top:1.4em}h3{font-size:11pt}h1,h2,h3{break-after:avoid}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:9pt}th,td{border-bottom:1px solid #ccd5de;padding:5px;vertical-align:top}tr{break-inside:avoid}blockquote{border-left:3px solid #647180;padding:5px 12px;background:#f2f5f7}pre{white-space:pre-wrap;background:#f2f5f7;padding:10px}a{color:#245b80}p{orphans:3;widows:3}@media print{body{margin:0;max-width:none}}'''
    html='<!doctype html><html lang="zh"><meta charset="utf-8"><title>STTW current research draft</title><style>'+css+'</style><body>'+body+'</body></html>'
    (HERE/'MANUSCRIPT.html').write_text(html)
    receipt=dict(images=image_inputs,html_sha256=digest(HERE/'MANUSCRIPT.html'),pdf_requested=pdf,
        disclosure='Offline local rendering only, no publication or source-data upload',physics_steps=0)
    if pdf:
        chrome=Path('/usr/bin/google-chrome')
        if not chrome.exists():raise RuntimeError('local Chrome unavailable; no dependency installation attempted')
        target=HERE/'MANUSCRIPT.pdf'
        with tempfile.TemporaryDirectory(prefix='sttw-paper-') as profile:
            proc=subprocess.run([str(chrome),'--headless','--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
                '--disable-background-networking',f'--user-data-dir={profile}','--no-pdf-header-footer',
                f'--print-to-pdf={target}',(HERE/'MANUSCRIPT.html').as_uri()],capture_output=True,text=True,timeout=90)
            if proc.returncode or not target.exists():raise RuntimeError('local PDF renderer failed; HTML retained')
        if target.read_bytes()[:5]!=b'%PDF-':raise ValueError('invalid PDF output')
        receipt.update(pdf_sha256=digest(target),pdf_bytes=target.stat().st_size,renderer='existing local headless Chrome')
    (HERE/'export_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group()
    group.add_argument('--figures-only',action='store_true');group.add_argument('--export-only',action='store_true')
    parser.add_argument('--pdf',action='store_true');args=parser.parse_args()
    if not args.export_only:control_diagram()
    evidence_manifest()
    if not args.figures_only:export_draft(pdf=args.pdf)
