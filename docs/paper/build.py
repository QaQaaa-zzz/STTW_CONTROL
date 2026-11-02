"""Build the current task-specified STTW manuscript from local evidence only."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "middle_reward": ROOT / "learning/src/sttw_control/path_command_reward.py",
    "middle_config": ROOT / "learning/configs/h75_low_speed_mixed_finetune_200.json",
    "lower_reward": ROOT / "learning/src/sttw_control/lower_command.py",
    "lower_config": ROOT / "learning/configs/lower_domain_finetune.json",
    "lower_selection": ROOT / "docs/results/sttw-tolerance-20261101/lower_base/selection.json",
    "lower_summary": ROOT / "docs/results/sttw-tolerance-20261101/lower_base/summary.json",
    "nominal_rows": HERE / "evidence/nominal_five_alpha_lower_selection.json",
    "nominal_aggregation": HERE / "evidence/nominal_five_alpha_summary.json",
    "nominal_aggregation_code": HERE / "analysis/nominal_five_alpha.py",
    "disturbance_comparison": ROOT / "docs/results/sttw-tolerance-20261101/cross_speed/comparison.json",
    "original_xy": ROOT / "docs/results/sttw-tolerance-20261101/lower_base/bank0_low_single_left_xy.png",
    "task_screen_config": ROOT / "learning/configs/task_preference_nominal40.json",
    "task_screen_diagnostic": ROOT / "docs/results/sttw-tolerance-20261101/task_preference_nominal40/task_preference_diagnostic.json",
    "task_screen_rows": ROOT / "docs/results/sttw-tolerance-20261101/task_preference_nominal40/results.json",
    "task_screen_code": HERE / "analysis/task_preference_screen.py",
    "task_score_config": ROOT / "learning/configs/task_preference_task40.json",
    "task_score_rows": ROOT / "docs/results/sttw-tolerance-20261101/task_preference_task40/task_scores.json",
    "task_score_native": ROOT / "docs/results/sttw-tolerance-20261101/task_preference_task40/results.json",
    "task_score_code": HERE / "analysis/task_preference_task_score.py",
    "middle_comparison_h75": ROOT / "learning/configs/task_preference_middle_h75.json",
    "middle_comparison_h75_extension": ROOT / "learning/configs/task_preference_middle_h75_extension.json",
    "middle_comparison_w400": ROOT / "learning/configs/task_preference_middle_w400.json",
    "middle_comparison_w400_extension": ROOT / "learning/configs/task_preference_middle_w400_extension.json",
    "middle_comparison_u325": ROOT / "learning/configs/task_preference_middle_u325.json",
    "middle_comparison_four_routes": ROOT / "docs/results/sttw-tolerance-20261101/frozen_middle_comparison/full_route_comparison.json",
    "middle_comparison_two_routes": ROOT / "docs/results/sttw-tolerance-20261101/frozen_middle_comparison/comparison.json",
    "middle_comparison_index": ROOT / "docs/results/sttw-tolerance-20261101/frozen_middle_comparison/README.md",
    "middle_comparison_code": HERE / "analysis/compare_full_route_models.py",
    "middle_comparison_two_route_code": HERE / "analysis/compare_frozen_middles.py",
    "middle_training_log": Path("/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_low_speed_mixed_finetune_200_v1/metrics.jsonl"),
    "middle_training_review": ROOT / "docs/results/sttw-tolerance-20261101/training_review/training_review.json",
    "middle_training_review_code": HERE / "analysis/review_middle_training.py",
    "task_combination_h75_config": ROOT / "learning/configs/task_preference_holdout_h75.json",
    "task_combination_u170_config": ROOT / "learning/configs/task_preference_holdout_u170.json",
    "task_combination_results": ROOT / "docs/results/sttw-tolerance-20261101/task_preference_holdout/holdout_comparison.json",
    "task_combination_code": HERE / "analysis/compare_holdout_middles.py",
    "geometric_rule_dev_selection": ROOT / "docs/results/sttw-tolerance-20261101/traditional_rule/selection.json",
    "geometric_rule_dev_code": HERE / "analysis/select_traditional_rule.py",
    "geometric_rule_holdout_config": ROOT / "learning/configs/task_traditional_rule_holdout.json",
    "geometric_rule_comparison": ROOT / "docs/results/sttw-tolerance-20261101/traditional_rule/comparison.json",
    "geometric_rule_comparison_code": HERE / "analysis/compare_traditional_rule.py",
    "signed_pulse_u170_config": ROOT / "learning/configs/task_preference_disturbance_u170.json",
    "signed_pulse_rule_config": ROOT / "learning/configs/task_preference_disturbance_rule.json",
    "signed_pulse_comparison": ROOT / "docs/results/sttw-tolerance-20261101/task_preference_disturbance/comparison.json",
    "signed_pulse_comparison_code": HERE / "analysis/compare_task_disturbance.py",
    "physical_domain_rescore": ROOT / "docs/results/sttw-tolerance-20261101/physical_domain_rescore/domain_task_rescore.json",
    "physical_domain_rescore_code": HERE / "analysis/rescore_physical_domain.py",
    "physical_domain_task_rule_config": ROOT / "learning/configs/task_preference_physical_domain_rule.json",
    "physical_domain_task_u170_config": ROOT / "learning/configs/task_preference_physical_domain_u170.json",
    "physical_domain_task_comparison": ROOT / "docs/results/sttw-tolerance-20261101/task_preference_physical_domain/comparison.json",
    "physical_domain_task_comparison_code": HERE / "analysis/compare_task_domains.py",
    "right_angle_u170_config": ROOT / "learning/configs/task_right_angle_u170.json",
    "right_angle_rule_config": ROOT / "learning/configs/task_right_angle_rule.json",
    "right_angle_comparison": ROOT / "docs/results/sttw-tolerance-20261101/task_right_angle/comparison.json",
    "right_angle_comparison_code": HERE / "analysis/compare_right_angle.py",
    "supplement_generator": HERE / "analysis/make_supplement.py",
    "article_figure_code": HERE / "analysis/make_article_figures.py",
    "article_figure_sources": HERE / "figures/article_figure_sources.json",
    "earlier_nominal_screen": Path("/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_nominal_five_alpha_v1/analysis/README.md"),
    "earlier_tight_turn": Path("/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_low_speed_tight_turn_v1/analysis/README.md"),
}
SOURCES.update({f"xy_{path.stem}": path for path in sorted((ROOT / "docs/results/sttw-tolerance-20261101/lower_base").glob("*xy.png"))})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--engine", default="tectonic")
    args = parser.parse_args()
    subprocess.run([sys.executable, str(HERE / "analysis/make_supplement.py")], check=True, cwd=ROOT)
    source = (HERE / "MANUSCRIPT.tex").read_text()
    figures = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", source)
    supplement = (HERE / "SUPPLEMENT.tex").read_text()
    supplement_figures = re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", supplement)
    missing = [str(HERE / path) for path in figures + supplement_figures if not (HERE / path).exists()]
    if missing:
        raise ValueError("missing manuscript figures: " + ", ".join(missing))
    if args.pdf:
        subprocess.run([args.engine, "MANUSCRIPT.tex", "--outdir", ".", "--keep-logs"], check=True, cwd=HERE)
        subprocess.run([args.engine, "SUPPLEMENT.tex", "--outdir", ".", "--keep-logs"], check=True, cwd=HERE)
    manifest = {
        "scope": "Task-specified preference LaTeX working article",
        "manuscript_sha256": digest(HERE / "MANUSCRIPT.tex"),
        "supplement_sha256": digest(HERE / "SUPPLEMENT.tex"),
        "builder_sha256": digest(__file__),
        "submission_ready": False,
        "human_verification": "not_recorded",
        "data_role": "development model selection and prospective exact nominal combinations; later signed-force, physical-domain and 90-degree panels are exploratory",
        "sources": {key: {"path": str(path), "sha256": digest(path), "verification": "AI artifact_checked; human pending"} for key, path in SOURCES.items()},
        "external_source": {"Huo_2025": "https://doi.org/10.1016/j.robot.2025.104996", "access": "local user-supplied full article"},
        "image_inputs": {path: digest(HERE / path) for path in figures},
        "supplement_image_inputs": {path: digest(HERE / path) for path in supplement_figures},
        "figures_checked": len(figures) + len(supplement_figures),
        "physics_steps": 0,
        "optimizer_updates": 0,
    }
    (HERE / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {"manuscript_sha256": manifest["manuscript_sha256"], "evidence_manifest_sha256": digest(HERE / "evidence_manifest.json"), "pdf_requested": args.pdf, "disclosure": "Local LaTeX compilation; no manuscript or unpublished data sent to external service"}
    if args.pdf:
        target = HERE / "MANUSCRIPT.pdf"
        receipt.update(pdf_sha256=digest(target), pdf_bytes=target.stat().st_size)
        supplement_target = HERE / "SUPPLEMENT.pdf"
        receipt.update(supplement_pdf_sha256=digest(supplement_target), supplement_pdf_bytes=supplement_target.stat().st_size)
    (HERE / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
