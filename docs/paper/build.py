"""Build the current task-specified STTW manuscript from local evidence only."""
import argparse
import base64
import hashlib
import json
import mimetypes
import re
import subprocess
import tempfile
from pathlib import Path

import markdown


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
    "earlier_nominal_screen": Path("/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_nominal_five_alpha_v1/analysis/README.md"),
    "earlier_tight_turn": Path("/media/qy/6A7E3DB07E3D75C3/STTW_CONTROL_archives/tolerance_calibration_disk_recovery_20261025/h75_friction08_low_speed_tight_turn_v1/analysis/README.md"),
}
SOURCES.update({f"xy_{path.stem}": path for path in sorted((ROOT / "docs/results/sttw-tolerance-20261101/lower_base").glob("*xy.png"))})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    source = (HERE / "MANUSCRIPT.md").read_text()
    links = re.findall(r"\]\(([^)]+)\)", source)
    local_links = [(HERE / link).resolve() for link in links if not link.startswith(("https://", "http://"))]
    missing = [str(path) for path in local_links if not path.exists()]
    if missing:
        raise ValueError("missing manuscript evidence links: " + ", ".join(missing))
    image_inputs = {}

    def embed(match):
        alt, relative = match.groups()
        path = (HERE / relative).resolve()
        image_inputs[str(path)] = digest(path)
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        payload = base64.b64encode(path.read_bytes()).decode()
        return f"![{alt}](data:{mime};base64,{payload})"

    rendered = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed, source)
    body = markdown.markdown(rendered, extensions=["tables", "fenced_code"])
    css = "@page{size:A4;margin:16mm}body{font-family:'Noto Sans CJK SC',sans-serif;font-size:10pt;line-height:1.55;color:#243548;max-width:1000px;margin:24px auto}h1{font-size:18pt}h2{font-size:13pt;margin-top:1.3em}h3{font-size:10.5pt}h1,h2,h3{break-after:avoid}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:8.5pt;break-inside:avoid}th,td{border-bottom:1px solid #ccd5de;padding:4px;vertical-align:top}tr{break-inside:avoid}a{color:#245b80}p{orphans:3;widows:3}@media print{body{margin:0;max-width:none}}"
    html = "<!doctype html><html lang='en'><meta charset='utf-8'><title>STTW path-speed preference draft</title><style>" + css + "</style><body>" + body + "</body></html>"
    (HERE / "MANUSCRIPT.html").write_text(html)
    manifest = {
        "scope": "Task-specified preference working manuscript",
        "manuscript_sha256": digest(HERE / "MANUSCRIPT.md"),
        "builder_sha256": digest(__file__),
        "submission_ready": False,
        "human_verification": "not_recorded",
        "data_role": "nominal DEV, bounded frozen task-score validation and related-source middle selection; tuned-rule and independent-domain evidence pending",
        "sources": {key: {"path": str(path), "sha256": digest(path), "verification": "AI artifact_checked; human pending"} for key, path in SOURCES.items()},
        "external_source": {"Huo_2025": "https://doi.org/10.1016/j.robot.2025.104996", "access": "publisher abstract and section snippets; numerical tables unverified"},
        "image_inputs": image_inputs,
        "links_checked": len(links),
        "physics_steps": 0,
        "optimizer_updates": 0,
    }
    (HERE / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {"manuscript_sha256": manifest["manuscript_sha256"], "html_sha256": digest(HERE / "MANUSCRIPT.html"), "evidence_manifest_sha256": digest(HERE / "evidence_manifest.json"), "pdf_requested": args.pdf, "disclosure": "Local offline rendering; no manuscript or unpublished data sent to external service"}
    if args.pdf:
        target = HERE / "MANUSCRIPT.pdf"
        with tempfile.TemporaryDirectory(prefix="sttw-paper-") as profile:
            proc = subprocess.run(["/usr/bin/google-chrome", "--headless", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--disable-background-networking", f"--user-data-dir={profile}", "--no-pdf-header-footer", f"--print-to-pdf={target}", (HERE / "MANUSCRIPT.html").as_uri()], capture_output=True, text=True, timeout=90)
        if proc.returncode or target.read_bytes()[:5] != b"%PDF-":
            raise RuntimeError("local PDF rendering failed")
        receipt.update(pdf_sha256=digest(target), pdf_bytes=target.stat().st_size)
    (HERE / "export_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
