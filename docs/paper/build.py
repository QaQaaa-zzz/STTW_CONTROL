"""Build the current manuscript and inventory its evidence inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    source = HERE / "MANUSCRIPT.md"
    text = source.read_text()
    figures = [HERE / name for name in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)]
    if args.pdf:
        subprocess.run([sys.executable, str(ROOT / "learning/cli/export_paper.py"),
                        "--source", str(source)], check=True)
    files = [source, *figures]
    if args.pdf:
        files += [source.with_suffix(".html"), source.with_suffix(".pdf")]
    evidence = {
        "E1": "/home/qy/STTW_CONTROL/runs/worktrees/tolerance-calibration/docs/results/sttw-tolerance-20261101/frozen_middle_comparison/README.md",
        "E2": "docs/evidence/h75_evidence_queue_20261110/final_results/README.md",
        "E3": "docs/evidence/h75_mathematical_verification/README.md",
        "E4": "docs/evidence/h75_theory_atlas/README.md",
        "E5": "docs/evidence/h75_state_alpha_scan/README.md",
    }
    manifest = {
        "source": "docs/paper/MANUSCRIPT.md",
        "scope": "Preference-conditioned reference coordination and off-road recovery",
        "inputs": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "evidence": evidence,
        "figures": len(figures),
        "new_physics_steps": 0,
        "new_optimizer_updates": 0,
        "cpu_test_command": "PYTHONPATH=learning/src python -m unittest discover -s learning/tests -p test_paper_preference_theory.py",
        "data_roles": "Historical flat DEV, off-road ablations, nominal model calculations, natural schedules; see Table 2.",
    }
    (HERE / "evidence_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    main()
