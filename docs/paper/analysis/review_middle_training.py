"""Summarize the archived U170 continuation log without treating reward as task success."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: review_middle_training.py METRICS_JSONL OUTPUT_DIR")
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in source.read_text().splitlines()]
    if len(records) != 150 or [r["update"] for r in records] != list(range(53, 203)):
        raise ValueError("expected the complete U170 continuation update 53–202 log")
    update = np.array([r["update"] for r in records])
    reward = np.array([r["mean_step_reward"] for r in records])
    domain = np.array([r["domain_failures"] for r in records])
    kl = np.array([r["optimizer"]["exact_kl"] for r in records])
    value = np.array([r["optimizer"]["value"] for r in records])
    rolling = np.convolve(reward, np.ones(20) / 20, "valid")
    segments = {}
    for name, group in (("early", records[:40]), ("middle", records[55:95]), ("late", records[-40:])):
        values = np.array([r["mean_step_reward"] for r in group])
        segments[name] = dict(first_update=group[0]["update"], last_update=group[-1]["update"],
                              mean_sample_reward=float(values.mean()), sample_reward_std=float(values.std(ddof=1)),
                              physical_failures=sum(r["physical_failures"] for r in group),
                              domain_exits=sum(r["domain_failures"] for r in group),
                              mean_exact_kl=float(np.mean([r["optimizer"]["exact_kl"] for r in group])),
                              accepted_minibatches=sum(r["optimizer"]["accepted_minibatches"] for r in group))
    summary = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   updates=len(records), update_range=[int(update[0]), int(update[-1])],
                   total_physical_failures=sum(r["physical_failures"] for r in records),
                   total_domain_exits=int(domain.sum()),
                   accepted_minibatches=sum(r["optimizer"]["accepted_minibatches"] for r in records),
                   target_kl_shortened_updates=[r["update"] for r in records if r["optimizer"]["stop_reason"] == "target_kl"],
                   nonfinite_stops=sum(r["optimizer"]["nonfinite_stop"] for r in records),
                   segments=segments,
                   interpretation="The sampling distribution and scene mix vary. A better late sample-reward mean does not prove fixed-task convergence or safety.")
    (output / "training_review.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    axes[0, 0].plot(update, reward, color="#7298b6", lw=.8, label="Training batch mean")
    axes[0, 0].plot(update[19:], rolling, color="#1c4e80", lw=2, label="20-update rolling mean")
    axes[0, 0].set(ylabel="Mean reward / lower step", title="Sampled training reward")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].step(update, np.cumsum(domain), where="mid", color="#a64b3c")
    axes[0, 1].set(ylabel="Cumulative domain exits", title="Training rollout domain exits")
    axes[1, 0].plot(update, kl, color="#5f6e9b", lw=1)
    axes[1, 0].set(xlabel="Update", ylabel="Exact KL", title="PPO update size")
    axes[1, 1].plot(update, value, color="#786e91", lw=1)
    axes[1, 1].set(xlabel="Update", ylabel="Value loss", title="Critic loss", yscale="symlog")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle("U170 continuation: updates 53–202; changing sampled tasks, no fixed-task validation curve")
    fig.savefig(output / "training_progress.png", dpi=180)
    plt.close(fig)
    lines = ["# U170 training log review", "", "![Reward, domain exits, KL and critic loss](training_progress.png)", "",
             "The source is the archived 150-record continuation `metrics.jsonl`, updates 53–202. The plot uses sampled training batches; the rollout mix changes, so its reward curve is not a fixed-task validation curve.", "",
             "| Window | Updates | Mean reward/step | SD | Physical failures | Domain exits | Mean exact KL | Accepted minibatches |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in segments.items():
        lines.append(f"| {name} | {row['first_update']}–{row['last_update']} | {row['mean_sample_reward']:.4f} | {row['sample_reward_std']:.4f} | {row['physical_failures']} | {row['domain_exits']} | {row['mean_exact_kl']:.4f} | {row['accepted_minibatches']} |")
    lines += ["", f"Across all 150 records: {summary['total_physical_failures']} physical failures, {summary['total_domain_exits']} domain exits, {summary['accepted_minibatches']} accepted minibatches; target-KL shortened updates {summary['target_kl_shortened_updates']}, with {summary['nonfinite_stops']} nonfinite stops. The late mean improves over the middle window, but no fixed-task performance plateau is demonstrated. The frozen checkpoint is judged by physical task panels, not this training trace alone.", ""]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
