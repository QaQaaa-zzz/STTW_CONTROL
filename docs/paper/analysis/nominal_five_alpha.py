"""Recompute the manuscript's descriptive five-alpha nominal table."""
import hashlib
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "evidence/nominal_five_alpha_lower_selection.json"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    rows = [row for row in json.loads(SOURCE.read_text()) if row["model"] == "last200"]
    contexts = sorted({(row["bank"], row["route"]) for row in rows})
    if len(rows) != 40 or len(contexts) != 8 or any(sum(row["alpha"] == alpha and (row["bank"], row["route"]) == context for row in rows) != 1 for context in contexts for alpha in ALPHAS):
        raise ValueError("expected eight matched contexts with five unique alpha rows each")
    means = []
    for alpha in ALPHAS:
        group = [row for row in rows if row["alpha"] == alpha]
        means.append({
            "alpha": alpha,
            "n": len(group),
            "arrived": sum(row["arrived"] for row in group),
            "survived": sum(row["survived"] for row in group),
            "turn_speed_mean_m_s": sum(row["turn_speed_mean_m_s"] for row in group) / len(group),
            "common_prefix_original_cross_rmse_mean_m": sum(row["common_prefix_original_cross_rmse_m"] for row in group) / len(group),
            "arrival_mean_s": sum(row["arrival_s"] for row in group) / len(group),
        })
    speed_monotone = cross_monotone = later_alpha1_arrival = 0
    for context in contexts:
        group = sorted((row for row in rows if (row["bank"], row["route"]) == context), key=lambda row: row["alpha"])
        speed_monotone += all(b["turn_speed_mean_m_s"] > a["turn_speed_mean_m_s"] for a, b in zip(group, group[1:]))
        cross_monotone += all(b["common_prefix_original_cross_rmse_m"] > a["common_prefix_original_cross_rmse_m"] for a, b in zip(group, group[1:]))
        later_alpha1_arrival += group[-1]["arrival_s"] > group[0]["arrival_s"]
    print(json.dumps({"source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(), "data_role": "previously inspected nominal DEV", "contexts": len(contexts), "episodes": len(rows), "means": means, "strict_speed_monotone_contexts": speed_monotone, "strict_cross_monotone_contexts": cross_monotone, "later_alpha1_arrival_contexts": later_alpha1_arrival}, indent=2))


if __name__ == "__main__":
    main()
