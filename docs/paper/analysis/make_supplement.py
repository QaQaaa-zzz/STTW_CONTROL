"""Render the article's full grouped simulation evidence as a LaTeX supplement."""
import json
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
RESULTS = PAPER.parent / "results/sttw-tolerance-20261101"
nominal = json.loads((RESULTS / "task_preference_holdout/holdout_comparison.json").read_text())
pulse = json.loads((RESULTS / "task_preference_disturbance/comparison.json").read_text())
domain = json.loads((RESULTS / "task_preference_physical_domain/comparison.json").read_text())
angle = json.loads((RESULTS / "task_right_angle/comparison.json").read_text())
training = json.loads((RESULTS / "training_review/training_review.json").read_text())

names = {
    "holdout_low_single_left_76_R1p45_v1p90": "Low single L",
    "holdout_low_S_right_72_R1p55_v2p00": "Low S R",
    "holdout_high_single_right_96_R3p15_v3p05": "High single R",
    "holdout_high_S_left_70_R3p10_v3p10": "High S L",
    "right_angle_low_left_90_R1p6_v2p0": "Low 90 L",
    "right_angle_low_right_90_R1p6_v2p0": "Low 90 R",
    "right_angle_high_left_90_R3p0_v3p05": "High 90 L",
    "right_angle_high_right_90_R3p0_v3p05": "High 90 R",
}


def table(caption, columns, headings, rows):
    lines = [r"\begin{longtable}{" + columns + "}", r"\caption{" + caption + r"}\\",
             r"\toprule", " & ".join(headings) + r" \\", r"\midrule", r"\endfirsthead",
             r"\toprule", " & ".join(headings) + r" \\", r"\midrule", r"\endhead"]
    lines += [" & ".join(map(str, row)) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    return "\n".join(lines)


out = [r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=19mm]{geometry}
\usepackage{booktabs,longtable,graphicx,amsmath,hyperref}
\hypersetup{hidelinks}
\setlength{\LTpre}{5pt}
\setlength{\LTpost}{9pt}
\begin{document}
\begin{center}
{\Large Supplementary simulation results}\\[4pt]
{\large Task-Specified Path--Speed Preference Control for an Autonomous Single-Track Robot}\\[5pt]
2026-11-02
\end{center}

\section*{Reading the evidence}
This supplement gives the route-level data omitted from the focused article. All scores use the same original world-frame path and destination, a 10~s observation window, and the frozen task contract. ``Goal'' means entry into the original 1~m destination disk; H10 means upright survival throughout. A narrow pass additionally needs a buffered path corridor and 8--10~s original-path return. A wide pass adds an actual mean turn speed of at least 85\% of the request. The 0.5~m circular buffer is an anchor-based vehicle-envelope proxy. Counts by route use two paired prepared states, not independent policy-training seeds.

U170 is one frozen preference-conditioned middle network. Each $\alpha\in\{0,0.25,0.5,0.75,1\}$ is an input to the same weights. H75 is a separate earlier checkpoint. The geometric rule uses no middle-network output; all matched rows share the lower balance stack and 10~Hz causal path reconnection. The new nominal task combinations were declared before execution. Signed pulses, physical-domain extrema and 90$^\circ$ routes were added after earlier outcomes had been inspected and are exploratory. Physical-domain and nominal panels have different prepared states and must be read separately.

\section{Training review}
The U170 continuation record spans updates 53--202. Its sampled training reward changes with the rollout distribution and is not a fixed-task performance curve. The source log records 150 updates, 2376 accepted minibatches, zero physical failures, 119 domain exits and three target-KL-shortened updates (155, 175 and 188). The frozen checkpoint is assessed by the executed task panels below. Figure~\ref{fig:training} shows the saved reward, domain-exit, KL and critic-loss trends.

\begin{figure}[ht]
\centering\includegraphics[width=.85\textwidth]{../results/sttw-tolerance-20261101/training_review/training_progress.png}
\caption{Sampled training diagnostics from the archived U170 continuation.}\label{fig:training}
\end{figure}
"""]

training_rows = []
for key in ("early", "middle", "late"):
    x = training["segments"][key]
    training_rows.append((key.title(), f"{x['first_update']}--{x['last_update']}",
                          f"{x['mean_sample_reward']:.4f}", f"{x['sample_reward_std']:.4f}",
                          x["domain_exits"], f"{x['mean_exact_kl']:.4f}"))
out.append(table("U170 training windows; reward values are mean per sampled step.",
                 "lrrrrr", ["Window", "Updates", "Reward", "SD", "Domain exits", "Exact KL"], training_rows))

out.append(r"""\section{New nominal task combinations}
Four exact low/high single/S route--speed combinations were run from prepared banks 2 and 5. Table~\ref{tab:nominal} reports every route and preference for both H75 and U170. Each row has two physical episodes; all 80 episodes reached the original goal and survived H10. The task completion columns add the path, return and speed tests stated above. Turn speed is the true vehicle-forward speed averaged over the declared turn interval. ``Max cross'' is maximum absolute original-path cross-track error before first goal entry.
""")
nominal_rows = []
for x in sorted(nominal["by_route"], key=lambda x: (x["model"], x["route"], x["alpha"])):
    nominal_rows.append((x["model"], names[x["route"]], f"{x['alpha']:g}",
                         f"{x['narrow']}/2", f"{x['wide']}/2", f"{x['mean_turn_speed_m_s']:.3f}",
                         f"{x['mean_pre_goal_max_cross_m']:.3f}"))
out.append(table(r"Nominal task outcomes by frozen checkpoint, route and preference.\label{tab:nominal}",
                 "llrrrrr", ["Model", "Route", r"$\alpha$", "Narrow", "Wide", "Speed", "Max cross"], nominal_rows))

out.append(r"""\section{Signed-force pulse extension}
A world-fixed signed 36~N force was applied at 2.8--2.95~s on the same four routes and banks. Table~\ref{tab:pulse} groups the full 48-case matrix by route, sign and method (two states per row). All 48 remained upright. The positive example in the article is the $+36$~N low S right and high single right subset. Table~\ref{tab:positive} exposes those four matched route--state contexts without aggregating away speed, path deviation, goal entry or the 8--10~s return error. The subset was identified after viewing the full matrix; the negative sign and other routes remain in Table~\ref{tab:pulse}.
""")
pulse_rows = []
for route in sorted({x["route"] for x in pulse["u170_rows"]}):
    for sign in ("-36", "+36"):
        for label, rows in (("Rule", pulse["rule_rows"]),
                            (r"U170 $\alpha=0$", [x for x in pulse["u170_rows"] if x["alpha"] == 0]),
                            (r"U170 $\alpha=1$", [x for x in pulse["u170_rows"] if x["alpha"] == 1])):
            group = [x for x in rows if x["route"] == route and x["case_id"].endswith("F" + sign)]
            pulse_rows.append((names[route], sign, label, f"{sum(x['original_goal_arrived'] for x in group)}/2",
                               f"{sum(x['narrow_success'] for x in group)}/2",
                               f"{sum(x['wide_success'] for x in group)}/2",
                               f"{sum(x['turn_speed_mean_m_s'] for x in group)/2:.3f}",
                               f"{sum(x['pre_goal_max_abs_original_cross_track_m'] for x in group)/2:.3f}"))
out.append(table(r"Full signed-pulse matrix grouped over two prepared states.\label{tab:pulse}",
                 "lllr rrrr".replace(" ", ""),
                 ["Route", "Force", "Method", "Goal", "Narrow", "Wide", "Speed", "Max cross"], pulse_rows))

positive_rows = []
for route in ("holdout_low_S_right_72_R1p55_v2p00", "holdout_high_single_right_96_R3p15_v3p05"):
    for bank in (2, 5):
        for label, rows in (("Rule", pulse["rule_rows"]),
                            (r"U170 $\alpha=0$", [x for x in pulse["u170_rows"] if x["alpha"] == 0]),
                            (r"U170 $\alpha=1$", [x for x in pulse["u170_rows"] if x["alpha"] == 1])):
            x = next(x for x in rows if x["route"] == route and x["bank"] == bank and x["case_id"].endswith("F+36"))
            positive_rows.append((names[route], bank, label, int(x["original_goal_arrived"]), int(x["wide_success"]),
                                  f"{x['turn_speed_mean_m_s']:.3f}", f"{x['pre_goal_max_abs_original_cross_track_m']:.3f}",
                                  f"{x['last_2s_original_cross_rmse_m']:.3f}"))
out.append(table(r"Individual matched cases in the illustrated $+36$~N positive subset.\label{tab:positive}",
                 "lrlrrrrr", ["Route", "Bank", "Method", "Goal", "Wide", "Speed", "Max cross", "Return RMS"], positive_rows))

out.append(r"""\section{Physical-domain and prepared-state transfer}
The two domain models were selected from an existing four-draw TEST set by the maximum and minimum wheel-friction values. Domain 0 used wheel friction 0.932 and chassis mass scale 0.975; domain 3 used 0.612 and 1.074. Each domain used its own complete prepared banks 0 and 6. Table~\ref{tab:domain} gives every domain/route/method grouping. All 48 physical episodes reached the original destination and survived H10. The new task protocol was applied after earlier outcomes were inspected, so these data show behavior in two sampled domains rather than an untouched generalization estimate.
""")
domain_rows = []
for d in (0, 3):
    for route in sorted({x["route"] for x in domain["rows"]}):
        for label in ("Rule", "U170 alpha=0", "U170 alpha=1"):
            group = [x for x in domain["rows"] if x["domain"] == d and x["route"] == route and x["method"] == label]
            method = "Rule" if label == "Rule" else r"U170 $\alpha=" + label[-1] + "$"
            domain_rows.append((d, names[route], method,
                                f"{sum(x['narrow_success'] for x in group)}/2",
                                f"{sum(x['wide_success'] for x in group)}/2",
                                f"{sum(x['turn_speed_mean_m_s'] for x in group)/2:.3f}",
                                f"{sum(x['pre_goal_max_abs_original_cross_track_m'] for x in group)/2:.3f}"))
out.append(table(r"Task transfer by physical domain, original route and method.\label{tab:domain}",
                 "rllrrrr", ["Domain", "Route", "Method", "Narrow", "Wide", "Speed", "Max cross"], domain_rows))

out.append(r"""\section{Smooth 90-degree turn experiment}
The added finite-radius original routes change heading by exactly $\pm90^\circ$ at the end of each turn; the source NPZ paths were independently checked. Low-speed requests use 2.0~m/s and radius 1.6~m; high-speed requests use 3.05~m/s and radius 3.0~m, each in left and right directions. Every method used the same two prepared states, original destination and task thresholds. All 48 episodes entered the original goal and survived H10. Table~\ref{tab:angle} gives the complete route/preference grouping. The rule completed both task scores in 8/8 contexts; U170 exhibits a consistent preference response while not exceeding that rule on these turns. Figure~\ref{fig:right} shows the right-turn trajectories omitted from the main article's preselected left-turn figure.
""")
angle_rows = []
for route in sorted({x["route"] for x in angle["u170_rows"]}):
    for label, rows in (("Rule", angle["rule_rows"]),) + tuple((r"U170 $\alpha=" + f"{a:g}" + "$", [x for x in angle["u170_rows"] if x["alpha"] == a]) for a in (0, .25, .5, .75, 1)):
        group = [x for x in rows if x["route"] == route]
        angle_rows.append((names[route], label, f"{sum(x['narrow_success'] for x in group)}/2",
                           f"{sum(x['wide_success'] for x in group)}/2",
                           f"{sum(x['turn_speed_mean_m_s'] for x in group)/2:.3f}",
                           f"{sum(x['pre_goal_max_abs_original_cross_track_m'] for x in group)/2:.3f}"))
out.append(table(r"All smooth 90$^\circ$ route/preference groupings.\label{tab:angle}",
                 "llrrrr", ["Route", "Method", "Narrow", "Wide", "Speed", "Max cross"], angle_rows))

out.append(r"""\begin{figure}[ht]
\centering
\includegraphics[width=.98\textwidth]{../results/sttw-tolerance-20261101/task_right_angle/xy/u170/bank2_right_angle_low_right_90_R1p6_v2p0_zero_fixed_candidate_xy.png}
\includegraphics[width=.98\textwidth]{../results/sttw-tolerance-20261101/task_right_angle/xy/u170/bank2_right_angle_high_right_90_R3p0_v3p05_zero_fixed_candidate_xy.png}
\caption{Right 90$^\circ$ turns at low and high requested speed, prepared bank 2: all five U170 preferences, original path, actual speed and roll. The corresponding rule and bank-5 trajectories are in the full result index.}\label{fig:right}
\end{figure}

\begin{figure}[ht]
\centering
\includegraphics[width=.98\textwidth]{../results/sttw-tolerance-20261101/task_preference_disturbance/xy/u170/bank2_holdout_low_S_right_72_R1p55_v2p00_t2.8_F+36_fixed_candidate_xy.png}
\includegraphics[width=.98\textwidth]{../results/sttw-tolerance-20261101/task_preference_disturbance/xy/u170/bank2_holdout_low_S_right_72_R1p55_v2p00_t2.8_F-36_fixed_candidate_xy.png}
\caption{Same low-speed S right route and prepared bank 2 under the two signed force pulses. The article's positive example uses the upper $+36$~N case; the lower $-36$~N case is retained for direction-specific comparison.}\label{fig:pulsesign}
\end{figure}

\section*{Complete case and raw-trajectory access}
The machine-readable companions retain every individual outcome, including non-arrivals and task failures: \href{../results/sttw-tolerance-20261101/task_preference_holdout/holdout_comparison.json}{nominal cases}, \href{../results/sttw-tolerance-20261101/task_preference_disturbance/comparison.json}{signed pulses}, \href{../results/sttw-tolerance-20261101/task_preference_physical_domain/comparison.json}{physical domains}, and \href{../results/sttw-tolerance-20261101/task_right_angle/comparison.json}{90-degree turns}. Their adjacent Markdown indices link all original/actual XY images and archived raw NPZ trajectories. The current paper package has no new policy training and no hardware results.
\end{document}
""")

(PAPER / "SUPPLEMENT.tex").write_text("\n".join(out))
