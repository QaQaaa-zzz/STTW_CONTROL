# Fixed-Actor path repair diagnostic — 2026-10-12

![All four case local XY](DETAIL_XY.png)

[Complete20s actual/reference XY, all cases and methods](ALL_XY.png)

**Fixed Actor interventions only; no training update and no natural-overlimit recovery.**

Execution ACCOUNTING state: **error**, retained unchanged. All eight physical traces exist; straight2.6 strict prefix failed at 2e-6. Its latefade outcome is descriptive, not accepted causal qualification.

Active lower ticks: 32000/32000. Training transitions: 0.

| Intervention/case | Roll rad | Path peak m | Primary normalized error | Final hold | Physical case gates | Strict prefix |
|---|---:|---:|---:|---|---|---|
| left90_R2_2.6_beta0.75_backoff0 | 0.296432 | 0.580355 | 0.930 | True | False | N/A (intervention from reset) |
| right90_R2_2.6_beta0.75_backoff0 | 0.335745 | 0.368344 | 0.614 | True | False | N/A (intervention from reset) |
| left90_R2_2.6_beta0.5_backoff0 | 0.350525 | 0.254407 | 1.180 | True | False | N/A (intervention from reset) |
| right90_R2_2.6_beta0.5_backoff0 | 0.354089 | 0.216348 | 0.666 | True | False | N/A (intervention from reset) |
| left90_R2_2.6_beta0.5_backoff0.05 | 0.338576 | 0.246675 | 1.939 | True | False | N/A (intervention from reset) |
| right90_R2_2.6_beta0.5_backoff0.05 | 0.337772 | 0.215084 | 1.415 | True | False | N/A (intervention from reset) |
| straight_2.0_late_fade | 0.048555 | 0.112573 | 0.734 | False | False | True |
| straight_2.6_late_fade | 0.113928 | 0.291044 | 1.900 | False | False | False |

Turn-repair support gate: **False**. Ordinary latefade support gate: **False**.
Conditional training decision: **Stop automatic training under declared diagnostic gate**.

[left90_R2_2.6: reward/error/detail](left90_R2_2.6_diagnostic.png) · [command chain](left90_R2_2.6_command_chain.png) · [all signed components](left90_R2_2.6_reward_components.png) · [cumulative components](left90_R2_2.6_cumulative_components.png)
[right90_R2_2.6: reward/error/detail](right90_R2_2.6_diagnostic.png) · [command chain](right90_R2_2.6_command_chain.png) · [all signed components](right90_R2_2.6_reward_components.png) · [cumulative components](right90_R2_2.6_cumulative_components.png)
[straight_2.0: reward/error/detail](straight_2.0_diagnostic.png) · [command chain](straight_2.0_command_chain.png) · [all signed components](straight_2.0_reward_components.png) · [cumulative components](straight_2.0_cumulative_components.png)
[straight_2.6: reward/error/detail](straight_2.6_diagnostic.png) · [command chain](straight_2.6_command_chain.png) · [all signed components](straight_2.6_reward_components.png) · [cumulative components](straight_2.6_cumulative_components.png)

[Calculated per-case gates and pair decision](metrics.json) · [Offline source](analyze.py)

[All-channel strict prefix audit](prefix_audit.json). Both new traces apply exactly the original Actor latent before10s; low-speed trace matches all original columns exactly. High-speed trace differs in measured motion before intervention despite identical batch shape and frozen source. Numerical compilation/rounding is a hypothesis, not an established root cause. Tolerance was not relaxed and no physical retry was performed.

[Original NPZ, exact Actor/config/bank identities and execution accounting](/home/qy/STTW_CONTROL/runs/worktrees/path-safety-repair/runs/path_safety_repair_20261012/diagnostic)

Original six-case parent and paired200 evidence remain available in [prior report](/home/qy/STTW_CONTROL/runs/worktrees/path-safety-budget-200/docs/evidence/path_safety_preference_v1/paired200_analysis/INDEX.md).

All cases use original speed/path requests. .05chi speed backoff may fail original speed-primary gate; it does not change the gate. No incomplete bins, shifted references or zero-padded failed tails are accepted. One historical DEV seed, no holdout claim.
