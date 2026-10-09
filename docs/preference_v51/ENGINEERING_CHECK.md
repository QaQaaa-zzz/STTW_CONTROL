# Preference V5.1 engineering check

- Source parent: `6a264f77468cc4090f8e79919027b333740e5dfa`
- Attachment ZIP SHA-256: `0fb16f2db889ffcd51de4833378c0ffd329586ebce50028476389978839f44b3`
- Focused V5.1 tests: 6 passed.
- Existing V5 reward/task/PPO tests: 74 passed, one pre-existing tensor conversion warning.
- Attachment `reward_delta_reference.py`: output exactly matches supplied `math_checks.json`.
- Final engineering run after logging/reporting changes: `runs/preference_v51_engineering_8x16x2_run02_20261009`, 8 environments, 16 policy steps, 2 accepted updates, 256 policy transitions, 1024 control ticks.
- Initialization: fresh Actor/Critic/Adam, zero Actor output layer, standard deviations `[0.30, 0.10]`, seed87.
- Reset isolation forced case: family3 at env9, e0 `-0.0439624190rad`; qpos/qvel unchanged; frozen lower initialized from actual pose; external evaluation rows give e0=0.
- Scope: engineering validation only. It does not establish primary tracking, roll compliance, heading recovery, or low ordinary intervention.
