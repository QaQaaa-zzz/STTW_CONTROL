# R196 Preference V5 implementation

User-authorized contract: [attachment](attachment/STTW_R196_PreferenceV5_Codex.md), both endpoint JSON specifications. Actual implementation starts at `e1a6cc1f5a57b90b780207a69ce5bfa543cc1642`; report `bb71425` is evidence only. Current research facts are maintained under `/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md`.

| V4 | V5 actual semantics |
|---|---|
| Frozen R196 + governed-centered ECBC/ESO | Preserved exactly, lower alpha1; all physics/authority unchanged |
| Independent endpoints | Independent fresh Actor/Critic/Adam/RNG; seed83; no V4 upper |
| Actual motion and heading recovery | Preserved; ordinary/recovery speed deadband .03, ordinary steer .003 |
| No primary reference priority term | Conflict nonrecovery priority cost4 with steer .01/speed .05 tolerance |
| No request compatibility cost | Governed steer / existing ECBC q(governed speed), cost only |
| Roll component cap100 | Roll cap900, sum independent caps1680; no global clip |
| Stronger motion smoothing | Rate .03 and acceleration .01, conflict/debt rho floor .2 |
| Temporal regularization from beginning | First20 effective updates0; ramp to.002 at40, both endpoints settled |
| Equal initial latent std | Speed .30, steer .10; per-channel bounds |
| 16s finite task everywhere | Stage1 true5s, Stage2 true16s; Critic remaining seconds /16 |
| Full scenarios during initial training | Stage1 left/right only at20/40; gate required for Stage2 |

Resolved authoritative configs are written per endpoint/stage under `runs/preference_v5_20261009`. Original JSONs are retained verbatim; compatibility adapter fields are explicit in `smooth_command_config.py`. Retained legacy wall bookkeeping values are disabled by `wall_limits_enabled=false` and impose no cutoff.

Stage1: max40 batches each,512x128=2,621,440 policy transitions per endpoint. Stage2 only after both left/right gates: at most80 more each, total120. Failed stage1 stops expansion; nonfinite/three successive hard KL rejects stop. No automatic250/500 and no automatic push. One smoke8x16x2 adds256 policy transitions/1024 control ticks; evaluation ticks and compilation are separately recorded. Complete episode statistics are separate from rollout/mode summaries; partial episodes at stage boundary are excluded and logged.

Execution: `learning/cli/train_smooth_command.py --preference-v5 --output <fresh-path>`; engineering smoke adds `--smoke`. Root stage1 physical/control figures precede reward interpretation. Stage1 passing is a pilot coordination gate, not full10s/16s direction completion.
