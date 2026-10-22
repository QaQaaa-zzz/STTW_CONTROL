# Model-margin shaping implementation plan

Goal: test whether derived directional recovery reserve helps the existing two middle policies preserve disturbance recovery while retaining the frozen local300 and original task contracts. This is an experimental shaping hypothesis, not adoption of a safety predictor.

## Evidence and design decision

The outcome-window audit reconstructs all264 original labels. Among66 speed-transfer cases with valid pre-pulse prefixes,36 recover; static/dynamic frozen effort classifiers miss30/25 failures respectively, with0/4 conservative errors. Removing past failures does not validate the model. Accordingly, the earlier prerequisite of a validated task-success classifier is not met. We do not promote the model to a constraint. A separately declared shaping experiment can test utility despite this limitation; its failure must remain a valid result.

Alternatives considered: more scalar-threshold calibration cannot add missing task/actuator dynamics; a learned closed-loop predictor needs more independent states and introduces a separate model. The first experimental comparison instead tests the original analytic reserve directly, while original roll/path/speed costs continue to enforce task demand. No online shield, endpoint mixing, changed chi, or claimed force certificate.

## Frozen candidate definition

For the controller's frozen-speed scalar model, lambda=sqrt(a4), b=a1+a2/lambda, z=a2*delta/lambda+lambda*phi+roll_rate+dhat/lambda. Use current rear-wheel speed proxy, matching the controller. Let T=min(1 second, positive remaining task time), R=1, pulse duration0.1 second,

W=b^2*(1-exp(-2 lambda T))/(2 lambda), K=gamma*(1-exp(-lambda*0.1))/lambda.

Use gamma=0.815560907125473, the median of12 calibration-only first-tick mappings, and common E_b=0.030839085298575167 for both static/dynamic training arms. This is a normalization inherited from calibration, not a physical certified budget. Fixed gamma removes runtime probes and is an approximation. Full online coefficient/force mapping validity remains unproven.

When abs(z)<=sqrt(E_b W), directional model force margins are (sqrt(E_b W)-sign(K)z)/abs(K) and (sqrt(E_b W)+sign(K)z)/abs(K). Otherwise both are zero because there is no nominal-feasible connected interval. Static arm sets z=0; dynamic uses current z. Cost rate=10*clip(1-min(m_plus,m_minus)/6N,0,1). Six newtons is within the already evaluated amplitude range; no claim it is optimal. The same gamma, budget, duration, force target and coefficient apply to both arms. The symmetric minimum is intentional for unknown disturbance direction; it does not exploit a known one-sided force schedule.

No-cost arm leaves original reward unchanged. Disabled calculation bypasses invalid model inputs. Enabled nonfinite/degenerate W/K/horizon is an engineering UNKNOWN, not a safe margin; surface a fault rather than reward it. Finite but insufficient-budget states have zero margins and full cost. Optional component cap10 raises failure cost-rate bound from5080.2 to5090.2. Integrate once per0.005s lower tick, retaining four-tick0.02s policy aggregation. Do not introduce network observation dimensions in this first experiment.

## Implementation tasks

- [x] Pure JAX scalar margin/cost kernel with exact NumPy quadratic-reference tests, directional symmetry, insufficient budget, degenerate inputs, disabled bypass, vmap/JIT checks.
- [x] Current-state coefficient adapter matching controller._system; reward integration with optional component schema, all accumulators and failure bounds. Zero coefficient must preserve existing transitions/reward.
- [ ] Explicit weights-only new reward fork for all three arms from identical frozen Coverage u200 endpoint parents; actor and critic copied, optimizer reset identically. Preserve/check source hashes and original identity; no weakened resume checks. Freeze full run configs and seed/budget before any PPO.
- [ ] Real short numerical/accounting smoke, then matched training with checked live TensorBoard reward and registered Bark. Reuse adopted vmap/scan/cache. No production training has started under this document.
- [ ] Evaluate original task and pulse success, task error, nominal regression and incremental costs on a newly declared comparison panel, retaining current DEV for diagnostics. Fixed final update primary comparison avoids reward-scale-dependent best selection. Multi-seed replication and roll-only cost-matched control required before claiming dynamic-specific paper benefit.

Training step count and resources will be fixed in concrete launch manifests after integration/smoke, not guessed here. This document freezes the first cost candidate, not an already executable campaign. Current2.8 DEV is already used and cannot become unseen TEST. A poor candidate is rejected; do not redefine task thresholds to make it succeed.

Terminal-window integration rule: do not initiate a hypothetical full0.1s pulse when less than0.1s remains. The adapter must mark this cost inactive for that terminal window (and preserve all original task costs); the standalone enabled kernel reports UNKNOWN for that invalid query. It must not silently shorten the declared pulse.

Integration verification: b2dd78e,29 CPU tests; shaping_integration_smoke_v1 completed1600 lower ticks with frozen actors, exact zero-weight match in this smoke, valid accounting and bounded active dynamic cost. See docs/evidence/shaping_integration_smoke_20261022/REPORT.md. Learner fork and production campaign remain pending.
