# Single-alpha development candidate selection

Selected update 325 from the completed V3 curriculum400 run. This is one alpha-conditioned network with frozen local300; no policy-action blending. Checkpoint identities are pinned in candidate_selection.json. Production pointers and original qualification labels are unchanged.

## Evidence

- 30/30 cases reached the goal and met the original common recovery checks, with zero physical/domain failures. Maximum final-second errors: path 0.06424 m, heading 0.01044 rad, speed 0.01787 m/s. Maximum roll 0.29764 rad.
- In every one of the four turning cases, speed RMSE decreased at each alpha step (0, .25, .5, .75, 1). Endpoint speed benefits were 0.2687–0.4949 m/s.
- Alpha0 endpoint path RMSE was better in three turns; the fourth differed by only 0.0008 m in alpha1's favor. This negligible difference is accepted for development, consistent with the user's tolerance for small errors.
- Intermediate alphas sometimes have lower path RMSE than alpha0. The useful behavior is selectable speed/path trade-off, not deliberate degradation to force a monotone error curve.
- Visual inspection of update325 and update400 XY_TURN_EXIT confirms separated turn trajectories and improved terminal recovery in update325. Straight panels are too compressed for quantitative visual inference; raw trajectory metrics above supply that evidence.
- Original strict endpoint-primary qualification remains false: worst alpha0 turn path RMSE 0.1464 m and alpha1 turn speed RMSE 0.1310 m/s. Do not relabel strict qualification as passed.

## Comparison and next action

Under the previously examined 0.20 m final-path / 0.33 rad roll practical sensitivity, update325 and update375 pass 30/30; update400 passes 24/30. Worst final path error grows from 0.064 m to 0.185 m to 0.240 m. Further training is therefore not justified solely by the later update number.

Use update325 as the current development candidate and retain the later checkpoints for comparison. No additional PPO is launched. These six routes/speeds times five constant alphas are development evidence, not unseen-route generalization or online alpha-switching validation. Dynamic preference switching remains unvalidated. Resume primary effort on the ongoing tolerance comparison once this selection is recorded; revisit alpha training only if a concrete application test reveals a deficiency.
