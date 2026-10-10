# No-alpha local lower: 200-update results

## Actual / issued tracking and XY

[All seven matched XY/tracking panels](INDEX.md). Original issued stream is identical for candidate and frozen R196; each2000 ticks, bank3,10s,200Hz. XY is descriptive, not a new path reward.

## Physical result

Formal200/200,26,214,400 control transitions,200 accepted updates. Best=update200, SHA `5d2cf35739f81429b7f6cbff39066dc1ad6d72f939dd8524e955e0609ef37df4`; random training-reward candidate=update164, separate. Best was reloaded; repeated physical traces differ by at most0.0001212m XY /0.00002908rad steering and counts are unchanged. Both full datasets retained.

Score tuple = physical failures / working-limit failures / primary tracking failures / final-hold failures / normalized Q (lower is better).

|update|score tuple|qualified|
|---|---|---|
|100|[0, 0, 6, 3, 6.242980957851922]|False|
|150|[0, 0, 6, 3, 4.458281985402324]|False|
|200|[0, 0, 5, 2, 2.8460162728627396]|False|

R196 same protocol: `[0, 0, 4, 0, 3.506210242645785]`; no overall new-controller dominance. New complete qualification2/7 (straight speed-change, negative turn-return); R1963/7. Zero physical/working failures in this panel does not qualify local tracking.

|case|new stable v RMSE m/s|R196 v|new stable delta RMSE rad|R196 delta|new small delta RMSE|R196 small delta|new primary/final fail|
|---|---|---|---|---|---|---|
|straight_speed_change|0.03095|0.01863|0.00274|0.00293|N/A|N/A|False/False|
|steady_positive|0.06042|0.03945|0.01932|0.00520|N/A|N/A|True/True|
|steady_negative|0.05333|0.05186|0.02523|0.00414|N/A|N/A|True/True|
|return_positive|0.05802|0.03941|0.01435|0.01810|N/A|N/A|True/False|
|return_negative|0.04826|0.05644|0.01889|0.02211|N/A|N/A|False/False|
|post_return_small_positive|0.05023|0.05857|0.02027|0.02172|0.0069479080848395824|0.014098932035267353|True/False|
|post_return_small_negative|0.05972|0.04347|0.01605|0.01826|0.00794238317757845|0.014544850215315819|True/False|

Negative steady tracking has a continuous steering error>.02rad from2.165 to10s (7.835s). The new small-positive/negative windows improve over R196 to.006948/.007942rad versus.014099/.014545rad, but their whole-case ordinary tracking still fails. Prior B0[6.315,8.745) mismatch has not been retested: local qualification gate remains closed, so no claim it is eliminated.

## Training and extension

Q150→200 improves36.16%; primary failures6→5 and final-hold failures3→2. Last25 complete episodes cost improves10.4% versus preceding25; all four families below improve cost/speed/steer. These are descriptive changing-policy training episodes, not paired fixed evaluations.

|family|episodes151–175 /176–200|mean cost/tick before→after|speed RMSE before→after|steer RMSE before→after|
|---|---|---|---|---|
|ordinary|391/379|0.001264→0.001139|0.02523→0.02373|0.02152→0.02046|
|turn_return|534/563|0.003126→0.002869|0.02844→0.02809|0.04158→0.03897|
|post_return_small|374/356|0.003366→0.002969|0.02904→0.02818|0.04266→0.03898|
|reversal|237/238|0.005689→0.004919|0.03125→0.03078|0.05989→0.05479|

Continue one authorized100-update block to total300, restoring update200 full Actor/Critic/Adam/std/RNG and512-environment physics/ESO/actuator/history/commands/episode statistics. No reset or new learner initialization. Extra13,107,200 transitions; total39,321,600, hard cap400/52,428,800. Numeric validation250/300; reuse R196 cache. Preserve stage200best/last/archive. At300 reload physical best then stop for evidence review.

No original mismatch replay, adapter registration/installation, frozen-upper comparison or upper training in this stage because local gate is not passed. Frozen upper best0added200/best1added250 identities remain protected.

Live TensorBoard: http://localhost:6006 (`formal/.`); exact launch PID/command in `../resume_launch_0200.json`. User preference: visible CMD output, no ongoing Codex polling; stage/completion desktop notifications remain.

## Training coverage caveat

Fixed steady reference values are{'steady_positive': 0.12092247605323792, 'steady_negative': -0.12092247605323792}; the ordinary training family has abs steer≤.10rad. Dynamic turn families hold larger turns at low1.7–2.1m/s, so prolonged2.6m/s/approximately±.122rad steady tracking is not directly covered by the ordinary target support. These references remain within declared global core ranges and acceptance remains unchanged; this is a joint-distribution coverage gap, not proof of the observed error mechanism or an exemption. The authorized extension retains the frozen distribution.
