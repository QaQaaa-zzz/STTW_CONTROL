# 原source/RB对称适配：输入准备，未训练

名义矩阵中两原策略各18次可达交权且腾空，均无严格落台。按任务书条件准备有界同待遇适配；不能用update2替代两者初始策略。

现有推理包只有Actor，已用原export_pendulum_skill从绑定原checkpoint分别导出Actor、Critic和归一化，CPU校验各16输入：最大动作误差2.83e−7，目标误差2.86e−6。0物理/0优化；原包不覆盖。权重留数据盘chapter3_skill_adaptation_inputs/bundles，本包仅身份/校验。

真实入口库18条来自一个名义上游根、三距离六接近；不能称18独立根。保存交权root/关节位置速度、原76维历史观测和上一动作，仅平移世界x使台沿与训练模板3.6m一致，姿态与速度不改。不是完整隐藏PhysX快照，未连接训练环境。

计划各25更新、每5轮配对评价，4096×24采样，累计4915200策略转移/39321600物理lane步；积分2.5ms、策略20ms，原架构/映射与Go2随机化保持。入口库reset及冻结9工况面板已准备，尚缺分阶段评价编排与资源分配；当前状态明确prepared_not_training_ready，不可直接启动。没有新增训练，不自动延长。

### Optional reset integration

`PhaseUEnv` now accepts `reachable_entry_bank` (NPZ) and
`reachable_entry_probability` (0.50 in the prepared plan). Unset preserves the
existing nominal/airborne reset path. One draw partitions reachable, airborne
(original unconditional 0.08), and nominal resets (remaining 0.42). Live joint
names determine bank column order. Recorded physical state, 76-D policy input,
three history frames and previous action are restored; episode events restart
from the translated position. Go2 randomization and delay buffers reset normally.
This is an observed-state restart, **not** restoration of hidden PhysX contact
or actuator-delay history. The first recorded policy input is the original
nominal observation; later frames use the configured noise/delay pipeline.

CPU checks: bank joint reorder/history/empty selection and existing Phase U
reward tests: 4 passed. Executed the actual reset method with a CPU plant stub
and 10,000 lanes: 5,001 reachable / 764 airborne / 4,235 nominal; disjoint RSI,
recorded history retained, episode counters cleared. These are wiring checks,
not physical validation. No optimizer update or extra GPU simulation was run.
The frozen retention/entry evaluation panel and its budget remain to be connected
before launching the two 25-update adaptations.

### Frozen DEV panel and physical configuration correction

`panel/panel_plan.json` freezes update0/5/10/15/20/25, per policy three
nominal-start speed conditions (1.8/2.0/2.2m/s,8s) and six full-chain conditions
(three distances,2m/s,alpha0/1,22s). Both policies together: at most108
evaluation rollouts/748800physics lane steps. Nominal retention is relative
to each original policy in this same Isaac domain, not the donor MuJoCo
benchmark or its full recovery distribution. Hard cases are seen DEV.

The historical training template still selected sectors/convexHull wheels;
prepared configs now inherit the actual current evaluation asset and cylinder
profile (radius.07m,width.05m,21slices). GPU replicated training layout and Go2
randomization remain. No physics was run with the mismatched template.
The runner reuses landing analysis and XY plotting for nominal jump-only cases.
Staged evaluation/selection orchestration remains unconnected; no PPO launched.

### Sequential execution and checkpoint selection

`run_skill_adaptation.py --plan .../panel/panel_plan.json --output NEW_RUN`
now connects training, exact checkpoint export, both frozen panels and selection.
Run with `PYTHONPATH=learning/src` and the Isaac Python environment after resource
allocation. It is **not launched**. The sequence performs each bounded25 training
run, exits its GPU process, then evaluates update0/5/10/15/20/25 sequentially.
This is evaluation of every fifth saved update after the budget, not immediate
online evaluation or early stopping within25. It avoids retaining the4096-lane
plant during single-lane evaluation. Source and RB get identical treatment.

The upstream periodic filename uses the zero-based loop index. New optional
`evaluation_checkpoint_every=5` saves `model_update_00005.pt` etc only after the
corresponding optimizer update, with accurate `iter` and `completed_updates`.
Existing historical files and sampled pre-update reward selection are unchanged.

Selection rejects loss of any baseline nominal full-window/support condition,
then maximizes complete hard-case successes; tied counts compare time only on
shared successful cases, retaining earlier updates on ties. If the original has
no positive nominal condition, the report explicitly marks the retention evidence
uninformative. No claim of broad donor capability preservation follows.

Seven CPU tests passed (reset bank, reward, selection including rejection of a
nominal regression and retaining an earlier best). CLI syntax checked. No actual
PPO or end-to-end orchestration run yet; TensorBoard/notifications and GPU
allocation remain launch-time work. Frozen task/data/checkpoint evidence must
still be reviewed after execution.
