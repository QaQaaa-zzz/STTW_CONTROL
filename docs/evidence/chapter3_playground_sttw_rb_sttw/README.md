# 第三章：当前结果与大纲进度

[完整 E2 实际/参考轨迹与逐例结果](p4_joint_test/RESULTS.md) · [完整 E3 实际/参考轨迹与逐例结果](p4_model_test/RESULTS.md)

当前依据：[《STTW 第三章：Codex 实施任务书》2026-11-15](taskbook_20261115/STTW_第三章_Codex任务书.md)。用户已明确替换旧DOCX；旧方案与实验保留为历史证据。目标是冻结技能的全链执行预测、后续可行性影响前段选择，以及落地实测更新。优先有限路线与已验证离散速度，暂停扩大连续地面CEM。

## 近期结果精简总览（2026-11-18）

已有STTW越野和适配技能完整串联成功证据；尚无自动α选择或完整RecoveryBridge方法的额外整程收益证明。不同模型、数据角色与分母不能混用。

|结果|实际证据|边界与入口|
|---|---|---|
|锁摆STTW越野|96/96组合初差、8/8连续路线通过|4cm最高脊坡、1.8m/s、α0；[全部XY与验收](../chapter3_playground_isaac_recovery/README.md)|
|适配shared225完整骑—跳—骑|已知DEV九α组合399/648最终成功；固定0→0为72/72|shared225来自PPO-only分支后适配，不等于完整RB；[全部条件及XY](taskbook_20261115/p5_alpha_grid/README.md)|
|条件地面预测|跳前/跳后两结构×三种子均完成1200更新|保留最佳/末轮与长时域反例，未认定可靠收敛；[跳前](taskbook_20261115/p5_alpha_models/phase0_continue600/RESULTS.md)、[跳后](taskbook_20261115/p5_alpha_models/phase2_continue600/RESULTS.md)|
|条件串联与入口暴露|648已知DEV：正确准入242→336，误准入41→56|召回改善但错误准入增加；[串联及诊断](taskbook_20261115/p5_alpha_models/conditional_chain/README.md)|
|有限α选择|4个已知DEV任务，固定α0与完整枚举均4/4|18查询预算全部9种候选排列保留，部分退步；[全部选择与XY](taskbook_20261115/p5_alpha_models/alpha_selection/README.md)|
|原source与完整RB|各36条：远岸接触22/30，交回1/0，最终0/0|map00 TRAIN原权重迁移，不能把远岸接触当成功；[72条XY及逐例](taskbook_20261115/p5_source_full_rb/RESULTS.md)|

当前P5未完成，P6落地实测更新与任务重配置未完成。原始轨迹、模型和训练日志保留在数据盘；本页精简结论，以下报告保留逐例失败与身份，不新增重复大附件。

## 历史基线（新任务书前）

|比较|全程完成|结论边界|
|---|---|---|
|E3：M0 / M1结构化修正 / M2直接学习|156/200 · 200/200 · 200/200|学习模型加校准裕量减少无计划；未证明假可行减少或M1独立优势|
|E2：贪心分段 / 强回溯分段 / 联合|104/200 · 200/200 · 200/200|联合与强回溯仅差约1.45ms，未证明联合优势|
|滚动规划DEV|物理12/12；状态覆盖11/12|落地后前轮速度历史超界仍未解决|

E2/E3同50张冻结新布局、四新初态；分别600方法条件、484/312条唯一物理轨迹。四个根跨图复用，按地图组统计。E2在查看E3结果后实施，算法/阈值沿用既有DEV，不能称三组预先同时冻结。M0的44次失败与贪心分段的96次失败均为无计划回退后的禁止障碍碰撞；完整失败保留。联合−强回溯平均用时差−0.00145s，地图bootstrap95%区间[−0.00627,0.00325]s。

## 新任务书阶段（旧报告编号不代表此处验收）

|阶段|当前状态与下一项|
|---|---|
|P0 冻结与复现|已完成身份冻结、600条件复算、真实smoke 1/1及2700候选诊断；[复现记录](taskbook_20261115/baseline_reproduction.md)|
|P1 完整快照与续接|完整snapshot/planner续接与地面100=40+60已实测；固定顺序数值版本通过有限重复基准，原生版本超界保留；[边界与证据](taskbook_20261115/p1_snapshots/RESULTS.md)|
|P2 真实跨阶段案例|原两图36格未找到机制；DEV地图48两根发现并冻结重复确认，配对视频/状态/占据图完成；[证据与范围](taskbook_20261115/p2_oracle_wall/README.md)|
|P3 单段世界模型|单段训练与诊断完成：RB三种子累计2400、短事件读出、真实GPU续接及决策误差齐备；保留固定时序更优/旧裕量无计划负结果；[证据与P4入口](taskbook_20261115/p3_world_model/README.md)|
|P4 全链预测规划|规定的实验与对照已收束；当前候选可靠性不达标。[逐项核对/标定选路](taskbook_20261115/p4_sequence/stage_review/README.md)：原始C2/2但标定后无计划，保留全部负结果；不以可靠部署作为阻止P5采集的额外门槛|
|P5 α与RB融合|九组合TRAIN/DEV采集、三种子条件地面模型、条件串联和同预算α选择已完成；原source/完整RB首图对照完成但均未完成任务。详见上表；匹配预测接口和最终比较仍未完成|
|P6 落地更新与任务重配置|保留历史与承诺前缀；C-R及任务组合泛化未完成|
|P7 可选LLM|仅P0–P6稳定且确需复杂任务结构时考虑|

每阶段完成代码、测试、实际运行和结论再进入下一阶段。不优先新PPO、连续轨迹优化或LLM；沿用共同成功标准与失败记录。

## 固定物理与证据入口

H75＋ECBC/ESO（无Local300）、RB225；STTW锁髋膝，RB独占四通道并按实测状态交回。implicitfast、摩擦0.8、物理1ms/底层5ms/策略20ms。碎石1–4mm、脊坡2/4cm、沟宽0.3m；沟前后各2m平地。跳前、跳后均有转弯。

- [地面恢复与越野证据](../chapter3_playground_isaac_recovery/README.md)
- [P1已有执行时域核查](p1_ground_model/execution_horizon/RESULTS.md)
- [E1已有DEV与负结果](e1_ground/forced_turn_hold/RESULTS.md)
- [滚动覆盖未通过的证据](p2_rolling_multimap_support/RESULTS.md)
- [预测模型及训练存档](p3_multimap_models/RESULTS.md) · [跳前续训比较](p3_approach_models/continuation/RESULTS.md)
- [方法与验收设计](../../CHAPTER3_EVENT_WORLD_MODEL.md) · [论文草稿](../../chapter3_paper/MANUSCRIPT.pdf)（已含E3；新E2尚待纳入）

原始权重、逐步NPZ和冻结配置在数据盘`STTW_runs/chapter3_playground_sttw_rb_sttw/`，未作为此次Git上传内容；远端报告含逐例数值、全部XY图和源数据路径。旧400行首页完整保留于Git提交`5e3fd90a`，各历史报告及失败记录保留。

取消三小时总限制；预计等待超过半小时暂停并等待“继续”。完成、暂停或停止均Bark；通知器使用`/usr/bin/python3`，以真实回执为准。

跳跃技能身份说明：[训练合同核对](taskbook_20261115/p5_snapshot_alpha/README.md#跳跃技能来源更正)。当前shared225初始化来自注册名rb_run3的PPO-only配方，工程名RB不等于完整RecoveryBridge方法；既有数值不变，P5方法比较须单独匹配。

P5条件预测接口与九组合覆盖入口：[进度/合同](taskbook_20261115/p5_alpha_grid/README.md)。九组合648条件已完成，成功矩阵72/55/16、72/53/14、63/43/11（每格72）；全部交回，失败均在驶离。条件输入与续接接口已实现、尚未训练；DEV保持原角色。
