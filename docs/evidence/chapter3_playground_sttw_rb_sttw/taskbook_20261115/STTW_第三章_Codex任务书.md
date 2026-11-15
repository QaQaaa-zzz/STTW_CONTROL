# STTW 第三章：Codex 实施任务书

版本：2026-11-15。本文是待执行的实现方案，不是已完成实验报告。

## 0. 给执行者的总指令

请在 STTW_CONTROL 仓库现有第三章代码上增量实现，不另起一套仿真系统。先核对最新分支、资产和接口，然后按 P0–P7 依次推进。每一阶段都完成代码、测试、实际运行和结论记录，再进入下一阶段。发现问题时按本文诊断顺序修改，不通过改变成功标准制造正面结果。

仓库：https://github.com/QaQaaa-zzz/STTW_CONTROL

工作分支基线：`experiment/sttw-offroad`；本任务书核查基准提交为 `5e3fd90a24d3ba306bf9fb79060cb38bd727919c`。开工时 fetch 并检查新提交，记录实际使用 SHA，不使用过期默认分支代替。保留已有用户改动，使用新实验目录，不覆盖旧证据。先阅读仓库 AGENTS.md 和相关技能。

目标不是再训练一套更强的底层控制器，也不是给现有网络换名。目标是：**在执行前预测已有地面与跳跃技能的整段运动，依据跳跃之后的任务可完成性选择跳跃之前的指令，并在落地后依据实测状态更新后续方案。**

本轮优先世界模型和任务规划；LLM 是最后的可选任务结构层，不是前置依赖。不默认新增付费 API、云计算或外部数据上传。

## 1. 背景、任务与三章关系

### 1.1 任务

已知地图、起点、目标区及禁行区域。机器人完成：

`越野接近 → 前向跳沟 → 落地 → 地面绕障 → 到达并保持目标状态`

机器人只能按现有能力向前跳，不增加滚转执行器，不要求斜向跳跃，不扩展为任意全局导航。起落支撑区采用现有技能适用的平整区域，区域之外可有脊坡和毫米级碎石。先复用有限路线与离散速度，不同时引入连续轨迹优化。

### 1.2 三章分工

- 第一章 STTW：地面侧倾—转向—速度耦合下的偏好条件控制，提供路线、速度请求、α 接口。
- 第二章 RecoveryBridge：拓展原技能受扰后的恢复能力，提供增强跳跃技能。
- 第三章：预测这些冻结技能接受不同指令后的实际运动，选择满足整程任务约束的方案。

核心案例：同一初态下两种接近方案都能跳过沟，但落地状态不同，只有一种仍能绕开后方障碍。必须从真实仿真找出这种案例，不预设 α、速度或侧倾方向必然对应成功。

### 1.3 要检验的研究主张

1. 跨技能传递完整动态状态与历史，比把各段接到典型理想状态更有利于整程预测和决策。
2. 后续任务可行性应影响前面的路线、速度、α 与触发选择，而不只判断能否跨沟。
3. 落地实测状态更新可以修正预测偏差并改善剩余任务选择。
4. 第一章的可调能力和第二章的恢复扩展能否转化为第三章的任务收益，由配对实验决定。

不是新意的内容：简单技能切换、冻结底层加高层、普通候选枚举、泛称世界模型或 LLM。与 Autonomous Trail Racing 的差异应落在冻结技能的可查询执行预测、明确任务约束和任务重配置，而不是声称对方不会跳跃或不会分层。

## 2. 当前真实基线及不能混淆的结果

P4：50 张新布局 × 4 个初态，每方法 200 个条件；静态 18 候选，跳前/跳跃使用 TRAIN 实测前缀，学习预测主要用于落地后驶离。

- M0 线性闭环预测：156/200。
- M1 名义模型＋地形/历史残差：200/200。
- M2 同输入直接学习预测：200/200。
- 600 方法条件对应 312 条唯一物理轨迹。
- M0 44 次失败都是无计划后默认直行撞障；所有方法已准入计划均完成。
- 各模型使用各自 DEV 校准裕量，属于完整标定后规划系统比较。
- α 仍固定 0；已有速度域为 1.8、1.9 m/s。

不要把上述结果说成残差结构独有优势、预测通过后的失败减少、已实现全链世界模型或已实现自动 α 选择。强回溯与联合规划旧结果均为 12/12，不以弱贪心作为唯一基线。

证据目录：

```
docs/evidence/chapter3_playground_sttw_rb_sttw/
  p3_multimap_collection/
  p3_multimap_models/
  p3_approach_models/continuation/
  p3_model_planning/predicted_terminal/
  p4_model_test/
  p2_rolling_multimap_support/
docs/chapter3_paper/source_manifest.json
docs/chapter3_paper/claims.csv
```

## 3. 现有代码复用与关键缺口

| 代码 | 职责 |
|---|---|
| `learning/cli/evaluate_playground_handoff.py` | 实际完整链采集和评估 |
| `learning/src/sttw_control/playground_handoff_evaluation.py` | STTW→RB→STTW 循环、交接、成功判定 |
| `learning/src/sttw_control/playground_adapters.py` | 地面指令发布、控制状态和跳跃动作所有权 |
| `learning/src/sttw_control/playground_snapshot.py` | 物理、控制器、RB、运行时快照 |
| `learning/src/sttw_control/playground_history_model.py` | 当前地面 HistoryDynamics |
| `learning/src/sttw_control/playground_prediction.py` | 名义预测及 BodyEnvelope |
| `learning/src/sttw_control/playground_terrain_features.py` | 地形描述 |
| `learning/src/sttw_control/playground_arx.py` | 状态提取、参考特征、名义模型 |
| `learning/src/sttw_control/playground_rolling.py` | 滚动筛选及控制历史特征 |
| `learning/src/sttw_control/playground_joint_planner.py` | 有限候选规划 |
| `learning/cli/prepare_playground_model_planning.py` | 静态模型规划准备 |
| `learning/cli/prepare_playground_multimap.py` | 多地图采集计划 |
| `learning/cli/prepare_playground_structured_models.py` | 同数据模型训练清单 |
| `learning/cli/run_playground_structured_models.py` | 监督训练运行 |
| `learning/cli/train_playground_history_model.py` | 地面预测训练 |

配置：`learning/configs/playground_validated_command_domain.json`、`chapter3_playground_world_map.json`、`playground_rb_skill_sources.json`。

### 3.1 当前网络不是跳跃模型

当前 `HistoryDynamics.rollout` 使用 381 维控制历史初始化 GRU64，每次调用重新初始化 hidden，不返回可续接 hidden；输出 9 维动态修正。它保持髋膝姿态不变，只推进转向和前后轮关节，因此不能预测 RB 跳跃中的髋膝运动。

新增模型必须包含全关节位置/速度、接触阶段、RB 观测/动作历史与可传递的模型状态。不要用旧地面 GRU 改名替代这项工作。

### 3.2 执行合同

以实际 resolved config 为权威复核以下当前合同：物理子步 1 ms、底层 5 ms、技能/中层 20 ms；CLI `--physics-substeps 5`。接触子步遥测要求该设置。当前任务 10 s，目标 x∈[16.5,18.5] m、|y|≤0.5 m，最低前进速度 0.5 m/s，路径误差≤0.3 m、航向误差≤0.2 rad、速度误差≤0.4 m/s，连续保持 0.5 s。

已验证速度为 1.8/1.9 m/s，且切速只验证了 RB 回交事件；不能自动开放任意时刻切速。旧基准原样保留。新任务族可以另立版本合同，但同组对照必须使用同一合同。17.3°不是通用摔倒阈值，不把偏好范围当任务失败。

## 4. 世界模型设计与接口

### 4.1 状态与指令

扩展状态 b=(x,h,z,q)：x 为物理状态，h 为控制器历史/滤波状态，z 为预测模型记忆，q 为技能及接触阶段。物理状态至少覆盖位置、姿态、线/角速度、全部关节及速度、必要接触状态。

地面指令 c_G=(路线参数,速度请求,α)。跳跃指令 c_J 为已验证技能及触发条件，不添加没有执行能力的空中转向自由度。

计划 c=(c_in,c_J,c_out)，任务 T=(目标区,禁行区,走廊,检查点顺序,时间限制)。

### 4.2 预测形式

采用状态型世界模型，不做视频生成：

\[
z_{t+1}=F_\theta(z_t,\hat x_t,\eta(\hat x_t),q_t,c_t),\qquad
\hat x_{t+1}=\operatorname{Integrate}(\hat x_t,D_\theta(z_{t+1})).
\]

地面分支可复用名义动力学＋残差；跳跃分支预测全关节与接触事件。姿态在旋转空间积分，不对四元数任意相加。输出阶段内轨迹、持续时间、事件、下一阶段可续接状态。模型记忆与控制器实际历史不是同一个对象，要明确定义，不能声称预测出了未建模的真实历史。

建议 API（新接口，待实现）：

```python
state = model.initialize(physical_state, controller_state, skill_state, context)
pred = model.rollout(state, command, max_steps, stop_event)
# pred: trajectory, events, duration, next_state, support_diagnostics
# next_state 能被下一阶段模型使用，不重新读取真实未来
```

建议模块：

```
skill_world_state.py
skill_world_model.py
skill_sequence_rollout.py
task_contract.py
continuation_planner.py
```

模块放在现有 `learning/src/sttw_control/` 下，允许根据实际代码结构调整，不重复造已有功能。

### 4.3 任务规划

预测整车占据区域：

\[
\hat{\mathcal B}_t=B(\hat x_t)\oplus E_t.
\]

在有限候选中求：

\[
c^*=\arg\min_{c\in\mathcal C}J(\hat\Xi(c);T),
\]

约束为占据区域不碰禁行物、在允许区域内、满足执行域和最终目标保持。裕量 E 由开发数据按冻结规则标定，是经验范围，不声称严格安全证明。

后续可行性表达：

\[
\hat s_{land}=\hat F_J(\hat F_G(s_0,c_{in}),c_J),\quad
\exists c_{out}:\ \operatorname{TaskSuccess}(\hat\Xi_G(\hat s_{land},c_{out}),T).
\]

不必训练额外黑盒“可行域分类器”；先直接展开后段候选检查。所有技能阶段内的碰撞都要检查，不只检查终点。

## 5. P0–P7 实施顺序与验收

### P0 冻结与复现

1. 记录代码 SHA、实际 actor hash、地图、模型、校准、任务合同和依赖版本。
2. 先重算 P4 汇总，确认 156/200、200/200、200/200；再选最小 smoke case 做真实运行。
3. 分离 no-plan、准入后失败、实际成功、未运行；旧直行 fallback 只用于原协议复现。新无计划不默认停住就安全，也不继续盲冲来制造失败，先在离线决策层报告，在线只能沿已验证的承诺段并采用预先验证的备用策略。
4. 增加拒绝原因日志：阶段、时刻、几何体、约束、原始净空、扩张后净空和裕量来源。

交付：`baseline_manifest.json`、`baseline_reproduction.md`、smoke 轨迹与结果。资产不存在时列出缺失项，不冒称已运行。

### P1 完整快照与续接接口

复用现有 snapshot，它已经含 MJX leaf、约束/warm start、GroundAdapter、RB state、runtime、tick/context。不得只保存 qpos/qvel。

测试两个层次：

- 物理连续运行与截断后快照恢复，在起跳前、触地后、交回地面时对比动作、轨迹、事件和终止。
- 世界模型一次预测 100 步，与预测40步、序列化next_state再预测60步一致。

当前 runner 拒绝 rolling planner 的快照恢复；先实现 planner state 保存，不绕过错误。

验收：误差不超过相同引擎重复运行的已测数值基线；不通过增加训练掩盖接口不连续。

### P2 先用真实仿真寻找有研究价值的跨阶段案例

从同一完整快照分叉多个接近路线、已验证速度，之后逐步加入 α。自然运行到落地，再从每个真实落地快照尝试允许的驶离候选。

生成矩阵：行=接近+跳跃方案，列=驶离方案，单元=整程成功、失败阶段、耗时、净空。

第一轮建议小规模开发筛查，覆盖2–4地图、每图2–4初态，先发现机制再扩大。寻找真实的“都能跳过，但只有一种有可行后续”案例，并重复验证。若找不到，检查候选是否被权限限制或输出夹紧为相同行为；必要时调整开发地图障碍布局，不更改统一成功标准。

Oracle 仅为离线诊断与候选库上界，不可在测试规划中读取真实未来。

交付：`oracle_matrix.csv`、两条配对视频、状态与占据图。找不到时也交付完整结果和下一步判断，不挑选删掉反例。

### P3 单段世界模型

先接入已有 phase0/phase2 地面模型，再新增 RB 跳跃全关节/接触分支。

数据须有全状态、实际动作、控制器历史、技能标识、路线/速度/α、地形、事件与最终结果。真实入口诊断与自由预测分别记录。

训练从已有监督设置起步：Adam 3e-4、clip1、batch64、多步时域；把原配置作为可调整起点而非不可改变真理。初始3随机种子；以开发集多步预测与决策错误选择模型，保存最佳及最后检查点。这里是监督世界模型训练，不叫1000轮PPO，不重训冻结控制技能。

损失：归一化物理状态多步误差＋旋转误差＋接触/事件损失＋交接状态损失。标注各项尺度与权重，训练只从 TRAIN 估计归一化。

评估单步、多步、自回归；位置、姿态、速度、全关节、起跳/触地时间、交接状态和占据误差。接触转换窗口单列，不能用平稳段平均掩盖。

### P4 全链预测与后续可行性规划

把前段预测状态/历史传递到跳跃，再传到后段。正式预测只在开始读取真实历史，其后不得使用真实未来状态、动作或未来接触标签。

实现四种对照：

| 方法 | 内容 |
|---|---|
| G | 仅参考几何路线 |
| I | 各段独立预测，使用训练确定的典型交接状态 |
| C | 连续传递预测状态和历史 |
| Oracle | 离线真实仿真候选结果上界 |

保留同候选、同预测器和约束的强回溯规划。模型结构对照继续保留直接学习与残差模型。

至少报告：no-plan、准入后失败、整程完成、真实净空、任务耗时、查询时间/次数、各阶段错误。对比 I→C 是核心；C→Oracle 用于区分模型不足与候选不足。

### P5 第一章 α 与第二章 RB 融合

α 先采用每阶段固定值0/0.5/1，不做高频切换。先采数据证明各组合确实可执行，再训练条件模型。不得让只见α=0的模型直接外推其他值。

比较固定各α、仅选速度、速度+α选择；报告同预算结果以及完整候选枚举结果。必要时做实速匹配，避免把单纯速度效应称作独立偏好收益。

RB 比较原跳跃与增强跳跃：同地图、初态、扰动、地面控制和候选设计；各自建立匹配预测接口且采集预算一致。不要把RB专属预测器套给源策略。分解跨沟成功、后续存在可行解、规划是否选到、最终任务成功。统计新增/保留/丢失，不预设能力集合严格包含。

### P6 落地后更新与任务重配置

新增 C-R：C 加实测落地状态更新。保留真实控制历史，异步预测；保留承诺前缀和 plan_version，不在飞行中随意换跳跃策略。超时继续已验证当前计划，不能等待网络使车失去控制。

测试同物理地图改变目标、禁行区、走廊或检查点；冻结控制技能与世界模型，只改变任务合同。所有方法在同一任务使用同标准。这叫任务组合泛化，不叫新地形泛化。

分别测试新地图族泛化与任务重配置，避免混为一谈。若加入学习高层基线，提供同样地图、目标和约束信息，明确是启发式复现基线而非完整复现 Trail Racing。

### P7 可选 LLM 任务结构层

只有 P0–P6 稳定且任务确实包含先后关系、可选目标和约束变更时，再接入。固定“骑—跳—骑”不需要LLM。

LLM 输入任务文字、简化拓扑、技能接口、硬/软约束，输出受JSON schema约束的任务图。世界模型负责连续运动可行性，LLM不输出扭矩、不预测物理安全。

反馈包含失败阶段、约束ID、可行替代、已完成前缀。LLM只能调整未执行后缀，不能删除用户硬约束。飞行及快速控制不等待LLM。

对照：规则任务图+同世界模型、LLM无动态反馈、LLM+世界模型反馈。衡量任务解析、约束保持、整程成功、延迟/调用量；规则同样好则把LLM定位为接口而非主创新。

## 6. 数据、切分与标定规范

`case_id = map_geometry_id + task_constraints_id + initial_snapshot_id + disturbance_schedule_id`。

`rollout_id = case_id + candidate_id + ground_policy_hash + jump_policy_hash + execution_config_hash + simulator_seed`。

同一物理运行可复用给多个规划方法，但分别报告方法条件数与唯一运行数。

按地图族、完整rollout、快照来源组划分TRAIN/DEV/TEST。同一运行相邻窗口和同快照反事实分支不得跨集合。归一化只用TRAIN，校准/模型选择只用DEV。测试失败后修改模型，要标为对已知测试的再分析，最终另用冻结新测试。

两类校准结果同时保存：各自按同规则校准的实际系统结果，以及共同/交叉裕量诊断。后者用于分离中心轨迹预测与误差膨胀的作用，不自动认为更安全。

日志记录每候选通过/拒绝、约束位置与原因。通过真实执行标签区分真可行准入、真不可行准入、可行误拒绝、不可行拒绝。这里真可行仅指该仿真协议下的结果，不是全局物理证明。

## 7. 数据字段及时间对齐

当前轨迹含 qpos、qvel、trace、world_flags、wheel_contact_flags、reference_xy、published_reference_xy、contact_substep。保存 resolved_config、state schema、完整snapshot、terrain.mjb和terrain_source.npz。

trace包含时间/位置/姿态/速度、ground_active、path_distance、steer/rear/hip/knee target、approaching、rear_rate、requested_speed、phase。

现历史为phase events之后、control之前，和qpos同时间戳；接触子步row t聚合前5ms，row0为NaN。明确输入与标签的时间方向，不能将未来接触泄漏为当前输入。

原始大数据可能在用户机器 `/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/`；开工检查实际存在与hash。GitHub汇总存在不代表原始轨迹可用，不猜路径或悄悄用合成数据代替。

## 8. 已核验命令模板

下面是现有入口，变量须由本机manifest解析。先运行help确认新版本未改变接口。

```bash
PYTHONPATH=learning/src:$PYTHONPATH "$PYTHON" learning/cli/evaluate_playground_handoff.py --help

PYTHONPATH=learning/src:$PYTHONPATH \
 "$PYTHON" learning/cli/evaluate_playground_handoff.py \
 --registry learning/configs/playground_rb_skill_sources.json \
 --config "$RESOLVED_CONFIG" --output "$NEW_RUN_DIR" \
 --skills rb_run3 --modes h75 --routes 0 1 2 3 4 5 6 7 8 \
 --seconds 10 --physics-substeps 5

PYTHONPATH=learning/src:$PYTHONPATH \
 "$PYTHON" learning/cli/prepare_playground_structured_models.py \
 --evaluation "$FIRST_TRAIN_DEV_EVALUATION" \
 --additional-evaluations "${OTHER_EVALUATIONS[@]}" \
 --output "$NEW_MODEL_QUEUE" --phase 2 --updates 600

PYTHONPATH=learning/src:$PYTHONPATH \
 "$PYTHON" learning/cli/run_playground_structured_models.py "$NEW_MODEL_QUEUE"
```

phase0用于跳前；现训练仅接受0/2，不能传非法跳跃phase。新增跳跃训练入口或扩展schema。config中的rb_adaptation_bundles可能替换registry actor，必须记录最终权重hash。上述内部参数名仅供工程调用，论文正文使用抽象技能命名。

建议新增CLI：prepare_skill_world_data.py、train_skill_world_model.py、evaluate_skill_world_model.py、evaluate_continuation_planner.py。具体参数实现后提供可实际执行的命令，不把建议入口声称为现成工具。

## 9. 失败后的迭代顺序

1. 快照或历史不连续：先修状态恢复/动作所有权/滤波。
2. Oracle候选全失败：扩展合理候选或调整开发任务，不盲目扩大网络。
3. 有真实可行方案却全部拒绝：检查中心预测与裕量，不统一缩小安全裕量。
4. 预测准入实际撞障：定位几何体、阶段、事件时刻、分布外状态。
5. 单段好串联差：真实/预测交接互换诊断，再检查误差传播。
6. 模型准但规划无收益：任务可能不需要跨阶段选择，报告该结论。
7. α无收益：查执行权限、实速、候选覆盖；不强迫出现三档各胜。
8. RB无收益：检查任务饱和与既定扰动层级，不选择性删样本。

每轮只改变少量可解释因素，记录假设、改动、预算、结果、保留/回退决定。达到预设训练预算而无改进时先诊断，不无限续训。优先复用已有模型与数据，不为了论文热点重建昂贵系统。

## 10. 最终交付

每阶段交付：代码差异、单元/集成测试、实际运行命令、配置与hash、模型、数据清单、逐例结果、错误归因、简短结论。

最终重点图和视频：

1. 同地图同起点“两种方案都跳成功，但只有一种完成后续任务”的真实配对。
2. 参考线、预测线、实际线、车身包络、障碍和误差位置同图。
3. 真实/预测交接状态随时间，起跳触地时刻清楚。
4. 固定α/可选α、源跳跃/RB的整程结果。
5. 落地更新前后方案与最终执行对比。

所有图标明运动方向、地图、任务合同及方法，不重复堆砌数字。示意图与实测图分开。模型输入/输出和数据流框图必须与代码一致。

最终小包只含代码、必要模型、配置、说明和小型复现实例。大轨迹单独存储并给manifest，不把巨大原始数据塞给用户。论文不设置补充材料来隐藏关键方法；正文应能讲清任务、预测、决策、基线和结果。

提交前回答：完成到哪个P阶段？哪些真实运行？哪些仅实现？哪些主张获支持？下一步最有信息量的实验是什么？不要将计划写成成果。

## 11. 文献定位

- Autonomous Trail Racing with a Bicycle Robot：https://arxiv.org/html/2611.15280v1
- SkiMo：https://proceedings.mlr.press/v205/shi23a.html
- STAP：https://sites.google.com/stanford.edu/stap/home
- SayCan：https://say-can.github.io/
- Inner Monologue：https://proceedings.mlr.press/v205/huang23c.html

现有研究已覆盖技能世界模型、技能序列可行性、语言规划和反馈修订。我们的研究重点是双轮连续动态技能交接、阶段内占据与后续任务可行性，并通过同条件实验验证它们的价值，不宣称首次将这些通用模块组合。

## 12. 第一轮立即执行清单

- [ ] 检查最新分支、用户改动与资产。
- [ ] 固定基线manifest，重算P4与一个smoke真实运行。
- [ ] 审查动作所有权、快照和时间对齐。
- [ ] 完成物理续接一致性测试。
- [ ] 建立小规模真实跨阶段oracle矩阵。
- [ ] 输出首个可复现的两方案机制案例，或明确未找到及原因。
- [ ] 在上述基础上实现可续接世界模型API，再训练跳跃分支。

**第一轮不要优先做LLM、视频世界模型、连续轨迹优化或新PPO底层训练。先证明确实存在值得预测和规划的跨阶段选择，再扩大模型和实验。**
