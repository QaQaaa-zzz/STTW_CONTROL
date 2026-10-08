# 固定指令六场景：fixed_command_six_v1

后续指令协调器的默认共同开发测试，用户2026-10-08确认复用之前场景。机器配置：learning/configs/fixed_command_six.json；执行入口：learning/cli/train_upper_endpoints.py --review-alias（在声明队列中调用），upper_endpoint_review共用。替换本轮原main/random两场景，不扩成额外叠加测试。训练与冻结历史结果不改。

| 场景 | 原始目标指令（时刻s：速度m/s，转角rad） |
|---|---|
| straight_hold：直行 | 0：2.3，0；保持至10s |
| speed_changes：直线变速 | 0：2.3，0；1：2.6，0；3.5：2.0，0；6：2.3，0 |
| gentle_positive：正向缓转后回正 | 0：2.3，0；1：2.3，+0.08；4：2.3，0 |
| gentle_negative：反向缓转后回正 | 0：2.3，0；1：2.3，−0.08；4：2.3，0 |
| steer_reversal：正反转向 | 0：2.3，0；1：2.3，+0.15；3：2.3，−0.15；5：2.3，0 |
| fast_turn：加速急转后回正 | 0：2.3，0；1：2.6，+0.25；4：2.3，0 |

正负采用控制接口符号，不凭直觉改成左右。所有原始目标通过既有slew：速度0.5m/s²、转角0.3rad/s。上层不能改变原始参考、窗口或目标积分。

固定seed77001、prepared_bank[3]（3.5s原ECBC准备，2.3m/s初速），无外扰，10s观察。两模型准备bank逐叶对照一致；身份不同不复用旧基线。物理200Hz、上层50Hz、原权限/ESO不变，lower_alpha固定1。10s只是观察截断；原16s网络任务剩余时间和失败吸收奖励保留，不把10s冒充原任务自然结束。

每个底层固定三方法：B0=仅该冻结底层（零上层修正）、独立upper_alpha0、独立upper_alpha1。6×3=18物理回合，不新增种子/扰动笛卡尔积。best_model.json优先，无best才last_completed；图/manifest记录各自checkpoint。旧ECBC+ESO与旧底层面板是历史参考，除非全闭环身份和初态核验一致，不冒充本轮配对基线。

统一指标：全观察时序；指令窗口[1,6)s速度/转角RMSE；最终[9.5,10)s连续保持，速度误差≤0.1m/s、转角误差≤0.04rad、航向误差≤0.05rad，三项分别报告且共同保持另列。航向按原始指令独立积分，不重置参考，不把轮角回零等同方向恢复。峰值侧倾、>0.3rad历史诊断持续时间和原配置工作侧倾界违反分别报告；不更改原工作界/物理失败阈值。失败处截断，缺失完整窗口记missing，不填零，不当成功。

逐场景交付速度、转角、侧倾、航向、实际两通道残差与XY叠图、每步/累计奖励、奖励分项和CSV/NPZ。奖励基线按upper alpha分别从同物理轨迹重评分；不同alpha总reward不用于物理性能排名。单种子开发测试不称泛化成功率。

历史证据：/home/qy/STTW_CONTROL/runs/tracking_candidate_review_20261008/commands/INDEX.md（底层候选），/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/lower_tracking_probe_20261008/INDEX.md（旧底层）。这些不是R196新上层250/143的已完成评价。
