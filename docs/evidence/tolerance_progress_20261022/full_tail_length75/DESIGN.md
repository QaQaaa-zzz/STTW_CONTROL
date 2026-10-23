# 固定其它因素的训练长度对照

原full_tail uniform的27轮含2critic+25PPO，延伸至77轮即75PPO，仅新增50PPO（3,276,800策略转移）。父策略u325、完整尾段10s、320状态库、V3、local300、seed、力档、学习率、KL锚均不变。选择full_tail是因为它包含完整任务截止窗口，不按本次双网结果挑选最好分支。

旧checkpoint SHA39e2055869bd803845e9d36da1f267a20b197481eac17e5ceef7528c35efb12e已认证；保留Actor/Critic/log_std/Adam及随机状态、父锚游标。物理/控制器/history从保存的route counters+1重新抽完整bank状态，**不是帧连续续训**。schedule_resume_identity仅放行updates_initial/max延长，其余配置和lineage必须一致；12项针对测试通过。初始工程认证并不证明续训有效。

主比较固定最终75PPO对旧25PPO，另取现有保存节奏update50（48PPO）作描述，不能挑最好点替代主比较。先复用固定开发面板的旧25结果（身份/协议完全匹配才复用），同时报告原始u325及当前教学候选。报告受扰原任务成功、配对改善/退化、无扰保留、方向与alpha子组以及连续误差。该试验隔离长度因素，不是等交互成本方法优势证明；增加的训练交互完整计入。

2026-10-23已排队，等待双网完成后启动；TensorBoard http://localhost:6040 已HTTP200，首个新reward已通过HTTP读取：step28=-2.2504148483。训练pid2663458运行中。计划文件runs/preference_full_tail_length75_v1/uniform/plan.json。启动前接入独立TensorBoard并核验实际reward标量HTTP，登记中文STTW Bark完成/报错；等待双网评测释放GPU，单训练worker、512lane批量。没有固定墙钟截止，不无上限续训；终局结果后决定是否有必要继续。
