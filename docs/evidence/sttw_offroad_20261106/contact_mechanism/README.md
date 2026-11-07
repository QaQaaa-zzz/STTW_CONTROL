# 接触机制验证：精简结果包

先看 [改善轨迹](ridge/example.png)、[退化轨迹](ridge/counterexample.png)、[平地对照](flat/example.png)，再读 [结论](CONCLUSION.md)。

**结论：当前补偿/接触特征方案没有稳定的恢复或选档净收益，按用户授权结束本支线。物理作用和个别改善仍成立，原STTW偏好主线保留。**

- 单脊150、平地250条均完成10秒。恢复段0.4m约束原29/50、抵消负段28/50、恢复参考27/50。
- G6基础状态与接触增强均20/30可行，增强平均速度RMSE略差；全条件指标及反例保留。
- 未执行的G4替代切换、连续脊与G7已取消，不冒充完成。
- 无新训练/奖励修改；冻结原H75+橙Local300。configs含模型身份及参数；source仅提供相关分析/干预源码快照，非独立运行安装包。可引用源码提交274fcaac，完整运行资产与原始5kHz数据保留本地。

## 数据

ridge/flat的contact_intervention_effects.json保留全部100/200配对条件指标；config.json记录所有lane身份及参数。xy_50hz.npz为原200Hz记录每4点取最后一点，无插值/平滑，覆盖全部lane和最终样本；原时间索引明确保留，不适于分析5kHz接触峰值。参考路线见reference_route.npz。道路约束是3–10秒中心偏差，不是全程到达指标。

paired_contact_summary与recovery_time来自已完成P0；contact_branch_outcomes与contact_selection文件来自G3/G6。JSON中runs路径是本地原始数据来源，不代表远端包含原始大文件。统计均为DEV固定参数组，非独立训练种子或实车结论。

包内MANIFEST记录文件大小/校验和。旧顶层MANIFEST仍对应先前交付，本次增量以本目录为准。
