# H75近期结果与可执行输入（2026-11-09，美国西部时间）

[共同起点：有外部推扰的原H75/150轮XY对照](results/common_start/evaluation/plots/README.md) · [150轮无外部推扰五alpha图](results/common_start/evaluation_no_push/plots/README.md) · [密集石子与10cm单脊](results/dense_corridor_ridge10/RESULTS.md)

| 阶段 | 结果 | 应如何解释 |
|---|---|---|
| 密集轨迹石子带、10cm单脊 | 72条件，原冻结控制器；全部失败原因及alpha图保留 | 地形难度扩展，不是新训练成果 |
| 旧中途初态续训76轮 | 原/续训完整44/65→53/65；速度误差明显增大 | 用户否决初始化方案；历史诊断，禁止冒充共同起点模型 |
| 共同起点续训150轮，有外部推扰 | 原/续训完整24/65→25/65；共同窗口路径RMSE .5831→.4616m，速度 .1774→.3523m/s | 路径改善伴随明显减速，不是全面增强 |
| 同150轮，无外部推扰 | 完整56/65；6个运行域终止、3个侧倾阈值终止；平均速度偏差-.3811m/s | 外部推扰对旧失败贡献很大，减速依然存在 |
| 原H75无外部推扰配对 | 本快照尚未完成，已排队 | 不用上行56/65声称续训优于原H75 |

完整观察不等于最终跟踪保持合格，侧倾阈值不直接等同摔倒。13工况×5alpha不是65个独立地形种子。无外部推扰只关闭起步横向力与周期速度冲击，仍保留车辆参数随机化、传感噪声、延迟及地形。

## 模型身份

- `inputs/training_bank/original_policy.pt`：原H75，评测before模型。
- `inputs/bundle/actor.npz`：原H75部署表示及任务/奖励配置。
- `inputs/lower_actor/actor.npz`：冻结橙色Local300，配ECBC/ESO。
- `models/common_start_update150.pt`：从原H75起步、共同路径起点续训150轮；本轮after模型，非已证明最优模型。
- `models/deprecated_midroute_update76.pt`：旧中途初态76轮，历史保留，禁止默认采用。

原始checkpoint及输入保持字节一致；SHA256见`manifest.json`。训练配置、逐轮日志在`training_records/`，现有75/150等历史模型不混用。重复文件通过包内相对符号链接去重。

## 代码与运行

代码就在本仓库同一提交的`learning/src/sttw_control/`和`learning/cli/`中。关键入口：`train_h75_random_terrain.py`、`plot_random_terrain_alphas.py`、`analyze_common_start_results.py`、`render_isaac_scene_media.py`。原H75奖励/PPO和网络配置在`inputs/bundle/config.json`，控制及随机化源码一并推送。

需安装Isaac Sim 5.1及兼容Isaac Lab、CUDA/PyTorch、tensordict、tensorboard等；训练使用的RSL runtime源码已包含。外部GPU SDK未打包，不承诺裸机直接运行。视频仅用MuJoCo做保存姿态的显示，不用其重新计算物理。

从仓库根目录，在已配置的Isaac Lab Python环境运行一个13条件alpha评测（输出目录须不存在）：

```bash
PYTHONPATH=learning/src python learning/cli/train_h75_random_terrain.py \
 --config docs/evidence/h75_recent_20261109/evaluate_no_push.json \
 --output /tmp/h75_no_push_after_alpha05 --evaluate --eval-alpha 0.5 \
 --checkpoint docs/evidence/h75_recent_20261109/models/common_start_update150.pt
```

省略`--checkpoint`即使用包内原H75。五alpha均跑完后用同仓库绘图与分析入口。完整原始轨迹和逐条件初始化/随机化在`results/common_start/`，不需要重跑仿真即可分析。旧报告中的本机绝对路径属原始出处；本包相对路径和manifest为交付索引。密集石子72条件仅保留精简统计和XY图，未复制约1GB原始全身轨迹；共同起点比较保留全部逐步NPZ。

## 当前视频与待补结果

用户要求视频直接开始，已将13环境alpha=.5补录从GPU队列拆出；原H75五alpha比较仍排队。预选case1的石子直径110–160mm、配置最大高度11.78mm，按实际碰撞网格显示，含位置图与1m比例尺。当前快照不把未完成视频或比较写成完成。

验证范围：包内文件哈希、配置输入路径、CPU checkpoint加载及原始轨迹可读；不是新机器GPU复现实验。
