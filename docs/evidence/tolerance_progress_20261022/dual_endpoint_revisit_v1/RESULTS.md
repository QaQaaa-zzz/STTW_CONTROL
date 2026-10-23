# 双网络回访：原生结果与共同窗口诊断

| 端点 | 窗口 | 成功/18 | 域退出 | 无扰成功/2 |
|---|---|---:|---:|---:|
| alpha0 | native20 | 9 | 7 | 2 |
| alpha0 | clipped10 | 8 | 3 | 2 |
| alpha1 | native20 | 18 | 0 | 2 |
| alpha1 | clipped10 | 18 | 0 | 2 |

| 单网络策略 | α | 原10s成功/18 |
|---|---:|---:|
| parent | 0.0 | 9 |
| seed918501 | 0.0 | 16 |
| parent | 1.0 | 18 |
| seed918501 | 1.0 | 18 |

历史双网原生20s结果完整保留，前10s重新按相同任务恢复标签计算；未把后10s成功记为前10s成功。双网历史reward、最高速度3.0m/s与单网V3/3.5m/s不同，单网与双网的起扰前状态也不同。这是实际方案比较，不是纯网络结构因果消融。18例共享少数任务前缀，不能作为18个独立随机初态。

数值不表示单网络达到能力上限，也不否定双网络作为参考或潜在教师的价值。是否作为教师需要同一完整受扰状态下的跨策略接续见证，当前面板尚未执行这种接续。

[alpha0_left105_R2p4_10s_xy](figures/alpha0_left105_R2p4_10s_xy.png)
[alpha0_left105_R2p4_20s_xy](figures/alpha0_left105_R2p4_20s_xy.png)
[alpha0_right105_R2p4_10s_xy](figures/alpha0_right105_R2p4_10s_xy.png)
[alpha0_right105_R2p4_20s_xy](figures/alpha0_right105_R2p4_20s_xy.png)
[alpha1_left105_R2p4_10s_xy](figures/alpha1_left105_R2p4_10s_xy.png)
[alpha1_left105_R2p4_20s_xy](figures/alpha1_left105_R2p4_20s_xy.png)
[alpha1_right105_R2p4_10s_xy](figures/alpha1_right105_R2p4_10s_xy.png)
[alpha1_right105_R2p4_20s_xy](figures/alpha1_right105_R2p4_20s_xy.png)

共同10s逐工况互补性：α0双网相对原单网有2新增、3丢失（新增为右弯-20/-30N）；相对教学后单网0新增、8丢失。α1三策略均18/18。该面板不支持主线切回旧双网；双网作为教师仍需共同完整状态接续实验证据。
