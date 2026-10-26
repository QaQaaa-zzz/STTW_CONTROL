# 同初始函数的Critic/完整回合估计器在线检验

状态：设计，未实现/prepare/运行。依据600历史轨迹和8冻结策略价值拟合；不将预测改善当控制改善。

## 唯一问题与三个角色

能否通过更合理的有限时域任务价值信号，减少抗扰训练后的偏好/末段跟踪损失？原始用户alpha、V3任务/失败奖励、冻结local300/ECBC/ESO及参考限幅/滤波/slew不改，profile0不加任何gate。

A臂free_GAE：V=-.16D(t)+NN，自由残差，lambda=.99。
B臂finite_GAE：V=-D(t)softplus(NN)，lambda=.99。
C臂free_MC：与A相同自由残差，lambda=1，完整实际回合目标。

D(t)=(1−.997^(500*remaining_fraction))/(1−.997)。.16来自本轮train数据成本率量级的共同固定先验，不由validation或未来评价拟合，不扫r。A/B/C初始V逐点相同，隐藏表征复制同原task64 Critic，输出头按各自形式设置常数；Actor/logstd从同一原task64复制，fresh Adam。原旧Critic/Actor全冻结保留。A对B比较参数化包（同时有边界/符号/优化几何差别）；A对C比较GAE与MC回合目标（同时影响价值目标与优势），不是其它机制消融。

## 实现路径与必要检查

- 新建finite_value_online.py：独立learner适配器，复用现有PPO数学/合法数据。Torch和JAX的Critic参数化一致，352维剩余时间输入不改；Actor351标准导出不改。mode/rate必须进入checkpoint/采样身份，不能用旧schema冒充可恢复新Critic。
- 新建finite_value_run.py及薄CLI：180lane动态参数/脉冲scan50；不改旧live_ppo_run.py、旧采集或冻结源码。三臂同任务/扰动/RNG配对，不能使用未来事件作为Actor/Critic输入。
- 扩展独立测试：初始函数/Actor一致性，Torch/JAX输出误差，lambda1与完整回合MC等价，回退奖励折入/无伪动作，恢复checkpoint身份。先shape0物理，再登记Bark/台账启动。

## 固定训练与评价

三臂各64轮×60完整10s回合；首2轮Critic-only，之后62轮Actor更新，每轮4epoch、Actor1e−4/Critic3e−4/clip.2/entropy.001、fixed denominator30000、同fixed-slot熵归一。原gamma=.997，唯一lambda差如上。预计最大23040000个5ms控制slots，实际成本另记，不设墙钟截止或自动扩轮。初值误差、每alpha/左右/力档真实样本占比、Critic末段误差与优化步/TensorBoard实际reward全部记录。

schedule seed2026102571、learner seed2026102572、collection seed2026102573；五alpha、左右原R1.1请求1.8/1.5、无扰/±32N/.1s、起扰在[1.6,3.4]秒隐藏抽取。开发试验的一次优化重复，不称独立父策略重复。

固定第64终点，不选best。评价seed2026102574：两路线×五alpha×每组一个zero与双方向各两个隐藏时刻=50共同事件；六角色为原A、原A因果速度支持、原A恒定支持、三新Actor，共300逻辑回合。新Actor均确定性均值执行、相同live/16clean合法交回；不把随机训练与确定性评价混为同权重噪声消融。先声明准确条件再执行，不重复zero或事后放宽标签。

## 裁决与路线边界

原alpha任务成本、防摔新增/丢失、无扰任务保留、受扰末段和连续误差作为主要控制证据，保留失败，允许不完美。若MC已经覆盖结构收益则保留普通方法；若结构只改善RMSE而控制无增量，停止该结构，不扫r/尾窗/奖励。若有控制收益再在独立路线/时刻/持续时间及真正独立训练重复确认，并检验alpha取舍是否仍可用。

Critic工程修正本身不承担首次创新；主要研究问题仍是有限参考权限下的偏好—抗扰耦合与最小任务代价协调。当前无新容扰估计器、奖励重构或硬件开发。
