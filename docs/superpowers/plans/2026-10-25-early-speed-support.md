# 提前平滑速度支持：一次机会验证

状态：固定80已真实启动，结果PENDING；STTW_CONTROL，DEV，0训练。前轮floor组合120裁决为依据，不能称独立确认。

问题：固定1.5参考下限补一次摔倒但损害普通策略任务，能否在入弯准备/早期提供支持、随后释放，保留更多原alpha任务？本次只试一个几何阶段定义，不扫节点/阈值/长度。

## 同接口实现

设原Actor物理速度修正dv、当前已发布名义速度v_nom。支持目标dv_support=max(dv,1.5-v_nom)，再通过原inverse_offset_target映射到同一[-1,+.25]速度修正；不降低原修正，转向latent逐值保留。全部动作经过原correction_tick滤波/变化率/限幅和冻结完整闭环。这个软目标不是实际速度下限，也不保证参考或真实速度达到1.5。

EARLY(profile6)：g= smooth((s-(s_begin-1.5))/1.5) × (1-smooth((s-s_begin)/((s_end-s_begin)/2)))；smooth(u)=clip(u,0,1)^2*(3-2clip(u,0,1))。s_begin/s_end来自原路径入弯与出弯，1.5m沿用既有准备距离；入弯前平滑启用，至弯中点完全释放。目标为dv+g*(dv_support-dv)，g=0精确保留原latent。

CONSTANT(profile7)：同一dv_support、同一反映射与滤波，全程g=1。它是必要匹配对照：否则阶段支持与旧硬floor的差异可能来自执行实现而非时机。禁止在运行中跳改配置下限。

两profile都只用当前projection.progress、名义参考、原Actor输出和路径几何，禁止实际未来pulse/time/direction/force信息。原alpha不变；同16完整有效块交回，disabled/故障/接触/时域均不得发布支持动作。

## 固定矩阵与成本

父Actor与普通task64各跑EARLY/CONSTANT，左右原R1.1请求1.8/1.5、alpha.5/.75、zero及t1.9/t2.9×世界方向±32N/.1s，共2Actor×2profile×2route×2alpha×5event=80新，≤160000实际5ms步，0optimizer。原设想40仅EARLY；在新物理前增加40同接口CONSTANT，目的是分离时机与硬clip实现，不因结果追加。

复用80已认证共同handback轨迹（parent、parent+硬floor、task、task+硬floor），8配方合160逻辑记录。physical-credit不另扩，因为现有强对照没有独立增益。40/route，按原架构分组batch20、scan50、10秒；不改频率/物理。全部从原完整task0/physical700运行，保留真实history/ESO/filter/actuator/clock。

## 检验与停止

- [x] 先写测试：几何envelope光滑启停/弯中点归零；原steer及无介入latent精确保留；饱和下不减原动作、不扩大权限；通过真实correction_tick验证受原速率限制；声明profile矩阵无缺/重复、原alpha/源Actor/force/profile配置可定位。
- [x] 准备层显式profile声明，默认旧20/50/30/40行为不变；当前模型/原状态/cache SHA认证、全部源封存，旧原始结果不改。
- [x] 两route完整状态GPUshape、0积分；相关CPU测试通过，Bark仅现有中文通道/repairfalse，台账启动前登记后一次运行；900秒检查或真实进程退出事件。
- [ ] 80新+80cache完整配对：EARLY vs同Actor CONSTANT、原Actor、硬floor；父Actor同样干预作简单规则。分无扰与受扰、逐route/alpha/事件、原alpha任务成本/真实触地/最大偏离/参考与实际速度/roll/steer/原任务接回；全部共享XY和reward曲线。

若EARLY没有同接口CONSTANT覆盖不了的任务—未摔增量，停止这一阶段过程，不调弯中点/1.5/16块/幅值/观察窗继续追正例。若有可解释增量，仍须与同规则父Actor、普通训练和新起始情境比较；阶段规则本身不是方法新颖性。旧解析Emin不参与选择，本轮不拟合或开发新网络。若基本接口不成立，报告工程事实，不把过程输出越权硬塞给执行器。

启动前验证：5360dde；35CPU通过2.01s；左右各2batch20 fullstate GPUshape exit0/0积分；根80cache+80new唯一160，子profile多次来源引用不当独立轨迹。11关键runtime SHA及plant step/16块资格AST核对，新增仅candidate规则与准备声明。

真实启动：controller2237066/left2237074均/proc live，session77853，root/实际stage running，监控900秒。
