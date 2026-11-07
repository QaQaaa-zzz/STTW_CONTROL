# 第三章当前代码与运行输入闭包

本目录与同一次 Git 提交中的 `learning/src/sttw_control`、`learning/cli`、`learning/configs`、`learning/scripts` 一起交付。不是只有结果或依赖数据盘的入口。所有运行命令从仓库根执行；配置中的运行输入已改为仓库相对路径，输出写入被忽略的 `runs/chapter3_portable/`。

## 已包含

- 当前项目源码、实际执行的完整链分析器与打包入口；manifest记录1158个项目源码/配置/脚本的实际内容hash。
- 冻结H75 actor_0077、Local300、heading125和update2推理数组、归一化、动作映射与身份；当前地面执行 **zero_lower=true**，Local300仍因原加载接口需要随包提供，不表示正在使用残差。
- 原RB带摆车辆URDF、相邻接触契约、全部导入网格；另附原物理模型XML和网格。
- 两套实际起伏/平台地形、地面初态、原始路径、分流地图与路径、六候选记录、核尺度/支持度/概率门槛。
- 六份可直接运行的配置：当前heading125已运行成功的000_C6、预声明TEST012/019的fixed/C6四条，以及旧update2的30秒三查询完整链。
- Go2实际使用的rsl_rl/legged_gym源码和原LICENSE；历史heading训练配置、初始训练checkpoint、训练bundle、真实入口bank、当前选定125 checkpoint。训练配置保留当时初始化语义，没有把它伪装成某次完整优化器精确恢复。

`manifest.json`的`inputs`保留源→包内映射；`files`是输入hash清单，`source_files`指向同提交中的当前源码。identity/contract内的历史训练路径用于溯源，不是推理时需要访问的文件。包内README和closure_check为派生说明，不自引用hash。

## 外部环境

Isaac Lab/Isaac Sim/CUDA和Python第三方依赖是已声明的外部安装，不复制数十GB的SDK，也不宣称在裸系统上零安装运行。实测环境见 `environment.json`：Python3.11.15、IsaacLab0.54.3、IsaacSim5.1.0.0、PyTorch2.7.0+cu128、RTX4090D。IsaacLab源版本与Go2源版本一并记录。使用已配置IsaacLab的Python；推理不要求访问原DVGC/JIT目录或下载旧训练仓库。

## 运行当前heading125接续

```bash
export PYTHONPATH="$PWD/learning/src"
python learning/cli/evaluate_isaac_pendulum_jump.py --config learning/deployment/chapter3/chapter3_heading_online_000_C6.json
```

`python`须是上面已配置IsaacLab的解释器。该配置使用本包权重/模型/地图/能力记录，物理400Hz（dt=.0025s）、地面控制/传感200Hz、H75/跳跃策略50Hz；候选仅作用到跳跃交权，落地续行使用共同1.5m/s规则。地面锁摆，跳跃释放，交回锁实际摆姿态。

旧update2长接近与三查询配置：

```bash
python learning/cli/evaluate_isaac_pendulum_jump.py --config learning/deployment/chapter3/chapter3_task_reselection_config.json
```

这是独立的旧技能接口证据，不替代heading125，不混作泛化结果。以上命令不自动训练。

## 用现有分析器查看完整轨迹

以第一条输出为例：

```bash
R=runs/chapter3_portable/chapter3_heading_online_000_C6
python learning/cli/analyze_isaac_pendulum_landing.py --run "$R"
python learning/cli/analyze_rb_continuous_task.py --run "$R"
python learning/cli/plot_isaac_pendulum_jump.py --run "$R"
```

图和物理分项生成在同一输出目录。已完成的实际结果见 `docs/evidence/chapter3_jump_heading/RESULTS.md`，不需要重跑便可查看。新测试四条全保留；它们在选档前域退出，不当作跳跃或选档器已失败的证据。

## 本次检查范围

`closure_check.json`：源文件/输入hash核对通过；两种跳跃策略、H75、Local300数组均已用CPU实际加载。6个重定位配置运行输入均在提交中。未因打包重跑物理；因此这不是新机器上的GPU仿真复现认证。源码中保留项目其他历史实验入口，包内6个配置才是本次明确闭合的当前运行入口；历史身份路径不要求在新机器上存在。
