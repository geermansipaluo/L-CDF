# LCDF本地实验开发手册与修改记录

本文档是本仓库后续修改的唯一总入口。每次代码修改都应新增一个版本条目，
记录“为什么改、改了什么、怎么运行、结果在哪里、如何解释”，并保留Git提交号。

## 当前版本

| 版本 | 日期 | 内容 | 状态 |
|---|---|---|---|
| `v0.1.2` | 2026-07-30 | 持续接触只计一次碰撞事件，20场静态随机环境正式测试 | 已完成：13/20成功 |
| `v0.1.1` | 2026-07-30 | 固定随机圆柱为静态模型；修复动态圆柱导致的净空统计失真 | 第1场主动中止，事件计数需修复 |
| `v0.1.0` | 2026-07-30 | `cdf_r_ego=0.36`，20个随机圆柱环境及完整结果落盘 | 已运行，但动态圆柱使净空结果无效 |

版本规则：小修复增加补丁号（如 `v0.1.1`），新增实验增加次版本号（如
`v0.2.0`）。运行时会把版本号、时间、Git分支、提交号和工作区是否有未提交修改
写进 `config.json`。建议每个确认可用的版本执行一次Git提交：

```bash
cd /home/guo/L-CDF
git status
git add README.md EXPERIMENT_README.md results \
  src/scene/launch/traj_eval_sweep.launch \
  src/sensor_cdf/scripts/experiment_utils.py \
  src/sensor_cdf/scripts/traj_eval_sweep.py \
  src/sensor_cdf/scripts/run_random_env_eval.sh
git commit -m "feat(eval): add v0.1.2 random environment benchmark"
```

上面只是建议命令，本次修改没有替你自动提交。

## v0.1.2修改分析

`v0.1.1` 启动静态障碍测试后，持续接触会被旧逻辑每0.5秒重复计为碰撞事件。
`v0.1.2` 将事件定义改为“从非碰撞区进入碰撞区的一次边沿”；只有机器人离开后
再次进入才算第二次。回合是否碰撞、真实净空和到达判定保持不变。

### v0.1.1静态障碍修复

`v0.1.0` 首轮20场运行后，失败场景出现“机器人中心几乎经过圆柱初始中心，但
LiDAR仍看到约0.8 m障碍距离”的矛盾。回查 `world.world` 发现8个圆柱原本均为
动态模型：机器人会把圆柱推走，而评测节点仍按初始坐标统计。这会让净空和碰撞事件
失真。因此从 `v0.1.1` 起将 `cylinder_0...7` 全部设为静态模型，再用完全相同的种子
重跑。旧运行目录内已加入 `VALIDITY_WARNING.md`，不得用于正式结论。

以下场景设计、参数与输出结构均延续自 `v0.1.0`。

### v0.1.2正式结果

正式结果目录：

`results/random_environment/v0.1.2_20260730_095639/`

20场中13场到达且无碰撞，7场首次碰撞终止，成功率65%、碰撞率35%、超时率0。
qpth失败与JAX回退均为0。详细的成功/失败场景、效率、净空与实时性分析见该目录
的 `analysis.md`。这轮只能回答 `0.36` 的表现；判断增加0.05 m是否有效仍需运行
同场景 `0.31` 配对基线。

### 为什么要分开三个半径

这次把 `cdf_r_ego` 从 `0.31` 增大 `0.05` 到 `0.36`。这可以作为明确的
安全裕量实验，但不能把它和机器人真实尺寸混在一起：

- 机器人真实半径：`0.31 m`，不变。
- Gazebo圆柱真实半径：`0.25 m`，来自 `world.world`。
- CDF计算半径：`0.36 m`，比真实机器人半径多 `0.05 m`。
- 真实接触临界中心距：`0.31 + 0.25 = 0.56 m`。
- CDF安全包络临界中心距：`0.36 + 0.25 = 0.61 m`。

因此结果同时报告“真实几何碰撞率”和“CDF安全包络违反率”。即使成功率提高，
也能看出提高是否来自额外的 `0.05 m` 裕量，而不会把调参效果误写成模型能力。
正式做结论时，建议用完全相同的20个场景再运行一组 `cdf_r_ego=0.31` 作为基线。

### 随机环境定义

- 共20个环境，每个环境运行一轮。
- 起点固定为 `(0, 0, 0)`，终点固定为 `(15, 0)`。
- 每场3至6个圆柱，使用世界中已有的 `cylinder_0` 到 `cylinder_7`。
- 圆柱中心范围默认是 `x∈[2,13]`、`y∈[-3,3]`。
- 场景总种子默认 `20260730`；第 `i` 场使用 `20260730+i`，可以单独复现。
- 第一个障碍物会放在起点到终点直线路径附近，避免大量“无障碍直行”的简单场景。
- 障碍物之间、起点附近和终点附近都有最小间距约束。
- 正式脚本默认在首次碰撞时结束该回合（`TERMINATE_ON_COLLISION=true`），避免静态
  障碍前持续顶撞；需要观察碰撞后恢复行为时可显式设为 `false`。

随机的是场景布局，不是运行时不断移动的动态障碍物。`scenarios.json` 会保存20场
的全部坐标，因此同一组场景可用于 `0.31` 和 `0.36` 公平比较。

### 代码修改位置

- `src/sensor_cdf/scripts/experiment_utils.py`
  负责无ROS依赖的随机场景生成、JSON/CSV写入、指标汇总和Markdown报告。
- `src/sensor_cdf/scripts/traj_eval_sweep.py`
  负责逐场景切换Gazebo障碍物、逐控制周期采样、逐场指标和中断保护。
- `src/scene/launch/traj_eval_sweep.launch`
  增加版本、结果目录、随机场景和碰撞半径参数；本版本默认
  `cdf_l_k=0.33`、`cdf_r_ego=0.36`。
- `src/sensor_cdf/scripts/run_random_env_eval.sh`
  一键运行本测试，并将完整终端输出保存为 `run.log`。
- `results/`
  统一的训练、一般评测、随机环境评测目录。实际运行产物被Git忽略。

修正了一个会影响多场景测试的旧流程问题：过去重置Gazebo后会先摆放上一场景，
再增加回合编号；现在会先切换环境编号和目标，再摆放新障碍物。

## 傻瓜式运行说明

### 第一次运行前

打开一个终端，只需要执行：

```bash
cd /home/guo/L-CDF
source /opt/ros/noetic/setup.bash
catkin build
source devel/setup.bash
chmod +x src/sensor_cdf/scripts/run_random_env_eval.sh
```

如果项目已经成功编译过，之后不修改C++消息或依赖时通常不需要重复
`catkin build`。

一键脚本默认自动激活这台机器上已有的 `cdflearn` Conda环境，因为当前base环境
没有 `torch_geometric`。如果以后环境名改变，可这样指定：

```bash
CONDA_ENV=新的环境名 bash src/sensor_cdf/scripts/run_random_env_eval.sh
```

### 运行20场 `cdf_r_ego=0.36`

```bash
cd /home/guo/L-CDF
bash src/sensor_cdf/scripts/run_random_env_eval.sh
```

脚本会自动启动Gazebo、加载默认的48演示模型、依次运行20场并关闭。最长评测时间
约为 `20 × 80 s`，另有Gazebo启动和每场重置时间。终端可以实时查看进度；
不要同时启动另一套Gazebo或 `roscore` 实验。

运行完成后执行：

```bash
cd /home/guo/L-CDF
ls -dt results/random_environment/v0.1.2_* | head -1
```

进入显示出的最新目录，首先阅读 `summary.md`。

### 用同一随机种子运行 `0.31` 基线

```bash
cd /home/guo/L-CDF
CDF_R_EGO=0.31 bash src/sensor_cdf/scripts/run_random_env_eval.sh
```

两次运行的 `RANDOM_ENV_SEED` 都是 `20260730`，所以20个环境完全相同。比较两个
目录的 `summary.json` 和 `episodes.csv`，才能判断增加0.05 m带来的安全性、
到达率和效率变化。

### 常用覆盖参数

```bash
# 先快速跑2场检查系统
NUM_EVAL_EPISODES=2 bash src/sensor_cdf/scripts/run_random_env_eval.sh

# 更换模型
MODEL_PATH=/绝对路径/model_best_parametric_bc.pt \
  bash src/sensor_cdf/scripts/run_random_env_eval.sh

# 换一组随机环境
RANDOM_ENV_SEED=12345 bash src/sensor_cdf/scripts/run_random_env_eval.sh

# 打开RViz
USE_RVIZ=true bash src/sensor_cdf/scripts/run_random_env_eval.sh
```

## 结果目录与文件

每次运行目录格式：

```text
results/random_environment/v0.1.2_YYYYMMDD_HHMMSS/
├── config.json
├── scenarios.json
├── episodes.csv
├── episodes.json
├── steps.jsonl
├── summary.json
├── summary.md
├── trajectories.pt
├── aggregate_compat.csv
├── launcher_settings.txt
├── roslaunch_exit_code.txt
├── run.log
└── legacy_trajectories/
```

- `summary.md`：人直接阅读的总报告。
- `summary.json`：程序绘图和跨版本比较用的聚合指标。
- `episodes.csv/json`：每个随机环境一行/一项，适合找失败场景。
- `steps.jsonl`：每个控制周期一行，含位姿、目标距离、LiDAR点数和最近距离、
  nominal/safe控制量、控制修正、真实/CDF净空、推理耗时和QP回退。
- `scenarios.json`：所有障碍物坐标、场景种子、起终点。
- `config.json`：模型、参数、半径、版本、Git状态和运行状态。
- `trajectories.pt`：兼容原有PyTorch轨迹绘图代码。
- `run.log`：原来只显示在终端的ROS日志。

程序每完成一场就覆盖保存一次完整快照。若手动中断，`config.json` 中状态为
`interrupted`，已经完成的场景仍可分析。

## 建议重点分析的指标

- `arrival_rate`：进入目标半径的比例，不考虑是否碰撞。
- `success_rate`：到达且没有触发评测碰撞阈值的比例。
- `physical_collision_rate`：按真实半径之和 `0.56 m` 计算的碰撞比例。
- `cdf_envelope_violation_rate`：是否进入 `0.61 m` 的CDF安全包络。
- `path_efficiency`：本场朝目标取得的直线进展除以实际路径长度，越接近1越直接；
  这样不会因为机器人进入目标半径但未到目标中心而出现大于1的虚假效率。
- `min_physical_clearance_m` / `min_cdf_clearance_m`：最小净空，负值表示违反。
- `mean/max_control_correction`：安全层相对nominal控制修改了多少。
- `mean/p95/max_inference_ms`：推理实时性。
- `qpth_failures`、`qpth_fallback_steps`：QP稳定性。本一键测试禁用了JAX二次回退；
  safety layer自身的 `qp_fail_mode=fallback` 事件仍会计数，不能忽略。

分析时不要只展示成功率。至少同时给出到达率、真实碰撞率、CDF包络违反率、
超时率、路径效率、最小净空和QP失败数。
