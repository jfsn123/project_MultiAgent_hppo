# 基于分层 PPO 的人形机器人爬梯控制系统

本项目在 Webots 中使用 ROBOTIS OP2 人形机器人研究爬梯任务，并采用分层 PPO（Hierarchical PPO，HPPO）组织高层决策和各阶段连续动作控制。

完整任务被拆分为四个阶段：

1. 高层决策：根据压力、手脚距离和足部位置选择下一阶段；
2. 地面抓取：控制手臂抓住台阶或支撑点；
3. 台阶踏步：控制腿部完成抬脚和落脚；
4. 台阶上抓取：在更高位置完成左右手抓取，为继续攀爬提供支撑。

项目同时提供分阶段训练、单阶段批量测试和完整爬梯流程测试。

## 项目结构

```text
project_MultiAgent_hppo/
├─ controllers/
│  ├─ Train_main/
│  │  └─ Train_main.py          # 机器人主控制器；选择训练或测试模式
│  └─ new_ti_zi/
│     └─ new_ti_zi.py           # Supervisor；重置仿真和随机化初始位置
├─ python_scripts/
│  ├─ PPO/
│  │  ├─ checkpoint/            # decision/catch/tai/upper 模型文件
│  │  ├─ episode/               # 四阶段训练流程
│  │  ├─ hppo/                  # HPPO 网络、动作采样和更新逻辑
│  │  ├─ log/                   # 各阶段训练日志和测试结果
│  │  ├─ Robot_Run/             # 地面抓取、踏步、台阶上抓取的底层控制
│  │  ├─ test/                  # 分阶段及完整流程测试
│  │  └─ utils/                 # 配置、Webots 接口、日志和绘图工具
│  ├─ resetFlag.txt             # 主控制器与 Supervisor 的重置信号
│  ├─ resetFlag1.txt            # 辅助重置信号
│  └─ robot_pos.txt             # Supervisor 写入的随机初始位置
├─ worlds/
│  └─ jiu_jia_zhua3.wbt         # Webots R2021a 仿真世界
├─ hppo.pdf                     # HPPO 参考资料
├─ ppo.pdf                      # PPO 参考资料
└─ readme.md
```

核心训练文件：

| 文件 | 作用 |
| --- | --- |
| `PPO_episoid_0.py` | 总调度器，连接高层决策和三个动作阶段 |
| `PPO_episoid_1.py` | 地面抓取训练 |
| `PPO_episoid_2.py` | 台阶踏步训练 |
| `PPO_episoid_3.py` | 台阶上抓取训练 |
| `hppo.py` | 高层离散决策策略 |
| `hppo_01.py` | 各动作阶段使用的混合/连续动作策略 |

## 运行环境

- Windows（项目当前路径配置使用 Windows 绝对路径）；
- Webots R2021a，或能够兼容该世界文件和 ROBOTIS OP2 控制器的版本；
- Python 3.7；
- NumPy；
- PyTorch；
- Pillow；
- Matplotlib（仅运行日志绘图工具时需要）；
- Tkinter（仅运行日志绘图工具时需要，通常随 Python 安装）。

示例安装命令：

```powershell
pip install numpy torch pillow matplotlib
```

## 运行前配置

### 1. 修改绝对路径

项目默认根目录写为：

```text
D:/project_MultiAgent_hppo
```

如果项目存放在其他位置，需要检查并修改：

- `controllers/Train_main/Train_main.py` 中的 `sys.path.append(...)`；
- `controllers/new_ti_zi/new_ti_zi.py` 中 `robot_pos.txt`、`resetFlag.txt` 的路径；
- `Project_config.py` 中的 `path_list`；
- `Webots_interfaces.py` 中可能存在的项目路径。

主要配置项包括：

```python
path_list = {
    "resetFlag": ".../python_scripts/resetFlag.txt",
    "resetFlag1": ".../python_scripts/resetFlag1.txt",
    "model_path_decision_PPO_h": ".../checkpoint/decision",
    "model_path_catch_PPO_h": ".../checkpoint/catch",
    "model_path_tai_PPO_h": ".../checkpoint/tai",
    "model_path_upper_PPO_h": ".../checkpoint/upper",
    "decision_log_path_PPO": ".../log/decision_log/",
    "catch_log_path_PPO": ".../log/catch_log/",
    "tai_log_path_PPO": ".../log/tai_log/",
    "upper_log_path_PPO": ".../log/upper_log/",
}
```

`gps_goal`、`gps_goal1` 和 `gps_goal2` 分别对应地面抓取、踏步和台阶上抓取阶段的目标位置。修改世界模型或台阶尺寸后，应同步校准这些目标。

### 2. 检查 Webots 控制器

打开 `worlds/jiu_jia_zhua3.wbt`。该世界已配置两个控制器：

- Supervisor：`new_ti_zi`；
- ROBOTIS OP2：`Train_main`。

Supervisor 负责接收重置信号、重置物理状态并随机化机器人的初始位置；`Train_main` 负责训练和测试。

## 训练

在 `controllers/Train_main/Train_main.py` 中设置 `TRAIN_MODE`：

| 模式 | 训练内容 |
| --- | --- |
| `"full"` | 决策 + 地面抓取 + 踏步 + 台阶上抓取 |
| `"no_upper"` | 决策 + 地面抓取 + 踏步 |
| `"catch_only"` | 仅训练地面抓取 |
| `"tai_only"` | 仅训练踏步，加载已有决策和抓取模型 |
| `"upper_only"` | 仅训练台阶上抓取，加载已有前置阶段模型 |

例如：

```python
TRAIN_MODE = "full"
```

然后在 Webots 中打开世界文件并启动仿真。主控制器会调用：

```python
PPO_episoid_0(
    robot=robot,
    decision_model_path=None,
    catch_model_path=None,
    tai_model_path=None,
    upper_model_path=None,
    train_upper=train_upper,
    train_decision=train_decision,
    train_catch=train_catch,
    train_tai=train_tai,
)
```

模型路径传入 `None` 时，初始化逻辑会从配置目录查找已有 checkpoint；是否继续训练和保存由对应 `train_*` 开关控制。当前训练上限在 `PPO_episoid_0.py` 中设为 10000 个 episode，checkpoint 通常每 500 个 episode 保存一次。

### 模型文件命名

```text
decision/decision_hppo_<episode>.ckpt
catch/catch_hppo_<episode>.ckpt
tai/tai_agent_<total_episode>_<episode>.ckpt
upper/upper_agent_<total_episode>_<episode>.ckpt
```

涉及前置阶段的训练或测试会自动选择目录中编号最新的匹配模型。缺少必要模型时会抛出 `FileNotFoundError`。

## 测试

同样通过 `TRAIN_MODE` 选择测试模式：

| 模式 | 测试内容 | 返回值 |
| --- | --- | --- |
| `"test_decision_batch"` | 批量测试所有决策模型 | 每个模型的成功率列表 |
| `"test_catch_batch"` | 批量测试所有地面抓取模型 | 每个模型的成功率列表 |
| `"test_tai_batch"` | 批量测试所有踏步模型 | 每个模型的成功率列表 |
| `"test_upper_grasp_only"` | 批量测试所有台阶上抓取模型 | 每个模型的成功率列表 |
| `"test_full_climb"` | 使用各阶段最新模型测试完整爬梯流程 | 完整流程汇总字典 |

批量测试默认对每个模型执行 100 次有效测试。列表中的单项通常采用以下格式：

```json
{
  "model_path": ".../model.ckpt",
  "episodes": 100,
  "success_rate": 0.87
}
```

完整爬梯测试返回模型路径、测试参数、每轮攀爬阶数以及各阶段累计统计。例如：

```python
results = full_climb_test(robot)
cycle_count = len(results["cycles"])
max_stairs = results["stats"]["max_stairs"]
```

测试结果默认写入 `python_scripts/PPO/log/test_log/`：

```text
decision_model_success_rates.json
catch_ground_success_rates.json
tai_ground_success_rates.json
upper_grasp_model_success_rates.json
full_climb_test_result.json
```

其中前四个文件的 JSON 顶层为模型结果数组；完整爬梯测试文件记录各轮的攀爬结果。

## 日志可视化

`python_scripts/PPO/utils/Draw.py` 提供基于 Tkinter 和 Matplotlib 的多文件曲线查看工具：

```powershell
python python_scripts/PPO/utils/Draw.py
```

该工具适合加载训练产生的 JSON 日志、选择横纵坐标字段并比较多条训练曲线。若系统 Python 不包含 Tkinter，需要先安装对应的 Tk 组件。

## 常见问题

### `ModuleNotFoundError: No module named 'controller'`

`controller` 是 Webots Python API。请从 Webots 启动 `Train_main`，并确认 Webots 使用的 Python 解释器已安装 NumPy、PyTorch 和 Pillow。

### 找不到 `python_scripts.PPO.utils.Project_config`

确认 `Train_main.py` 添加的项目根目录正确，并确认 `python_scripts/PPO/utils/Project_config.py` 存在。

### 找不到 checkpoint

确认四类模型位于 `path_list` 指定目录，并符合约定的文件名格式。`tai_only`、`upper_only` 和完整流程测试依赖前置阶段模型。

### 修改代码后 Webots 仍运行旧版本

停止仿真并重新加载控制器；同时确认世界文件绑定的是本项目的 `Train_main` 和 `new_ti_zi` 控制器。

### 仿真无法正常重置

检查 `resetFlag.txt`、`robot_pos.txt` 的绝对路径和读写权限，并确认 `new_ti_zi` 节点启用了 Supervisor 权限。

## 目前还存在的问题

踩踏台阶模型测试时发现机器人总是踩错台阶，可以尝试修改奖励函数或者修改状态输入，放大机器人位置带来的影响
## 参考资料

- `ppo.pdf`：Proximal Policy Optimization 相关资料；
- `hppo.pdf`：Hierarchical PPO 相关资料；
- Webots 文档：机器人控制器、Supervisor 和 Python API；
- ROBOTIS OP2 文档：关节、传感器和运动控制接口。
