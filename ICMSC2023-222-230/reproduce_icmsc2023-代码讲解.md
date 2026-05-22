# reproduce_icmsc2023.py 代码讲解

## 论文来源

复现论文: **"Path Planning in Dynamic Environments Based on Q-Learning"**
- 作者: Xiangqi Li, Ocean University of China
- 会议: ICMSC 2023, Highlights in Science, Engineering and Technology, Vol.63, pp.222-230

---

## 使用的算法: Q-Learning (强化学习，非深度学习)

本论文使用的是 **Q-Learning（Q学习）**，这是强化学习中的一种经典 **无模型（model-free）时序差分（Temporal Difference）** 算法，**不属于深度学习**。它通过维护一张 Q 表（Q-table）来存储每个状态-动作对的价值估计，而非使用神经网络。

### 核心思想

Q-learning 通过迭代更新 Q 值来学习最优策略：

```
Q(s,a) = Q(s,a) + α × [r + γ × max Q(s',a') - Q(s,a)]
```

其中:
- `α` (学习率): 初始 0.99，每 100 回合衰减 0.01
- `γ` (折扣因子): 固定 0.25
- `r`: 即时奖励
- `max Q(s',a')`: 下一状态的最大 Q 值

### 关键特性

| 特性 | 说明 |
|------|------|
| **动作选择** | 纯贪心（取最大 Q 值，平局随机），**无 ε-greedy 探索** |
| **学习率** | 0.99 初始，每 100 回合减 0.01，模拟从探索到利用的过渡 |
| **环境** | 10×10 栅格，3 种不同复杂度的环境 |
| **动作空间** | 环境 1/2 为 4 动作（上下左右），环境 3 为 8 动作（含对角线） |
| **Trash Point** | 必经点：无人机必须先收集垃圾点才能到达终点获得奖励 |

---

## 文件结构总览

```
reproduce_icmsc2023.py  (约 720 行)
├── DynamicObstacle          # 动态障碍物类（沿线段往返移动）
├── GridEnvironment          # 环境类（10×10 栅格）
├── QLearningAgent           # Q-learning 核心算法
├── astar()                  # A* 路径规划对比
├── plot_env()               # 绘制环境地图
├── plot_path()              # 在图上叠加路径
├── plot_convergence()       # 绘制训练收敛曲线
├── format_q_table()         # 格式化输出 Q 值表
├── run_solution()           # 一次完整实验运行
├── print_action_table()     # 打印起点 Q 值表
└── main()                   # 主程序入口
```

---

## 逐类/逐函数讲解

### 1. `DynamicObstacle` 类 (第 38-77 行)

动态障碍物，沿指定线段做往返运动。

| 方法 | 作用 |
|------|------|
| `__init__(path_start, path_end, start_pos)` | 初始化障碍物的移动路径（线段两端点）和起始位置，自动判断移动轴（水平或垂直）和初始方向 |
| `move()` | 每步沿轴向移动 1 格，碰到端点自动反向（往返运动） |
| `get_pos()` | 返回障碍物当前坐标 |
| `reset()` | 重置障碍物到起始位置和方向 |

### 2. `GridEnvironment` 类 (第 83-248 行)

10×10 栅格环境，管理静态/动态障碍物、trash point 和无人机位置。

| 方法 | 作用 |
|------|------|
| `__init__(env_id, seed)` | 根据 env_id (1/2/3) 初始化对应的环境配置 |
| `_add_block(r_start, r_end, c_start, c_end)` | 辅助方法，批量添加矩形静态障碍物块 |
| `_setup_env1()` | 配置环境 1：1 个静态障碍物 + 1 个动态障碍物 + trash(4,4)，4 动作 |
| `_setup_env2()` | 配置环境 2：2 个静态障碍物 + 1 个动态障碍物 + trash(4,4)，4 动作 |
| `_setup_env3()` | 配置环境 3：多个静态障碍物 + 2 个动态障碍物 + trash(6,3)，8 动作 |
| `is_obstacle(pos)` | 判断位置是否被静态或动态障碍物占据 |
| `is_valid(pos)` | 判断位置是否合法（在界内且无障碍物） |
| `reset()` | 重置环境：无人机回起点，trash 状态重置，动态障碍物回初始位置 |
| `step(action)` | 执行动作，返回 `(next_state, reward, done)`。特殊规则：必须先收集 trash 后到达终点才得奖励 |

#### step() 奖励规则

| 情况 | 奖励 | 是否终止 |
|------|------|----------|
| 越界 | -1 | 是 |
| 碰到障碍物（静态或动态） | -1 | 是 |
| 到达 trash point（首次） | +5 或 +10 | 否（继续） |
| 到达终点（已收集 trash） | +5 | 是 |
| 到达终点（未收集 trash） | 0 | 否（穿过） |
| 其他 | 0 | 否 |

### 3. `QLearningAgent` 类 (第 255-383 行)

Q-learning 算法的核心实现。

| 方法 | 作用 |
|------|------|
| `__init__(env, alpha_start, ...)` | 初始化 Q 表 `(10, 10, num_actions)`，全部为零 |
| `get_alpha(episode)` | 计算当前回合的学习率：α = 0.99 - (episode // 100) × 0.01，不小于 0.01 |
| `choose_action(state)` | **纯贪心选择**：取当前状态下 Q 值最大的动作，多个最大时随机选。无 ε-greedy |
| `update_q(state, action, reward, next_state)` | Q-learning 核心更新：`Q += α × [r + γ × max Q(s') - Q(s,a)]` |
| `train(episodes, verbose, print_interval)` | 训练主循环：每回合重置环境，循环执行 choose_action → step → update_q，直到终止或达到最大步数 |
| `get_path()` | 从学到的 Q 表中用贪心策略提取路径（临时关闭动态障碍物进行评估） |

### 4. `astar(env)` 函数 (第 390-434 行)

A* 路径规划算法作为对比基线（仅处理静态障碍物）。

| 内部函数 | 作用 |
|----------|------|
| `astar_segment(from_pos, to_pos)` | 用 A* 计算两点间最短路径，支持 4 方向或 8 方向，对角线代价为 √2 |
| 主逻辑 | 分两段规划：start → trash → end，拼接成完整路径 |

### 5. 可视化函数 (第 441-513 行)

| 函数 | 作用 |
|------|------|
| `plot_env(ax, env, title, show_dynamic)` | 在 matplotlib 坐标轴上绘制环境：白色空地、天蓝静态障碍、红色动态障碍，标注起点(红圆)、终点(深蓝方)、trash(黄菱) |
| `plot_path(ax, path, color, linewidth, label)` | 在环境图上叠加绘制规划路径 |
| `plot_convergence(episode_rewards, title, save_path, window)` | 绘制训练收敛曲线（原始数据 + 滑动平均），保存为 PNG |
| `format_q_table(q_table, state)` | 格式化输出某个状态的 Q 值，用于打印 Q 表 |

### 6. 实验管理函数 (第 519-583 行)

| 函数 | 作用 |
|------|------|
| `run_solution(env_id, episodes, num_runs, seed_base)` | 对指定环境运行多次训练，统计成功率、路径长度、训练时间 |
| `print_action_table(runs, env_id)` | 打印起点 (0,0) 处的 Q 值表，对应论文 Table 1/2/3 |

### 7. `main()` 函数 (第 585-720 行)

主程序入口，执行流程：
1. 对 3 个环境各运行 3 次 Q-learning 训练（env1: 1000 回合, env2: 2000 回合, env3: 8000 回合）
2. 运行 A* 对比实验
3. 生成 7 张图表：Fig1（3 个环境地图）、每个环境的 3 条最优路径图、每个环境的收敛曲线图
4. 输出性能汇总表

---

## 与 main-2 论文的关键区别

| 方面 | ICMSC 论文 (本文件) | main-2 论文 |
|------|---------------------|-------------|
| 动作选择 | **纯贪心**（无探索） | ε-greedy（ε=0.9）或 SDP |
| 学习率 α | 0.99→衰减 | 0.3 固定 |
| 折扣因子 γ | 0.25 固定 | 0.1→0.9 线性增长 |
| 环境规模 | 10×10 | 25×25 |
| Trash point | **有**（必经点） | 无 |
| 经验回放 | 无 | 有（缓冲区 5000） |
| 动态障碍物 | 往返移动 | 随机移动 |
