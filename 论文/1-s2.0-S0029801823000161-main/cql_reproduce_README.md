# CQL (Classical Q-learning) + ε-greedy 复现 — 函数说明文档

> 对应文件: `cql_reproduce.py`  
> 论文: *A path planning approach for unmanned surface vehicles based on dynamic and fast Q-learning*  
> Hao B., Du H., Yan Z. — Ocean Engineering 270 (2023) 113632

---

## 目录

1. [全局常量](#1-全局常量)
2. [GridEnvironment (栅格环境类)](#2-gridenvironment-栅格环境类)
3. [地图生成函数](#3-地图生成函数)
4. [ClassicalQLearning (Q-learning核心类)](#4-classicalqlearning-q-learning核心类)
5. [评价函数](#5-评价函数)
6. [可视化函数](#6-可视化函数)
7. [main() 主程序](#7-main-主程序)

---

## 1. 全局常量

### `ACTION_DELTAS` (dict)

8 方向动作的位移增量映射。键为动作编号 (0~7)，值为 `(dx, dy)` 元组。

```python
0: (0, 1)   → 上
1: (1, 1)   → 右上
2: (1, 0)   → 右
3: (1, -1)  → 右下
4: (0, -1)  → 下
5: (-1, -1) → 左下
6: (-1, 0)  → 左
7: (-1, 1)  → 左上
```

### `N_ACTIONS` (int)

动作总数，固定为 8。

### `REWARD_GOAL` (float)

到达目标点的奖励值：**10.0**（论文公式14）。

### `REWARD_OBSTACLE` (float)

碰到障碍物/边界的惩罚值：**-10.0**（代替论文中的 `-inf`）。

### `REWARD_MOVE` (float)

普通移动的奖励值：**0.0**。标准 Q-learning 路径规划中使用零步奖励，使 Q 值从目标点向外梯度传播。

### `ALPHA` (float)

学习率：**0.9**（论文 Table 1）。

### `GAMMA` (float)

折扣因子：**0.9**（论文 Table 1）。

### `EPSILON_START` / `EPSILON_END` (float)

探索率 ε 的初始值 **0.5** 和终止值 **0.05**。每 episode 线性衰减。

---

## 2. GridEnvironment (栅格环境类)

论文 Section 2.1 / Section 4.2

### `__init__(self, grid, start, goal)`

初始化 30×20 栅格海洋环境。

| 参数 | 类型 | 说明 |
|------|------|------|
| `grid` | np.ndarray (rows×cols) | 二值矩阵，0=可行区域，1=障碍物 |
| `start` | tuple (x, y) | USV 起始位置 |
| `goal` | tuple (x, y) | USV 目标位置 |

### `is_valid(self, pos) → bool`

判断坐标 `pos` 是否在网格内且非障碍物。

### `reset(self) → tuple`

将智能体位置重置为起始点，返回起始坐标。

### `step(self, action) → (next_state, reward, done)`

执行一步动作，返回 `(新位置, 奖励, 是否终止)`。

| 参数 | 类型 | 说明 |
|------|------|------|
| `action` | int (0~7) | 8方向动作编号 |

**返回逻辑**:
- 新位置 = 目标 → 奖励 +10，done=True
- 新位置 = 障碍物/越界 → 奖励 -10，done=False，原地不动
- 新位置 = 空地 → 奖励 0，done=False，移动到新位置

> 注意：碰到障碍物**不终止 episode**，智能体原地不动并继续探索。

---

## 3. 地图生成函数

对应论文 Fig.5 的 M01-M06，使用起点 (1, 1) 和终点 (28, 18)。

### `build_map_m01() → (grid, start, goal)`

**M01**: 简单稀疏环境（群岛场景），7 个障碍块。

### `build_map_m02() → (grid, start, goal)`

**M02**: 中等复杂度（河流/狭道场景），7 个障碍块。

### `build_map_m03() → (grid, start, goal)`

**M03**: 较复杂（多岛/港口场景），12 个障碍块。

### `build_map_m04() → (grid, start, goal)`

**M04**: 复杂狭道（港口场景），12 个障碍块，包含横向狭窄通道。

### `build_map_m05() → (grid, start, goal)`

**M05**: 高密度障碍，21 个散落障碍块。

### `build_map_m06() → (grid, start, goal)`

**M06**: 迷宫型，13 个障碍块，需绕行。

### `MAP_BUILDERS` (dict)

地图名称到构建函数的映射字典，方便批量调用。

---

## 4. ClassicalQLearning (Q-learning核心类)

论文 Section 2.2 — 标准 Q-learning + ε-greedy 策略。

### `__init__(self, env, alpha, gamma, eps_start, eps_end, max_episodes, max_steps, seed)`

初始化 Q-learning 算法。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env` | GridEnvironment | — | 栅格环境 |
| `alpha` | float | 0.9 | 学习率 |
| `gamma` | float | 0.9 | 折扣因子 |
| `eps_start` | float | 0.5 | 初始探索率 |
| `eps_end` | float | 0.05 | 最终探索率 |
| `max_episodes` | int | 2000 | 训练总轮数 |
| `max_steps` | int | 3000 | 每轮最大步数 |
| `seed` | int | None | 随机种子 |

**内部状态**:
- `self.q_table` — 形状 `(rows, cols, 8)` 的 Q 值表
- `self.episode_steps` — 记录每轮步数
- `self.episode_rewards` — 记录每轮总奖励
- `self.episode_success` — 记录每轮是否到达目标

---

### `_decay_epsilon(self, ep)`

**ε 线性衰减**。将 ε 从 `eps_start` 线性衰减到 `eps_end`。

```
epsilon = eps_start + (eps_end - eps_start) * (ep / (max_episodes - 1))
```

早期高 ε (0.5) 鼓励探索，后期低 ε (0.05) 改为贪心利用。

---

### `update_q(self, state, action, reward, next_state)`

**Q 表更新** — 论文公式 (1)。

```
Q(s, a) ← (1 - α) × Q(s, a) + α × [r + γ × max_a Q(s', a')]
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `state` | tuple (x, y) | 当前状态 |
| `action` | int (0~7) | 执行的动作 |
| `reward` | float | 即时奖励 |
| `next_state` | tuple (x, y) | 下一状态 |

实现步骤:
1. 取下一状态所有动作的最大 Q 值
2. 计算 TD 目标: `reward + γ × max_Q`
3. 计算 TD 误差: `TD_target - Q(s, a)`
4. 更新 Q 值: `Q(s, a) += α × TD_error`

---

### `choose_action(self, state) → int`

**ε-greedy 动作选择**。

- 以概率 ε 随机选择一个动作 (0~7)
- 以概率 1-ε 选择 Q 值最大的动作，平局时随机

---

### `train(self, verbose=True)`

**训练主循环**，共 `max_episodes` 轮。

每轮执行：
1. 调用 `_decay_epsilon(ep)` 更新探索率
2. 重置环境到起始点
3. 循环执行直到到达目标或达到 `max_steps` 上限：
   - `choose_action` 选动作
   - `env.step` 执行动作
   - `update_q` 更新 Q 表
4. 记录本轮步数和奖励

当 `verbose=True` 时，每 500 轮打印一次训练进度。

---

### `get_path(self) → (path, success)`

利用训练好的 Q 表，**从起点贪心提取路径**。

策略：
1. 从起始点出发
2. 每一步：取当前状态 Q 值最大的动作方向
3. 如果该方向指向已访问节点，按 Q 值降序尝试下一个方向
4. 到达目标点返回 True，无法移动返回 False

返回 `(路径点列表, 是否成功到达目标)`。

---

## 5. 评价函数

论文 Section 4.1

### `compute_path_length(path) → float`

**路径总长度** — 论文公式 (20)。

计算路径相邻点的欧几里得距离之和：

```
PathLength = Σ_i sqrt((x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2)
```

---

### `compute_turning_angle(path) → float`

**总转弯角度** (弧度) — 论文公式 (21)-(24)。

使用三边公式计算每个中间点的转向角：

```
a_i = distance(i-1, i)    # 前一段长度
b_i = distance(i, i+1)    # 后一段长度
c_i = distance(i-1, i+1)  # 弦长
angle_i = π - arccos((a_i^2 + b_i^2 - c_i^2) / (2 × a_i × b_i))
```

对所有中间点求和得到总转弯角度。值越小表示路径越平滑。

---

## 6. 可视化函数

### `plot_environment(ax, env, title)`

在 matplotlib 坐标轴上绘制栅格环境：
- 白色 = 可行区域
- 黑色 = 障碍物
- 红色圆 = 起点
- 绿色星 = 终点

### `plot_path(ax, path, color, label, linewidth)`

在已绘制的环境图上叠加路径线条。

### `plot_convergence(episode_data, title, ylabel, filepath, window)`

绘制训练收敛曲线：
- 半透明线 = 原始每轮数据
- 深色线 = 滑动平均（默认窗口 50）

保存为 PNG 文件。

---

## 7. main() 主程序

`main()` 函数执行完整的实验流程：

1. **遍历 6 张地图 M01-M06**
2. 每张地图**重复训练 30 次**（不同随机种子）
3. 记录每次的路径长度、转弯角度、计算时间
4. 打印每张地图的平均值、标准差、成功率
5. 打印汇总对比表
6. 生成两类 PNG 图表（结果保存到 `Results/` 目录）：
   - `CQL_{map}_path.png` — 路径规划可视化
   - `CQL_{map}_convergence.png` — 收敛曲线

### 运行方式

```bash
cd "D:\QiangHuaXueXi\论文\1-s2.0-S0029801823000161-main"
python cql_reproduce.py
```

### 输出示例

```
 Map     Success    PathLen      Std    Angle      Std  Time(s)      Std
--------------------------------------------------------------------------------
 M01        100%      35.57     1.82    9.896    3.139     1.24     0.11
 M02        100%      39.16     2.06   12.409    2.280     1.47     0.14
 M03        100%      37.26     1.68    9.817    2.473     1.28     0.10
 M04        100%      37.48     1.12   10.341    1.572     1.36     0.17
 M05        100%      35.11     1.31    5.733    1.649     1.24     0.11
 M06        100%      37.14     1.62    8.325    2.625     1.46     0.15
```
