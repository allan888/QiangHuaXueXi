# reproduce_main2.py 代码讲解

## 论文来源

复现论文: **"Q-learning-based unmanned aerial vehicle path planning with dynamic obstacle avoidance"**
- 作者: Amala Sonny, Sreenivasa Reddy Yeduri, Linga Reddy Cenkeramaddi
- 期刊: Applied Soft Computing 147 (2023) 110773

---

## 使用的算法: Q-Learning + SDP 策略（强化学习，非深度学习）

本论文同样使用的是 **Q-Learning（Q学习）** 及其改进变体，属于强化学习中的 **无模型（model-free）时序差分（Temporal Difference）** 算法，**不涉及深度学习**。它通过 Q 表（而非神经网络）存储状态-动作价值。

### 核心公式

```
Q(s,a) ← Q(s,a) + α × [r + γ × max_a' Q(s',a') - Q(s,a)]
```

其中:
- `α` (学习率): 固定 **0.3**
- `γ` (折扣因子): 从 **0.1 线性增长到 0.9**（随训练进程）
- `ε` (探索率): 固定 **0.9**（ε-greedy 策略）
- 经验回放缓冲区大小: **5000**

### 论文贡献: SDP 策略

论文提出了 **SDP（Shortest Distance Prioritization，最短距离优先）** 策略来替代传统 ε-greedy：

- **ε-greedy**: 以 ε=0.9 概率随机探索，0.1 概率选择最大 Q 值动作 → 探索过多，收敛困难
- **SDP**: 对每个候选动作，计算执行后位置到终点的曼哈顿距离，**贪心地选择距离最小的动作** → 利用先验知识加速收敛

### 实现的四种算法

| 算法 | Q-learning 类型 | 动作选择 | 障碍物 |
|------|-----------------|----------|--------|
| Alg1 | 原始 Q-learning | ε-greedy | 仅静态 |
| Alg2 | 原始 Q-learning | ε-greedy | 静态 + 2 动态 |
| Alg3 | 改进 Q-learning | **SDP** | 仅静态 |
| Alg4 | 改进 Q-learning | **SDP** | 静态 + 2/4 动态 |

### 对比算法

- **SARSA**: On-policy 时序差分学习（仅静态环境）
- **A\***: 经典启发式搜索（仅静态，作为上界参照）
- **Dijkstra**: 最短路径搜索（仅静态，作为上界参照）

---

## 文件结构总览

```
reproduce_main2.py  (约 992 行)
├── UAVEnvironment              # 环境类（25×25 栅格）
├── manhattan_dist()            # 曼哈顿距离辅助函数
├── QLearningBase               # Q-learning 基类（核心算法）
├── Alg1_OriginalQL_Static      # 算法1: 原始 QL + ε-greedy，仅静态
├── Alg2_OriginalQL_Dynamic     # 算法2: 原始 QL + ε-greedy，静态+动态
├── Alg3_ProposedQL_Static      # 算法3: QL + SDP，仅静态
├── Alg4_ProposedQL_Dynamic     # 算法4: QL + SDP，静态+动态
├── SarsaStatic                 # SARSA 对比算法
├── astar()                     # A* 路径规划
├── dijkstra()                  # Dijkstra 路径规划
├── plot_env()                  # 绘制环境地图
├── plot_path_on()              # 在图上叠加路径
├── plot_convergence()          # 绘制训练收敛曲线
└── main()                      # 主程序入口
```

---

## 逐类/逐函数讲解

### 1. `UAVEnvironment` 类 (第 50-197 行)

25×25 栅格环境，模拟无人机（UAV）飞行空间。

| 方法 | 作用 |
|------|------|
| `__init__(grid_size, n_dynamic, seed)` | 初始化 25×25 栅格，构建静态障碍物，初始化指定数量的动态障碍物 |
| `_build_static_obstacles()` | 按论文 Fig.1 布局创建 4 个静态障碍物：横向长条(左上)、竖向长条(中右)、L形(右下)、方块(右上) |
| `_init_dynamic_obstacles()` | 随机初始化动态障碍物位置，确保不与静态障碍物、起点、终点重合 |
| `move_dynamic_obstacles()` | 每个动态障碍物**随机选择方向**移动 1 格（与无人机同速），包含停留动作 |
| `is_obstacle(pos)` | 判断位置是否被任何障碍物占据（静态+动态） |
| `is_valid(pos)` | 判断位置是否合法（界内且无障碍物） |
| `reset()` | 重置环境：无人机回起点，动态障碍物重新随机初始化 |
| `step(action)` | 执行动作，返回 `(next_state, reward, done)` |

#### step() 奖励规则

| 情况 | 奖励 | 是否终止 |
|------|------|----------|
| 越界（超出 25×25 范围） | -1 | 是 |
| 到达终点 (24,24) | +1 | 是 |
| 碰到障碍物（静态或动态） | -1 | 是 |
| 正常移动 | 0 | 否 |

### 2. `manhattan_dist(p1, p2)` 函数 (第 203-204 行)

计算两点间的曼哈顿距离: `|x1-x2| + |y1-y2|`，用于 SDP 策略中评估动作优劣。

### 3. `QLearningBase` 类 (第 207-346 行)

Q-learning 基类，封装核心算法逻辑，被四个算法子类继承。

| 方法 | 作用 |
|------|------|
| `__init__(env, alpha, gamma_start, ...)` | 初始化 Q 表 `(25, 25, 4)`，经验回放缓冲区，记录列表 |
| `get_gamma(episode)` | γ 从 0.1 **线性增长**到 0.9：`γ = 0.1 + 0.8 × (episode / episodes)` |
| `choose_action_egreedy(state)` | **ε-greedy 策略**（论文公式5）：以 ε=0.9 概率随机探索，0.1 概率选择最大 Q 值动作 |
| `choose_action_sdp(state)` | **SDP 策略**（论文 Algorithm 3/4）：对 4 个方向分别计算执行后位置到终点的曼哈顿距离，选最小距离动作；若碰撞/越界则距离为 ∞ |
| `update_q(state, action, reward, next_state, gamma)` | Q-learning 更新（论文公式7-9）：`Q += α × [r + γ × max Q(s') - Q(s,a)]`，同时存入经验回放缓冲区 |
| `experience_replay(gamma, batch_size)` | 从回放缓冲区随机采样 32 条经验进行额外学习，打破样本间相关性 |
| `get_path()` | 从学到的 Q 表用贪心策略提取规划路径（按 Q 值从高到低尝试动作，避免回环） |

### 4. 四种算法实现类 (第 352-503 行)

这四个类都继承 `QLearningBase`，区别仅在于 `train()` 方法中调用的动作选择函数和环境配置。

| 类 | 对应论文 | 动作选择 | 环境特征 |
|----|----------|----------|----------|
| `Alg1_OriginalQL_Static` | Algorithm 1 | ε-greedy | 仅静态障碍物 |
| `Alg2_OriginalQL_Dynamic` | Algorithm 2 | ε-greedy | 静态+动态障碍物，每步检测碰撞，每步移动动态障碍物 |
| `Alg3_ProposedQL_Static` | Algorithm 3 | SDP | 仅静态障碍物 |
| `Alg4_ProposedQL_Dynamic` | Algorithm 4 | SDP | 静态+动态障碍物，每步检测碰撞，每步移动动态障碍物 |

每个类的 `train()` 方法结构相同:
1. 每回合重置环境
2. 循环：选择动作 → 执行动作 → 更新 Q 表 → 移动动态障碍物（如有）→ 检查碰撞 → 直到终止
3. 每回合结束后执行经验回放

### 5. `SarsaStatic` 类 (第 505-555 行)

SARSA（State-Action-Reward-State-Action）对比算法。

| 关键区别 | Q-learning (off-policy) | SARSA (on-policy) |
|----------|------------------------|-------------------|
| 更新公式 | `Q += α × [r + γ × max Q(s') - Q(s,a)]` | `Q += α × [r + γ × Q(s', a') - Q(s,a)]` |
| 下一动作 | 取 max（与实际执行的动作无关） | 取实际执行的下一个动作 a' |

SARSA 使用与当前策略相同的 ε-greedy 选择下一个动作，因此更保守——它会"知道"自己可能探索到危险区域。

### 6. `astar(env)` 函数 (第 562-592 行)

A* 最短路径搜索。使用曼哈顿距离作为启发式函数 `f(n) = g(n) + h(n)`，保证在栅格地图上找到最优路径。仅能处理静态障碍物。

### 7. `dijkstra(env)` 函数 (第 595-623 行)

Dijkstra 最短路径搜索。等价于 A* 在 `h(n)=0` 时的特例，搜索范围更大。仅能处理静态障碍物。

### 8. 可视化函数 (第 631-681 行)

| 函数 | 作用 |
|------|------|
| `plot_env(ax, env, title)` | 绘制环境地图：白色空地、蓝色静态障碍物、红色动态障碍物，标注起点（红圆）和终点（黄方） |
| `plot_path_on(ax, path, color, label)` | 在环境图上叠加绘制规划路径 |
| `plot_convergence(data, title, ylabel, save_path, window)` | 绘制收敛曲线：原始数据（半透明）+ 滑动平均曲线，保存为 PNG |

### 9. `main()` 函数 (第 690-992 行)

主程序入口，执行流程：

1. **训练四种算法** (5000 回合): Alg1(QL+ε-greedy,静态) → Alg2(QL+ε-greedy,动态) → Alg3(QL+SDP,静态) → Alg4(QL+SDP,动态)
2. **训练对比算法**: SARSA（仅静态）
3. **运行经典算法**: A* 和 Dijkstra
4. **不同动态障碍物数量测试**: SDP + 0/2/4 个动态障碍物的性能对比
5. **输出性能汇总表**（对应论文 Table 2）
6. **生成 7 张图表** (对应论文 Fig.4 ~ Fig.10):
   - Fig.4: 静态环境 4 种算法路径对比
   - Fig.5: 动态环境 SDP-QL 在不同动态障碍物数量下的表现
   - Fig.6-7: 原始 Q-learning 收敛曲线（静态/动态）
   - Fig.8-10: SDP-QL 收敛曲线（0/2/4 个动态障碍物）

---

## 关键观察

- SDP 策略的算法（Alg3/Alg4）始终能找到最优路径（长度 49），而 ε-greedy 变体（Alg1/Alg2）在 5000 回合后仍很少成功——ε=0.9 的探索率对本环境过高
- A* 和 Dijkstra 能瞬间找到最优路径（约 0.001 秒），但无法处理动态障碍物
- 经验回放缓冲区（大小 5000）帮助打破训练样本间的时序相关性

---

## 与 ICMSC 论文的关键区别

| 方面 | main-2 论文 (本文件) | ICMSC 论文 |
|------|----------------------|------------|
| 动作选择 | ε-greedy (ε=0.9) 或 SDP | 纯贪心（无探索） |
| 学习率 α | 0.3 固定 | 0.99→衰减 |
| 折扣因子 γ | 0.1→0.9 线性增长 | 0.25 固定 |
| 环境规模 | **25×25** | 10×10 |
| 静态障碍物 | 4 个（论文 Fig.1） | 1~多个（3 种复杂度环境） |
| Trash point | 无 | **有**（必经点，先收集才能到终点） |
| 经验回放 | **有**（缓冲区 5000） | 无 |
| 动态障碍物 | 随机移动 | 往返移动 |
| 动作空间 | 仅 4 方向 | 环境 3 支持 8 方向 |
| 对比算法 | SARSA, A*, Dijkstra | A* |
