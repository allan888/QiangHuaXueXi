# IRL 与 SA-CADRL：奖励函数、策略与数学原理详解

> 本文详细剖析**逆强化学习（IRL）**和**社会感知碰撞避免深度强化学习（SA-CADRL）**的数学原理，重点聚焦 R(s,a) 奖励函数、策略 π 以及核心推导过程。

---

## 一、IRL — 逆强化学习

**代表论文**:  
- Vasquez, Okal, Arras. *"Inverse Reinforcement Learning algorithms and features for robot navigation in crowds: An experimental comparison."* IROS 2014.
- 基础理论: Abbeel & Ng (2004) — Max-Margin IRL; Ziebart et al. (2008) — Maximum Entropy IRL

**核心思想**: 传统 RL 已知 R(s,a) 求解最优策略 π\*；IRL **反过来**——已知专家演示轨迹 τ_E，反推出专家在优化的奖励函数 R(s,a)。IRL 的前提是：专家轨迹是对某个未知奖励函数的最优响应。

---

### 1.1 奖励函数 R(s,a) 的数学形式

IRL 最常见的奖励建模是**特征线性加权**形式：

$$
\boxed{R(s,a) = \omega^\top \phi(s,a) = \sum_{j=1}^{k} \omega_j \cdot \phi_j(s,a)}
$$

其中：
- $\phi(s,a): S \times A \to \mathbb{R}^k$ — **特征映射**，将状态-动作对映射为 k 维特征向量
- $\omega \in \mathbb{R}^k$ — **待学习的权重向量**（IRL 的核心任务）
- k — 特征维度

**路径规划中的典型特征设计**（Vasquez et al. 2014）：

| 特征类别 | $\phi_j$ 含义 | 期望符号 $\omega_j$ |
|----------|----------------|---------------------|
| 默认代价 | $\phi_{\text{default}} = 1$（常开特征，对应步数惩罚） | 负（每走一步都有代价） |
| 人群密度 | 机器人局部邻域内行人数量（分 3 个离散等级） | 负（人多的地方代价高） |
| 速度+朝向 | 行人相对速度和朝向的统计量 | 正/负取决于场景 |
| 相对速度分量 | 按方向分箱：toward / perpendicular / away | toward 为负，away 为正 |
| 社会力（Social Force） | 基于 Helbing 社会力模型的各向异性影响力 | 负（靠近行人代价高） |
| 社会力+相对速度 | 社会力的扩展，加入相对速度项 | 负 |

**示例：基于社会力特征的奖励函数**：

$$
R(s,a) = \underbrace{\omega_0}_{\text{步数惩罚}} + \sum_{i=1}^{N_{\text{peds}}} \underbrace{\omega_1 \cdot f(d_i)}_{\text{距离影响力}} + \underbrace{\omega_2 \cdot f(d_i, \theta_i)}_{\text{各向异性项}} + \underbrace{\omega_3 \cdot f(v_{\text{rel}})}_{\text{相对速度项}}
$$

---

### 1.2 策略 π 的数学形式

#### 1.2.1 Max-Margin IRL（学徒学习，Abbeel & Ng 2004）

Max-Margin IRL 直接学习策略，不显式输出概率分布。策略通过特征期望匹配获得：

$$
\pi^* = \arg\min_{\pi} \left\| \mu(\pi) - \hat{\mu}_E \right\|_2
$$

其中**特征期望**定义为：

$$
\boxed{\mu(\pi) = \mathbb{E}_{s_t \sim \pi}\left[ \sum_{t=0}^{\infty} \gamma^t \cdot \phi(s_t, a_t) \right] \in \mathbb{R}^k}
$$

**经验特征期望**（从 N 条专家轨迹估计）：

$$
\boxed{\hat{\mu}_E = \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T_i} \gamma^t \cdot \phi(s_t^{(i)}, a_t^{(i)})}
$$

**学习算法** — 迭代投影：

```
1. 随机初始化策略 π(0)，计算 μ(0) = μ(π(0))
2. 对于迭代 i = 0, 1, 2, ...:
   a. 设置奖励权重: ω(i) = μ̂_E - μ(i)   （奖赏特征差距大的方向）
   b. 以 R = ω(i)ᵀφ 为奖励，用RL求解最优策略 π(i+1)
   c. 计算新特征期望 μ(i+1)
   d. 设 μ̄(i+1) = 投影(μ̂_E) 到 {μ(0), ..., μ(i+1)} 凸包
   e. 若 ||μ̂_E - μ̄(i+1)|| < ε，停止
   f. ω(i+1) = μ̂_E - μ̄(i+1)
3. 输出: 混合策略或最后一个 ω
```

#### 1.2.2 Maximum Entropy IRL（最大熵 IRL，Ziebart 2008）

这是 Vasquez et al. 论文中对比的第二种方法，也是现代 IRL 的基础。

**核心原则**: 在满足特征匹配约束的前提下，选择熵最大的轨迹分布（最无偏假设）。

**策略形式** — **Softmax over Q-values**：

$$
\boxed{\pi_\omega(a \mid s) = \frac{\exp\big(Q_\omega(s,a)\big)}{\displaystyle\sum_{a' \in \mathcal{A}} \exp\big(Q_\omega(s,a')\big)}}
$$

等价写法（用 partition function）：

$$
\pi_\omega(a_j \mid s_i) = \frac{Z_{s_i, a_j}}{Z_{s_i}}, \quad Z_{s_i} = \sum_{a_j} Z_{s_i, a_j}
$$

**关键区别**：与标准 RL 的确定性贪心策略 $a^* = \arg\max_a Q(s,a)$ 不同，MaxEnt IRL 的策略是**随机的**——Q 值越大的动作被选中的概率指数级地大于 Q 值小的动作，但所有动作都有非零概率。带温度参数 α 时：

$$
\pi_\omega(a \mid s) = \frac{\exp\big(Q_\omega(s,a) / \alpha\big)}{\sum_{a'} \exp\big(Q_\omega(s,a') / \alpha\big)}
$$

- α → 0：趋向贪心确定性策略
- α → ∞：趋向均匀随机策略

---

### 1.3 MaxEnt IRL 的核心算法：Soft Value Iteration

#### 软 Bellman 方程

**标准 RL (Hard)**:
$$
V(s) = \max_a Q(s,a)
$$
$$
Q(s,a) = R(s,a) + \gamma \sum_{s'} P(s' \mid s,a) \cdot V(s')
$$

**MaxEnt IRL (Soft)**:
$$
\boxed{V^{\text{soft}}(s) = \log \sum_{a \in \mathcal{A}} \exp\big(Q^{\text{soft}}(s,a)\big)}
$$

$$
\boxed{Q^{\text{soft}}(s,a) = R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s,a) \cdot V^{\text{soft}}(s')}
$$

log-sum-exp 是 max 的平滑近似：$\lim_{\alpha \to 0} \alpha \log\sum e^{x_i/\alpha} = \max(x_i)$

#### 算法流程（有限视界 H）

**后向传播（Backward Pass）**— 计算软值函数和策略：

```
初始化: V_H(s) = 0 (终结状态)

for t = H-1, H-2, ..., 0:
    for each state s:
        for each action a:
            Q_t(s,a) = R_ω(s,a) + γ · Σ_{s'} P(s'|s,a) · V_{t+1}(s')
        V_t(s) = log( Σ_a exp(Q_t(s,a)) )   # softmax over actions

for each state s, action a:
    π_t(a|s) = exp( Q_t(s,a) - V_t(s) )     # 等价于 softmax
```

**前向传播（Forward Pass）**— 计算状态访问频率：

```
初始化: D_0(s_start) = 1

for t = 0, 1, ..., H-2:
    for each state s':
        D_{t+1}(s') = Σ_{s} Σ_{a} D_t(s) · π_t(a|s) · P(s'|s,a)
```

#### 学习权重 ω — 最大似然梯度

给定专家轨迹集 $\mathcal{D} = \{\tau_1, \tau_2, \ldots, \tau_N\}$，最大化对数似然：

$$
\omega^* = \arg\max_\omega \mathcal{L}(\omega) = \arg\max_\omega \sum_{\tau \in \mathcal{D}} \log P(\tau \mid \omega)
$$

**梯度** — 经验特征期望与模型特征期望的差：

$$
\boxed{\nabla_\omega \mathcal{L}(\omega) = \underbrace{\frac{1}{N}\sum_{\tau \in \mathcal{D}} \sum_{t} \phi(s_t, a_t)}_{\text{专家特征期望 } \hat{\mu}_E} \;-\; \underbrace{\sum_{s} D_s \cdot \phi(s)}_{\text{模型特征期望}}}
$$

更新规则：$\omega \leftarrow \omega + \eta \cdot \nabla_\omega \mathcal{L}$

**Intuition**: 梯度推动 ω 朝着增大专家常用特征权重、减小模型偏好特征权重的方向移动。收敛时两者特征期望相等。

---

## 二、SA-CADRL — 社会感知碰撞避免深度强化学习

**代表论文**:  
- **CADRL**: Chen, Y. F., Liu, M., Everett, M., & How, J. P. *"Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning."* ICRA 2017.
- **SA-CADRL**: Chen, Y. F., Everett, M., Liu, M., & How, J. P. *"Socially Aware Motion Planning with Deep Reinforcement Learning."* IROS 2017. （🏆 IROS 2017 Best Student Paper Award）

**核心思想**: 与其建模人类"该做什么"（极其复杂），不如建模人类"**不该做什么**"（违反社会规范的行为）。用深度强化学习学出一个时间高效的导航策略，该策略同时尊重常见的社会规范。

---

### 2.1 状态空间与动作空间

#### 状态空间 — 联合状态 (Joint State)

SA-CADRL 将场景中所有智能体的信息编码为联合状态：

$$
\boxed{s_t = \big[ s_t^{\text{self}}, s_t^{(1)}, s_t^{(2)}, \ldots, s_t^{(n)} \big]}
$$

其中每个智能体（机器人或行人）的状态：

$$
s_t^{(i)} = \big[ p_x, p_y, v_x, v_y, r, \theta, v_{\text{pref}}, \psi \big]^{(i)}
$$

| 符号 | 含义 | 维度 |
|------|------|------|
| $p_x, p_y$ | 二维位置坐标 | 2 |
| $v_x, v_y$ | 二维速度分量 | 2 |
| $r$ | 智能体半径 | 1 |
| $\theta$ | 当前朝向角 | 1 |
| $v_{\text{pref}}$ | 偏好速度大小 | 1 |
| $\psi$ | 目标方向角 | 1 |

**输入处理**: 将行人/机器人状态从世界坐标系转换到**以机器人为中心的坐标系**（ego-centric），保证平移和旋转不变性。

#### 动作空间 — 速度指令离散化

动作空间不直接操作底层控制，而是输出**高层速度指令**：

$$
a_t = (v_t, \omega_t) \quad \text{或} \quad a_t = (\dot{x}, \dot{y})_t
$$

离散化为有限集合：
- **线速度** $v \in \{0, v_{\text{pref}}\}$（或更多离散档位）
- **角速度/航向** $\omega \in \{-\frac{\pi}{4}, -\frac{\pi}{8}, 0, +\frac{\pi}{8}, +\frac{\pi}{4}\}$（典型 5-11 个离散方向）

或者直接离散为速度向量：$a \in \{(\dot{x}_1, \dot{y}_1), \ldots, (\dot{x}_k, \dot{y}_k)\}$，通常 11-33 个动作。

机器人运动学：$p_{t+1} = p_t + \Delta t \cdot v_t$

---

### 2.2 奖励函数 R(s,a) — "不该做什么"的思路

这是 SA-CADRL 最核心的设计。整个奖励函数围绕**惩罚社会违规**和**奖励达成目标**来构造：

$$
\boxed{R(s_t, a_t) = R_{\text{goal}} + R_{\text{collision}} + R_{\text{discomfort}} + R_{\text{step}}}
$$

#### 2.2.1 各分项详解

**① 目标到达奖励 $R_{\text{goal}}$**:

$$
R_{\text{goal}}(s_t) = 
\begin{cases}
+r_{\text{reach}} & \text{if } \|p_t - p_{\text{goal}}\| < d_{\text{goal}} \\
0 & \text{otherwise}
\end{cases}
$$

典型取值：$r_{\text{reach}} = +1.0$ 到 $+10.0$，$d_{\text{goal}} \approx 0.2\text{m}$

---

**② 碰撞惩罚 $R_{\text{collision}}$**:

$$
R_{\text{collision}}(s_t) = 
\begin{cases}
-r_{\text{col}} & \text{if } \min_i \|p_t^{\text{robot}} - p_t^{(i)}\| < r_{\text{robot}} + r_{(i)} \\
0 & \text{otherwise}
\end{cases}
$$

即当机器人与任何行人/障碍物的距离小于两者半径之和时触发碰撞。典型取值：$r_{\text{col}} = -10.0$ 到 $-25.0$（高惩罚保证安全）。

---

**③ 不适感惩罚 $R_{\text{discomfort}}$** — SA-CADRL 的核心创新:

这是 CADRL 与 SA-CADRL 的最大区别。引入一个**不适感区域**（比碰撞半径更大的圆）：

$$
R_{\text{discomfort}}(s_t) = 
\begin{cases}
-r_{\text{disc}} & \text{if } d_{\text{col}} < \min_i \|p_t^{\text{robot}} - p_t^{(i)}\| < d_{\text{disc}} \\
0 & \text{otherwise}
\end{cases}
$$

典型参数：
- $d_{\text{col}} = r_{\text{robot}} + r_{(i)}$ ≈ 0.3m（碰撞边界）
- $d_{\text{disc}}$ ≈ 0.6m–1.0m（不适感边界，约为碰撞边界的 2-3 倍）
- $r_{\text{disc}} = -0.1$ 到 $-0.5$（适中的不适感惩罚）

这种设计让智能体不仅学会不撞人，还学会**保持舒适的社交距离**——像人类在拥挤场景中自然的绕行行为。

---

**④ 时间步惩罚 $R_{\text{step}}$**:

$$
R_{\text{step}} = -r_{\text{time}} \quad \text{（每个时间步固定惩罚）}
$$

典型取值：$r_{\text{time}} = -0.01$ 到 $-0.1$，鼓励智能体尽快到达目标而非无限徘徊。

---

**⑤ 接近目标塑形奖励 $R_{\text{potential}}$**（部分版本使用）:

$$
R_{\text{potential}}(s_t, s_{t+1}) = \eta \cdot \big( d(p_t, p_{\text{goal}}) - d(p_{t+1}, p_{\text{goal}}) \big)
$$

基于势能的塑形奖励，靠近目标得正奖励，远离得负奖励。势能函数 $\Phi(s) = -\eta \cdot d(p, p_{\text{goal}})$

---

#### 2.2.2 完整奖励函数汇总

$$
\boxed{
R(s_t, a_t) = 
\begin{cases}
+r_{\text{reach}} & \text{到达目标} \\
-r_{\text{col}} & \text{碰撞行人/障碍物} \\
-r_{\text{disc}} & d_{\text{col}} < \text{距离} < d_{\text{disc}} \quad \text{（不适感区域）} \\
-r_{\text{time}} & \text{每个时间步} \\
0 & \text{otherwise}
\end{cases}
}
$$

**设计哲学总结**：只需要告诉智能体"别撞人"、"别靠太近"、"快点到"，而不需要显式建模复杂的社交规则——符合社会规范的行为会**涌现**地产生。

---

### 2.3 策略 π — 值函数网络与动作选择

#### 2.3.1 网络架构：Value Network

SA-CADRL 使用**值函数近似**（value-based）而非策略网络（policy-based）。网络结构为对称双流架构：

```
输入层（联合状态 s_t，扁平化）
    │
    ├── 机器人状态流 ──┐
    │   [p, v, r, ψ, v_pref]   │
    │                          ├── 拼接 ──→ 全连接层1 (ReLU, ~150 units)
    ├── 行人1状态流 ──┘        │
    │   [p, v, r, θ]           │
    ├── 行人2状态流 ──┘        ├── 全连接层2 (ReLU, ~100 units)
    │   ...                    │
    ├── 行人n状态流 ──┘        ├── 全连接层3 (ReLU, ~50 units)
    │                          │
                                ↓
                        输出层（|A| 个节点，无激活函数）
                        输出：Q(s, a_1), Q(s, a_2), ..., Q(s, a_k)
```

**对称性的关键**: 行人状态流共享相同的网络权重（parameter sharing），保证排列不变性——交换两个行人的顺序不改变输出。这使得网络能自然泛化到不同数量的行人。

#### 2.3.2 Q 值更新 — Double DQN

SA-CADRL 使用 Double DQN 来缓解过估计问题：

**标准 DQN 更新**:
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ R(s,a) + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a) \right]
$$

**Double DQN 更新**（SA-CADRL 采用）:
$$
\boxed{
y_t^{\text{target}} = R_t + \gamma \cdot Q\big(s_{t+1}, \arg\max_{a'} Q(s_{t+1}, a'; \theta); \theta^-\big)
}
$$

**损失函数**（Huber Loss，对离群点鲁棒）:

$$
\boxed{\mathcal{L}(\theta) = \mathbb{E}_{(s,a,R,s') \sim \mathcal{D}} \left[ \mathcal{H}\big( y^{\text{target}} - Q(s,a; \theta) \big) \right]}
$$

其中 $\mathcal{H}(x) = \begin{cases} 0.5x^2 & |x| < 1 \\ |x| - 0.5 & |x| \geq 1 \end{cases}$ 为 Huber 损失。

---

#### 2.3.3 从 Q 值到策略

推理时，SA-CADRL 使用**单步前瞻**（One-Step Lookahead）从值网络提取策略：

**步骤**:
1. 对当前状态 s，考虑所有候选动作 $a \in \mathcal{A}$
2. 利用运动学模型**预测**：假设行人以当前速度匀速运动一个时间步，计算机器人执行每个 a 后的下一状态 s'
3. 对每个 s' 用值网络评估 $Q(s', a')$
4. 选择最优动作：

$$
\boxed{\pi(s) = \arg\max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \cdot \max_{a'} Q(s', a') \right]}
$$

这个单步前瞻等价于在线执行一步 Model Predictive Control (MPC)，利用了对其他智能体运动的简单预测模型（匀速假设）。

---

#### 2.3.4 训练算法 — GA3C（GPU-accelerated Asynchronous Advantage Actor-Critic 变体）

SA-CADRL 使用了 GA3C 框架进行高效训练：

- **多进程并行**: 多个环境实例并行运行，每个与不同的随机行人配置交互
- **经验回放**: 存储 $(s, a, R, s')$ 四元组到 replay buffer
- **目标网络**: 定期将在线网络 $\theta$ 的参数复制到目标网络 $\theta^-$（如每 500 步）
- **ε-greedy 探索**（训练时）：以概率 ε 选随机动作，否则选 Q 值最大的动作；推理时 ε=0

---

### 2.4 CADRL vs SA-CADRL 的数学区别

| 维度 | CADRL | SA-CADRL |
|------|-------|----------|
| 奖励 | $R = R_{\text{goal}} + R_{\text{collision}} + R_{\text{step}}$ | $R = R_{\text{goal}} + R_{\text{collision}} + \mathbf{R_{\text{discomfort}}} + R_{\text{step}}$ |
| 最小安全距离 | $d_{\text{col}}$（仅物理碰撞） | $d_{\text{disc}} > d_{\text{col}}$（物理+社交双重安全边际） |
| 学习到的行为 | 极近距离擦过行人（bare minimum） | 绕行时保持舒适社交距离 |
| 行人模型假设 | 匀速直线（一个时间步） | 匀速直线（一个时间步） |

SA-CADRL 仅仅在 CADRL 的奖励函数中增加了一项 $R_{\text{discomfort}}$，就能从训练中涌现出显著更自然、更符合人类预期的绕行行为。这体现了**奖励函数设计**在强化学习中的核心重要性。

---

## 三、IRL vs SA-CADRL 的数学对比

| 维度 | IRL (MaxEnt) | SA-CADRL |
|------|-------------|----------|
| **问题设定** | 已知专家轨迹 τ_E，反推奖励函数 R | 已知奖励函数 R，学习策略 π |
| **奖励函数** | $R(s,a) = \omega^\top \phi(s,a)$，ω 是学习目标 | $R = R_{\text{goal}} + R_{\text{col}} + R_{\text{disc}} + R_{\text{step}}$，手工设计 |
| **策略形式** | 随机策略: $\pi(a\mid s) \propto \exp(Q(s,a))$ | 确定性策略: $\pi(s) = \arg\max_a [R + \gamma \max Q(s',a')]$ |
| **值函数** | Soft Bellman: $V = \log\sum\exp(Q)$ | Hard Bellman: $V = \max Q$ |
| **优化目标** | max 对数似然: $\sum \log P(\tau_E\mid\omega)$ | min Bellman 误差: $\mathcal{L}(\theta) = \mathbb{E}[(y - Q)^2]$ |
| **需要专家数据** | 是（专家演示轨迹） | 否（通过环境交互学习） |
| **泛化能力** | 理论上可泛化（学的是奖励） | 需在新场景重新训练 |
| **输出** | 奖励函数参数 ω | 神经网络参数 θ |
| **可解释性** | 高（权重 ω_j 直接解释特征重要性） | 低（神经网络黑箱）|

**互补性**: IRL 学到的奖励函数 R_ω 可以**嵌入**到 SA-CADRL 框架中（替代手工设计的 R），从而结合两者的优势——从人类数据中自动学习社交规范的量化标准（IRL），再通过深度 RL 在多智能体环境中高效训练（SA-CADRL）。
