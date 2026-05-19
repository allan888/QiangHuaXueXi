# 基于Q学习的无人机路径规划与动态避障

**原文标题**: Q-learning-based unmanned aerial vehicle path planning with dynamic obstacle avoidance

**作者**: Amala Sonny, Sreenivasa Reddy Yeduri, Linga Reddy Cenkeramaddi

**机构**: Department of ICT, University of Agder, Grimstad, 4879, Aust-Agder, Norway

**期刊**: Applied Soft Computing 147 (2023) 110773

**出版**: Elsevier B.V. (开放获取，CC BY 4.0协议)

---

## 摘要

近年来，无人机（UAV）在自主感知方面展现了令人瞩目的成果。无人机已被部署用于多种应用，包括监视、测绘、跟踪和搜索行动。在起点与目标之间寻找一条高效路径是一个关键问题，也是近期探索的焦点。许多路径规划算法被用于为无人机寻找从起点到目标的高效路径并进行避障。尽管已有大量文献和众多关于路径规划的研究方案，但**动态避障尚未通过机器学习方法得到解决**。当障碍物是动态的（即它们的位置会随时间变化）时，路径规划算法的约束条件变得更加苛刻，从而为路径规划算法增加了一层复杂性。为了解决这一挑战，本文提出了一种Q学习算法，用于实现无人机在静态和动态障碍物环境下的高效路径规划。我们在学习过程中引入了**最短距离优先策略（Shortest Distance Prioritization）**，该策略能够略微缩短无人机到达目标所需飞行的距离。此外，所提出的Q学习算法采用基于网格图的方法来解决路径规划问题，通过智能体在环境中的行为学习最大化奖励。通过实验结果，我们将所提出的方法与A*、Dijkstra和Sarsa等现有先进路径规划方法在**学习时间和路径长度**方面进行了性能比较。结果表明，所提出的方法在性能上优于现有先进方法。此外，我们还评估了障碍物数量增加对所提方法性能的影响。

---

## 1. 引言

无人机的路径规划是自主导航领域的关键组成部分。其任务是为无人机找到一条最优路径以到达给定目标，同时受到障碍物、能耗和时间限制等各种参数的约束[1,2]。路径规划旨在确保无人机能够安全高效地到达目的地，同时避免碰撞和其他危险[3]。

现有的无人机路径规划方法包括：

- **基于网格的方法（Grid-based methods）** [4,5]：将环境划分为网格单元，利用A*[13]或Dijkstra[14,15]等搜索算法寻找最优路径。
- **势场法（Potential field methods）** [6,7]：利用排斥力和吸引力的组合引导无人机飞向目标，同时避开障碍物。排斥力引导无人机远离障碍物，吸引力则将无人机引向目标。
- **基于采样的方法（Sampling-based methods）** [8,9]：在环境中随机采样点，使用概率框架（如快速探索随机树RRT[16]或概率路标图PRM[17]）寻找最优路径。
- **模型预测控制（MPC）方法** [10,11]：利用无人机动力学和环境的数学模型预测未来行为，然后优化代价函数以找到最佳控制输入。
- **混合方法（Hybrid methods）** [12]：将上述两种或多种方法组合以找到最优路径。

文献[18]提出了一种用于四轮全向COVID-19芳香疗法机器人的轨迹跟踪控制方法。该方法有效解决了机器人运动中的潜在误差，如速度设置不当、运动方向偏差和机器人姿态变化等，采用PID控制方法调节电机速度、机器人运动方向和姿态方向。文献[19]提出了一种用于医疗远程临场机器人的扰动抑制控制方法，结合了PID控制和卡尔曼滤波器。GY-521 MPU6050传感器读数与卡尔曼滤波器的结合旨在获得精确读数，仿真和实际机器人测试表明机器人能够在外部扰动下保持平衡并自我恢复平衡。文献[20]提出了一种基于粒子群优化（PSO）的路径规划方法，利用改进的PSO算法进行无人机路径规划以支持用户的速率需求。虽然该方法关注了用户的速率需求和能耗优化，但**未考虑路径规划中的动态障碍物**。

方法的选择取决于无人机的具体需求和操作环境。基于网格的方法可能适用于几何结构简单、障碍物较少的场景，而基于采样的方法可能更适合几何结构复杂、障碍物众多的情况。

无人机路径规划面临的一大挑战是需要**对环境的实时变化做出及时响应**[21]。这可以通过使用在线算法来持续根据新环境信息更新路径来实现。另一个挑战是需要考虑无人机的动力学特性，如速度和加速度限制，以确保平稳安全的飞行。

### Q学习在路径规划中的应用

Q学习是近年来最常用于路径规划的先进技术之一。Q学习的核心是训练智能体利用从行动中获得的奖励在给定环境中做出最优决策。在基于Q学习的无人机路径规划中，智能体就是无人机，环境则是其需要飞行的场景。智能体的主要目标是以最大奖励找到一条安全可靠的高效路径到达目标点，同时避免与障碍物碰撞。它利用Q表来跟踪每个状态-动作组合的预期奖励，使其能够基于最高预期奖励做出决策。Q表随着智能体与环境的互动而更新，并估计每个状态中能够获得最大奖励的最佳动作。

然而，在路径规划中使用Q学习面临一些挑战：一是**状态-动作空间很大**，可能使算法运行缓慢且不切实际；另一个是在学习环境的同时规划最高效的路径。针对这些挑战，文献[22]提出了一种将Q学习与PSO结合的群体机器人路径规划方法；文献[23]提出了一种基于离线Q学习的方法，策略集中在路径长度、安全性和能耗上；文献[24]提出了一种确定性Q学习方法，利用关于距离的先验知识使路径规划更省时；文献[25]提出了改进的Q学习方法用于未知对抗场景中的无人机路径规划；文献[26]提出了A*与Q学习的混合路径规划方法；文献[27]提出了自适应随机探索（ARE）方法；文献[28,29]提出了不确定动态环境下的Q学习路径规划方法。

然而，上述方法都**回避了对动态障碍物的考虑**。尽管文献中有很多路径规划的研究方案，但动态避障尚未通过机器学习方法得到探索。当障碍物是动态的（位置随时间变化）时，路径规划算法的约束条件更加挑战性，增加了复杂性。

**基于此，本文提出了一种用于静态和动态障碍物环境下无人机路径规划的Q学习算法。** 主要贡献如下：

- 提出了一种带有**最短距离优先策略的Q学习方法**，用于静态和动态障碍物避障的无人机路径规划。
- 提出的Q学习算法采用**基于网格图的方法**解决路径规划问题，通过智能体在环境中的行为学习最大化奖励。
- 分析了环境**动态障碍物数量不断增加**对所提方法性能的影响。
- 将提出的路径规划方法与**A*算法、Dijkstra算法和Sarsa算法**等现有先进方法进行了比较。
- 通过数值结果，展示了所提方法**优于现有先进方法的性能**。

本文后续章节安排如下：第2节详细介绍系统模型；第3节描述提出的Q学习算法；第4节进行性能评估；最后，第5节给出总结和未来展望。

---

## 2. 系统模型

环境的深入了解是构建鲁棒路径规划方法的基础。对环境中所存在的各种障碍物进行详细描述，能够使避障过程更加高效和连贯。成功避障方法的关键在于识别障碍物的种类并区别对待。

在低空环境中，无人机飞行区域内存在诸多障碍物，如建筑物、树木和山丘，这些都构成重大风险。精确地为路径规划建模这些障碍物十分困难。为了简化建模过程并降低计算复杂度，我们用几何形状来表示它们，其尺寸可以根据实际物体（如树木、山丘和建筑物）进行变化调整。本研究考虑的基础环境结构如图1所示。我们考虑一个25×25单元的网格，每个单元固定为20像素。红色圆圈表示无人机，黄色方块表示无人机的目标位置。无人机的位置代表起始位置。四个蓝色矩形是环境中各种尺寸和形状的**静态障碍物**。无人机需要高效地发现从起点到目标位置的最优路径，以降低成本和时间，同时避开障碍物以减少碰撞风险。

### 2.1 包含动态障碍物的环境

路径规划算法旨在为机器人或无人机寻找从起点到目标的高效路径并避开障碍物。然而，当障碍物是动态的（即位置可能随时变化）时，路径规划算法的约束条件变得更加苛刻。算法需要在整个过程中持续处理障碍物信息，做出实时决策以避免碰撞，这需要快速且高处理能力的算法。此外，算法还需要准确检测和跟踪这些动态障碍物。

随着动态障碍物数量的增加，计算复杂度也随之增加，使路径规划成为一个具有挑战性且耗时的过程，尤其在复杂环境中。同时，预测动态障碍物的未来位置并保持安全裕度以确保无人机安全，使得整个流程更加复杂，并可能对路径规划算法的准确性产生不利影响。由于这些因素，动态障碍物的存在为路径规划算法增加了**一层复杂性**，使其更具挑战性。在本工作中，我们假设无人机与动态障碍物**以相同速度移动**。

---

## 3. 提出的基于Q学习的方法

本研究提出了一种Q学习算法，用于在静态和动态障碍物并存的条件下进行无人机路径规划。该方法采用一种策略，允许智能体选择对应最大奖励和最短路径到达目的地的动作。无人机路径规划与碰撞避免的训练框架流程图如图2所示。

所提出的方法包含两个主要组成部分：

1. **最短路径选择**：智能体基于最短距离优先策略从当前状态选择下一个动作。
2. **碰撞避免**：智能体尝试避开路径上所有障碍物，采用安全轨迹。

本工作除静态障碍物外，还考虑了多达**四个动态障碍物**，其位置随时间随机变化。智能体遵循最短路径选择和碰撞避免的策略来学习环境，以经济高效且无碰撞的方式到达目的地。

### 3.1 Q学习算法

Q学习是一种基于时间差分的强化学习算法，通过Q表表示状态-动作值进行任务求解。它是主要的无模型强化学习方法，不需要完整的环境先验知识，因为它采用基于网格图的方法来解决问题。Q学习通过智能体在环境中的行为学习最大化奖励。Q学习算法的框架如图3所示。

Q学习的结构由**五个关键要素**组成：

- **智能体（Agent）**
- **环境（Environment）**
- **状态（State）**
- **动作（Action）**
- **奖励（Reward）**

它在探索环境的同时实施奖励和惩罚的概念。智能体根据奖励和对应策略选择下一个动作，然后执行该动作。环境包含多种状态和对应的奖励，智能体可以从中选择下一个状态。动作是从一个状态到另一个状态的移动。每次智能体执行动作后，环境转移到下一状态并提供相应的奖励。智能体选择的下一个动作取决于策略，奖励对应不同状态。此过程持续进行直到智能体到达预定义的目的地。通过这个学习过程，智能体学会在每个状态采取合适动作以获得最大奖励。

Q学习的主要优势之一是能够利用**时间差分（TD）方法**进行离线学习，并利用**贝尔曼方程**求解马尔可夫过程的最优策略。因此，Q学习仍然是一种广泛使用且研究深入的强化学习算法。它是学习复杂环境中最优策略的一种简单而有效的方法。

#### 数学公式

强化学习的基础在于解决由**马尔可夫决策过程（MDP）**[30]定义的序列决策问题。在智能体的学习过程中，价值函数被用来构建贝尔曼方程，然后通过Q学习进行求解[31]。

在概率环境中，执行动作后的转移状态和奖励都是随机的。在这种环境中为特定状态选择未来动作时，必须遵循称为**策略**的某些规则。这种类型的方法称为强化学习算法。通常，强化学习算法使用MDP来形式化。强化学习中的参数包括状态、动作、奖励、折扣因子、策略和状态转移概率矩阵。

策略是从状态和动作到采取该动作的可能性的映射过程，表达为：

$$
Pr(\lambda|\gamma) = P[\Lambda_t = \lambda | \Gamma_t = \gamma] \qquad (1)
$$

其中，$Pr$表示智能体在特定状态$\gamma$下选择特定动作$\lambda$的概率。

价值函数被用作判断哪种策略会在下一个动作中产生更好奖励的标准。Q学习中有两种不同的价值函数：

**状态价值函数（State-value function）**：

$$
v_p(\gamma) = E_p[R_{t+1} + \delta v_p(\Gamma_{t+1}) | \Gamma_t = \gamma] \qquad (2)
$$

其中$p$是智能体选择的策略，$R_{t+1}$是下一时间步的预期奖励，$\delta$是折扣因子。状态价值函数是对选择某一状态$\gamma$的总奖励期望估计。

**动作价值函数（Action-value function）**，通常称为Q函数，评估在状态$\gamma$下采取动作$\lambda$并遵循策略$p$的期望回报。Q学习算法以达成最优动作价值函数$Q^*$为目标[35]。智能体基于最优贝尔曼方程学习一系列轨迹，定义为：

$$
Q^*(\gamma, \lambda) = E_p\left[R_{t+1} + \delta \max_{\lambda' \in \Lambda} Q^*(\Gamma_{t+1}, \lambda') \mid \Gamma_t = \gamma, \Lambda_t = \lambda\right] \qquad (3)
$$

其中$\lambda'$是下一动作，$Q^*(\gamma, \lambda)$在时间$t$可近似为$\tilde{Q}(\gamma, \lambda)$。

Q函数与价值函数的关系为：

$$
v_p(\gamma) = \sum_{\lambda \in \Lambda} p(\lambda|\gamma) Q^*(\gamma, \lambda) \qquad (4)
$$

#### ϵ-贪心策略

Q学习算法包含两个主要部分。第一部分是**训练数据的收集**，由预定义策略引导。策略可根据问题目标进行调整。最常见的**行为策略**是**ϵ-贪心策略**，定义为：

$$
\begin{cases}
\arg\max_{\lambda} \tilde{Q}(\Gamma_t, \lambda), & 1 - \epsilon \\
\text{Uniform}, & \epsilon
\end{cases} \qquad (5)
$$

ϵ-贪心算法用于在状态$\Gamma_t$中选择动作$\lambda$。智能体执行动作并收到奖励$R_t$，然后转移到状态$\gamma'$。本研究将$\epsilon$值设为0.9[36]。

#### 经验回放与Q值更新

第二部分是**经验回放**，这是一种强化学习中用于存储智能体每个时间步经验的记忆回放技术。从经验回放记忆中随机抽取四元组$(\Gamma_t, \Lambda_t, R_t, \Gamma_{t+1})$。

元组$(\Gamma_t, \Lambda_t, R_t, \Gamma_{t+1}, \tilde{Q}_{\text{now}}, \tilde{Q}_{\text{new}})$用于记录智能体-环境交互的状态、动作、奖励和结果状态。

计算下一状态的最大Q值：

$$
\hat{q}_{j+1} = \max_{\lambda} \tilde{Q}_{\text{now}}(\gamma_{j+1}, \lambda) \qquad (6)
$$

计算时间差分（TD）目标：

$$
\hat{y}_j = r_j + \gamma \hat{q}_{j+1} \qquad (7)
$$

计算TD误差：

$$
\delta_j = \hat{q}_j - \hat{y}_j \qquad (8)
$$

新的动作价值函数估计：

$$
\tilde{Q}_{\text{new}}(\gamma_j, \lambda_j) \leftarrow (1 - \alpha)\tilde{Q}_{\text{now}}(\gamma_j, \lambda_j) + \alpha\delta_j \qquad (9)
$$

其中$\alpha$和$\gamma$是超参数，分别表示**学习率**和**折扣因子**。$\alpha$的值介于0和1之间，用于确定新获取的信息在多大程度上覆盖现有Q值。当$\gamma = 0$时，智能体优先考虑近期奖励；$\gamma = 1$则表示对更高奖励的长期承诺。如果折扣因子超过1，动作价值Q可能会发散。在此情况下，逐渐将折扣因子从较低值增加到最终值可提高学习效率。例如，随着迭代次数的增加，$\gamma$从0.1增加到0.9。

经验回放共考虑5000个样本（$b = 5000$），$\beta$设为0.3。实验参数列于表1。

| 参数 | 值 |
|------|------|
| b    | 5000 |
| β    | 0.3 |
| γ    | [0.1, 0.9] |
| ϵ    | 0.9 |

Q值使用贝尔曼方程（公式9）持续更新每个状态的值。在开始学习过程之前，所有可能的奖励被初始化在Q表中。智能体基于预定义策略采取动作，并转移到下一状态，重复此过程直到Q值收敛到特定阈值[37]。

#### 算法伪代码

**算法1**: 仅存在静态障碍物时的Q学习

```
输入: 起点位置, 目标位置, 解空间
输出: 无人机从起点到目标的最优路径

1. 初始化 Q(γ, λ) ← 0 (Γ状态集, Λ动作集)
2. for each episode do
3.   从状态集Γ中随机选择状态γt
4.   while (γt ≠ target) do
5.     for each λ_i^t ∈ Λ where Λ = [up, down, left, right] do
6.       使用适当策略从γt选择λ_i^t (如ϵ-greedy)
7.     end
8.     执行动作λ_i^t并接收惩罚或奖励
9.     更新 Q(γt, λt)
10.   end
11. end
```

**算法2**: 存在静态和动态障碍物时的Q学习

```
输入: 起点位置, 目标位置, 解空间
输出: 无人机从起点到目标的最优路径

1. 初始化 Q(γ, λ) ← 0, J个动态障碍物
2. for each episode do
3.   从状态集Γ中随机选择状态γt
4.   确定每个动态障碍物j ∈ J的位置loc_d^j
5.   while (γt ≠ target) do
6.     for each λ_i^t ∈ Λ do
7.       使用适当策略选择λ_i^t
8.       确定每个动态障碍物j ∈ J的位置loc_d^j
9.     end
10.    if (loc_λ_i^t == loc_d) then break
11.    执行动作λ_i^t并接收惩罚或奖励
12.    更新 Q(γt, λt)
13.   end
14. end
```

**算法3**（带最短距离优先策略）和**算法4**的原理：在每个状态，当智能体需要选择一个能提供最大奖励的动作时，算法同时检查距离标准。计算从每个可能的下一个位置到目标的距离，使智能体**更接近目标点的动作获得更高权重**。若到达目标位置，奖励值设为1；若到达障碍物（静态或动态），奖励设为-1；其他所有情况奖励设为0。这样，智能体学习目标点在环境中的位置，从而缩短学习时间和无人机飞行距离。

---

## 4. 性能评估

本节阐述实验发现，评估所提方法相对其他方法的有效性。

### 4.1 实验条件

路径规划的目标是确保系统的安全性和效率，同时避免与静态和动态障碍物的碰撞。本研究分析了算法在静态和动态障碍物条件下的性能。我们考虑如图1所示的含4个不同尺寸静态障碍物的场景。为评估原始Q学习和所提算法在动态障碍物条件下的性能，以及算法在拥挤环境中的有效性，我们考察了两种场景：第一种包含**2个动态障碍物**，第二种包含**4个动态障碍物**。动态障碍物随机放置在环境中，不限制其运动范围。

### 4.2 带最短距离优先策略的Q学习

每个无人机应用的主要目标是在无碰撞的前提下，以成本、燃料和时间的最高效率完成目标任务。最小化燃料消耗需要减少无人机到达目标的飞行距离。高效的路径规划方法可以通过减少不必要的飞行距离来实现这一目标。我们提出的最短距离优先策略能够相比使用ϵ-贪心策略的原始Q学习略微缩短无人机的飞行距离。

### 4.3 仅含静态障碍物的原始Q学习性能

原始Q学习算法的有效性取决于环境中存在的障碍物类型。Q学习是一种强大的算法，能够成功处理多种障碍物。图4(d)给出了在本文环境中通过原始Q学习估计的最优路径，图6给出了各episode的步数和代价。Q学习通过更新基于成功和不成功动作奖励的Q值，能够有效学习避开静态障碍物。

### 4.4 含静态与动态障碍物的原始Q学习性能

Q学习的一个显著优势是能够从先前经验中学习并适应不断变化的环境，这使得算法能够有效应对位置和速度随时间变化的动态障碍物。我们在有静态和动态障碍物的环境中评估了原始Q学习算法的有效性。图5(a)给出了算法估计的路径，图7给出了相应的步数和代价。

### 4.5 仅含静态障碍物时带最短距离优先策略的Q学习性能

为塑造奖励信号，我们修改了奖励函数以提供更具信息量的信号。例如，当无人机遇到新障碍物或到达前往目标途中的里程碑时，可以增加奖励。提出的最短距离优先策略显著缩短了无人机到达目的地的飞行距离。策略计算从所有可能的下一个位置到目标的距离，赋予使智能体更接近目标的动作更高的权重。图5(b)展示了仅含静态障碍物时提出算法的性能，图8给出了相应的步数和代价。

### 4.6 含静态与动态障碍物时提出的Q学习性能

表2显示，在存在动态障碍物的情况下，原始Q学习算法估计最优路径所需时间约为无动态障碍物场景的**十倍**。因此，我们评估了提出的Q学习算法在含有静态和动态障碍物环境中的导航性能。结果显示，提出的Q学习算法能够**持续稳定地估计最优路径**，无论环境中存在2个还是4个动态障碍物（除静态障碍物外），见表2。图5(c)和(d)分别显示了含2个和4个动态障碍物时智能体的路径，图9和图10给出了相应的步数和代价。

### 4.7 性能对比

为评估所提方法的有效性，我们基于**执行时间和路径长度**与多种现有先进方法进行了对比分析：

- **A*算法**[39]：广泛使用的图遍历和启发式搜索算法，搜索速度快，实时性能可靠。图4(a)显示了A*算法估计的路径。
- **Dijkstra算法**[40]：经典的寻找图中源节点到所有其他节点最短路径的算法。图4(b)显示了Dijkstra算法估计的路径。
- **SARSA算法**[41]：Q学习的改进版本。Q学习与SARSA的关键区别在于：Q学习旨在最大化未来预期奖励，SARSA关注当前所采取行动将产生的奖励。SARSA采用同策略（On-policy）学习方法，与Q学习不同，它独立于先前学习。图4(c)显示了SARSA算法的路径。

由于部分算法不适用于动态障碍物条件下的路径规划，所有对比均在仅含静态障碍物的环境假设下进行。路径长度和训练时间的对比见表2。

**表2：各算法性能对比（时间与距离）**

| 参数 | 训练时间 | 最短距离 | 最长距离 |
|------|---------|---------|---------|
| A* [39] | 0.01199s | 24 | - |
| Dijkstra [40] | 0.01559s | 37 | - |
| SARSA [41] | 1116.75s | 240 | 2087 |
| 原始Q-learning [36] | 347.2888s | 354 | 1025 |
| 原始Q-learning（含动态+静态障碍物） | 3128.7966s | 38 | 1794 |
| 提出Q-learning（仅静态障碍物） | 326.4840s | 50 | 104 |
| 提出Q-learning（静态+2个动态障碍物） | 336.5344s | 60 | 131 |
| 提出Q-learning（静态+4个动态障碍物） | 357.3923s | 70 | 112 |

根据表2：
- **A*和Dijkstra算法**速度很快，距离最短，但**不适用于含动态障碍物的环境**，因此不适合大多数实时无人机应用。
- **SARSA算法**学习速度较慢，估计路径包含更多步骤，即使在无动态障碍物时也是如此。
- **原始Q学习**性能相对较快，但路径距离较长；在含动态障碍物的环境中，执行时间是所有对比算法中**最长的**（3128.8秒）。
- **提出的方法**在静态和动态障碍物条件下，在**路径长度和执行速度**方面均表现良好。即使在增加动态障碍物数量的情况下，训练时间也仅从326.5秒小幅增加到357.4秒，展现了良好的可扩展性。

---

## 5. 结论与未来工作

本文提出了一种Q学习算法，用于在含有静态和动态障碍物的环境中高效规划无人机路径。所提方法在学习过程中引入了**最短距离优先策略**，能够略微缩短无人机到达目标点所需的飞行距离。所提方法的性能在学习时间和路径长度两个维度上进行了评估，并与多种现有先进路径规划方法进行了比较。结果表明，所提方法在**最小化无人机总飞行距离**方面优于其他方法。

**未来的研究方向**包括：

- 将Q学习与其他AI技术整合，以进一步提高路径规划的准确性和效率，使无人机能更好地应对复杂环境并做出更明智的决策。
- 开发更先进的Q学习策略，能够同时处理多重目标和约束，如**最小化飞行时间、避开障碍物和最大化能效**。

---

## 作者贡献声明

- **Amala Sonny**: 概念设计、方法论、软件开发、撰写-审查与编辑
- **Sreenivasa Reddy Yeduri**: 撰写-审查与编辑、指导
- **Linga Reddy Cenkeramaddi**: 概念设计、指导、审查与编辑、资金获取

---

## 利益冲突声明

作者声明没有已知的可能影响本文工作的竞争性经济利益或个人关系。

## 数据可用性

本文所述研究未使用任何数据。

## 致谢

本研究得到以下项目资助：
- INTPAR项目：印挪威自主信息物理系统合作（INCAPS），挪威项目号：287918
- IKTPLUSS项目：低空无人机通信与跟踪（LUCAT），挪威项目号：280835
- 以上均由挪威研究理事会资助

---

## 参考文献

[1] P. Chen, J. Pei, W. Lu, M. Li, A deep reinforcement learning based method for real-time path planning and dynamic obstacle avoidance, Neurocomputing 497 (2022) 64–75.

[2] L. Yang, J. Qi, J. Xiao, X. Yong, A literature review of UAV 3D path planning, in: Proceeding of the 11th World Congress on Intelligent Control and Automation, 2014, pp. 2376–2381.

[3] F. Borrelli, D. Subramanian, A. Raghunathan, L. Biegler, MILP and NLP techniques for centralized trajectory planning of multiple unmanned air vehicles, in: 2006 American Control Conference, 2006, p. 6.

[4] K. Fransen, J. Van Eekelen, A. Pogromsky, M.A. Boon, I.J. Adan, A dynamic path planning approach for dense, large, grid-based automated guided vehicle systems, Comput. Oper. Res. 123 (2020) 105046.

[5] M. Kanehara, S. Kagami, J.J. Kuffner, S. Thompson, H. Mizoguhi, Path shortening and smoothing of grid-based path planning with consideration of obstacles, in: 2007 IEEE International Conference on Systems, Man and Cybernetics, 2007, pp. 991–996.

[6] J. Barraquand, B. Langlois, J.-C. Latombe, Numerical potential field techniques for robot path planning, in: Fifth International Conference on Advanced Robotics 'Robots in Unstructured Environments', 1991, pp. 1012–1017.

[7] F. Bounini, D. Gingras, H. Pollart, D. Gruyer, Modified artificial potential field method for online path planning applications, in: 2017 IEEE Intelligent Vehicles Symposium (IV), 2017, pp. 180–185.

[8] L. Schmid, M. Pantic, R. Khanna, L. Ott, R. Siegwart, J. Nieto, An efficient sampling-based method for online informative path planning in unknown environments, IEEE Robot. Autom. Lett. 5 (2) (2020) 1500–1507.

[9] L. Jaillet, J. Cortés, T. Siméon, Sampling-based path planning on configuration-space cost maps, IEEE Trans. Robot. 26 (4) (2010) 635–646.

[10] J. Ji, A. Khajepour, W.W. Melek, Y. Huang, Path planning and tracking for vehicle collision avoidance based on model predictive control with multiconstraints, IEEE Trans. Veh. Technol. 66 (2) (2017) 952–964.

[11] C. Liu, S. Lee, S. Varnhagen, H.E. Tseng, Path planning for autonomous vehicles using model predictive control, in: 2017 IEEE Intelligent Vehicles Symposium (IV), 2017, pp. 174–179.

[12] J. Li, G. Deng, C. Luo, Q. Lin, Q. Yan, Z. Ming, A hybrid path planning method in unmanned air/ground vehicle (UAV/UGV) cooperative systems, IEEE Trans. Veh. Technol. 65 (12) (2016) 9585–9596.

[13] F.H. Tseng, T.T. Liang, C.H. Lee, L.D. Chou, H.C. Chao, A star search algorithm for civil UAV path planning with 3g communication, in: 2014 Tenth International Conference on Intelligent Information Hiding and Multimedia Signal Processing, 2014, pp. 942–945.

[14] Z. He, L. Zhao, The comparison of four UAV path planning algorithms based on geometry search algorithm, in: 2017 9th International Conference on Intelligent Human-Machine Systems and Cybernetics (IHMSC), Vol. 2, 2017, pp. 33–36.

[15] Y. Deng, Y. Chen, Y. Zhang, S. Mahadevan, Fuzzy dijkstra algorithm for shortest path problem under uncertain environment, Appl. Soft Comput. 12 (3) (2012) 1231–1237.

[16] B. Li, B. Chen, An adaptive rapidly-exploring random tree, IEEE/CAA J. Autom. Sin. 9 (2) (2022) 283–294.

[17] L. Kavraki, P. Svestka, J.-C. Latombe, M. Overmars, Probabilistic roadmaps for path planning in high-dimensional configuration spaces, IEEE Trans. Robot. Autom. 12 (4) (1996) 566–580.

[18] A. Ma'arif, N.M. Raharja, G. Supangkat, F. Arofiati, R. Sekhar, D.U. Rijalusalam, et al., PID-based with odometry for trajectory tracking control on four-wheel omnidirectional COVID-19 aromatherapy robot, Emerg. Sci. J. 5 (2021) 157–181.

[19] I. Suwarno, A. Ma'arif, N.M. Raharja, T.K. Hariadi, M.A. Shomad, Using a combination of PID control and Kalman filter to design of IoT-based telepresence self-balancing robots during COVID-19 pandemic, Emerg. Sci. J. 4 (2020) 241–261.

[20] A. Sonny, S.R. Yeduri, L.R. Cenkeramaddi, Autonomous UAV path planning using modified PSO for UAV-assisted wireless networks, IEEE Access (2023).

[21] S. Aggarwal, N. Kumar, Path planning techniques for unmanned aerial vehicles: A review, solutions, and challenges, Comput. Commun. 149 (2020) 270–299.

[22] S.I.A. Meerza, M. Islam, M.M. Uzzal, Q-learning based particle swarm optimization algorithm for optimal path planning of swarm of mobile robots, in: 2019 1st International Conference on Advances in Science, Engineering and Robotics Technology (ICASERT), 2019, pp. 1–5.

[23] K.B. de Carvalho, I.R.L. de Oliveira, D.K.D. Villa, A.G. Caldeira, M. Sarcinelli-Filho, A.S. Brandão, Q-learning based path planning method for UAVs using priority shifting, in: 2022 International Conference on Unmanned Aircraft Systems (ICUAS), 2022, pp. 421–426.

[24] A. Konar, I. Goswami Chakraborty, S.J. Singh, L.C. Jain, A.K. Nagar, A deterministic improved Q-learning for path planning of a mobile robot, IEEE Trans. Syst. Man Cybern.: Syst. 43 (5) (2013) 1141–1153.

[25] C. Yan, X. Xiang, A path planning algorithm for UAV based on improved Q-learning, in: 2018 2nd International Conference on Robotics and Automation Sciences (ICRAS), 2018, pp. 1–5.

[26] D. Li, W. Yin, W.E. Wong, M. Jian, M. Chau, Quality-oriented hybrid path planning based on A* and Q-learning for unmanned aerial vehicle, IEEE Access 10 (2022) 7664–7674.

[27] Z. Yijing, Z. Zheng, Z. Xiaoyi, L. Yang, Q learning algorithm based UAV path learning and obstacle avoidance approach, in: 2017 36th Chinese Control Conference (CCC), 2017, pp. 3397–3402.

[28] J.-h. Cui, R.-x. Wei, Z.-c. Liu, K. Zhou, UAV motion strategies in uncertain dynamic environments: A path planning method based on Q-learning strategy, Appl. Sci. 8 (11) (2018) 2169.

[29] Y. Gao, Y. Li, Z. Guo, A Q-learning based UAV path planning method with awareness of risk avoidance, in: 2021 China Automation Congress (CAC), 2021, pp. 669–673.

[30] M.L. Puterman, Markov decision processes, Handb. Oper. Res. Manag. Sci. 2 (1990) 331–434.

[31] R.S. Sutton, Generalization in reinforcement learning: Successful examples using sparse coarse coding, in: Advances in Neural Information Processing Systems, Vol. 8, 1995.

[32] A.R. Cassandra, Exact and Approximate Algorithms for Partially Observable Markov Decision Processes, Brown University, 1998.

[33] M.L. Littman, Value-function reinforcement learning in Markov games, Cogn. Syst. Res. 2 (1) (2001) 55–66.

[34] R.S. Sutton, D. McAllester, S. Singh, Y. Mansour, Policy gradient methods for reinforcement learning with function approximation, in: Advances in Neural Information Processing Systems, Vol. 12, 1999.

[35] C. Ye, J. Borenstein, A method for mobile robot navigation on rough terrain, in: IEEE International Conference on Robotics and Automation, 2004 (ICRA '04), Vol. 4, 2004, pp. 3863–3869.

[36] C. Wang, X. Yang, H. Li, Improved Q-learning applied to dynamic obstacle avoidance and path planning, IEEE Access 10 (2022) 92879–92888.

[37] M. Langlois, R.H. Sloan, Reinforcement learning via approximation of the Q-function, J. Exp. Theor. Artif. Intell. 22 (3) (2010) 219–235.

[38] E.I. Grøtli, T.A. Johansen, Path planning for UAVs under communication constraints using SPLAT! and MILP, J. Intell. Robot. Syst. 65 (1–4) (2012) 265–282.

[39] G. Tang, C. Tang, C. Claramunt, X. Hu, P. Zhou, Geometric A-star algorithm: An improved A-star algorithm for AGV path planning in a port environment, IEEE Access 9 (2021) 59196–59210.

[40] M. Luo, X. Hou, J. Yang, Surface optimal path planning using an extended dijkstra algorithm, IEEE Access 8 (2020) 147827–147838.

[41] H. Bomning, L. Wei, M. Fuzeng, F. Huahao, Research for UAV path planning method based on guided SARSA algorithm, in: 2022 IEEE 2nd International Conference on Software Engineering and Artificial Intelligence (SEAI), 2022, pp. 220–224.
