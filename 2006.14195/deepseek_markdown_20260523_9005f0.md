# 密集动态环境中机器人导航的路径规划方法综述

**作者机构：**  
1. 哈尔滨工业大学（深圳）机械工程与自动化学院，深圳 518055  
2. 香港中文大学电子工程系，香港特别行政区 999077  
3. 不列颠哥伦比亚大学，温哥华，BC，加拿大  

**摘要：** 在密集人口动态环境中，机器人导航面临诸多挑战。本文综述了面向密集环境机器人导航的路径规划方法。特别地，在移动机器人导航框架下，路径规划根据规划范围和可执行性分为全局路径规划和局部路径规划。在此框架内，本文介绍了路径规划方法的最新进展，并分析了其优缺点。值得注意的是，本文对近年来发展的速度障碍法（Velocity Obstacle）及其作为局部规划器的变体进行了全面分析。此外，作为当前机器人应用中广泛使用的无模型方法，本文详细阐述了基于强化学习的路径规划算法。

**关键词：** 导航；动态环境；局部规划器；全局规划器；人体运动；强化学习

---

## 1 引言

自主机器人，例如工厂中的工业机器人和公共区域的服务机器人，在过去几十年中引起了广泛关注并日益发展。例如，文献[1]中开发的多模式商场娱乐机器人（MuMMER）可以在露天广场环境中为公众提供所需服务。波士顿动力公司开发的Atlas机器人[2]能够在室内外环境中执行特殊任务。最近，Tanaka[3]开发了一种类人机器人Pepper，可作为家庭伴侣，在家庭环境中协助老年人。一个趋势是，这些机器人越来越多地用于或与人类共存于越来越复杂的环境中，例如购物中心、城市街道或火车站。在这些环境中，机器人规划出一条安全、无碰撞的路径以有效为人类提供服务是先决条件。

近年来，已发表了许多关于机器人路径规划的综述文章。例如，Kruse等人[4]综述了社交感知轨迹规划的主题。该文更侧重于导航过程中机器人的行为，其中研究了人类舒适度和社交性等因素对路径规划的影响。Chik等人[5]将机器人导航的路径规划器分为全局规划器和局部规划器。他们的综述概述了几种机器人导航框架的类型，但缺乏对局部路径规划器的充分描述。Douthwaite等人[6]针对多智能体的速度障碍法（VO）进行了比较研究。他们还提出了几种评估指标，以应对低分辨率传感器引起的不确定性。Mohanan等人[7]回顾了复杂环境中机器人运动规划的研究。他们对运动规划方法进行了分类，并对其性能进行了比较研究。Zafar等人[8]将运动规划方法分为典型方法和启发式方法。两类的比较分析表明，启发式方法在路径规划中表现出更好的性能。最近，Cheng等人[9]将当前方法分为基于反应的方法、基于预测的方法、基于模型的方法和基于学习的方法。这些综述按时间顺序归纳于表1，并突出了这些方法的主要特点。

尽管许多现有的综述从不同方面介绍了机器人导航中的路径规划，但尚未有全面系统地介绍包含全局路径规划器和局部路径规划器的分层路径规划框架的研究。此外，作为避障局部路径规划器的速度障碍法（VO）尚未得到充分研究。为了解决这些不足，本文详细介绍了可作为全局和局部路径规划器的不同方法。此外，讨论从基于模型的方法扩展到基于学习的框架。能够全面解决路径规划问题的强化学习（RL）方法在本文中进行了研究。本文概述了这些方法及其在解决路径规划问题方面的最新进展。

本文的其余部分组织如下。第2节介绍了经典的导航框架。介绍了经典全局规划器的最新进展以及专注于VO的局部路径规划传统方法。第3节介绍了用于路径规划的强化学习方法。第4节给出结论。

**表1 机器人避障文献总结**

| 作者 | 年份 | 主要内容 |
|------|------|----------|
| Thibault Kruse [4] | 2013 | 机器人在导航中需要考虑的因素 |
| S. F. Chik [5] | 2016 | 全局规划器、局部规划器、四种导航框架类型 |
| Douthwaite J.A [6] | 2018 | 速度障碍方法的比较研究 |
| M.G.Mohanan [7] | 2018 | 机器人运动规划 |
| MN Zafar [8] | 2018 | 经典方法、启发式方法 |
| Jiyu Cheng [9] | 2018 | 基于反应的方法、基于预测的方法、基于模型的方法、基于学习的方法 |

---

## 2 分层路径规划

路径规划模块在引导机器人在动态环境中安全移动方面起着关键作用。路径规划的目标是引导机器人从起点移动到目标点，并满足车辆运动约束。通常，基于机器人在导航过程中能够获取的环境信息，路径规划可分为两个阶段[10]：全局规划器和局部规划器。

机器人的导航框架由全局路径规划器和局部路径规划器组成，如图1所示。形式上，全局路径规划器负责规划从起点到目标点的无碰撞路径。此过程仅涉及静态全局地图，因此生成的全局路径是静态的，不考虑动态物体。局部路径规划器则对全局路径进行分段优化，考虑动态物体信息和机器人运动约束。在实际应用中采用这种分层路径规划框架有许多好处。以下各节将回顾全局路径规划算法和局部路径规划算法的最新发展。

<center>**图1 全局规划器和局部规划器规划的路径。** 起点标记为点A。机器人目标点为点C。绿色路径由全局规划器生成，红色轨迹由局部规划器生成。</center>

<center>**图2 包含全局路径规划器和局部路径规划器的导航框架。**</center>

### 2.1 全局路径规划器

全局路径规划方法根据全局地图和目标点为机器人生成一条路径。必须利用在动态环境中获得的地图信息来规划机器人可以遵循的路径，以应对随时可能出现的任何障碍。

关于全局路径规划有大量研究。已开发的算法可分为三类：基于图搜索的算法[11]、随机采样算法[12]和智能仿生算法[13]，如图3所示。用于图搜索的经典基于图搜索的算法主要包括Dijkstra算法[14]、A\*算法[15]、DFS算法[16]和BFS算法[17]。Dijkstra算法和A\*算法在过去几十年中得到了深入研究，并通过在机器人操作系统（ROS）[18]中的广泛实现证明了其在现实世界机器人应用中的能力。凭借启发式搜索策略，这些方法在相对简单的二维环境中是有效的。然而，当在大规模或高维环境中实现时，这些方法面临沉重的计算负担。

通常，如图3所示，随机采样算法包括批量 informed树（BIT）[19]、区域加速批量 informed树（RABIT）[20]、快速探索随机树（RRT）[21]和基于风险的双树快速探索随机树（Risk-DTRRT）[22]等。与基于图搜索的算法相比，这些算法效率更高，并广泛用于动态或高维环境。

全局路径规划方法的另一个重要分支是基于智能仿生的方法，这是一种模拟昆虫进化行为的智能算法。它通常包括遗传算法（GA）[23]、蚁群算法（ACO）[24]、人工蜂群算法（ABC）[25]和粒子群优化算法（PSO）[26]。为了进一步提高计算效率并避免局部最优问题，提出了许多先进算法。Wang等人[27]提出了遗传算法-粒子群优化算法（OGA-PSO）的优化，以解决焊接机器人的最短无碰撞路径规划问题。Liu等人[28]将人工势场和几何局部优化方法与ACO相结合，搜索全局最优路径。Mac等人[29]提出了一种带加速方法的约束多目标粒子群优化算法，以生成最优全局轨迹。

<center>**图3 不同经典全局规划器方法的示意图。**</center>

### 2.2 局部路径规划器

局部路径规划器专注于利用机器人周围环境的可用信息生成局部路径，以便机器人能够有效地避开局部障碍。局部路径规划器被广泛使用，因为在动态环境中，传感器系统捕获的信息是实时变化的。与全局路径规划方法相比，局部路径规划方法更高效、更实用，是连接全局路径与控制的桥梁。然而，一个显著的缺点是局部规划器可能会陷入局部最小值。

通常，在机器人导航中，人类被视为障碍物。Nishitani等人[30]开发了X-Y-T空间运动规划方法来避开人类。该方法考虑了人类的方向和人类个人空间。然而，计算效率很大程度上取决于导航地图的网格大小。Kollmitz等人[31]提出了一种用于复杂环境中导航的分层社会成本地图。此外，作为A\*算法[32]的扩展，定时A\*方法能够通过使用社会成本函数预测人类轨迹。

在局部规划器中有许多经典算法用于获得最优路径并避免局部最小值问题，例如人工势场法[33]、模糊逻辑算法[34]、模拟退火算法[35]、粒子算法[36]以及与遗传算法结合的混合方法[37]。然而，这些方法没有考虑智能体与动态物体之间的相对运动，更糟糕的是，有时很难明确获取动态物体的速度分布。

最近，一种不依赖明确速度分布信息的局部规划器被开发出来。在该系统中，环境中每个智能体的导航是独立的，一个智能体不需要与其他智能体通信[38]。特别地，P. Fiorini[39]提出了VO理论，通过定义一个速度约束（描述为一个几何区域）来定义，智能体的速度落入该区域将导致下一步智能体之间发生碰撞。该方法在考虑智能体速度的同时有效避障。然而，使用VO时，当两个智能体处于碰撞航向时会发生振荡。这些振荡的产生是因为两个机器人在避障开始时都选择了较大的当前速度偏移。为了减少当前速度的偏移并提高性能，Van den Berg等人[38]提出了互惠速度障碍（RVO）方法。他们将机器人的新速度分布视为其当前速度与超出其他智能体VO的速度的平均值。RVO被认为是规划平滑且安全的无振荡路径的有效方法，用于多智能体导航。然而，它仍然有一个缺点：多个机器人可能无法就哪一侧通过达成共识，这会导致所谓的“互惠舞蹈”问题。为了解决这个问题，Snape[40]将RVO扩展为混合互惠速度障碍（HRVO）。该方法通过考虑机器人的运动学和传感器不确定性，已应用于多机器人导航。

然而，如果应用场景中存在大量动态物体，机器人的速度将在速度空间中趋近于初始点[41]。结果，机器人可能被困在一个区域。这个问题可以通过截断方法[42]消除，使用该方法，机器人在截断后的定义时间步内不会发生碰撞。图4显示了一个VO示例，其中灰色区域标记了可能导致智能体之间碰撞的速度分布。更多细节可参见[43]。智能体的新速度必须选择在这些灰色区域之外。为此，从不同方面提出了多种方法。现在概述近年来提出的三种常用方法。第一种方法是Berg等人[44]提出的最优互惠碰撞避免（ORCA）。使用该方法，可以为每个智能体计算并分配无碰撞速度的半平面。然后可以通过求解线性规划问题来定义最优速度区域。智能体选择最接近最优速度的速度分布并随之移动。第二种估计无碰撞速度的方法是[45]中提出的ClearPath。ClearPath是一种鲁棒的方法，优于先前基于VO的碰撞避免方法。ClearPath有两种计算无碰撞速度的方法。一种是选择任意VO两条边界线交点处的速度。另一种方法是通过将首选速度分布投影到最近的VO上确定的速度[44]。第三种方法是[46]中提出的考虑定位不确定性的碰撞避免（CALU）方法，该方法结合了最优互惠碰撞避免（ORCA）和非完整机器人最优互惠碰撞避免（NH-ORCA）[43]，以减少对环境先验知识的需求。

<center>**图4 VO的示例图。** 图片致谢[42]。</center>

VO可以有效地处理复杂环境中由不精确定位和通信引起的问题。尽管如此，它仍需要关于智能体形状、速度和位置的足够信息。然而，它在两种情况下无效：a) 机器人底盘不能被视为圆盘；b) AMCL的位姿置信分布向一个方向漂移。Claes[41]引入了有界定位不确定性下的碰撞避免（COCALU）来解决这个问题。它改变了粒子云的形式而不是外接圆。这些VO的演变过程如图5所示。表2总结了现有的局部路径规划器。

<center>**图5 基于速度障碍的经典算法之间的关系。**</center>

**表2 局部规划器的经典算法**

| 局部规划器 | 优点 | 缺点 |
|------------|------|------|
| 人工势场法 [33] | 方案效率高，能解决传统算法中的局部最小值问题 | 存在陷阱区域，机器人通过狭窄通道时会振荡 |
| 模糊逻辑算法 [34] | 减少对环境信息的依赖，具有鲁棒性好、有效性高的优点 | 模糊规则通常由人的经验预先确定，因此无法学习，灵活性差 |
| 模拟退火算法 [35] | 描述简单、使用灵活、效率高、初始条件少 | 收敛速度慢，随机性高 |
| 速度障碍法 (VOs) [39] | 考虑了障碍物速度 | 未考虑社会之间的复杂关系 |
| X-Y-T空间法 [30] | 考虑了人类的方向区域和个人空间 | 效率取决于定义的地图网格大小 |
| 定时A\*算法 [32] | 可以预测人类轨迹 | 未考虑障碍物运动 |

---

## 3 路径规划中的强化学习

强化学习（RL）是一种有效的机器学习方法，它利用过去行动的结果，根据行动的成功或失败来强化或削弱这些行动。在移动机器人领域，该方法使用环境反馈作为路径规划的输入。它通过不断与外部环境交互，为机器人输出一个动作。通过强化学习机制，机器人尝试采取行动并接收反馈，然后基于反馈做出决策。特别地，对于正确的行动，算法将给予机器人一个正强化值；而对于错误的行动，算法将给予机器人一个负值。在整个过程中，机器人强化其正确行为，弱化其错误行为。因此，当机器人在环境中遇到人和其他机器人时，可以生成一个合理的解决方案。此外，如同任何类型的学习一样，性能随着经验的积累而提高。强化学习的过程如图6所示。策略，即导航过程中的路径，是通过与外部环境交互生成的。

<center>**图6 强化学习的过程**</center>

机器人路径规划中的经典RL算法包括Q-learning算法[47]、SARSA算法[48]、R-learning算法。其中Q-learning是研究最多的RL算法。通常，该算法通过机器人从环境获得的反馈，为机器人状态和动作输出一个奖励值。特别地，正确动作的Q值通过降低错误动作的比率而增加。然后，基于Q值的方法在Q值被过滤后输出最优策略。Q-learning算法也有一些局限性。第一，内存需求大。第二，学习时间长。第三，收敛速率低。为了解决Q-learning的问题，Peng[49]提出了Q(λ)算法，该算法利用了回溯的思想。在这种方法中，后续数据可以及时回传，从而使方法能够以时间有效的方式预测下一步行为。在更新过程中，引发的错误行为逐渐被遗忘。

RL及其变体已广泛应用于机器人导航。为了以优雅的方式与人类交互，机器人需要理解和遵循某些规则。基于此目标，Kuderer[50]提出了一种对人类协作导航行为进行建模的方法。它能够实时获取人体轨迹。最近，逆强化学习（IRL）获得了大量研究兴趣。它为决策过程包含一个奖励函数。一些研究人员应用IRL来获取用于协作导航的人类舒适度模型[51]。为了在人口密集环境中实现优雅的路径规划，Chen等人[52]提出了一种基于深度强化学习的分散式多智能体碰撞避免算法，该算法有效地将在线计算转移到离线学习。该算法优于ORCA算法，但可能导致振荡（边际稳定）路径。

SA-CALDRL[53]已被提出来应对人类行为的随机性，其中使用了时间有效的导航策略。值得注意的是，该方法可以在人口密集的动态环境中实现机器人车辆的低速自主导航。然而，该算法没有考虑与行人的关系。Everett[54]通过在算法中引入长短期记忆（LSTM）方法扩展了SA-CALDRL算法。该算法的优点是不需要假设特定的行为模型。此外，它试图以简单的方式预测机器人的运行方向。另外，Ciou等人[55]提出了复合强化学习（CRL）框架。在该框架中，机器人通过传感器输入学习优雅的社交导航。实验表明，CRL方法可以学习在环境中安全导航。然而，该框架需要先验知识，限制了其在现实世界中的广泛应用。Long等人[56]提出了各种环境和各种类型的阶段训练框架。该策略可以很好地扩展到训练阶段未出现的新场景。近年来强化学习在动态路径规划中的应用如表3所示，其中指出了RL方法的优缺点。

**表3 近年来强化学习在动态运动规划中的应用**

| 方法 | 优点 | 缺点 |
|------|------|------|
| IRL [51] | 在不同环境中建立人体模型，可提供协作导航 | 计算昂贵，高度依赖特征选择性能 |
| CADRL [52] | 实时性和高路径质量 | 可能导致振荡路径 |
| SA-CADRL [53][54] | 解决人类行为的随机性 | 未考虑与行人之间常见社会规范的关系 |
| CRL [55] | 既能解决社交感知运动规划问题，又能与人类交互 | 需要先验知识 |

---

## 4 结论

本文综述了自主机器人导航的路径规划算法。研究了将规划方法分为全局路径规划器和局部路径规划器的框架下的路径规划问题。这些方法在解决不同应用中的导航问题方面是有效的。然而，仍有进一步改进的空间。本文探讨了基于强化学习的路径规划方法的进展，并指出了它们在复杂环境中导航的能力。从长远来看，强化学习方法可能被编码到分层路径规划框架中。

---

## 参考文献

[1] Foster, M. E., Alami, R., Gestranius, O., Lemon, O., Niemela, M., Odobez, J. M., & Pandey, A. K. (2016, November). The MuMMER project: Engaging human-robot interaction in real-world public spaces. In *International Conference on Social Robotics* (pp. 753-763). Springer, Cham.

[2] Feng, S., Whitman, E., Xinjilefu, X., & Atkeson, C. G. (2014, November). Optimization based full body control for the atlas robot. In *2014 IEEE-RAS International Conference on Humanoid Robots* (pp. 120-127). IEEE.

[3] Tanaka, F., Isshiki, K., Takahashi, F., Uekusa, M., Sei, R. & Hayashi, K. (2015, November). Pepper learns together with children: Development of an educational application. In *2015 IEEE-RAS 15th International Conference on Humanoid Robots (Humanoids)* (pp. 270-275). IEEE.

[4] Kruse, T., Pandey, A. K., Alami, R., & Kirsch, A. (2013). Human-aware robot navigation: A survey. *Robotics and Autonomous Systems*, 61(12), 1726-1743.

[5] Chik, S. F., Yeong, C. F., Su, E. L. M., Lim, T. Y., Subramaniam, Y., & Chin, P. J. H. (2016). A review of social-aware navigation frameworks for service robot in dynamic human environments. *Journal of Telecommunication, Electronic and Computer Engineering (JTEC)*, 8(11), 41-50.

[6] Douthwaite, J. A., Zhao, S., & Mihaylova, L. S. (2018, September). A Comparative Study of Velocity Obstacle Approaches for Multi-Agent Systems. In *2018 UKACC 12th International Conference on Control (CONTROL)* (pp. 289-294). IEEE.

[7] Mohanan, M. G., & Salgoankar, A. (2018). A survey of robotic motion planning in dynamic environments. *Robotics and Autonomous Systems*, 100, 171-185.

[8] Zafar, M. N., & Mohanta, J. C. (2018). Methodology for Path Planning and Optimization of Mobile Robots: A Review. *Procedia computer science*, 133, 141-152.

[9] J. Cheng, H. Cheng, M. Q.-H. Meng, H. Zhang, Autonomous Navigation by Mobile Robots in Human Environments: A Survey, in: *2018 IEEE International Conference on Robotics and Biomimetics (ROBIO)*, IEEE, 2018.

[10] Wang, C., Meng, L., She, S., Mitchell, I. M., Li, T., Tung, F., ... & de Silva, C. W. (2017, September). Autonomous mobile robot navigation in uneven and unstructured indoor environments. In *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 109-116). IEEE.

[11] Chen, P. C., & Hwang, Y. K. (1998). SANDROS: a dynamic graph search algorithm for motion planning. *IEEE Transactions on Robotics and Automation*, 14(3), 390-403.

[12] Karaman, S., & Frazzoli, E. (2011). Sampling-based algorithms for optimal motion planning. *The international journal of robotics research*, 30(7), 846-894.

[13] Chen, T. (2009, April). A simulative bionic intelligent optimization algorithm: artificial searching swarm algorithm and its performance analysis. In *2009 International Joint Conference on Computational Sciences and Optimization* (Vol. 2, pp. 864-866). IEEE.

[14] Broumi, S., Bakal, A., Talea, M., Smarandache, F., & Vladareanu, L. (2016, November). Applying Dijkstra algorithm for solving neutrosophic shortest path problem. In *2016 International Conference on Advanced Mechatronic Systems (ICAMechS)* (pp. 412-416). IEEE.

[15] Duchoň, F., Babinec, A., Kajan, M., Beňo, P., Florek, M., Fico, T., & Jurišica, L. (2014). Path planning with modified a star algorithm for a mobile robot. *Procedia Engineering*, 96, 59-69.

[16] Guo, M., Johansson, K. H., & Dimarogonas, D. V. (2013, May). Revising motion planning under linear temporal logic specifications in partially known workspaces. In *2013 IEEE International Conference on Robotics and Automation* (pp. 5025-5032). IEEE.

[17] Yu, J., & LaValle, S. M. (2013, May). Planning optimal paths for multiple robots on graphs. In *2013 IEEE International Conference on Robotics and Automation* (pp. 3612-3617). IEEE.

[18] Quigley, M., Conley, K., Gerkey, B., Faust, J., Foote, T., Leibs, J., ... & Ng, A. Y. (2009, May). ROS: an open-source Robot Operating System. In *ICRA workshop on open source software* (Vol. 3, No. 3.2, p. 5).

[19] Gammell, J. D., Srinivasa, S. S., & Barfoot, T. D. (2015, May). Batch informed trees (BIT\*): Sampling-based optimal planning via the heuristically guided search of implicit random geometric graphs. In *2015 IEEE International Conference on Robotics and Automation (ICRA)* (pp. 3067-3074). IEEE.

[20] Choudhury, S., Gammell, J. D., Barfoot, T. D., Srinivasa, S. S., & Scherer, S. (2016, May). Regionally accelerated batch informed trees (rabit\*): A framework to integrate local information into optimal path planning. In *2016 IEEE International Conference on Robotics and Automation (ICRA)* (pp. 4207-4214). IEEE.

[21] Wang C, Meng M Q H. Variant step size RRT: An efficient path planner for UAV in complex environments[C]//*2016 IEEE International Conference on Real-time Computing and Robotics (RCAR)*. IEEE, 2016: 555-560.

[22] Chi, W., Wang, C., Wang, J., & Meng, M. Q. H. (2018). Risk-DTRRT-Based Optimal Motion Planning Algorithm for Mobile Robots. *IEEE Transactions on Automation Science and Engineering*.

[23] Hu, Y., & Yang, S. X. (2004, April). A knowledge based genetic algorithm for path planning of a mobile robot. In *IEEE International Conference on Robotics and Automation, 2004. Proceedings. ICRA'04. 2004* (Vol. 5, pp. 4350-4355). IEEE.

[24] Wang, J., Wang, N., & Jiang, H. (2015, October). Robot global path planning based on improved ant colony algorithm. In *5th International Conference on Advanced Design and Manufacturing Engineering*. Atlantis Press.

[25] Liu, H., Xu, B., Lu, D., & Zhang, G. (2018). A path planning approach for crowd evacuation in buildings based on improved artificial bee colony algorithm. *Applied Soft Computing*, 68, 360-376.

[26] Tharwat, A., Elhoseny, M., Hassanien, A. E., Gabel, T., & Kumar, A. (2018). Intelligent Bézier curve-based path planning model using Chaotic Particle Swarm Optimization algorithm. *Cluster Computing*, 1-22.

[27] Wang, X., Shi, Y., Ding, D., & Gu, X. (2016). Double global optimum genetic algorithm-particle swarm optimization-based welding robot path planning. *Engineering Optimization*, 48(2), 299-316.

[28] Liu, J., Yang, J., Liu, H., Tian, X., & Gao, M. (2017). An improved ant colony algorithm for robot path planning. *Soft Computing*, 21(19), 5829-5839.

[29] Mac, T. T., Copot, C., Tran, D. T., & De Keyser, R. (2017). A hierarchical global path planning approach for mobile robots based on multi-objective particle swarm optimization. *Applied Soft Computing*, 59, 68-76.

[30] Nishitani, I., Matsumura, T., Ozawa, M., Yorozu, A., & Takahashi, M. (2015). Human-centered X-Y-T space path planning for mobile robot in dynamic environments. *Robotics and Autonomous Systems*, 66, 18-26.

[31] Kollmitz, M., Hsiao, K., Gaa, J., & Burgard, W. (2015, September). Time dependent planning on a layered social cost map for human-aware robot navigation. In *2015 European Conference on Mobile Robots (ECMR)* (pp. 1-6). IEEE.

[32] Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In *Thirtieth AAAI Conference on Artificial Intelligence*.

[33] Le Gouguec, A., Kemeny, A., Berthoz, A., & Merienne, F. (2017). Artificial Potential Field Simulation Framework for Semi-Autonomous Car Conception.

[34] Bakdi, A., Hentout, A., Boutami, H., Maoudj, A., Hachour, O., & Bouzouia, B. (2017). Optimal path planning and execution for mobile robots using genetic algorithm and adaptive fuzzy-logic control. *Robotics and Autonomous Systems*, 89, 95-109.

[35] Turker, T., Sahingoz, O. K., & Yilmaz, G. (2015, June). 2D path planning for UAVs in radar threatening environment using simulated annealing algorithm. In *2015 International Conference on Unmanned Aircraft Systems (ICUAS)* (pp. 56-61). IEEE.

[36] Petrović, M., Vuković, N., Mitić, M., & Miljković, Z. (2016). Integration of process planning and scheduling using chaotic particle swarm optimization algorithm. *Expert systems with Applications*, 64, 569-588.

[37] Paulo, P., Branco, F., de Brito, J., & Silva, A. (2016). BuildingsLife-The use of genetic algorithms for maintenance plan optimization. *Journal of cleaner production*, 121, 84-98.

[38] Van den Berg, J., Lin, M., & Manocha, D. (2008, May). Reciprocal velocity obstacles for real-time multi-agent navigation. In *2008 IEEE International Conference on Robotics and Automation* (pp. 1928-1935). IEEE.

[39] Fiorini, P., & Shiller, Z. (1998). Motion planning in dynamic environments using velocity obstacles. *The International Journal of Robotics Research*, 17(7), 760-772.

[40] Snape, J., Van Den Berg, J., Guy, S. J., & Manocha, D. (2009, October). Independent navigation of multiple mobile robots with hybrid reciprocal velocity obstacles. In *2009 IEEE/RSJ International Conference on Intelligent Robots and Systems* (pp. 5917-5922). IEEE.

[41] Claes, D., Hennes, D., Tuyls, K., & Meeussen, W. (2012, October). Collision avoidance under bounded localization uncertainty. In *2012 IEEE/RSJ International Conference on Intelligent Robots and Systems* (pp. 1192-1198). IEEE.

[42] Bloembergen, D., Tuyls, K., Hennes, D., & Kaisers, M. (2015). Evolutionary dynamics of multi-agent learning: A survey. *Journal of Artificial Intelligence Research*, 53, 659-697.

[43] Alonso-Mora, J., Breitenmoser, A., Rufli, M., Beardsley, P., & Siegwart, R. (2013). Optimal reciprocal collision avoidance for multiple non-holonomic robots. In *Distributed Autonomous Robotic Systems* (pp. 203-216). Springer, Berlin, Heidelberg.

[44] Van Den Berg, J., Snape, J., Guy, S. J., & Manocha, D. (2011, May). Reciprocal collision avoidance with acceleration-velocity obstacles. In *2011 IEEE International Conference on Robotics and Automation* (pp. 3475-3482). IEEE.

[45] Guy, S. J., Chhugani, J., Kim, C., Satish, N., Lin, M., Manocha, D., & Dubey, P. (2009, August). Clearpath: highly parallel collision avoidance for multi-agent simulation. In *Proceedings of the 2009 ACM SIGGRAPH/Eurographics Symposium on Computer Animation* (pp. 177-187). ACM.

[46] Hennes D, Claes D, Meeussen W, et al. Multi-robot collision avoidance with localization uncertainty[C]//*Proceedings of the 11th International Conference on Autonomous Agents and Multiagent Systems- Volume 1*. International Foundation for Autonomous Agents and Multiagent Systems, 2012: 147-154.

[47] Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In *Thirtieth AAAI Conference on Artificial Intelligence*.

[48] Xu, D., Fang, Y., Zhang, Z., & Meng, Y. (2017, December). Path Planning Method Combining Depth Learning and Sarsa Algorithm. In *2017 10th International Symposium on Computational Intelligence and Design (ISCID)* (Vol. 2, pp. 77-82). IEEE.

[49] Peng, J., & Williams, R. J. (1994). Incremental multi-step Q-learning. In *Machine Learning Proceedings 1994* (pp. 226-232). Morgan Kaufmann.

[50] Kuderer, M. (2015). *Socially Compliant Mobile Robot Navigation* (Doctoral dissertation, Verlag nicht ermittelbar).

[51] Vasquez, D., Okal, B., & Arras, K. O. (2014, September). Inverse reinforcement learning algorithms and features for robot navigation in crowds: an experimental comparison. In *2014 IEEE/RSJ International Conference on Intelligent Robots and Systems* (pp. 1341-1346). IEEE.

[52] Chen, Y. F., Everett, M., Liu, M., & How, J. P. (2017, September). Socially aware motion planning with deep reinforcement learning. In *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 1343-1350). IEEE.

[53] Chen, Y. F., Everett, M., Liu, M., & How, J. P. (2017, September). Socially aware motion planning with deep reinforcement learning. In *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 1343-1350). IEEE.

[54] Everett, M., Chen, Y. F., & How, J. P. (2018, October). Motion planning among dynamic, decision-making agents with deep reinforcement learning. In *2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 3052-3059). IEEE.

[55] Ciou, P. H., Hsiao, Y. T., Wu, Z. Z., Tseng, S. H., & Fu, L. C. (2018, October). Composite Reinforcement Learning for Social Robot Navigation. In *2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)* (pp. 2553-2558). IEEE.

[56] Long, P., Fanl, T., Liao, X., Liu, W., Zhang, H., & Pan, J. (2018, May). Towards optimally decentralized multi-robot collision avoidance via deep reinforcement learning. In *2018 IEEE International Conference on Robotics and Automation (ICRA)* (pp. 6252-6259). IEEE.