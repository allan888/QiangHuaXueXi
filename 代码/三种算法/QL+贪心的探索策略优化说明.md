# 探索策略优化说明 (Alg2)

## 问题

原始参数 `epsilon=0.7` 全程固定，70% 随机动作。在 100×100 网格 + 稀疏奖励（撞墙 -1，到达 +1）下，随机探索几乎不可能偶然到达目标，Q 值无法从目标反向传播，导致训练无效、上来就撞墙。

## 三项关键修改

### 1. Epsilon 衰减

| 参数 | 修改前 | 修改后 |
|------|--------|--------|
| `epsilon` | 0.7（固定） | 1.0（起始） |
| `epsilon_decay` | 无 | 0.995 / episode |
| `epsilon_min` | 无 | 0.02 |

每 episode 结束时执行 `self.epsilon = max(0.02, self.epsilon * 0.995)`：

```
Episode   1: ε = 1.000  (100% 探索)
Episode 100: ε = 0.606
Episode 300: ε = 0.223
Episode 500: ε = 0.082
Episode 780: ε = 0.020  (触底, 2% 探索)
Episode2000: ε = 0.020
```

- **前期**（~0-500 集）：高探索率，充分覆盖状态空间，积累碰撞经验（哪些动作会撞墙）
- **中期**（~500-800 集）：探索逐步降低，利用已学到的 Q 值同时保留一定随机性
- **后期**（~800-2000 集）：98% 利用已学策略，偶尔随机尝试避免局部最优

### 2. 距离奖励塑形（Reward Shaping）

```python
if reward >= 0:  # 未碰撞时
    cur_dist = self._distance_to_goal(x, y)  # 当前位置到目标距离
    reward += (prev_dist - cur_dist) * self.distance_reward / WORLD_SIZE
```

原理：每一步未碰撞时，根据距离目标的变化给予微调奖励：
- **靠近目标** → 正奖励（`distance_reward=0.05`，除以 WORLD_SIZE=1000 缩放）
- **远离目标** → 负奖励
- **距离不变** → 0

这在稀疏奖励（只有终点+1）的基础上提供了**稠密的梯度信号**，使 Q 值能逐步向目标方向传播，而不是完全依赖随机撞到目标。

### 3. 初始 epsilon 设为 1.0

修改前 `epsilon=0.7` 意味着 30% 概率走 Q 值最高动作，但 Q 表初始全是 0，选哪个都一样。改为 1.0 起步，前期 100% 随机探索，等积累足够经验后再逐步衰减。

## 效果验证

```
修改前: epsilon=0.7 固定 → 撞墙，无法到达目标
修改后:
  Ep 500:   steps=33  eps=0.082
  Ep 1000:  steps=37  eps=0.020
  Ep 1500:  steps=38  reward=1.03   (到达目标)
  Ep 2000:  steps=39  reward=1.03
  最终路径: 39 pts, GoalReached=True
  训练成功率: 375/2000 (18.75%)
```

## 代码位置

- 基类 `QLearningBase.__init__`：新增 `epsilon_decay`、`epsilon_min`、`distance_reward` 参数
- 基类 `QLearningBase.decay_epsilon()`：每 episode 衰减 epsilon
- 基类 `QLearningBase._distance_to_goal()`：计算当前位置到目标欧氏距离
- `Alg2_OriginalQL_Dynamic.train()`：加入距离奖励计算和 epsilon 衰减调用
- `main()` 中 Alg2 初始化参数：
  ```python
  alg2 = Alg2_OriginalQL_Dynamic(env2, episodes=EPISODES, epsilon=1.0,
                                  epsilon_decay=0.995, epsilon_min=0.02,
                                  alpha=0.4, step_penalty=-0.001,
                                  distance_reward=0.05, seed=SEED)
  ```
