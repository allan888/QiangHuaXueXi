"""
严格按照论文复现:
"Path Planning in Dynamic Environments Based on Q-Learning"
Xiangqi Li, Ocean University of China
ICMSC 2023, Highlights in Science, Engineering and Technology, Vol.63, pp.222-230

============================================================
环境:
  - 10×10 栅格 (MATLAB 1-indexed → Python 0-indexed)
  - 起点 (0,0), 终点 (9,9)
  - 3个不同复杂度的环境
  - 动态障碍物沿直线往返移动, 与agent同速

关键规则:
  - agent 必须先到达 trash point, 终点奖励才生效
  - 动作选择: 纯贪心 (选最大Q值动作, 平局随机)
  - 无 ε-greedy 探索

参数:
  - α (学习率): 0.99 初始, 每100回合减0.01
  - γ (折扣率): 0.25 固定
  - 无 ε-greedy 探索, 纯贪心+随机破平局
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import time

OUTDIR = os.path.dirname(os.path.abspath(__file__)) + '\\'

# ============================================================
# 1. 动态障碍物
# ============================================================

class DynamicObstacle:
    """沿线段往返移动的动态障碍物, 每步1格, 碰到端点即反向."""

    def __init__(self, path_start, path_end, start_pos):
        self.path_start = np.array(path_start)
        self.path_end = np.array(path_end)
        self.start_pos = np.array(start_pos)
        self.pos = np.array(start_pos)
        # 确定移动轴 (仅支持水平或垂直)
        diff = self.path_end - self.path_start
        self.axis = int(np.argmax(np.abs(diff)))
        self.signed_axis_range = (min(self.path_start[self.axis],
                                      self.path_end[self.axis]),
                                  max(self.path_start[self.axis],
                                      self.path_end[self.axis]))
        # 初始方向: 远离最近端点
        lo, hi = self.signed_axis_range
        mid = (lo + hi) / 2.0
        self.direction = 1 if self.pos[self.axis] <= mid else -1

    def move(self):
        new_val = self.pos[self.axis] + self.direction
        lo, hi = self.signed_axis_range
        if new_val > hi:
            new_val = hi - 1
            self.direction = -1
        elif new_val < lo:
            new_val = lo + 1
            self.direction = 1
        self.pos[self.axis] = new_val

    def get_pos(self):
        return tuple(self.pos)

    def reset(self):
        self.pos = self.start_pos.copy()
        lo, hi = self.signed_axis_range
        mid = (lo + hi) / 2.0
        self.direction = 1 if self.pos[self.axis] <= mid else -1


# ============================================================
# 2. 环境
# ============================================================

class GridEnvironment:
    """
    10×10 栅格环境, 包含:
      - 静态障碍物 (灰色)
      - 动态障碍物 (红色, 每步移动)
      - trash point (黄色, 必经点)
      - 起点=红色, 终点=深蓝色
    """

    def __init__(self, env_id, seed=None):
        """
        env_id: 1, 2, 或 3 (对应论文的三种环境)
        """
        self.env_id = env_id
        self.grid_size = 10
        self.rng = np.random.RandomState(seed)

        # 坐标: Python 0-indexed (论文 MATLAB 1-indexed → 全部-1)
        self.start = (0, 0)       # 论文: (1,1)
        self.end = (9, 9)         # 论文: (10,10)

        # 静态障碍物, trash point, 动态障碍物 — 按环境配置
        self.static_obstacles = set()
        self.dynamic_obstacles = []
        self.trash_collected = False

        if env_id == 1:
            self._setup_env1()
        elif env_id == 2:
            self._setup_env2()
        elif env_id == 3:
            self._setup_env3()

        self.agent_pos = self.start

    # --------------------------------------------------------
    # 静态障碍物布局 (论文 Fig.1, 从图中推断)
    # --------------------------------------------------------

    def _add_block(self, r_start, r_end, c_start, c_end):
        """添加矩形障碍物块"""
        for i in range(r_start, r_end + 1):
            for j in range(c_start, c_end + 1):
                if (i, j) != self.start and (i, j) != self.end:
                    self.static_obstacles.add((i, j))

    def _setup_env1(self):
        # 1个静态障碍物: 右上区域, 避开动态障碍物路径(row=6) 和 trash(4,4)
        self._add_block(2, 4, 6, 8)
        # 1个动态障碍物: 论文 (7,2)→(7,10), 起点(7,2) → Python (6,1)→(6,9), 起点(6,1)
        self.dynamic_obstacles = [
            DynamicObstacle(path_start=(6, 1), path_end=(6, 9), start_pos=(6, 1))
        ]
        self.trash_point = (4, 4)   # 论文: (5,5)
        self.num_actions = 4        # 上下左右
        self.trash_reward = 5.0
        self.end_reward = 5.0

    def _setup_env2(self):
        # 2个静态障碍物: 分散放置, 避开动态障碍物路径(row=6) 和 trash(4,4)
        self._add_block(2, 4, 1, 4)   # 左上区域
        self._add_block(7, 8, 5, 8)   # 右下区域
        # 1个动态障碍物: 同环境1
        self.dynamic_obstacles = [
            DynamicObstacle(path_start=(6, 1), path_end=(6, 9), start_pos=(6, 1))
        ]
        self.trash_point = (4, 4)   # 论文: (5,5)
        self.num_actions = 4        # 上下左右
        self.trash_reward = 5.0
        self.end_reward = 5.0

    def _setup_env3(self):
        # 多个静态障碍物 (最复杂), 不阻挡 trash(6,3) 和动态障碍物路径
        self._add_block(1, 2, 5, 7)     # 顶部小块
        self._add_block(3, 5, 0, 2)     # 左侧竖向
        self._add_block(4, 5, 5, 7)     # 中间小块
        self._add_block(7, 8, 4, 7)     # 底部横向
        # 2个动态障碍物
        # DO1: 论文 (10,1)→(10,9), 起点(10,1) → Python (9,0)→(9,8), 起点(9,0)
        # DO2: 论文 (1,9)→(10,9), 起点(5,9) → Python (0,8)→(9,8), 起点(4,8)
        self.dynamic_obstacles = [
            DynamicObstacle(path_start=(9, 0), path_end=(9, 8), start_pos=(9, 0)),
            DynamicObstacle(path_start=(0, 8), path_end=(9, 8), start_pos=(4, 8)),
        ]
        self.trash_point = (6, 3)   # 论文: (7,4)
        self.num_actions = 8        # 八个方向
        self.trash_reward = 10.0    # 环境3垃圾点奖励提高到10
        self.end_reward = 5.0

    # --------------------------------------------------------
    # 环境交互
    # --------------------------------------------------------

    def is_obstacle(self, pos):
        return (pos in self.static_obstacles
                or any(pos == dobj.get_pos() for dobj in self.dynamic_obstacles))

    def is_valid(self, pos):
        r, c = pos
        return (0 <= r < self.grid_size
                and 0 <= c < self.grid_size
                and pos not in self.static_obstacles
                and not any(pos == dobj.get_pos() for dobj in self.dynamic_obstacles))

    def reset(self):
        self.agent_pos = self.start
        self.trash_collected = False
        for dobj in self.dynamic_obstacles:
            dobj.reset()
        return self.agent_pos

    def step(self, action):
        """
        执行动作, 返回 (next_state, reward, done)
        规则:
          - 越界 → -1, 终止
          - 撞障碍物(静/动) → -1, 终止
          - 到达 trash point → +trash_reward, 继续
          - 到达终点 (trash已收集) → +end_reward, 终止
          - 到达终点 (trash未收集) → 0, 继续 (穿过终点)
          - 其他 → 0, 继续
        """
        # 4动作: {0:上, 1:下, 2:左, 3:右}
        # 8动作: {0:上, 1:下, 2:左, 3:右, 4:左上, 5:右上, 6:左下, 7:右下}
        if self.num_actions == 4:
            dirs = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        else:
            dirs = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                    4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)}

        dr, dc = dirs[action]
        new_pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)

        # 越界
        if not (0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size):
            return self.agent_pos, -1.0, True

        # 撞静态障碍物 (移动前检查)
        if new_pos in self.static_obstacles:
            return self.agent_pos, -1.0, True

        # 移动到新位置
        self.agent_pos = new_pos

        # 移动动态障碍物 (与agent同速)
        for dobj in self.dynamic_obstacles:
            dobj.move()

        # 检查是否与动态障碍物碰撞
        if any(self.agent_pos == dobj.get_pos() for dobj in self.dynamic_obstacles):
            return self.agent_pos, -1.0, True

        # 到达 trash point
        if self.agent_pos == self.trash_point and not self.trash_collected:
            self.trash_collected = True
            return self.agent_pos, self.trash_reward, False

        # 到达终点 — 仅在收集trash后有效
        if self.agent_pos == self.end:
            if self.trash_collected:
                return self.agent_pos, self.end_reward, True
            else:
                return self.agent_pos, 0.0, False

        return self.agent_pos, 0.0, False


# ============================================================
# 3. Q-learning 核心
# ============================================================

class QLearningAgent:
    """
    Q-learning, 论文 Section 2.3.1 参数:
      α (学习率) = 0.99 初始, 每100回合 -0.01
      γ (折扣率) = 0.25 固定
      动作选择: 纯贪心 + 随机破平局 (无 ε-greedy)
    """

    def __init__(self, env, alpha_start=0.99, alpha_decay_interval=100,
                 alpha_decay=0.01, gamma=0.25, seed=None):
        self.env = env
        self.alpha_start = alpha_start
        self.alpha_decay_interval = alpha_decay_interval
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.rng = np.random.RandomState(seed)

        g = env.grid_size
        na = env.num_actions
        # Q-table: (grid_size, grid_size, num_actions), 初始化为0
        self.q_table = np.zeros((g, g, na))

        # 记录
        self.episode_rewards = []
        self.episode_steps = []

    def get_alpha(self, episode):
        """α 每 alpha_decay_interval 回合减少 alpha_decay"""
        num_decays = episode // self.alpha_decay_interval
        alpha = self.alpha_start - num_decays * self.alpha_decay
        return max(alpha, 0.01)

    def choose_action(self, state):
        """
        纯贪心选择: 取最大Q值的动作, 多个最大时随机选一个.
        论文 Section 2.3.3 原文:
        "if more than one actions has same highest Q value,
         then the action will be selected randomly"
        """
        q_vals = self.q_table[state[0], state[1], :]
        max_q = np.max(q_vals)
        best_actions = np.where(q_vals == max_q)[0]
        return self.rng.choice(best_actions)

    def update_q(self, state, action, reward, next_state):
        """
        Q-learning 更新公式 (论文 Equation 1):
        ΔQ(s,a) = α [r + γ·max Q(s',a') - Q(s,a)]
        """
        td_target = reward + self.gamma * np.max(
            self.q_table[next_state[0], next_state[1], :])
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.alpha * td_error

    def train(self, episodes, verbose=True, print_interval=None):
        if print_interval is None:
            print_interval = max(1, episodes // 10)

        for ep in range(episodes):
            state = self.env.reset()
            self.alpha = self.get_alpha(ep)
            total_reward = 0.0
            steps = 0
            max_steps = self.env.grid_size * 200  # 2000步上限, 保证充分探索

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q(state, action, reward, next_state)
                total_reward += reward
                steps += 1
                state = next_state
                if done:
                    break

            self.episode_steps.append(steps)
            self.episode_rewards.append(total_reward)

            if verbose and (ep + 1) % print_interval == 0:
                print(f"  Ep {ep+1}/{episodes}  "
                      f"steps={steps}  reward={total_reward:.1f}  "
                      f"alpha={self.alpha:.3f}")

    def get_path(self):
        """从学到的Q-table用贪心策略提取路径 (无动态障碍物, 仅静态环境)"""
        # 临时关闭动态障碍物来评估路径
        saved_dynamic = self.env.dynamic_obstacles
        self.env.dynamic_obstacles = []

        path = [self.env.start]
        state = self.env.start
        self.env.agent_pos = state
        self.env.trash_collected = False
        visited = set()
        max_steps = self.env.grid_size * 50

        dirs_4 = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dirs_8 = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1),
                  4: (-1, -1), 5: (-1, 1), 6: (1, -1), 7: (1, 1)}
        dirs = dirs_8 if self.env.num_actions == 8 else dirs_4

        for _ in range(max_steps):
            if state == self.env.end and self.env.trash_collected:
                break
            visited.add(state)
            q_vals = self.q_table[state[0], state[1], :]
            sorted_actions = np.argsort(q_vals)[::-1]

            moved = False
            for a in sorted_actions:
                dr, dc = dirs[a]
                np_ = (state[0] + dr, state[1] + dc)
                if (0 <= np_[0] < self.env.grid_size
                        and 0 <= np_[1] < self.env.grid_size
                        and np_ not in self.env.static_obstacles
                        and (np_ not in visited or np_ == self.env.end
                             or np_ == self.env.trash_point)):
                    state = np_
                    path.append(state)
                    if state == self.env.trash_point:
                        self.env.trash_collected = True
                    moved = True
                    break
            if not moved:
                break

        # 恢复动态障碍物
        self.env.dynamic_obstacles = saved_dynamic
        return path


# ============================================================
# 4. A* 对比 (仅静态障碍物)
# ============================================================

def astar(env):
    """A* 算法, 只能处理静态障碍物, 需要经过 trash point"""
    import heapq

    # 分两段: start → trash, trash → end
    def astar_segment(from_pos, to_pos):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if env.env_id == 3:
            dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        open_set = [(0, from_pos)]
        came_from = {}
        g_score = {from_pos: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == to_pos:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for d in dirs:
                nb = (current[0] + d[0], current[1] + d[1])
                if not (0 <= nb[0] < env.grid_size and 0 <= nb[1] < env.grid_size):
                    continue
                if nb in env.static_obstacles:
                    continue
                cost = 1.414 if abs(d[0]) + abs(d[1]) == 2 else 1.0
                tg = g_score[current] + cost
                if nb not in g_score or tg < g_score[nb]:
                    g_score[nb] = tg
                    f = tg + abs(nb[0] - to_pos[0]) + abs(nb[1] - to_pos[1])
                    heapq.heappush(open_set, (f, nb))
                    came_from[nb] = current
        return []

    p1 = astar_segment(env.start, env.trash_point)
    if not p1:
        return []
    p2 = astar_segment(env.trash_point, env.end)
    if not p2:
        return []
    return p1[:-1] + p2


# ============================================================
# 5. 可视化
# ============================================================

def plot_env(ax, env, title="", show_dynamic=True):
    """绘制环境地图"""
    g = env.grid_size
    grid = np.zeros((g, g))
    for pos in env.static_obstacles:
        grid[pos] = 1  # 静态障碍物
    if show_dynamic:
        for dobj in env.dynamic_obstacles:
            pos = dobj.get_pos()
            if 0 <= pos[0] < g and 0 <= pos[1] < g:
                grid[pos] = 2  # 动态障碍物

    # 自定义颜色: 空地=白色, 静态障碍物=天蓝, 动态障碍物=红色
    cmap = ListedColormap(['white', 'skyblue', 'red'])
    ax.imshow(grid.T, origin='lower', cmap=cmap, vmin=-0.5, vmax=2.5,
              extent=[-0.5, g - 0.5, -0.5, g - 0.5])

    ax.plot(env.start[0], env.start[1], 'o', color='red',
            markersize=11, markeredgecolor='black', markeredgewidth=1.5,
            label='Start')
    ax.plot(env.end[0], env.end[1], 's', color='navy',
            markersize=11, markeredgecolor='black', markeredgewidth=1.5,
            label='End')
    ax.plot(env.trash_point[0], env.trash_point[1], 'D', color='yellow',
            markersize=10, markeredgecolor='black', markeredgewidth=1.5,
            label='Trash')

    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_xticks(range(g))
    ax.set_yticks(range(g))
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=6)


def plot_path(ax, path, color='orange', linewidth=2.5, label='Path'):
    """在图上绘制路径"""
    if not path:
        return
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, color=color, linewidth=linewidth, marker='.',
            markersize=4, label=label)


def plot_convergence(episode_rewards, title, save_path, window=None):
    """绘制训练收敛曲线 (每步累计奖励)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = np.array(episode_rewards)
    ax.plot(data, alpha=0.15, color='steelblue', linewidth=0.5)
    if window is None:
        window = max(1, len(data) // 50)
    if len(data) >= window:
        smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(data)), smoothed,
                color='darkblue', linewidth=2,
                label=f'Moving avg (w={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward per Episode')
    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {os.path.basename(save_path)}")


def format_q_table(q_table, state):
    """格式化输出某状态的Q值"""
    q_vals = q_table[state[0], state[1], :]
    return ", ".join(f"{v:+.4f}" for v in q_vals)


# ============================================================
# 6. 主程序
# ============================================================

def run_solution(env_id, episodes, num_runs=3, seed_base=42):
    """
    对环境 env_id 运行 num_runs 次训练, 每次产生一条最优路径.
    """
    print(f"\n{'='*60}")
    print(f" Environment {env_id} — {num_runs} runs, {episodes} episodes each")
    print(f"{'='*60}")

    env_config = {
        1: {'n_actions': 4, 'desc': '1 static + 1 dynamic'},
        2: {'n_actions': 4, 'desc': '2 static + 1 dynamic'},
        3: {'n_actions': 8, 'desc': 'multiple static + 2 dynamic'},
    }

    runs = []
    for run_idx in range(num_runs):
        seed = seed_base + run_idx * 100
        print(f"\n--- Run {run_idx+1}/{num_runs} (seed={seed}) ---")
        t0 = time.time()

        env = GridEnvironment(env_id, seed=seed)
        agent = QLearningAgent(env, seed=seed)
        agent.train(episodes, verbose=True,
                    print_interval=max(1, episodes // 10))
        t1 = time.time()

        path = agent.get_path()
        success = (len(path) > 0
                   and env.trash_point in path
                   and path[-1] == env.end)

        # 计算: 到达trash和end的步骤
        trash_step = path.index(env.trash_point) if env.trash_point in path else -1

        runs.append({
            'agent': agent,
            'env': env,
            'path': path,
            'success': success,
            'time': t1 - t0,
            'path_len': len(path),
            'trash_step': trash_step,
        })
        print(f"  训练时间: {t1-t0:.2f}s  路径长度: {len(path)}  "
              f"到达trash+end: {success}")

    return runs


def print_action_table(runs, env_id):
    """打印起点处Q值表 (对照论文 Table 1/2/3)"""
    print(f"\n  Q-values at start point (0,0) for Environment {env_id}:")
    na = runs[0]['env'].num_actions
    if na == 4:
        print(f"  {'Action':<12} {'1(Down)':>10} {'2(Up)':>10} "
              f"{'3(Right)':>10} {'4(Left)':>10}")
    else:
        print(f"  {'Action':<12} {'1(D)':>8} {'2(U)':>8} {'3(R)':>8} "
              f"{'4(L)':>8} {'5(LU)':>8} {'6(RU)':>8} {'7(LD)':>8} {'8(RD)':>8}")

    for i, run in enumerate(runs):
        q_vals = run['agent'].q_table[0, 0, :]
        vals_str = " ".join(f"{v:>+8.4f}" for v in q_vals)
        print(f"  Path {env_id}-{i+1}:        {vals_str}")


def main():
    print("=" * 65)
    print(" 复现: Path Planning in Dynamic Environments")
    print("       Based on Q-Learning")
    print(" Xiangqi Li, Ocean University of China, ICMSC 2023")
    print("=" * 65)

    # 论文中不同环境的训练回合数 (从收敛图推断)
    EP_ENV1 = 1000   # 收敛在 [250, 500]
    EP_ENV2 = 2000   # 收敛在 [250, 1000]
    EP_ENV3 = 8000   # 收敛在 [1500, 8000]

    # ---- 三个环境, 各运行3次 ----
    all_results = {}

    for env_id, episodes in [(1, EP_ENV1), (2, EP_ENV2), (3, EP_ENV3)]:
        runs = run_solution(env_id, episodes, num_runs=3, seed_base=42)
        all_results[env_id] = runs
        print_action_table(runs, env_id)

    # ---- A* 对比 ----
    print(f"\n{'='*60}")
    print(" A* 对比 (仅静态障碍物, 须经过 trash point)")
    print(f"{'='*60}")
    for env_id in [1, 2, 3]:
        env = GridEnvironment(env_id)
        t0 = time.time()
        path = astar(env)
        t1 = time.time()
        print(f"  Env{env_id}: {t1-t0:.4f}s  路径长度: {len(path)}  "
              f"经过trash: {env.trash_point in path}")

    # ================================================================
    # 生成图表 (对照论文 Fig.1 ~ Fig.8)
    # ================================================================
    print(f"\n{'='*60}")
    print(" 生成可视化图表...")
    print(f"{'='*60}")

    # --- Fig.1: 三个环境地图 (含动态障碍物轨迹箭头) ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
    for idx, env_id in enumerate([1, 2, 3]):
        env = GridEnvironment(env_id)
        ax = axes1[idx]
        plot_env(ax, env, f"Environment {env_id}", show_dynamic=True)
        # 画动态障碍物路径箭头
        for dobj in env.dynamic_obstacles:
            ps = dobj.path_start
            pe = dobj.path_end
            ax.annotate('', xy=(pe[0], pe[1]), xytext=(ps[0], ps[1]),
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=2, alpha=0.7))
    fig1.suptitle("Fig. 1: Three Environment Maps", fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig1.savefig(OUTDIR + 'Fig1_environments.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  保存: Fig1_environments.png")

    # --- 每个环境的路径图和收敛图 ---
    for env_id in [1, 2, 3]:
        runs = all_results[env_id]
        env = runs[0]['env']

        # 路径图 (3条最优路径)
        fig_path, axes_path = plt.subplots(1, 3, figsize=(18, 6))
        for i, run in enumerate(runs):
            ax = axes_path[i]
            # 显示训练结束时的动态障碍物位置
            temp_env = GridEnvironment(env_id)
            plot_env(ax, temp_env,
                     f"Path {env_id}-{i+1} ({len(run['path'])} steps)",
                     show_dynamic=True)
            plot_path(ax, run['path'], color='orange')
            # 标注 trash
            ax.plot(temp_env.trash_point[0], temp_env.trash_point[1],
                    'D', color='yellow', markersize=10,
                    markeredgecolor='black', markeredgewidth=1.5)
        fig_path.suptitle(
            f"Environment {env_id}: Three Optimal Paths",
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_path.savefig(OUTDIR + f'Fig_paths_env{env_id}.png',
                         dpi=150, bbox_inches='tight')
        plt.close(fig_path)
        print(f"  保存: Fig_paths_env{env_id}.png")

        # 收敛图 (3条训练曲线)
        fig_conv, axes_conv = plt.subplots(1, 3, figsize=(18, 6))
        w = max(1, len(runs[0]['agent'].episode_rewards) // 50)
        for i, run in enumerate(runs):
            ax = axes_conv[i]
            data = np.array(run['agent'].episode_rewards)
            ax.plot(data, alpha=0.15, color='steelblue', linewidth=0.5)
            if len(data) >= w:
                smoothed = np.convolve(data, np.ones(w) / w, mode='valid')
                ax.plot(range(w - 1, len(data)), smoothed,
                        color='darkblue', linewidth=2,
                        label=f'Moving avg (w={w})')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.set_title(f'Path {env_id}-{i+1}')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig_conv.suptitle(
            f"Environment {env_id}: Training Progress",
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_conv.savefig(OUTDIR + f'Fig_conv_env{env_id}.png',
                         dpi=150, bbox_inches='tight')
        plt.close(fig_conv)
        print(f"  保存: Fig_conv_env{env_id}.png")

    # ================================================================
    # 结果汇总
    # ================================================================
    print(f"\n{'='*70}")
    print(" 性能汇总")
    print(f"{'='*70}")
    print(f" {'环境/路径':<25} {'训练时间':>10} {'路径长度':>8} "
          f"{'到达trash':>10} {'到达end':>8}")
    print("-" * 65)
    for env_id in [1, 2, 3]:
        for i, run in enumerate(all_results[env_id]):
            path = run['path']
            has_trash = run['env'].trash_point in path
            has_end = path[-1] == run['env'].end if path else False
            print(f" Env{env_id} Path {i+1:<2}"
                  f"{'':>13} {run['time']:>8.2f}s {len(path):>8}  "
                  f"{'Yes' if has_trash else 'No':>10}  "
                  f"{'Yes' if has_end else 'No':>8}")

    print(f"\n完成! 图表已保存到 {OUTDIR}")


if __name__ == '__main__':
    main()
