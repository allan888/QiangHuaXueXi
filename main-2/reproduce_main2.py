"""
严格按照论文复现:
"Q-learning-based unmanned aerial vehicle path planning
 with dynamic obstacle avoidance"
Amala Sonny, Sreenivasa Reddy Yeduri, Linga Reddy Cenkeramaddi
Applied Soft Computing 147 (2023) 110773

============================================================
环境:
  - 25×25 栅格, 每格20像素
  - 起点(0,0), 终点(24,24)
  - 4个不同大小的静态障碍物 (论文Fig.1)
  - 0/2/4 个动态障碍物, 随机移动, 与UAV同速

参数 (论文Table 1):
  - episodes: 5000
  - α (学习率): 0.3
  - γ (折扣率): 逐渐从0.1增加到0.9
  - ε (探索率): 0.9 固定 (ε-greedy策略)
  - 经验回放缓冲区大小: 5000

动作空间: 上、下、左、右 (4个动作)
奖励: 到达终点+1, 碰到障碍物-1, 其他0

四种算法:
  Alg1: 原始Q-learning + ε-greedy,  仅静态障碍物
  Alg2: 原始Q-learning + ε-greedy,  静态+动态障碍物
  Alg3: 提出的Q-learning + SDP,     仅静态障碍物
  Alg4: 提出的Q-learning + SDP,     静态+动态障碍物

SDP (Shortest Distance Prioritization) 策略:
  在每个状态, 对4个候选动作分别计算如果执行后到终点的
  距离, 选择距离最小的动作。若多个动作距离相同, 随机选。
============================================================
"""

import numpy as np
import time
from collections import deque
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
# 1. 环境 (论文 Section 2, Fig.1)
# ============================================================

class UAVEnvironment:
    """
    25×25 栅格环境
    - 4个静态障碍物 (按论文Fig.1布局)
    - N个动态障碍物 (随机初始化, 每步随机移动)
    """

    def __init__(self, grid_size=25, n_dynamic=0, seed=None):
        self.grid_size = grid_size
        self.n_dynamic = n_dynamic
        self.rng = np.random.RandomState(seed)

        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)

        self.static_obstacles = self._build_static_obstacles()
        self.dynamic_obstacles = []
        if n_dynamic > 0:
            self._init_dynamic_obstacles()

        self.agent_pos = self.start

    # --------------------------------------------------------
    def _build_static_obstacles(self):
        """
        论文Fig.1: 四个不同大小和形状的蓝色矩形障碍物。
        论文没有给出精确坐标, 根据Fig.1的视觉布局推断:
          - 障碍物1: 左上区域, 横向长条  (建筑物)
          - 障碍物2: 中部偏右, 竖向长条  (建筑物)
          - 障碍物3: 右下区域, L形       (建筑物)
          - 障碍物4: 右上区域, 方块      (建筑物)
        """
        obs = set()
        g = self.grid_size

        # 障碍物1: 横向长条, 大约在 (2,2) ~ (5,10)
        for i in range(2, 5):
            for j in range(2, 10):
                if 0 <= i < g and 0 <= j < g:
                    obs.add((i, j))

        # 障碍物2: 竖向长条, 大约在 (8,14) ~ (18,16)
        for i in range(8, 18):
            for j in range(14, 16):
                if 0 <= i < g and 0 <= j < g:
                    obs.add((i, j))

        # 障碍物3: L形, 大约在 (17,3) ~ (22,8)
        #   水平部分
        for i in range(17, 22):
            for j in range(3, 5):
                if 0 <= i < g and 0 <= j < g:
                    obs.add((i, j))
        #   垂直部分
        for j in range(5, 8):
            if 0 <= 21 < g and 0 <= j < g:
                obs.add((21, j))

        # 障碍物4: 方块, 大约在 (10,20) ~ (17,24)
        for i in range(10, 17):
            for j in range(20, 24):
                if 0 <= i < g and 0 <= j < g:
                    obs.add((i, j))

        return obs

    # --------------------------------------------------------
    def _init_dynamic_obstacles(self):
        """随机初始化动态障碍物, 不与静态障碍物/起点/终点重合"""
        self.dynamic_obstacles = []
        for _ in range(self.n_dynamic):
            while True:
                pos = (self.rng.randint(0, self.grid_size),
                       self.rng.randint(0, self.grid_size))
                if (pos not in self.static_obstacles
                        and pos != self.start
                        and pos != self.goal
                        and pos not in self.dynamic_obstacles):
                    self.dynamic_obstacles.append(pos)
                    break

    # --------------------------------------------------------
    def move_dynamic_obstacles(self):
        """每个动态障碍物随机选择一个方向移动 (与UAV同速, 每步1格)"""
        new_positions = []
        for pos in self.dynamic_obstacles:
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
            self.rng.shuffle(dirs)
            moved = False
            for d in dirs:
                np_ = (pos[0] + d[0], pos[1] + d[1])
                if (0 <= np_[0] < self.grid_size
                        and 0 <= np_[1] < self.grid_size
                        and np_ not in self.static_obstacles
                        and np_ != self.start
                        and np_ != self.goal):
                    new_positions.append(np_)
                    moved = True
                    break
            if not moved:
                new_positions.append(pos)
        self.dynamic_obstacles = new_positions

    # --------------------------------------------------------
    def is_obstacle(self, pos):
        return (pos in self.static_obstacles
                or pos in self.dynamic_obstacles)

    def is_valid(self, pos):
        return (0 <= pos[0] < self.grid_size
                and 0 <= pos[1] < self.grid_size
                and not self.is_obstacle(pos))

    # --------------------------------------------------------
    def reset(self):
        self.agent_pos = self.start
        if self.n_dynamic > 0:
            self._init_dynamic_obstacles()
        return self.agent_pos

    # --------------------------------------------------------
    def step(self, action):
        """
        执行动作。action ∈ {0:上, 1:下, 2:左, 3:右}
        返回 (next_state, reward, done)
        """
        directions = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        new_pos = (self.agent_pos[0] + directions[action][0],
                   self.agent_pos[1] + directions[action][1])

        # 越界 → -1, 终止
        if not (0 <= new_pos[0] < self.grid_size
                and 0 <= new_pos[1] < self.grid_size):
            return self.agent_pos, -1, True

        # 到达终点 → +1, 终止
        if new_pos == self.goal:
            self.agent_pos = new_pos
            return new_pos, 1, True

        # 碰到障碍物 → -1, 终止
        if self.is_obstacle(new_pos):
            return self.agent_pos, -1, True

        # 正常移动 → 0, 继续
        self.agent_pos = new_pos
        return new_pos, 0, False


# ============================================================
# 2. Q-learning 核心 (论文 Section 3)
# ============================================================

def manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class QLearningBase:
    """
    Q-learning 基类。
    参数严格按论文 Table 1:
      α=0.3, γ 从0.1渐增到0.9, ε=0.9固定
    """

    def __init__(self, env, alpha=0.3, gamma_start=0.1, gamma_end=0.9,
                 epsilon=0.9, episodes=5000, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.epsilon = epsilon          # 论文: ε=0.9 固定
        self.episodes = episodes
        self.rng = np.random.RandomState(seed)

        g = env.grid_size
        # Q-table: (grid_size, grid_size, 4动作)
        self.q_table = np.zeros((g, g, 4))

        # 经验回放缓冲区 (论文: b=5000)
        self.replay_buffer = deque(maxlen=5000)

        # 记录
        self.episode_steps = []
        self.episode_rewards = []

    # --------------------------------------------------------
    def get_gamma(self, episode):
        """γ 随训练进程从 gamma_start 线性增加到 gamma_end"""
        progress = min(episode / self.episodes, 1.0)
        return self.gamma_start + (self.gamma_end - self.gamma_start) * progress

    # --------------------------------------------------------
    def choose_action_egreedy(self, state):
        """
        论文公式(5): ε-greedy 策略
        以 ε=0.9 的概率随机探索, (1-ε) 概率贪心选择最大Q值动作
        """
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, 4)
        else:
            q_vals = self.q_table[state[0], state[1], :]
            max_q = np.max(q_vals)
            best = np.where(q_vals == max_q)[0]
            return self.rng.choice(best)

    # --------------------------------------------------------
    def choose_action_sdp(self, state):
        """
        论文 Algorithm 3/4: Shortest Distance Prioritization
        对4个候选动作, 计算执行后位置到终点的曼哈顿距离,
        选距离最小的动作。多个相同则随机。
        如果动作导致碰撞/越界, 该动作距离设为无穷大。
        """
        dirs = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        goal = self.env.goal
        g = self.env.grid_size

        distances = []
        for a in range(4):
            np_ = (state[0] + dirs[a][0], state[1] + dirs[a][1])
            if (0 <= np_[0] < g and 0 <= np_[1] < g
                    and not self.env.is_obstacle(np_)):
                distances.append(manhattan_dist(np_, goal))
            else:
                distances.append(float('inf'))

        min_d = min(distances)
        if min_d == float('inf'):
            return self.rng.randint(0, 4)
        best = [a for a, d in enumerate(distances) if d == min_d]
        return self.rng.choice(best)

    # --------------------------------------------------------
    def update_q(self, state, action, reward, next_state, gamma):
        """
        论文公式(9): Q-table 更新
        Q_new(s,a) = (1-α)·Q_now(s,a) + α·[r + γ·max_a' Q(s',a')]
        等价于标准Q-learning: Q += α·[r + γ·max Q(s',a') - Q(s,a)]
        """
        s, ns = state, next_state
        # TD target (论文公式(7))
        td_target = reward + gamma * np.max(self.q_table[ns[0], ns[1], :])
        # TD error (论文公式(8))
        td_error = td_target - self.q_table[s[0], s[1], action]
        # 更新 (论文公式(9))
        self.q_table[s[0], s[1], action] += self.alpha * td_error

        # 存入经验回放
        self.replay_buffer.append((state, action, reward, next_state))

    # --------------------------------------------------------
    def experience_replay(self, gamma, batch_size=32):
        """
        论文: 从经验回放缓冲区随机采样, 进行额外学习。
        缓冲区大小 b=5000。
        """
        if len(self.replay_buffer) < batch_size:
            return
        indices = self.rng.choice(len(self.replay_buffer), batch_size,
                                  replace=False)
        for idx in indices:
            s, a, r, ns = self.replay_buffer[idx]
            td_target = r + gamma * np.max(self.q_table[ns[0], ns[1], :])
            self.q_table[s[0], s[1], a] += self.alpha * (
                td_target - self.q_table[s[0], s[1], a])

    # --------------------------------------------------------
    def get_path(self):
        """利用学到的Q-table从起点规划一条路径 (贪心)"""
        path = [self.env.start]
        state = self.env.start
        visited = set()
        dirs = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        g = self.env.grid_size
        max_steps = g * g

        for _ in range(max_steps):
            if state == self.env.goal:
                break
            visited.add(state)
            q_vals = self.q_table[state[0], state[1], :]
            # 按Q值从高到低尝试动作, 避免回环
            sorted_actions = np.argsort(q_vals)[::-1]
            moved = False
            for a in sorted_actions:
                np_ = (state[0] + dirs[a][0], state[1] + dirs[a][1])
                if (0 <= np_[0] < g and 0 <= np_[1] < g
                        and np_ not in self.env.static_obstacles
                        and (np_ not in visited or np_ == self.env.goal)):
                    state = np_
                    path.append(state)
                    moved = True
                    break
            if not moved:
                break
        return path


# ============================================================
# 3. 四种算法实现 (论文 Algorithm 1~4)
# ============================================================

class Alg1_OriginalQL_Static(QLearningBase):
    """
    Algorithm 1: 原始 Q-learning + ε-greedy, 仅静态障碍物。
    """

    def train(self, verbose=True):
        for ep in range(self.episodes):
            state = self.env.reset()
            gamma = self.get_gamma(ep)
            total_reward = 0
            steps = 0
            max_steps = self.env.grid_size * 40  # 足够大以允许长路径

            for _ in range(max_steps):
                action = self.choose_action_egreedy(state)    # ε-greedy
                next_state, reward, done = self.env.step(action)
                self.update_q(state, action, reward, next_state, gamma)
                total_reward += reward
                steps += 1
                state = next_state
                if done:
                    break

            self.experience_replay(gamma)
            self.episode_steps.append(steps)
            self.episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 500 == 0:
                print(f"  Alg1(QL+ε-greedy,静态) Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward}")


class Alg2_OriginalQL_Dynamic(QLearningBase):
    """
    Algorithm 2: 原始 Q-learning + ε-greedy, 静态+动态障碍物。
    论文: 每步跟踪动态障碍物位置, 若agent碰到动态障碍物则break。
    """

    def train(self, verbose=True):
        for ep in range(self.episodes):
            state = self.env.reset()
            gamma = self.get_gamma(ep)
            total_reward = 0
            steps = 0
            max_steps = self.env.grid_size * 40

            for _ in range(max_steps):
                action = self.choose_action_egreedy(state)
                next_state, reward, done = self.env.step(action)

                # 论文Algorithm 2 line 10-12: 与动态障碍物碰撞检查
                if (self.env.n_dynamic > 0
                        and next_state in self.env.dynamic_obstacles):
                    reward = -1
                    done = True

                self.update_q(state, action, reward, next_state, gamma)
                total_reward += reward
                steps += 1
                state = next_state

                if self.env.n_dynamic > 0:
                    self.env.move_dynamic_obstacles()

                if done:
                    break

            self.experience_replay(gamma)
            self.episode_steps.append(steps)
            self.episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 500 == 0:
                print(f"  Alg2(QL+ε-greedy,动态) Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward}")


class Alg3_ProposedQL_Static(QLearningBase):
    """
    Algorithm 3: 提出的 Q-learning + SDP策略, 仅静态障碍物。
    SDP: 每个状态下直接选距离终点最近的动作。
    """

    def train(self, verbose=True):
        for ep in range(self.episodes):
            state = self.env.reset()
            gamma = self.get_gamma(ep)
            total_reward = 0
            steps = 0
            max_steps = self.env.grid_size * 40

            for _ in range(max_steps):
                action = self.choose_action_sdp(state)      # SDP策略
                next_state, reward, done = self.env.step(action)
                self.update_q(state, action, reward, next_state, gamma)
                total_reward += reward
                steps += 1
                state = next_state
                if done:
                    break

            self.experience_replay(gamma)
            self.episode_steps.append(steps)
            self.episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 500 == 0:
                print(f"  Alg3(QL+SDP,静态)   Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward}")


class Alg4_ProposedQL_Dynamic(QLearningBase):
    """
    Algorithm 4: 提出的 Q-learning + SDP策略, 静态+动态障碍物。
    结合SDP动作选择和动态障碍物碰撞检测。
    """

    def train(self, verbose=True):
        for ep in range(self.episodes):
            state = self.env.reset()
            gamma = self.get_gamma(ep)
            total_reward = 0
            steps = 0
            max_steps = self.env.grid_size * 40

            for _ in range(max_steps):
                action = self.choose_action_sdp(state)      # SDP策略
                next_state, reward, done = self.env.step(action)

                # 论文Algorithm 4 line 12-14
                if (self.env.n_dynamic > 0
                        and next_state in self.env.dynamic_obstacles):
                    reward = -1
                    done = True

                self.update_q(state, action, reward, next_state, gamma)
                total_reward += reward
                steps += 1
                state = next_state

                if self.env.n_dynamic > 0:
                    self.env.move_dynamic_obstacles()

                if done:
                    break

            self.experience_replay(gamma)
            self.episode_steps.append(steps)
            self.episode_rewards.append(total_reward)

            if verbose and (ep + 1) % 500 == 0:
                print(f"  Alg4(QL+SDP,动态)   Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward}")


# ============================================================
# 4. A* 和 Dijkstra 对比 (论文 Section 4.7)
# ============================================================

def astar(env):
    """A* 算法 (只能处理静态障碍物)"""
    start, goal = env.start, env.goal
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
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
            tg = g_score[current] + 1
            if nb not in g_score or tg < g_score[nb]:
                g_score[nb] = tg
                f = tg + manhattan_dist(nb, goal)
                heapq.heappush(open_set, (f, nb))
                came_from[nb] = current
    return []


def dijkstra(env):
    """Dijkstra 算法 (只能处理静态障碍物)"""
    start, goal = env.start, env.goal
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    open_set = [(0, start)]
    came_from = {}
    dist = {start: 0}

    while open_set:
        d, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for mv in dirs:
            nb = (current[0] + mv[0], current[1] + mv[1])
            if not (0 <= nb[0] < env.grid_size and 0 <= nb[1] < env.grid_size):
                continue
            if nb in env.static_obstacles:
                continue
            nd = d + 1
            if nb not in dist or nd < dist[nb]:
                dist[nb] = nd
                heapq.heappush(open_set, (nd, nb))
                came_from[nb] = current
    return []


# ============================================================
# 5. 可视化
# ============================================================

def plot_env(ax, env, title="Environment"):
    """绘制环境地图: 白色=空地, 灰色=静态障碍, 红色=动态障碍"""
    g = env.grid_size
    grid = np.zeros((g, g))
    for pos in env.static_obstacles:
        grid[pos] = 1
    for pos in env.dynamic_obstacles:
        grid[pos] = 2

    cmap = ListedColormap(['white', 'dimgray', 'red'])
    ax.imshow(grid.T, origin='lower', cmap=cmap, vmin=-0.5, vmax=2.5)
    ax.plot(env.start[0], env.start[1], 'o', color='royalblue',
            markersize=10, markeredgecolor='black', label='Start (UAV)')
    ax.plot(env.goal[0], env.goal[1], 's', color='gold',
            markersize=10, markeredgecolor='black', label='Goal')
    ax.set_xlim(-0.5, g - 0.5)
    ax.set_ylim(-0.5, g - 0.5)
    ax.set_xticks(range(0, g, 5))
    ax.set_yticks(range(0, g, 5))
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7)


def plot_path_on(ax, path, color='lime', label='Path'):
    """在图上绘制路径"""
    if not path:
        return
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, color=color, linewidth=2, marker='.', markersize=3,
            label=label)


def plot_convergence(data, title, ylabel, save_path, window=100):
    """绘制收敛曲线 (原始+滑动平均)"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data, alpha=0.15, color='steelblue', linewidth=0.5)
    if len(data) >= window:
        smoothed = np.convolve(data, np.ones(window) / window, mode='valid')
        ax.plot(range(window - 1, len(data)), smoothed,
                color='darkblue', linewidth=2, label=f'Moving avg (w={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  保存: {save_path}")


# ============================================================
# 6. 主程序
# ============================================================

OUTDIR = 'D:\\QiangHuaXueXi\\main-2\\'


def main():
    print("=" * 65)
    print(" 复现: Q-learning-based UAV Path Planning")
    print("       with Dynamic Obstacle Avoidance")
    print(" Amala Sonny et al., Applied Soft Computing 147 (2023)")
    print("=" * 65)

    EPISODES = 5000
    GRID_SIZE = 25
    SEED = 42

    results = {}

    # ----- Alg1: 原始 QL + ε-greedy, 仅静态 -----
    print("\n>>> Algorithm 1: 原始 Q-learning + ε-greedy (仅静态障碍物)")
    t0 = time.time()
    env1 = UAVEnvironment(grid_size=GRID_SIZE, n_dynamic=0, seed=SEED)
    alg1 = Alg1_OriginalQL_Static(env1, episodes=EPISODES, seed=SEED)
    alg1.train()
    t1 = time.time()
    path1 = alg1.get_path()
    results['Alg1_OriginalQL_Static'] = {
        'time': t1 - t0,
        'train_steps': alg1.episode_steps,
        'train_rewards': alg1.episode_rewards,
        'path': path1,
        'path_len': len(path1),
        'success': len(path1) > 0 and path1[-1] == env1.goal,
    }
    print(f"  训练时间: {t1-t0:.2f}s  路径长度: {len(path1)}  "
          f"到达终点: {results['Alg1_OriginalQL_Static']['success']}")

    # ----- Alg2: 原始 QL + ε-greedy, 静态+2动态 -----
    print("\n>>> Algorithm 2: 原始 Q-learning + ε-greedy (静态+2动态障碍物)")
    t0 = time.time()
    env2 = UAVEnvironment(grid_size=GRID_SIZE, n_dynamic=2, seed=SEED)
    alg2 = Alg2_OriginalQL_Dynamic(env2, episodes=EPISODES, seed=SEED)
    alg2.train()
    t1 = time.time()
    path2 = alg2.get_path()
    results['Alg2_OriginalQL_Dynamic'] = {
        'time': t1 - t0,
        'train_steps': alg2.episode_steps,
        'train_rewards': alg2.episode_rewards,
        'path': path2,
        'path_len': len(path2),
        'success': len(path2) > 0 and path2[-1] == env2.goal,
    }
    print(f"  训练时间: {t1-t0:.2f}s  路径长度: {len(path2)}  "
          f"到达终点: {results['Alg2_OriginalQL_Dynamic']['success']}")

    # ----- Alg3: 提出的 QL + SDP, 仅静态 -----
    print("\n>>> Algorithm 3: 提出的 Q-learning + SDP (仅静态障碍物)")
    t0 = time.time()
    env3 = UAVEnvironment(grid_size=GRID_SIZE, n_dynamic=0, seed=SEED)
    alg3 = Alg3_ProposedQL_Static(env3, episodes=EPISODES, seed=SEED)
    alg3.train()
    t1 = time.time()
    path3 = alg3.get_path()
    results['Alg3_ProposedQL_Static'] = {
        'time': t1 - t0,
        'train_steps': alg3.episode_steps,
        'train_rewards': alg3.episode_rewards,
        'path': path3,
        'path_len': len(path3),
        'success': len(path3) > 0 and path3[-1] == env3.goal,
    }
    print(f"  训练时间: {t1-t0:.2f}s  路径长度: {len(path3)}  "
          f"到达终点: {results['Alg3_ProposedQL_Static']['success']}")

    # ----- Alg4: 提出的 QL + SDP, 静态+2动态 -----
    print("\n>>> Algorithm 4: 提出的 Q-learning + SDP (静态+2动态障碍物)")
    t0 = time.time()
    env4 = UAVEnvironment(grid_size=GRID_SIZE, n_dynamic=2, seed=SEED)
    alg4 = Alg4_ProposedQL_Dynamic(env4, episodes=EPISODES, seed=SEED)
    alg4.train()
    t1 = time.time()
    path4 = alg4.get_path()
    results['Alg4_ProposedQL_Dynamic'] = {
        'time': t1 - t0,
        'train_steps': alg4.episode_steps,
        'train_rewards': alg4.episode_rewards,
        'path': path4,
        'path_len': len(path4),
        'success': len(path4) > 0 and path4[-1] == env4.goal,
    }
    print(f"  训练时间: {t1-t0:.2f}s  路径长度: {len(path4)}  "
          f"到达终点: {results['Alg4_ProposedQL_Dynamic']['success']}")

    # ----- A* & Dijkstra -----
    print("\n>>> 对比算法: A* 和 Dijkstra (仅静态障碍物)")
    env_static = UAVEnvironment(grid_size=GRID_SIZE, n_dynamic=0, seed=SEED)

    t0 = time.time()
    path_astar = astar(env_static)
    t_a = time.time() - t0

    t0 = time.time()
    path_dij = dijkstra(env_static)
    t_d = time.time() - t0

    results['Astar'] = {
        'time': t_a, 'path': path_astar, 'path_len': len(path_astar),
        'success': len(path_astar) > 0,
    }
    results['Dijkstra'] = {
        'time': t_d, 'path': path_dij, 'path_len': len(path_dij),
        'success': len(path_dij) > 0,
    }
    print(f"  A*:       {t_a:.4f}s  路径长度: {len(path_astar)}")
    print(f"  Dijkstra: {t_d:.4f}s  路径长度: {len(path_dij)}")

    # ----- 汇总 (对应论文 Table 2) -----
    print("\n" + "=" * 70)
    print(" 性能汇总 (对照论文 Table 2)")
    print("=" * 70)
    print(f" {'算法':<30} {'训练时间':>10} {'路径长度':>8} {'到达终点':>8}")
    print("-" * 60)
    for name in ['Alg1_OriginalQL_Static', 'Alg2_OriginalQL_Dynamic',
                  'Alg3_ProposedQL_Static', 'Alg4_ProposedQL_Dynamic',
                  'Astar', 'Dijkstra']:
        r = results[name]
        print(f" {name:<30} {r['time']:>8.2f}s {r['path_len']:>8}  "
              f"{'Yes' if r['success'] else 'No':>8}")

    # ----- 统计 Alg1 成功到达终点的轮次 -----
    alg1_success_eps = [i for i, r in enumerate(results['Alg1_OriginalQL_Static']['train_rewards']) if r > 0]
    print(f"\n Alg1 (原始QL) 在 {EPISODES} 轮训练中成功到达终点 {len(alg1_success_eps)} 次")

    # ----- 不同动态障碍物数量的 SDP 性能 -----
    print("\n>>> SDP 算法在不同动态障碍物数量下的性能 (论文 Section 4.6):")
    dyn_results = []
    for n_dyn in [0, 2, 4]:
        t0 = time.time()
        env_tmp = UAVEnvironment(grid_size=GRID_SIZE, n_dynamic=n_dyn, seed=SEED)
        alg_tmp = Alg4_ProposedQL_Dynamic(env_tmp, episodes=EPISODES, seed=SEED)
        alg_tmp.train(verbose=False)
        t1 = time.time()
        p = alg_tmp.get_path()
        ok = len(p) > 0 and p[-1] == env_tmp.goal
        print(f"  SDP + {n_dyn}动态障碍物:  {t1-t0:.2f}s  "
              f"路径长度={len(p)}  到达终点={'Yes' if ok else 'No'}")
        dyn_results.append((n_dyn, env_tmp, p, ok))

    # ================================================================
    # 生成图表
    # ================================================================
    print("\n>>> 生成可视化图表...")

    # --- Fig.1: 六种算法的路径对比 (对应论文 Fig.4 & Fig.5) ---
    fig1, axes = plt.subplots(2, 3, figsize=(18, 13))

    env_plot = UAVEnvironment(grid_size=GRID_SIZE, n_dynamic=0, seed=SEED)

    plot_env(axes[0, 0], env_plot, "A*  (Static Only)")
    plot_path_on(axes[0, 0], path_astar, color='orange', label='A* Path')

    plot_env(axes[0, 1], env_plot, "Dijkstra  (Static Only)")
    plot_path_on(axes[0, 1], path_dij, color='cyan', label='Dijkstra Path')

    plot_env(axes[0, 2], env1, "Alg1: Original QL + ε-greedy  (Static)")
    plot_path_on(axes[0, 2], path1, color='lime', label='QL Path')

    plot_env(axes[1, 0], env2, "Alg2: Original QL + ε-greedy  (Static + 2 Dyn)")
    plot_path_on(axes[1, 0], path2, color='red', label='QL Path')

    plot_env(axes[1, 1], env3, "Alg3: Proposed SDP-QL  (Static)")
    plot_path_on(axes[1, 1], path3, color='lime', label='SDP-QL Path')

    plot_env(axes[1, 2], env4, "Alg4: Proposed SDP-QL  (Static + 2 Dyn)")
    plot_path_on(axes[1, 2], path4, color='lime', label='SDP-QL Path')

    fig1.suptitle("Path Planning Comparison (Paper Fig.4 & Fig.5)",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig1.savefig(OUTDIR + 'path_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  保存: path_comparison.png")

    # --- Fig.2: 四种QL算法的收敛曲线 (对应论文 Fig.6~Fig.10) ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    names_conv = [
        ('Alg1: Original QL (Static)', results['Alg1_OriginalQL_Static']['train_steps']),
        ('Alg2: Original QL (Static+2Dyn)', results['Alg2_OriginalQL_Dynamic']['train_steps']),
        ('Alg3: Proposed SDP-QL (Static)', results['Alg3_ProposedQL_Static']['train_steps']),
        ('Alg4: Proposed SDP-QL (Static+2Dyn)', results['Alg4_ProposedQL_Dynamic']['train_steps']),
    ]
    w = max(1, EPISODES // 50)
    for idx, (title, data) in enumerate(names_conv):
        ax = axes2.flat[idx]
        ax.plot(data, alpha=0.15, color='steelblue', linewidth=0.5)
        if len(data) >= w:
            sm = np.convolve(data, np.ones(w) / w, mode='valid')
            ax.plot(range(w - 1, len(data)), sm, color='darkblue',
                    linewidth=2, label=f'MA({w})')
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps per Episode')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig2.suptitle("Convergence Comparison (Paper Fig.6~Fig.10)",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig2.savefig(OUTDIR + 'convergence_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  保存: convergence_comparison.png")

    # --- Fig.3: 不同数量动态障碍物下SDP算法表现 (论文 Fig.9 & Fig.10) ---
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (n_dyn, env_d, path_d, ok_d) in enumerate(dyn_results):
        plot_env(axes3[idx], env_d,
                 f"SDP-QL: {n_dyn} Dynamic Obstacle(s)  (len={len(path_d)})")
        plot_path_on(axes3[idx], path_d, color='lime', label='SDP-QL Path')

    fig3.suptitle("SDP-QL with Different Numbers of Dynamic Obstacles (Paper Fig.9 & Fig.10)",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig3.savefig(OUTDIR + 'dynamic_obstacle_comparison.png', dpi=150,
                 bbox_inches='tight')
    plt.close(fig3)
    print("  保存: dynamic_obstacle_comparison.png")

    print("\n完成! 图表已保存到 main-2 文件夹。")


if __name__ == '__main__':
    main()
