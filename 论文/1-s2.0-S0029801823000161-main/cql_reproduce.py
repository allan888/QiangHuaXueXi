"""
复现: 基于动态快速Q-learning的无人水面艇路径规划
    A path planning approach for unmanned surface vehicles
    based on dynamic and fast Q-learning
    Hao B., Du H., Yan Z.  Ocean Engineering 270 (2023) 113632

实现算法: CQL (Classical Q-learning) + epsilon-greedy 策略
    - 8方向动作空间 (Moore 邻域)
    - 静态奖励函数 (论文公式14)
    - 30x20 栅格环境 (对应论文 M01-M06 地图)

参数 (论文Table 1 / Section 4.3):
    - alpha (学习率): 0.9
    - gamma (折扣率): 0.9
    - epsilon (探索率): 0.5 初始, 每轮衰减去 0.05
"""

import os
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ============================================================
# 常量: 8方向动作定义 (论文 Fig.2 / Section 3.2)
# ============================================================
ACTION_DELTAS = {
    0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1),
    4: (0, -1), 5: (-1, -1), 6: (-1, 0), 7: (-1, 1),
}
N_ACTIONS = 8

REWARD_GOAL = 10.0
REWARD_OBSTACLE = -10.0
REWARD_MOVE = 0.0

ALPHA = 0.9
GAMMA = 0.9
EPSILON_START = 0.5
EPSILON_END = 0.05


# ============================================================
# 1. 环境类: USV栅格环境 (论文 Section 2.1 / Section 4.2)
# ============================================================

class GridEnvironment:
    """30x20 栅格海洋环境"""
    def __init__(self, grid, start, goal):
        self.grid = grid.copy()
        self.rows, self.cols = grid.shape
        self.start = start
        self.goal = goal
        self.agent_pos = start

    def is_valid(self, pos):
        x, y = pos
        return (0 <= x < self.cols and 0 <= y < self.rows
                and self.grid[y, x] == 0)

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action):
        """执行一步动作, 返回 (next_state, reward, done)"""
        dx, dy = ACTION_DELTAS[action]
        nx, ny = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        new_pos = (nx, ny)

        if new_pos == self.goal:
            self.agent_pos = new_pos
            return new_pos, REWARD_GOAL, True

        if not self.is_valid(new_pos):
            return self.agent_pos, REWARD_OBSTACLE, False

        self.agent_pos = new_pos
        return new_pos, REWARD_MOVE, False


# ============================================================
# 2. 地图生成 (对应论文 Fig.5 的 M01-M06)
# ============================================================

def build_map_m01():
    """M01: 简单稀疏障碍物 (群岛场景)"""
    rows, cols = 20, 30
    grid = np.zeros((rows, cols), dtype=int)
    grid[5:8, 3:6] = 1
    grid[6:9, 9:12] = 1
    grid[5:8, 15:18] = 1
    grid[8:11, 20:23] = 1
    grid[14:17, 5:8] = 1
    grid[14:17, 12:14] = 1
    grid[13:16, 24:27] = 1
    return grid, (1, 1), (28, 18)


def build_map_m02():
    """M02: 中等复杂度"""
    rows, cols = 20, 30
    grid = np.zeros((rows, cols), dtype=int)
    grid[2:8, 4:6] = 1
    grid[6:8, 8:16] = 1
    grid[6:8, 18:22] = 1
    grid[9:14, 14:16] = 1
    grid[13:15, 2:10] = 1
    grid[13:15, 18:26] = 1
    grid[16:19, 22:26] = 1
    return grid, (1, 1), (28, 18)


def build_map_m03():
    """M03: 较复杂"""
    rows, cols = 20, 30
    grid = np.zeros((rows, cols), dtype=int)
    grid[2:5, 3:6] = 1
    grid[3:6, 10:13] = 1
    grid[2:5, 17:19] = 1
    grid[3:7, 24:27] = 1
    grid[8:10, 4:8] = 1
    grid[8:10, 12:20] = 1
    grid[8:10, 23:26] = 1
    grid[13:16, 2:5] = 1
    grid[12:15, 8:10] = 1
    grid[13:16, 15:18] = 1
    grid[12:15, 22:24] = 1
    grid[14:17, 27:29] = 1
    return grid, (1, 1), (28, 18)


def build_map_m04():
    """M04: 复杂狭道 (港口场景)"""
    rows, cols = 20, 30
    grid = np.zeros((rows, cols), dtype=int)
    grid[2:6, 3:7] = 1
    grid[3:7, 12:15] = 1
    grid[2:6, 20:23] = 1
    grid[7:9, 0:9] = 1
    grid[7:9, 12:16] = 1
    grid[7:9, 20:30] = 1
    grid[11:15, 4:9] = 1
    grid[12:16, 14:18] = 1
    grid[11:15, 22:25] = 1
    grid[16:19, 2:6] = 1
    grid[16:19, 10:14] = 1
    grid[16:18, 27:29] = 1
    return grid, (1, 1), (28, 18)


def build_map_m05():
    """M05: 高密度障碍"""
    rows, cols = 20, 30
    grid = np.zeros((rows, cols), dtype=int)
    positions = [
        (2, 3, 2, 2), (2, 15, 2, 2), (3, 9, 2, 3), (2, 22, 3, 3),
        (3, 28, 2, 2), (6, 2, 3, 3), (6, 11, 2, 3), (5, 18, 3, 2),
        (6, 25, 2, 3), (9, 5, 3, 3), (10, 14, 2, 3), (9, 22, 3, 2),
        (9, 28, 2, 2), (13, 2, 3, 2), (12, 16, 3, 3), (13, 10, 2, 2),
        (12, 24, 2, 3), (16, 4, 2, 3), (15, 14, 3, 3), (16, 21, 2, 3),
        (16, 28, 2, 2),
    ]
    for y, x, h, w in positions:
        grid[y:y+h, x:x+w] = 1
    return grid, (1, 1), (28, 18)


def build_map_m06():
    """M06: 迷宫型"""
    rows, cols = 20, 30
    grid = np.zeros((rows, cols), dtype=int)
    grid[2:6, 3:5] = 1
    grid[8:12, 2:5] = 1
    grid[2:4, 10:20] = 1
    grid[2:5, 25:28] = 1
    grid[6:9, 6:10] = 1
    grid[7:10, 16:18] = 1
    grid[10:12, 10:20] = 1
    grid[5:10, 22:24] = 1
    grid[12:16, 22:24] = 1
    grid[14:17, 2:6] = 1
    grid[14:17, 9:12] = 1
    grid[15:17, 16:27] = 1
    grid[12:14, 26:29] = 1
    return grid, (1, 1), (28, 18)


MAP_BUILDERS = {
    'M01': build_map_m01, 'M02': build_map_m02, 'M03': build_map_m03,
    'M04': build_map_m04, 'M05': build_map_m05, 'M06': build_map_m06,
}


# ============================================================
# 3. CQL (Classical Q-learning) 核心算法
# ============================================================

class ClassicalQLearning:
    """经典 Q-learning + epsilon-greedy (论文中的CQL)"""
    def __init__(self, env, alpha=0.9, gamma=0.9,
                 eps_start=0.5, eps_end=0.05,
                 max_episodes=2000, max_steps=3000, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.epsilon = eps_start
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps
        self.rng = np.random.RandomState(seed)

        self.q_table = np.zeros((env.rows, env.cols, N_ACTIONS))
        self.episode_steps = []
        self.episode_rewards = []
        self.episode_success = []

    def _decay_epsilon(self, ep):
        """epsilon 从 eps_start 线性衰减到 eps_end"""
        progress = min(ep / max(1, self.max_episodes - 1), 1.0)
        self.epsilon = self.eps_start + (self.eps_end - self.eps_start) * progress

    def update_q(self, state, action, reward, next_state):
        """Q(s,a) <- (1-alpha)*Q(s,a) + alpha*[r + gamma*max_a Q(s',a')]"""
        s, ns = state, next_state
        max_next_q = np.max(self.q_table[ns[1], ns[0], :])
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - self.q_table[s[1], s[0], action]
        self.q_table[s[1], s[0], action] += self.alpha * td_error

    def choose_action(self, state):
        """epsilon-greedy 动作选择"""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, N_ACTIONS)
        else:
            q_vals = self.q_table[state[1], state[0], :]
            max_q = np.max(q_vals)
            best = np.where(q_vals == max_q)[0]
            return self.rng.choice(best)

    def train(self, verbose=True):
        """训练主循环"""
        for ep in range(self.max_episodes):
            self._decay_epsilon(ep)
            state = self.env.reset()
            total_reward = 0.0
            steps = 0

            for _ in range(self.max_steps_per_episode):
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
            self.episode_success.append(done and total_reward > 0)

            if verbose and (ep + 1) % 500 == 0:
                recent_ok = sum(self.episode_success[-100:])
                print(f"  Ep {ep+1:5d}/{self.max_episodes}  "
                      f"eps={self.epsilon:.3f}  steps={steps:5d}  "
                      f"reward={total_reward:.1f}  "
                      f"recent_success={recent_ok}/100")

    def get_path(self):
        """利用Q表从起点贪心提取路径"""
        env = self.env
        path = [env.start]
        state = env.start
        visited = set([state])
        max_steps = env.rows * env.cols * 4

        for _ in range(max_steps):
            if state == env.goal:
                return path, True

            q_vals = self.q_table[state[1], state[0], :]
            sorted_actions = np.argsort(q_vals)[::-1]

            moved = False
            for a in sorted_actions:
                dx, dy = ACTION_DELTAS[a]
                np_ = (state[0] + dx, state[1] + dy)
                if env.is_valid(np_) and np_ not in visited:
                    state = np_
                    visited.add(state)
                    path.append(state)
                    moved = True
                    break
            if not moved:
                break

        return path, (path[-1] == env.goal if path else False)


# ============================================================
# 4. 评价函数 (论文 Section 4.1)
# ============================================================

def compute_path_length(path):
    """路径总长度 (欧几里得距离之和) - 论文公式(20)"""
    if len(path) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path) - 1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        length += np.sqrt(dx**2 + dy**2)
    return length


def compute_turning_angle(path):
    """总转弯角度 (弧度) - 论文公式(21)-(24)"""
    if len(path) < 3:
        return 0.0
    total_angle = 0.0
    for i in range(1, len(path) - 1):
        xp, yp = path[i-1]; xc, yc = path[i]; xn, yn = path[i+1]
        a = np.sqrt((xc - xp)**2 + (yc - yp)**2)
        b = np.sqrt((xn - xc)**2 + (yn - yc)**2)
        c = np.sqrt((xn - xp)**2 + (yn - yp)**2)
        if a > 1e-9 and b > 1e-9:
            cos_val = np.clip((a**2 + b**2 - c**2) / (2*a*b), -1.0, 1.0)
            total_angle += np.pi - np.arccos(cos_val)
    return total_angle


# ============================================================
# 5. 可视化
# ============================================================

def plot_environment(ax, env, title="Environment"):
    """绘制栅格环境"""
    cmap = ListedColormap(['white', 'black'])
    ax.imshow(env.grid, origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax.plot(env.start[0], env.start[1], 'ro', markersize=8,
            markeredgecolor='black', markeredgewidth=1, label='Start')
    ax.plot(env.goal[0], env.goal[1], 'g*', markersize=12,
            markeredgecolor='black', markeredgewidth=1, label='Goal')
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_xticks(range(0, env.cols, 5))
    ax.set_yticks(range(0, env.rows, 5))
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)


def plot_path(ax, path, color='lime', label='Path', linewidth=2):
    """在环境图上叠加路径"""
    if not path:
        return
    px = [p[0] for p in path]; py = [p[1] for p in path]
    ax.plot(px, py, color=color, linewidth=linewidth,
            marker='.', markersize=4, label=label)


def plot_convergence(episode_data, title, ylabel, filepath, window=50):
    """绘制收敛曲线"""
    fig, ax = plt.subplots(figsize=(8, 4))
    data = np.array(episode_data)
    ax.plot(data, alpha=0.15, color='steelblue', linewidth=0.5)
    if len(data) >= window:
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(data)), smoothed,
                color='darkblue', linewidth=2,
                label=f'Moving Avg (w={window})')
    ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 6. 主程序
# ============================================================

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')


def main():
    print("=" * 65)
    print(" CQL (Classical Q-learning) + epsilon-greedy 复现")
    print(" 论文: DFQL - Ocean Engineering 270 (2023) 113632")
    print("=" * 65)

    os.makedirs(OUTDIR, exist_ok=True)

    SEED = 42
    EPISODES = 2000
    N_REPEAT = 30

    summary = {}

    for map_name, builder in MAP_BUILDERS.items():
        print(f"\n{'='*50}")
        print(f"  地图 {map_name}")
        print(f"{'='*50}")

        grid, start, goal = builder()
        path_lengths, angles, times = [], [], []

        for run_i in range(N_REPEAT):
            seed = SEED + run_i
            env_test = GridEnvironment(grid, start, goal)
            cql = ClassicalQLearning(
                env_test, alpha=ALPHA, gamma=GAMMA,
                eps_start=EPSILON_START, eps_end=EPSILON_END,
                max_episodes=EPISODES, seed=seed,
            )
            t0 = time.time()
            cql.train(verbose=False)
            t1 = time.time()
            path, success = cql.get_path()
            elapsed = t1 - t0

            if success and len(path) >= 2:
                path_lengths.append(compute_path_length(path))
                angles.append(compute_turning_angle(path))
            times.append(elapsed)

        mean_len = np.mean(path_lengths) if path_lengths else float('nan')
        std_len = np.std(path_lengths) if path_lengths else float('nan')
        mean_angle = np.mean(angles) if angles else float('nan')
        std_angle = np.std(angles) if angles else float('nan')
        mean_time = np.mean(times)
        std_time = np.std(times)
        success_rate = len(path_lengths) / N_REPEAT * 100

        summary[map_name] = {
            'path_length_mean': mean_len, 'path_length_std': std_len,
            'angle_mean': mean_angle, 'angle_std': std_angle,
            'time_mean': mean_time, 'time_std': std_time,
            'success_rate': success_rate,
        }

        print(f"  成功率: {success_rate:.0f}%")
        print(f"  路径长度: {mean_len:.2f} +- {std_len:.2f}")
        print(f"  转弯角度: {mean_angle:.3f} +- {std_angle:.3f} rad")
        print(f"  计算时间: {mean_time:.2f} +- {std_time:.2f} s")

    print(f"\n{'='*80}")
    print("  汇总结果 (30次重复实验)")
    print(f"{'='*80}")
    print(f" {'Map':<6} {'Success':>8} {'PathLen':>10} {'Std':>8} "
          f"{'Angle':>8} {'Std':>8} {'Time(s)':>8} {'Std':>8}")
    print("-" * 80)
    for mn in ['M01','M02','M03','M04','M05','M06']:
        r = summary[mn]
        print(f" {mn:<6} {r['success_rate']:>7.0f}% "
              f"{r['path_length_mean']:>10.2f} {r['path_length_std']:>8.2f} "
              f"{r['angle_mean']:>8.3f} {r['angle_std']:>8.3f} "
              f"{r['time_mean']:>8.2f} {r['time_std']:>8.2f}")

    print(f"\n>>> 生成路径可视化图表...")
    for map_name, builder in MAP_BUILDERS.items():
        grid, start, goal = builder()
        env_viz = GridEnvironment(grid, start, goal)
        cql_viz = ClassicalQLearning(
            env_viz, alpha=ALPHA, gamma=GAMMA,
            eps_start=EPSILON_START, eps_end=EPSILON_END,
            max_episodes=EPISODES, seed=SEED,
        )
        cql_viz.train(verbose=False)
        path, success = cql_viz.get_path()

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_environment(ax, env_viz, f"CQL: {map_name} (epsilon-greedy)")
        plot_path(ax, path, color='cyan', label='CQL Path')
        ax.set_title(f"CQL Path Planning - {map_name}", fontsize=13, fontweight='bold')
        fig.tight_layout()
        out_path = os.path.join(OUTDIR, f'CQL_{map_name}_path.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {out_path}   success={success}  len={len(path)}")

    print(f"\n>>> 生成收敛曲线...")
    for map_name, builder in MAP_BUILDERS.items():
        grid, start, goal = builder()
        env_conv = GridEnvironment(grid, start, goal)
        cql_conv = ClassicalQLearning(
            env_conv, alpha=ALPHA, gamma=GAMMA,
            eps_start=EPSILON_START, eps_end=EPSILON_END,
            max_episodes=EPISODES, seed=SEED,
        )
        cql_conv.train(verbose=False)

        out_path = os.path.join(OUTDIR, f'CQL_{map_name}_convergence.png')
        plot_convergence(
            cql_conv.episode_steps,
            f"CQL Convergence - {map_name}",
            'Steps per Episode', out_path, window=50,
        )
        print(f"  保存: {out_path}")

    print(f"\n完成! 所有结果已保存至: {OUTDIR}")


if __name__ == '__main__':
    main()
