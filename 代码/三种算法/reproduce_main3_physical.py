"""
复现: Q-learning-based UAV Path Planning with Dynamic Obstacle Avoidance
物理模型扩展版: USV with physical dimensions and turning radius.

与原始论文的差异:
  1. 世界尺寸: 1000m x 1000m (100x100栅格, 每格10m)
  2. USV和障碍物均具有物理尺寸 (长 x 宽矩形)
  3. USV具有回转半径, 转弯路径为圆弧
  4. 回转半径在 [Rmin, Rmax] 范围内由模型自行选择避免碰撞
  5. 碰撞检测使用OBB的分离轴定理 (SAT)
  6. 航向离散为8个方向, 动作: 直行/左转/右转

参数 (保留论文Table 1精神):
  - episodes: 5000
  - alpha: 0.3
  - gamma: 0.1递增到0.9
  - epsilon: 0.9
  - 经验回放缓冲区: 5000
"""

import os
import numpy as np
import time
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch

# ============================================================
# 0. 物理配置常量
# ============================================================

WORLD_SIZE = 1000.0
CELL_SIZE = 10.0
GRID_SIZE = int(WORLD_SIZE / CELL_SIZE)  # 100

USV_LENGTH = 25.0
USV_WIDTH = 8.0
USV_MIN_RADIUS = 15.0
USV_MAX_RADIUS = 60.0

N_HEADINGS = 8
HEADING_ANGLES = np.array([i * 2 * np.pi / N_HEADINGS for i in range(N_HEADINGS)])
HEADING_DELTA = 2 * np.pi / N_HEADINGS

FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
N_ACTIONS = 3

DYNAMIC_OBS_LENGTH = 30.0
DYNAMIC_OBS_WIDTH = 15.0


# ============================================================
# 1. 几何工具函数
# ============================================================

def obb_corners(cx, cy, length, width, angle):
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    hl, hw = length / 2, width / 2
    local = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return local @ rot.T + np.array([cx, cy])


def _project(corners, axis):
    dots = corners @ axis
    return np.min(dots), np.max(dots)


def obb_overlap(c1, l1, w1, a1, c2, l2, w2, a2):
    corners1 = obb_corners(c1[0], c1[1], l1, w1, a1)
    corners2 = obb_corners(c2[0], c2[1], l2, w2, a2)
    for angle in [a1, a1 + np.pi / 2, a2, a2 + np.pi / 2]:
        axis = np.array([np.cos(angle), np.sin(angle)])
        min1, max1 = _project(corners1, axis)
        min2, max2 = _project(corners2, axis)
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return False
    return True


def point_in_world(x, y):
    return 0.0 <= x <= WORLD_SIZE and 0.0 <= y <= WORLD_SIZE


def grid_center(gx, gy):
    return gx * CELL_SIZE + CELL_SIZE / 2, gy * CELL_SIZE + CELL_SIZE / 2


def pos_to_grid(x, y):
    gx = int(np.clip(x / CELL_SIZE, 0, GRID_SIZE - 1))
    gy = int(np.clip(y / CELL_SIZE, 0, GRID_SIZE - 1))
    return gx, gy


def manhattan_dist_m(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


# ============================================================
# 2. 弧线计算
# ============================================================

def compute_turn_arc(x, y, heading_idx, turn_action, radius, n_points=12):
    theta = HEADING_ANGLES[heading_idx]
    delta = HEADING_DELTA

    if turn_action == TURN_RIGHT:
        cx = x + radius * np.sin(theta)
        cy = y - radius * np.cos(theta)
        alpha_start = theta + np.pi / 2
        alpha_end = alpha_start - delta
    else:
        cx = x - radius * np.sin(theta)
        cy = y + radius * np.cos(theta)
        alpha_start = theta - np.pi / 2
        alpha_end = alpha_start + delta

    alphas = np.linspace(alpha_start, alpha_end, n_points)
    xs = cx + radius * np.cos(alphas)
    ys = cy + radius * np.sin(alphas)
    return xs, ys


def compute_forward_line(x, y, heading_idx, n_points=5):
    theta = HEADING_ANGLES[heading_idx]
    dx, dy = np.cos(theta) * CELL_SIZE, np.sin(theta) * CELL_SIZE
    xs = np.linspace(x, x + dx, n_points)
    ys = np.linspace(y, y + dy, n_points)
    return xs, ys


# ============================================================
# 3. 障碍物模型
# ============================================================

class PhysicalObstacle:
    def __init__(self, cx, cy, length, width, angle=0.0, is_static=True):
        self.cx = cx
        self.cy = cy
        self.length = length
        self.width = width
        self.angle = angle
        self.is_static = is_static

    def collides_with(self, cx, cy, length, width, angle):
        return obb_overlap(
            (self.cx, self.cy), self.length, self.width, self.angle,
            (cx, cy), length, width, angle,
        )

    def move_by(self, dx, dy):
        self.cx += dx
        self.cy += dy


# ============================================================
# 4. USV 物理模型
# ============================================================

class PhysicalUSV:
    def __init__(self, x, y, heading_idx=1, length=USV_LENGTH, width=USV_WIDTH,
                 min_radius=USV_MIN_RADIUS, max_radius=USV_MAX_RADIUS):
        self.x = x
        self.y = y
        self.heading_idx = heading_idx % N_HEADINGS
        self.length = length
        self.width = width
        self.min_radius = min_radius
        self.max_radius = max_radius

    @property
    def heading_angle(self):
        return HEADING_ANGLES[self.heading_idx]

    def get_rect(self):
        return (self.x, self.y, self.length, self.width, self.heading_angle)

    @property
    def grid_pos(self):
        return pos_to_grid(self.x, self.y)

    def _check_swept_collision(self, xs, ys, obstacles, occ_grid=None):
        for i in range(len(xs)):
            cx, cy = xs[i], ys[i]
            if not point_in_world(cx, cy):
                return True
            gx, gy = pos_to_grid(cx, cy)
            if occ_grid is not None and not occ_grid[gx, gy]:
                for obs in obstacles:
                    if not obs.is_static and obs.collides_with(cx, cy, self.length, self.width, self.heading_angle):
                        return True
                continue
            for obs in obstacles:
                if obs.collides_with(cx, cy, self.length, self.width, self.heading_angle):
                    return True
        return False

    def _choose_turn_radius(self, turn_action, obstacles, occ_grid=None):
        candidates = np.linspace(self.max_radius, self.min_radius, 5)
        for R in candidates:
            xs, ys = compute_turn_arc(self.x, self.y, self.heading_idx, turn_action, R)
            if not self._check_swept_collision(xs, ys, obstacles, occ_grid):
                return R
        return None

    def move_forward(self, obstacles, occ_grid=None):
        xs, ys = compute_forward_line(self.x, self.y, self.heading_idx)
        if self._check_swept_collision(xs, ys, obstacles, occ_grid):
            return False
        self.x = xs[-1]
        self.y = ys[-1]
        return True

    def turn(self, turn_action, obstacles, occ_grid=None):
        radius = self._choose_turn_radius(turn_action, obstacles, occ_grid)
        if radius is None:
            return False
        xs, ys = compute_turn_arc(self.x, self.y, self.heading_idx, turn_action, radius)
        self.x = xs[-1]
        self.y = ys[-1]
        if turn_action == TURN_LEFT:
            self.heading_idx = (self.heading_idx + 1) % N_HEADINGS
        else:
            self.heading_idx = (self.heading_idx - 1) % N_HEADINGS
        return True

    def try_action(self, action, obstacles, occ_grid=None):
        if action == FORWARD:
            xs, ys = compute_forward_line(self.x, self.y, self.heading_idx)
            if self._check_swept_collision(xs, ys, obstacles, occ_grid):
                return None
            return (xs[-1], ys[-1], self.heading_idx)
        else:
            radius = self._choose_turn_radius(action, obstacles, occ_grid)
            if radius is None:
                return None
            xs, ys = compute_turn_arc(self.x, self.y, self.heading_idx, action, radius)
            nh = (self.heading_idx + 1) % N_HEADINGS if action == TURN_LEFT else (self.heading_idx - 1) % N_HEADINGS
            return (xs[-1], ys[-1], nh)

    def reset(self, x, y, heading_idx=1):
        self.x = x
        self.y = y
        self.heading_idx = heading_idx % N_HEADINGS


# ============================================================
# 5. 环境
# ============================================================

class USVEnvironment:

    def __init__(self, n_dynamic=0, seed=None):
        self.rng = np.random.RandomState(seed)
        self.n_dynamic = n_dynamic

        max_dim = max(USV_LENGTH, USV_WIDTH) / 2 + CELL_SIZE
        self.start = (max_dim, max_dim)
        self.goal = (WORLD_SIZE - max_dim, WORLD_SIZE - max_dim)
        self.start_heading = 1

        self.usv = PhysicalUSV(*self.start, heading_idx=self.start_heading)
        self.static_obstacles = self._build_static_obstacles()
        self._static_occ_grid = self._build_occ_grid()
        self.dynamic_obstacles = []
        if n_dynamic > 0:
            self._init_dynamic_obstacles()

    def _build_static_obstacles(self):
        obs = []

        obs.append(PhysicalObstacle(
            cx=250, cy=180, length=140, width=50, angle=0.0, is_static=True))

        obs.append(PhysicalObstacle(
            cx=500, cy=550, length=340, width=40, angle=0.0, is_static=True))

        obs.append(PhysicalObstacle(
            cx=750, cy=220, length=60, width=100, angle=0.0, is_static=True))
        obs.append(PhysicalObstacle(
            cx=800, cy=240, length=140, width=40, angle=0.0, is_static=True))

        obs.append(PhysicalObstacle(
            cx=450, cy=830, length=160, width=60, angle=0.0, is_static=True))

        return obs

    def _build_occ_grid(self):
        occ = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        for obs in self.static_obstacles:
            corners = obb_corners(obs.cx, obs.cy, obs.length, obs.width, obs.angle)
            gx_min = max(0, int(np.floor(np.min(corners[:, 0]) / CELL_SIZE)))
            gx_max = min(GRID_SIZE - 1, int(np.ceil(np.max(corners[:, 0]) / CELL_SIZE)))
            gy_min = max(0, int(np.floor(np.min(corners[:, 1]) / CELL_SIZE)))
            gy_max = min(GRID_SIZE - 1, int(np.ceil(np.max(corners[:, 1]) / CELL_SIZE)))
            for i in range(gx_min, gx_max + 1):
                for j in range(gy_min, gy_max + 1):
                    if occ[i, j]:
                        continue
                    gcx, gcy = grid_center(i, j)
                    if obb_overlap((obs.cx, obs.cy), obs.length, obs.width, obs.angle,
                                   (gcx, gcy), CELL_SIZE, CELL_SIZE, 0.0):
                        occ[i, j] = True
        return occ

    def _init_dynamic_obstacles(self):
        self.dynamic_obstacles = []
        for _ in range(self.n_dynamic):
            for _ in range(200):
                cx = self.rng.uniform(CELL_SIZE * 5, WORLD_SIZE - CELL_SIZE * 5)
                cy = self.rng.uniform(CELL_SIZE * 5, WORLD_SIZE - CELL_SIZE * 5)
                temp = PhysicalObstacle(cx, cy, DYNAMIC_OBS_LENGTH, DYNAMIC_OBS_WIDTH,
                                        angle=0.0, is_static=False)
                if not self._collides_any_static(temp.cx, temp.cy,
                                                 DYNAMIC_OBS_LENGTH, DYNAMIC_OBS_WIDTH, 0.0):
                    if not temp.collides_with(*self.usv.get_rect()):
                        d_goal = np.hypot(cx - self.goal[0], cy - self.goal[1])
                        if d_goal > CELL_SIZE * 6:
                            self.dynamic_obstacles.append(temp)
                            break

    def _collides_any_static(self, cx, cy, length, width, angle):
        for obs in self.static_obstacles:
            if obs.collides_with(cx, cy, length, width, angle):
                return True
        return False

    def _move_dynamic_obstacles(self):
        dirs = np.array([[CELL_SIZE, 0], [-CELL_SIZE, 0],
                         [0, CELL_SIZE], [0, -CELL_SIZE]])
        self.rng.shuffle(dirs)
        for dob in self.dynamic_obstacles:
            moved = False
            for d in dirs:
                nx, ny = dob.cx + d[0], dob.cy + d[1]
                if (CELL_SIZE * 2 < nx < WORLD_SIZE - CELL_SIZE * 2 and
                        CELL_SIZE * 2 < ny < WORLD_SIZE - CELL_SIZE * 2 and
                        not self._collides_any_static(nx, ny, dob.length, dob.width, dob.angle)):
                    dob.cx, dob.cy = nx, ny
                    moved = True
                    break
            if not moved:
                pass

    def get_usv_rect(self):
        return self.usv.get_rect()

    @property
    def state(self):
        gx, gy = self.usv.grid_pos
        return (gx, gy, self.usv.heading_idx)

    def is_at_goal(self):
        d = np.hypot(self.usv.x - self.goal[0], self.usv.y - self.goal[1])
        return d < CELL_SIZE * 0.8

    def reset(self):
        self.usv.reset(*self.start, heading_idx=self.start_heading)
        if self.n_dynamic > 0:
            self._init_dynamic_obstacles()
        return self.state

    def step(self, action):
        all_obs = self.static_obstacles + self.dynamic_obstacles
        occ = self._static_occ_grid

        if action == FORWARD:
            ok = self.usv.move_forward(all_obs, occ)
        elif action == TURN_LEFT:
            ok = self.usv.turn(TURN_LEFT, all_obs, occ)
        elif action == TURN_RIGHT:
            ok = self.usv.turn(TURN_RIGHT, all_obs, occ)
        else:
            ok = False

        if not ok:
            return self.state, -1, True

        if self.is_at_goal():
            return self.state, 1, True

        if not point_in_world(self.usv.x, self.usv.y):
            return self.state, -1, True

        if self.n_dynamic > 0:
            self._move_dynamic_obstacles()

        for dob in self.dynamic_obstacles:
            if dob.collides_with(*self.get_usv_rect()):
                return self.state, -1, True

        return self.state, 0, False


# ============================================================
# 6. Q-learning 核心
# ============================================================

class QLearningBase:
    def __init__(self, env, alpha=0.3, gamma_start=0.1, gamma_end=0.9,
                 epsilon=0.9, episodes=5000, step_penalty=0.0, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.epsilon = epsilon
        self.episodes = episodes
        self.step_penalty = step_penalty
        self.rng = np.random.RandomState(seed)

        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, N_HEADINGS, N_ACTIONS))
        self.replay_buffer = deque(maxlen=5000)

        self.episode_steps = []
        self.episode_rewards = []

    def get_gamma(self, episode):
        progress = min(episode / self.episodes, 1.0)
        return self.gamma_start + (self.gamma_end - self.gamma_start) * progress

    def choose_action_egreedy(self, state):
        gx, gy, h = state
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, N_ACTIONS)
        else:
            q_vals = self.q_table[gx, gy, h, :]
            max_q = np.max(q_vals)
            best = np.where(q_vals == max_q)[0]
            return self.rng.choice(best)

    def choose_action_sdp(self, state):
        gx, gy, h = state
        wx, wy = grid_center(gx, gy)

        temp_usv = PhysicalUSV(wx, wy, heading_idx=h)
        all_obs = (self.env.static_obstacles +
                   self.env.dynamic_obstacles if hasattr(self.env, 'dynamic_obstacles')
                   else self.env.static_obstacles)
        occ = (self.env._static_occ_grid if hasattr(self.env, '_static_occ_grid')
               else None)

        goal = self.env.goal
        distances = []
        for a in range(N_ACTIONS):
            result = temp_usv.try_action(a, all_obs, occ)
            if result is None:
                distances.append(float('inf'))
            else:
                nx, ny, nh = result
                d = manhattan_dist_m((nx, ny), goal)
                distances.append(d)

        min_d = min(distances)
        if min_d == float('inf'):
            return self.rng.randint(0, N_ACTIONS)
        best = [a for a, d in enumerate(distances) if d == min_d]
        return self.rng.choice(best)

    def update_q(self, state, action, reward, next_state, gamma):
        gx, gy, h = state
        ngx, ngy, nh = next_state
        td_target = reward + gamma * np.max(self.q_table[ngx, ngy, nh, :])
        self.q_table[gx, gy, h, action] += self.alpha * (td_target - self.q_table[gx, gy, h, action])
        self.replay_buffer.append((state, action, reward, next_state))

    def experience_replay(self, gamma, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return
        indices = self.rng.choice(len(self.replay_buffer), batch_size, replace=False)
        for idx in indices:
            s, a, r, ns = self.replay_buffer[idx]
            gx, gy, h = s
            ngx, ngy, nh = ns
            td_target = r + gamma * np.max(self.q_table[ngx, ngy, nh, :])
            self.q_table[gx, gy, h, a] += self.alpha * (td_target - self.q_table[gx, gy, h, a])

    def get_path(self, max_attempts=3):
        best_path = []
        for attempt in range(max_attempts):
            wx, wy = self.env.start
            h = self.env.start_heading
            env_copy = USVEnvironment(n_dynamic=self.env.n_dynamic,
                                      seed=42 + attempt)
            env_copy.usv.reset(wx, wy, h)
            path_points = [(wx, wy, h)]

            all_obs = env_copy.static_obstacles + env_copy.dynamic_obstacles
            occ = env_copy._static_occ_grid
            visited_states = set()
            max_steps = GRID_SIZE * 6

            for _ in range(max_steps):
                if env_copy.is_at_goal():
                    break

                gx, gy, h = env_copy.state
                state_key = (round(env_copy.usv.x), round(env_copy.usv.y), h)
                if state_key in visited_states:
                    break
                visited_states.add(state_key)

                q_vals = self.q_table[gx, gy, h, :].copy()
                for a in range(N_ACTIONS):
                    temp = PhysicalUSV(env_copy.usv.x, env_copy.usv.y, heading_idx=h)
                    result = temp.try_action(a, all_obs, occ)
                    if result is None:
                        q_vals[a] = -np.inf
                    else:
                        nx, ny, nh = result
                        if (round(nx), round(ny), nh) in visited_states:
                            q_vals[a] -= 10.0

                sorted_actions = np.argsort(q_vals)[::-1]
                moved = False
                for a in sorted_actions:
                    if q_vals[a] == -np.inf:
                        continue
                    ns, r, done = env_copy.step(a)
                    if r >= 0:
                        path_points.append((env_copy.usv.x, env_copy.usv.y,
                                            env_copy.usv.heading_idx))
                        moved = True
                        break

                if not moved:
                    break

            if env_copy.is_at_goal() or len(path_points) > len(best_path):
                best_path = path_points
            if best_path and env_copy.is_at_goal():
                break

        return best_path


# ============================================================
# 7. 算法实现
# ============================================================

class Alg2_OriginalQL_Dynamic(QLearningBase):

    def train(self, verbose=True):
        for ep in range(self.episodes):
            state = self.env.reset()
            gamma = self.get_gamma(ep)
            total_reward = 0
            steps = 0
            max_steps = GRID_SIZE * 5

            for _ in range(max_steps):
                action = self.choose_action_egreedy(state)
                next_state, reward, done = self.env.step(action)
                reward += self.step_penalty
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
                print(f"  Alg2(QL+eps,动态) Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward:.2f}")


class Alg4_ProposedQL_Dynamic(QLearningBase):

    def train(self, verbose=True):
        for ep in range(self.episodes):
            state = self.env.reset()
            gamma = self.get_gamma(ep)
            total_reward = 0
            steps = 0
            max_steps = GRID_SIZE * 5

            for _ in range(max_steps):
                action = self.choose_action_sdp(state)
                next_state, reward, done = self.env.step(action)
                reward += self.step_penalty
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
                print(f"  Alg4(SDP-QL,动态) Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward:.2f}")


# ============================================================
# 8. 可视化
# ============================================================

def draw_obb(ax, cx, cy, length, width, angle, color, alpha=0.7, lw=1):
    corners = obb_corners(cx, cy, length, width, angle)
    poly = Polygon(corners, closed=True, facecolor=color,
                   edgecolor='black', linewidth=lw, alpha=alpha)
    ax.add_patch(poly)


def plot_env(ax, env, title="Environment"):
    ax.set_xlim(-20, WORLD_SIZE + 20)
    ax.set_ylim(-20, WORLD_SIZE + 20)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, WORLD_SIZE + 1, 200))
    ax.set_yticks(np.arange(0, WORLD_SIZE + 1, 200))
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)

    for obs in env.static_obstacles:
        draw_obb(ax, obs.cx, obs.cy, obs.length, obs.width, obs.angle,
                 color='royalblue', alpha=0.7, lw=1.5)

    for dob in env.dynamic_obstacles:
        draw_obb(ax, dob.cx, dob.cy, dob.length, dob.width, dob.angle,
                 color='red', alpha=0.5, lw=1.5)

    wx, wy = env.start
    ax.plot(wx, wy, 'o', color='orange', markersize=8,
            markeredgecolor='black', zorder=5, label='Start')
    ax.plot(env.goal[0], env.goal[1], 's', color='yellow', markersize=8,
            markeredgecolor='black', zorder=5, label='Goal')


def plot_path_with_arcs(ax, path_points, color='lime', label='Path'):
    if not path_points or len(path_points) < 2:
        return

    xs = [p[0] for p in path_points]
    ys = [p[1] for p in path_points]
    ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8,
            marker='.', markersize=2, label=label + ' (center)')

    step = max(1, len(path_points) // 30)
    for i in range(0, len(path_points), step):
        x, y, h = path_points[i]
        draw_obb(ax, x, y, USV_LENGTH, USV_WIDTH, HEADING_ANGLES[h],
                 color=color, alpha=0.25, lw=0.5)

    if path_points:
        x0, y0, h0 = path_points[0]
        xf, yf, hf = path_points[-1]
        draw_obb(ax, x0, y0, USV_LENGTH, USV_WIDTH, HEADING_ANGLES[h0],
                 color='orange', alpha=0.8, lw=2)
        draw_obb(ax, xf, yf, USV_LENGTH, USV_WIDTH, HEADING_ANGLES[hf],
                 color='yellow', alpha=0.8, lw=2)


def plot_convergence(data, title, ylabel, save_path, window=100):
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
# 9. 主程序
# ============================================================

OUTDIR = 'D:\\QiangHuaXueXi\\Result\\'


def main():
    print("=" * 60)
    print(" USV Path Planning with Physical Model & Turning Radius")
    print(f" World: {WORLD_SIZE:.0f}m x {WORLD_SIZE:.0f}m")
    print(f" Grid:  {GRID_SIZE}x{GRID_SIZE} ({CELL_SIZE:.0f}m/cell)")
    print(f" USV:   {USV_LENGTH:.0f}m x {USV_WIDTH:.0f}m")
    print(f" Turn radius: {USV_MIN_RADIUS:.0f}m ~ {USV_MAX_RADIUS:.0f}m")
    print(" Alg2: QL+epsilon-greedy  vs  Alg4: SDP-QL")
    print("=" * 60)

    EPISODES = 2000
    SEED = 42

    results = {}

    print("\n>>> Algorithm 2: Q-learning + epsilon-greedy (static+2 dynamic)")
    t0 = time.time()
    env2 = USVEnvironment(n_dynamic=2, seed=SEED)
    alg2 = Alg2_OriginalQL_Dynamic(env2, episodes=EPISODES, epsilon=0.7,
                                   alpha=0.4, step_penalty=-0.001, seed=SEED)
    alg2.train()
    t1 = time.time()
    path2 = alg2.get_path()
    success2 = len(path2) > 1 and np.hypot(path2[-1][0] - env2.goal[0],
                                            path2[-1][1] - env2.goal[1]) < CELL_SIZE * 2
    results['Alg2_OriginalQL_Dynamic'] = {
        'time': t1 - t0,
        'train_steps': alg2.episode_steps,
        'train_rewards': alg2.episode_rewards,
        'path': path2,
        'path_len': len(path2),
        'success': success2,
    }

    alg2_success_count = sum(1 for r in alg2.episode_rewards if r > 0)
    print(f"  Train: {t1 - t0:.1f}s  Path pts: {len(path2)}  "
          f"GoalReached: {success2}  TrainSuccess: {alg2_success_count}/{EPISODES}")

    print("\n>>> Algorithm 4: SDP-QL (static+2 dynamic)")
    t0 = time.time()
    env4 = USVEnvironment(n_dynamic=2, seed=SEED)
    alg4 = Alg4_ProposedQL_Dynamic(env4, episodes=EPISODES, seed=SEED)
    alg4.train()
    t1 = time.time()
    path4 = alg4.get_path()
    success4 = len(path4) > 1 and np.hypot(path4[-1][0] - env4.goal[0],
                                            path4[-1][1] - env4.goal[1]) < CELL_SIZE * 2
    results['Alg4_ProposedQL_Dynamic'] = {
        'time': t1 - t0,
        'train_steps': alg4.episode_steps,
        'train_rewards': alg4.episode_rewards,
        'path': path4,
        'path_len': len(path4),
        'success': success4,
    }

    alg4_success_count = sum(1 for r in alg4.episode_rewards if r > 0)
    print(f"  Train: {t1 - t0:.1f}s  Path pts: {len(path4)}  "
          f"GoalReached: {success4}  TrainSuccess: {alg4_success_count}/{EPISODES}")

    print("\n" + "=" * 60)
    print(" Comparison")
    print("=" * 60)
    print(f" {'Algorithm':<30} {'Train(s)':>8} {'Steps':>8} {'Success':>8}")
    print("-" * 58)
    for name in ['Alg2_OriginalQL_Dynamic', 'Alg4_ProposedQL_Dynamic']:
        r = results[name]
        print(f" {name:<30} {r['time']:>8.1f} {r['path_len']:>8}  "
              f"{'Yes' if r['success'] else 'No':>8}")

    print("\n>>> Generating figures...")
    os.makedirs(OUTDIR, exist_ok=True)
    w = max(1, EPISODES // 50)

    fig1, ax1 = plt.subplots(figsize=(10, 9))
    plot_env(ax1, env2, "Alg2: QL+epsilon-greedy (Static + 2 Dynamic)")
    plot_path_with_arcs(ax1, path2, color='lime', label='Alg2 Path')
    ax1.set_title("Fig.1: Alg2 Path (Physical Model, Turning Radius)",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    fig1.savefig(OUTDIR + 'Fig1_Alg2_physical_path.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  Saved: Fig1_Alg2_physical_path.png")

    fig2, ax2 = plt.subplots(figsize=(10, 9))
    plot_env(ax2, env4, "Alg4: SDP-QL (Static + 2 Dynamic)")
    plot_path_with_arcs(ax2, path4, color='lime', label='Alg4 Path')
    ax2.set_title("Fig.2: Alg4 Path (Physical Model, Turning Radius)",
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    fig2.savefig(OUTDIR + 'Fig2_Alg4_physical_path.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("  Saved: Fig2_Alg4_physical_path.png")

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    d2 = results['Alg2_OriginalQL_Dynamic']['train_steps']
    ax3.plot(d2, alpha=0.15, color='steelblue', linewidth=0.5)
    if len(d2) >= w:
        sm = np.convolve(d2, np.ones(w) / w, mode='valid')
        ax3.plot(range(w - 1, len(d2)), sm, color='darkblue',
                 linewidth=2, label='Alg2 (eps-greedy)')
    d4 = results['Alg4_ProposedQL_Dynamic']['train_steps']
    ax3.plot(d4, alpha=0.15, color='darkgreen', linewidth=0.5)
    if len(d4) >= w:
        sm = np.convolve(d4, np.ones(w) / w, mode='valid')
        ax3.plot(range(w - 1, len(d4)), sm, color='darkgreen',
                 linewidth=2, label='Alg4 (SDP-QL)')
    ax3.set_title("Fig.3: Convergence Comparison (Physical Model)")
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Steps per Episode')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.savefig(OUTDIR + 'Fig3_physical_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("  Saved: Fig3_physical_convergence.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
