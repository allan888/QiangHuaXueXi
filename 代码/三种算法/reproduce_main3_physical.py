"""
复现: Q-learning-based UAV Path Planning with Dynamic Obstacle Avoidance
物理模型扩展版: USV with physical dimensions and turning radius.
扩展: 三种障碍物 + GIF轨迹动画

障碍物类型:
  1. 静态障碍物 (蓝色) - 固定位置, 有物理尺寸
  2. 巡逻障碍物 (橙色) - 一直移动, 在两个航点间来回巡逻, 有物理尺寸
  3. 伏击障碍物 (紫色) - 初始不可探测, 只有临近(探测距离内)才发现,
     必定出现在既定路线上, 有物理尺寸
"""

import os
import io
import numpy as np
import time
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

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

SAFETY_MARGIN = 10.0

N_HEADINGS = 8
HEADING_ANGLES = np.array([i * 2 * np.pi / N_HEADINGS for i in range(N_HEADINGS)])
HEADING_DELTA = 2 * np.pi / N_HEADINGS

FORWARD = 0
TURN_LEFT = 1
TURN_RIGHT = 2
N_ACTIONS = 3

PATROL_OBS_LENGTH = 20.0
PATROL_OBS_WIDTH = 10.0
PATROL_SPEED = 4.0

AMBUSH_OBS_LENGTH = 18.0
AMBUSH_OBS_WIDTH = 12.0
DETECTION_RANGE = 120.0

STATIC_COLOR = 'royalblue'
PATROL_COLOR = 'darkorange'
AMBUSH_COLOR = 'darkviolet'
AMBUSH_DETECTED_COLOR = 'magenta'


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


def compute_forward_line(x, y, heading_idx, n_points=15):
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


class PatrolObstacle(PhysicalObstacle):
    """巡逻障碍物: 在两个航点间来回移动, 始终可探测"""

    def __init__(self, cx, cy, length, width, angle,
                 waypoint_a, waypoint_b, speed=PATROL_SPEED):
        super().__init__(cx, cy, length, width, angle, is_static=False)
        self.waypoint_a = np.array(waypoint_a, dtype=float)
        self.waypoint_b = np.array(waypoint_b, dtype=float)
        self.speed = speed
        self.t = 0.5
        self.direction = 1

    def update(self):
        total_len = np.linalg.norm(self.waypoint_b - self.waypoint_a)
        if total_len < 1e-6:
            return
        dt = self.speed / total_len
        self.t += dt * self.direction
        if self.t > 1.0:
            self.t = 2.0 - self.t
            self.direction = -1
        elif self.t < 0.0:
            self.t = -self.t
            self.direction = 1
        self.t = np.clip(self.t, 0.0, 1.0)
        self.cx = self.waypoint_a[0] + self.t * (self.waypoint_b[0] - self.waypoint_a[0])
        self.cy = self.waypoint_a[1] + self.t * (self.waypoint_b[1] - self.waypoint_a[1])

    def copy(self):
        p = PatrolObstacle(self.cx, self.cy, self.length, self.width, self.angle,
                           self.waypoint_a.copy(), self.waypoint_b.copy(), self.speed)
        p.t = self.t
        p.direction = self.direction
        return p


class AmbushObstacle(PhysicalObstacle):
    """伏击障碍物: 初始不可探测, 只有USV进入探测范围才发现, 必定在既定路线上"""

    def __init__(self, cx, cy, length, width, angle=0.0, detection_range=DETECTION_RANGE):
        super().__init__(cx, cy, length, width, angle, is_static=False)
        self.detected = False
        self.detection_range = detection_range

    def check_detection(self, usv_x, usv_y):
        if not self.detected:
            dist = np.hypot(usv_x - self.cx, usv_y - self.cy)
            if dist < self.detection_range:
                self.detected = True
        return self.detected

    def copy(self):
        a = AmbushObstacle(self.cx, self.cy, self.length, self.width,
                           self.angle, self.detection_range)
        a.detected = self.detected
        return a


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
        safe_l = self.length + 2 * SAFETY_MARGIN
        safe_w = self.width + 2 * SAFETY_MARGIN
        for i in range(len(xs)):
            cx, cy = xs[i], ys[i]
            if not point_in_world(cx, cy):
                return True
            for obs in obstacles:
                if obs.collides_with(cx, cy, safe_l, safe_w, self.heading_angle):
                    return True
        return False

    def _choose_turn_radius(self, turn_action, obstacles, occ_grid=None):
        candidates = np.linspace(self.max_radius, self.min_radius, 8)
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

    def __init__(self, n_patrol=2, n_ambush=0, seed=None):
        self.rng = np.random.RandomState(seed)
        self.n_patrol = n_patrol
        self.n_ambush = n_ambush

        max_dim = max(USV_LENGTH, USV_WIDTH) / 2 + CELL_SIZE
        self.start = (max_dim, max_dim)
        self.goal = (WORLD_SIZE - max_dim, WORLD_SIZE - max_dim)
        self.start_heading = 1

        self.usv = PhysicalUSV(*self.start, heading_idx=self.start_heading)
        self.static_obstacles = self._build_static_obstacles()
        self._static_occ_grid = self._build_occ_grid()
        self.patrol_obstacles = []
        self.ambush_obstacles = []
        if n_patrol > 0:
            self._init_patrol_obstacles()
        if n_ambush > 0:
            self._init_ambush_obstacles()

    def _build_static_obstacles(self):
        obs = []
        obs.append(PhysicalObstacle(
            cx=220, cy=180, length=160, width=60, angle=0.0, is_static=True))
        obs.append(PhysicalObstacle(
            cx=600, cy=550, length=300, width=40, angle=0.0, is_static=True))
        obs.append(PhysicalObstacle(
            cx=800, cy=830, length=120, width=80, angle=0.0, is_static=True))
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

    def _init_patrol_obstacles(self):
        self.patrol_obstacles = []

        p1 = PatrolObstacle(
            cx=300, cy=400, length=PATROL_OBS_LENGTH, width=PATROL_OBS_WIDTH, angle=0.0,
            waypoint_a=(200, 400), waypoint_b=(450, 400), speed=PATROL_SPEED)
        self.patrol_obstacles.append(p1)

        p2 = PatrolObstacle(
            cx=800, cy=650, length=PATROL_OBS_LENGTH, width=PATROL_OBS_WIDTH, angle=0.0,
            waypoint_a=(800, 580), waypoint_b=(800, 780), speed=PATROL_SPEED)
        self.patrol_obstacles.append(p2)

    def _init_ambush_obstacles(self):
        self.ambush_obstacles = []
        for _ in range(self.n_ambush):
            for _ in range(200):
                cx = self.rng.uniform(CELL_SIZE * 8, WORLD_SIZE - CELL_SIZE * 8)
                cy = self.rng.uniform(CELL_SIZE * 8, WORLD_SIZE - CELL_SIZE * 8)
                if self._collides_any_static(cx, cy, AMBUSH_OBS_LENGTH, AMBUSH_OBS_WIDTH, 0.0):
                    continue
                if self._collides_any_patrol(cx, cy, AMBUSH_OBS_LENGTH, AMBUSH_OBS_WIDTH, 0.0):
                    continue
                temp = AmbushObstacle(cx, cy, AMBUSH_OBS_LENGTH, AMBUSH_OBS_WIDTH)
                if not temp.collides_with(*self.usv.get_rect()):
                    d_goal = np.hypot(cx - self.goal[0], cy - self.goal[1])
                    if d_goal > CELL_SIZE * 6:
                        self.ambush_obstacles.append(temp)
                        break

    def place_ambush_on_path(self, path_points):
        """在规划路上放置伏击障碍物, 路过时探测到并绕行"""
        if not path_points or len(path_points) < 3:
            return None
        mid_idx = min(len(path_points) * 2 // 3, len(path_points) - 1)
        x, y, h = path_points[mid_idx]

        for _ in range(100):
            ox = x + self.rng.uniform(-CELL_SIZE * 3, CELL_SIZE * 3)
            oy = y + self.rng.uniform(-CELL_SIZE * 3, CELL_SIZE * 3)
            if not point_in_world(ox, oy):
                continue
            if (self._collides_any_static(ox, oy, AMBUSH_OBS_LENGTH, AMBUSH_OBS_WIDTH, 0.0) or
                    self._collides_any_patrol(ox, oy, AMBUSH_OBS_LENGTH, AMBUSH_OBS_WIDTH, 0.0)):
                continue
            aob = AmbushObstacle(ox, oy, AMBUSH_OBS_LENGTH, AMBUSH_OBS_WIDTH)
            self.ambush_obstacles.append(aob)
            return aob
        aob = AmbushObstacle(x, y, AMBUSH_OBS_LENGTH, AMBUSH_OBS_WIDTH)
        self.ambush_obstacles.append(aob)
        return aob

    def _collides_any_static(self, cx, cy, length, width, angle):
        for obs in self.static_obstacles:
            if obs.collides_with(cx, cy, length, width, angle):
                return True
        return False

    def _collides_any_patrol(self, cx, cy, length, width, angle):
        for pob in self.patrol_obstacles:
            if pob.collides_with(cx, cy, length, width, angle):
                return True
        return False

    def _move_patrol_obstacles(self):
        for pob in self.patrol_obstacles:
            pob.update()

    def _check_detection(self):
        usx, usy = self.usv.x, self.usv.y
        for aob in self.ambush_obstacles:
            aob.check_detection(usx, usy)

    def _check_detection_with_pos(self, x, y):
        for aob in self.ambush_obstacles:
            aob.check_detection(x, y)

    def get_visible_obstacles(self):
        visible = list(self.static_obstacles)
        visible.extend(self.patrol_obstacles)
        for aob in self.ambush_obstacles:
            if aob.detected:
                visible.append(aob)
        return visible

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
        if self.n_patrol > 0:
            self._init_patrol_obstacles()
        for aob in self.ambush_obstacles:
            aob.detected = False
        return self.state

    def step(self, action, move_patrol=True):
        self._check_detection()
        all_obs = self.get_visible_obstacles()
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

        if move_patrol:
            self._move_patrol_obstacles()

        for dob in self.patrol_obstacles:
            if dob.collides_with(*self.get_usv_rect()):
                return self.state, -1, True

        for aob in self.ambush_obstacles:
            if aob.detected and aob.collides_with(*self.get_usv_rect()):
                return self.state, -1, True

        return self.state, 0, False


# ============================================================
# 6. Q-learning 核心
# ============================================================

class QLearningBase:
    def __init__(self, env, alpha=0.3, gamma_start=0.1, gamma_end=0.9,
                 epsilon=0.9, epsilon_decay=0.997, epsilon_min=0.02,
                 episodes=5000, step_penalty=0.0, distance_reward=0.05, seed=None):
        self.env = env
        self.alpha = alpha
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.episodes = episodes
        self.step_penalty = step_penalty
        self.distance_reward = distance_reward
        self.rng = np.random.RandomState(seed)

        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, N_HEADINGS, N_ACTIONS))
        self.replay_buffer = deque(maxlen=5000)

        self.episode_steps = []
        self.episode_rewards = []
        self.episode_replans = []
        self.episode_collisions = []

    def get_gamma(self, episode):
        progress = min(episode / self.episodes, 1.0)
        return self.gamma_start + (self.gamma_end - self.gamma_start) * progress

    def get_epsilon(self):
        return max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _distance_to_goal(self, x, y):
        return np.hypot(x - self.env.goal[0], y - self.env.goal[1])

    def choose_action_egreedy(self, state):
        gx, gy, h = state
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, N_ACTIONS)
        else:
            q_vals = self.q_table[gx, gy, h, :]
            max_q = np.max(q_vals)
            best = np.where(q_vals == max_q)[0]
            return self.rng.choice(best)

    def rank_actions_sdp(self, state):
        gx, gy, h = state
        wx, wy = self.env.usv.x, self.env.usv.y
        temp_usv = PhysicalUSV(wx, wy, heading_idx=h)
        visible = self.env.get_visible_obstacles()
        occ = self.env._static_occ_grid
        goal = self.env.goal

        candidates = []
        for a in range(N_ACTIONS):
            result = temp_usv.try_action(a, visible, occ)
            if result is not None:
                nx, ny, nh = result
                d = manhattan_dist_m((nx, ny), goal)
                candidates.append((a, d))
        candidates.sort(key=lambda x: x[1])
        return candidates

    def choose_action_sdp(self, state):
        gx, gy, h = state
        wx, wy = self.env.usv.x, self.env.usv.y

        temp_usv = PhysicalUSV(wx, wy, heading_idx=h)
        visible = self.env.get_visible_obstacles()
        occ = self.env._static_occ_grid

        goal = self.env.goal
        distances = []
        for a in range(N_ACTIONS):
            result = temp_usv.try_action(a, visible, occ)
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

    def get_path_sdp(self):
        for attempt in range(10):
            env_copy = USVEnvironment(
                n_patrol=self.env.n_patrol,
                n_ambush=0,
                seed=42 + attempt * 100,
            )
            env_copy.patrol_obstacles = [p.copy() for p in self.env.patrol_obstacles]
            for p in env_copy.patrol_obstacles:
                p.t = 0.5; p.direction = 1
                p.cx = p.waypoint_a[0] + 0.5 * (p.waypoint_b[0] - p.waypoint_a[0])
                p.cy = p.waypoint_a[1] + 0.5 * (p.waypoint_b[1] - p.waypoint_a[1])
            env_copy.ambush_obstacles = [a.copy() for a in self.env.ambush_obstacles]
            env_copy.static_obstacles = self.env.static_obstacles
            env_copy._static_occ_grid = self.env._static_occ_grid
            env_copy.usv.reset(*env_copy.start, heading_idx=env_copy.start_heading)

            path_points = [(env_copy.usv.x, env_copy.usv.y, env_copy.usv.heading_idx)]
            max_steps = GRID_SIZE * 10
            stuck_counter = 0

            for _ in range(max_steps):
                if env_copy.is_at_goal():
                    break

                env_copy._check_detection()
                all_obs = env_copy.get_visible_obstacles()
                occ = env_copy._static_occ_grid

                h = env_copy.usv.heading_idx
                sdp_candidates = []
                temp_usv = PhysicalUSV(env_copy.usv.x, env_copy.usv.y, heading_idx=h)
                for a in range(N_ACTIONS):
                    result = temp_usv.try_action(a, all_obs, occ)
                    if result is not None:
                        nx, ny, nh = result
                        d = manhattan_dist_m((nx, ny), env_copy.goal)
                        sdp_candidates.append((a, d))

                if not sdp_candidates:
                    stuck_counter += 1
                    if stuck_counter > 8:
                        break
                    for a in range(N_ACTIONS):
                        ns, r, done = env_copy.step(a, move_patrol=False)
                        if r >= 0:
                            path_points.append((env_copy.usv.x, env_copy.usv.y,
                                                env_copy.usv.heading_idx))
                            stuck_counter = 0
                            break
                    else:
                        break
                    continue

                stuck_counter = 0
                sdp_candidates.sort(key=lambda x: x[1])
                best_action = sdp_candidates[0][0]

                ns, r, done = env_copy.step(best_action, move_patrol=False)
                if r < 0:
                    for a, _ in sdp_candidates[1:]:
                        ns, r, done = env_copy.step(a, move_patrol=False)
                        if r >= 0:
                            best_action = a
                            break
                    if r < 0:
                        break
                path_points.append((env_copy.usv.x, env_copy.usv.y,
                                    env_copy.usv.heading_idx))
                if done and r > 0:
                    break

            if env_copy.is_at_goal():
                return path_points

        return path_points

    def get_path_ql(self, max_attempts=5):
        best_path = []

        for attempt in range(max_attempts):
            env_copy = USVEnvironment(
                n_patrol=self.env.n_patrol,
                n_ambush=0,
                seed=42 + attempt,
            )
            env_copy.patrol_obstacles = [p.copy() for p in self.env.patrol_obstacles]
            for p in env_copy.patrol_obstacles:
                p.t = 0.5; p.direction = 1
                p.cx = p.waypoint_a[0] + 0.5 * (p.waypoint_b[0] - p.waypoint_a[0])
                p.cy = p.waypoint_a[1] + 0.5 * (p.waypoint_b[1] - p.waypoint_a[1])
            env_copy.ambush_obstacles = [a.copy() for a in self.env.ambush_obstacles]
            env_copy.static_obstacles = self.env.static_obstacles
            env_copy._static_occ_grid = self.env._static_occ_grid
            env_copy.usv.reset(*env_copy.start, heading_idx=env_copy.start_heading)

            path_points = [(env_copy.usv.x, env_copy.usv.y, env_copy.usv.heading_idx)]
            visited_states = set()
            max_steps = GRID_SIZE * 6

            for _ in range(max_steps):
                if env_copy.is_at_goal():
                    break

                env_copy._check_detection()
                all_obs = env_copy.get_visible_obstacles()
                occ = env_copy._static_occ_grid

                gx, gy, h = env_copy.state
                state_key = (round(env_copy.usv.x), round(env_copy.usv.y), h)
                if state_key in visited_states:
                    break
                visited_states.add(state_key)

                q_vals = self.q_table[gx, gy, h, :].copy()
                temp_usv = PhysicalUSV(env_copy.usv.x, env_copy.usv.y, heading_idx=h)
                for a in range(N_ACTIONS):
                    if temp_usv.try_action(a, all_obs, occ) is None:
                        q_vals[a] = -np.inf

                action_order = list(np.argsort(q_vals)[::-1])

                moved = False
                for a in action_order:
                    if q_vals[a] == -np.inf:
                        continue
                    ns, r, done = env_copy.step(a, move_patrol=False)
                    if r >= 0:
                        path_points.append((env_copy.usv.x, env_copy.usv.y,
                                            env_copy.usv.heading_idx))
                        moved = True
                        break

                if not moved:
                    break

            if (env_copy.is_at_goal() or len(path_points) > len(best_path)):
                best_path = path_points
            if best_path and env_copy.is_at_goal():
                break

        return best_path

    def get_path(self, max_attempts=5):
        return self.get_path_ql(max_attempts)


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
            replan_count = 0
            collision_count = 0
            max_steps = GRID_SIZE * 5

            prev_dist = self._distance_to_goal(self.env.usv.x, self.env.usv.y)

            for _ in range(max_steps):
                action = self.choose_action_egreedy(state)
                next_state, reward, done = self.env.step(action)

                if reward < 0:
                    collision_count += 1
                    replan_count += 1

                if reward >= 0:
                    cur_dist = self._distance_to_goal(self.env.usv.x, self.env.usv.y)
                    reward += (prev_dist - cur_dist) * self.distance_reward / WORLD_SIZE
                    prev_dist = cur_dist

                reward += self.step_penalty
                self.update_q(state, action, reward, next_state, gamma)
                total_reward += reward
                steps += 1
                state = next_state
                if done:
                    break

            self.decay_epsilon()
            self.experience_replay(gamma)
            self.episode_steps.append(steps)
            self.episode_rewards.append(total_reward)
            self.episode_replans.append(replan_count)
            self.episode_collisions.append(collision_count)

            if verbose and (ep + 1) % 500 == 0:
                print(f"  Alg2(QL+eps) Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward:.2f}  "
                      f"eps={self.epsilon:.3f}  replans={replan_count}  collisions={collision_count}")


class Alg4_ProposedQL_Dynamic(QLearningBase):

    def train(self, verbose=True):
        for ep in range(self.episodes):
            state = self.env.reset()
            gamma = self.get_gamma(ep)
            total_reward = 0
            steps = 0
            replan_count = 0
            collision_count = 0
            max_steps = GRID_SIZE * 5

            for _ in range(max_steps):
                candidates = self.rank_actions_sdp(state)
                moved = False
                first_try = True
                for a, d in candidates:
                    next_state, reward, done = self.env.step(a)
                    if reward < 0:
                        collision_count += 1
                        if first_try:
                            replan_count += 1
                        first_try = False
                        continue
                    else:
                        if not first_try:
                            replan_count += 1
                        reward += self.step_penalty
                        self.update_q(state, a, reward, next_state, gamma)
                        total_reward += reward
                        steps += 1
                        state = next_state
                        moved = True
                        if done:
                            break
                        break
                if not moved:
                    break
                if done:
                    break

            self.experience_replay(gamma)
            self.episode_steps.append(steps)
            self.episode_rewards.append(total_reward)
            self.episode_replans.append(replan_count)
            self.episode_collisions.append(collision_count)

            if verbose and (ep + 1) % 500 == 0:
                print(f"  Alg4(SDP-QL) Ep {ep+1}/{self.episodes}  "
                      f"steps={steps}  reward={total_reward:.2f}  "
                      f"replans={replan_count}  collisions={collision_count}")

    def get_path(self, max_attempts=5):
        return self.get_path_sdp()


# ============================================================
# 8. 可视化
# ============================================================

def draw_obb(ax, cx, cy, length, width, angle, color, alpha=0.7, lw=1, zorder=None):
    corners = obb_corners(cx, cy, length, width, angle)
    kwargs = dict(closed=True, facecolor=color, edgecolor='black',
                  linewidth=lw, alpha=alpha)
    if zorder is not None:
        kwargs['zorder'] = zorder
    poly = Polygon(corners, **kwargs)
    ax.add_patch(poly)


def plot_env(ax, env, title="Environment", show_patrol=True, show_ambush=True):
    ax.set_xlim(-20, WORLD_SIZE + 20)
    ax.set_ylim(-20, WORLD_SIZE + 20)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(0, WORLD_SIZE + 1, 200))
    ax.set_yticks(np.arange(0, WORLD_SIZE + 1, 200))
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.2)

    for obs in env.static_obstacles:
        draw_obb(ax, obs.cx, obs.cy, obs.length, obs.width, obs.angle,
                 color=STATIC_COLOR, alpha=0.7, lw=1.5)

    if show_patrol:
        for pob in env.patrol_obstacles:
            draw_obb(ax, pob.cx, pob.cy, pob.length, pob.width, pob.angle,
                     color=PATROL_COLOR, alpha=0.5, lw=1.5)

    if show_ambush:
        for aob in env.ambush_obstacles:
            if aob.detected:
                color = AMBUSH_DETECTED_COLOR
                label = None
            else:
                color = AMBUSH_COLOR
                label = None
            draw_obb(ax, aob.cx, aob.cy, aob.length, aob.width, aob.angle,
                     color=color, alpha=0.5, lw=2)

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
# 9. GIF 轨迹动画生成
# ============================================================

def generate_path_gif(env, path_points, save_path, fps=8, show_detection=True):
    if not _HAS_PIL:
        print("  [警告] PIL(Pillow)未安装, 无法生成GIF。请执行: pip install Pillow")
        _save_frames_as_pngs(env, path_points, save_path.replace('.gif', '_frames'), fps)
        return

    frames = []
    env_sim = USVEnvironment(n_patrol=len(env.patrol_obstacles),
                              n_ambush=0, seed=42)
    env_sim.patrol_obstacles = [p.copy() for p in env.patrol_obstacles]
    for p in env_sim.patrol_obstacles:
        p.t = 0.5; p.direction = 1
        p.cx = p.waypoint_a[0] + 0.5 * (p.waypoint_b[0] - p.waypoint_a[0])
        p.cy = p.waypoint_a[1] + 0.5 * (p.waypoint_b[1] - p.waypoint_a[1])
    env_sim.ambush_obstacles = [a.copy() for a in env.ambush_obstacles]
    env_sim.static_obstacles = env.static_obstacles
    env_sim._static_occ_grid = env._static_occ_grid

    total_frames = len(path_points)

    for i, (x, y, h) in enumerate(path_points):
        env_sim._move_patrol_obstacles()
        env_sim._check_detection_with_pos(x, y)

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.set_xlim(-20, WORLD_SIZE + 20)
        ax.set_ylim(-20, WORLD_SIZE + 20)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, WORLD_SIZE + 1, 200))
        ax.set_yticks(np.arange(0, WORLD_SIZE + 1, 200))
        ax.set_title(f"USV Path Animation (Step {i+1}/{total_frames})", fontsize=12)
        ax.grid(True, alpha=0.2)

        for obs in env_sim.static_obstacles:
            draw_obb(ax, obs.cx, obs.cy, obs.length, obs.width, obs.angle,
                     color=STATIC_COLOR, alpha=0.7, lw=1.5)

        for pob in env_sim.patrol_obstacles:
            draw_obb(ax, pob.cx, pob.cy, pob.length, pob.width, pob.angle,
                     color=PATROL_COLOR, alpha=0.5, lw=1.5)

        for aob in env_sim.ambush_obstacles:
            if aob.detected:
                color = AMBUSH_DETECTED_COLOR
                label = 'Ambush (detected)'
                draw_obb(ax, aob.cx, aob.cy, aob.length, aob.width, aob.angle,
                         color=color, alpha=0.6, lw=2)
                ax.plot(aob.cx, aob.cy, 'x', color='black', markersize=8, zorder=10)
            else:
                color = AMBUSH_COLOR
                draw_obb(ax, aob.cx, aob.cy, aob.length, aob.width, aob.angle,
                         color=color, alpha=0.2, lw=1, zorder=0)

        if i > 0:
            xs_trail = [p[0] for p in path_points[:i + 1]]
            ys_trail = [p[1] for p in path_points[:i + 1]]
            ax.plot(xs_trail, ys_trail, color='red', linewidth=1.0, alpha=0.5,
                    linestyle='-', zorder=3)

        draw_obb(ax, x, y, USV_LENGTH, USV_WIDTH, HEADING_ANGLES[h],
                 color='lime', alpha=0.9, lw=2.5)
        ax.plot(x, y, 'o', color='lime', markersize=5, zorder=10)

        ax.plot(env.start[0], env.start[1], 'o', color='orange', markersize=8,
                markeredgecolor='black', zorder=5)
        ax.plot(env.goal[0], env.goal[1], 's', color='yellow', markersize=8,
                markeredgecolor='black', zorder=5)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=STATIC_COLOR, alpha=0.7, label='Static'),
            Patch(facecolor=PATROL_COLOR, alpha=0.5, label='Patrol'),
            Patch(facecolor=AMBUSH_COLOR, alpha=0.5, label='Ambush (hidden)'),
            Patch(facecolor=AMBUSH_DETECTED_COLOR, alpha=0.6, label='Ambush (detected)'),
            Patch(facecolor='lime', alpha=0.7, label='USV'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = PILImage.open(buf)
        frames.append(img.copy())
        plt.close(fig)
        buf.close()

        if (i + 1) % 20 == 0 or i == total_frames - 1:
            print(f"  GIF 帧: {i+1}/{total_frames}")

    duration = int(1000 / fps)
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False,
    )
    print(f"  GIF 保存: {save_path}  ({len(frames)}帧, {fps}fps)")


def _save_frames_as_pngs(env, path_points, save_dir, fps=8):
    os.makedirs(save_dir, exist_ok=True)
    for i, (x, y, h) in enumerate(path_points):
        fig, ax = plt.subplots(figsize=(10, 9))
        plot_env(ax, env, f"USV Path - Frame {i}")
        draw_obb(ax, x, y, USV_LENGTH, USV_WIDTH, HEADING_ANGLES[h],
                 color='lime', alpha=0.9, lw=2)
        if i > 0:
            xs_trail = [p[0] for p in path_points[:i + 1]]
            ys_trail = [p[1] for p in path_points[:i + 1]]
            ax.plot(xs_trail, ys_trail, 'r-', linewidth=1, alpha=0.5)
        fig.savefig(f"{save_dir}/frame_{i:04d}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)
    print(f"  帧保存至: {save_dir}/  ({len(path_points)}帧)")


# ============================================================
# 10. 主程序
# ============================================================

OUTDIR = 'D:\\QiangHuaXueXi\\Result\\'


def main():
    print("=" * 60)
    print(" USV Path Planning with Physical Model & 3 Obstacle Types")
    print(f" World: {WORLD_SIZE:.0f}m x {WORLD_SIZE:.0f}m")
    print(f" Grid:  {GRID_SIZE}x{GRID_SIZE} ({CELL_SIZE:.0f}m/cell)")
    print(f" USV:   {USV_LENGTH:.0f}m x {USV_WIDTH:.0f}m")
    print(f" Turn radius: {USV_MIN_RADIUS:.0f}m ~ {USV_MAX_RADIUS:.0f}m")
    print(f" Obstacles: Static(blue) + Patrol({PATROL_OBS_LENGTH:.0f}x{PATROL_OBS_WIDTH:.0f}m, orange)")
    print(f"            + Ambush({AMBUSH_OBS_LENGTH:.0f}x{AMBUSH_OBS_WIDTH:.0f}m, purple,")
    print(f"            detection range={DETECTION_RANGE:.0f}m)")
    print(" Alg2: QL+epsilon-greedy  vs  Alg4: SDP-QL")
    print("=" * 60)

    EPISODES = 2000
    SEED = 42
    N_PATROL = 2

    results = {}

    print("\n>>> Algorithm 2: Q-learning + epsilon-greedy (static + 2 patrol)")
    t0 = time.time()
    env2 = USVEnvironment(n_patrol=N_PATROL, n_ambush=0, seed=SEED)
    alg2 = Alg2_OriginalQL_Dynamic(env2, episodes=EPISODES, epsilon=1.0,
                                    epsilon_decay=0.995, epsilon_min=0.02,
                                    alpha=0.4, step_penalty=-0.001,
                                    distance_reward=0.05, seed=SEED)
    alg2.train()
    t1 = time.time()
    path2 = alg2.get_path()
    success2 = len(path2) > 1 and np.hypot(path2[-1][0] - env2.goal[0],
                                             path2[-1][1] - env2.goal[1]) < CELL_SIZE * 2
    results['Alg2_OriginalQL_Dynamic'] = {
        'time': t1 - t0,
        'train_steps': alg2.episode_steps,
        'train_rewards': alg2.episode_rewards,
        'train_replans': alg2.episode_replans,
        'train_collisions': alg2.episode_collisions,
        'path': path2,
        'path_len': len(path2),
        'success': success2,
        'env': env2,
    }

    alg2_success_count = sum(1 for r in alg2.episode_rewards if r > 0)
    avg_reward2 = np.mean(alg2.episode_rewards[-500:]) if len(alg2.episode_rewards) >= 500 else np.mean(alg2.episode_rewards)
    total_replans2 = sum(alg2.episode_replans)
    total_collisions2 = sum(alg2.episode_collisions)
    print(f"  Train: {t1 - t0:.1f}s  Path pts: {len(path2)}  "
          f"GoalReached: {success2}  TrainSuccess: {alg2_success_count}/{EPISODES}")
    print(f"  AvgReward: {avg_reward2:.4f}  TotalReplans: {total_replans2}  TotalCollisions: {total_collisions2}")

    print("\n>>> Algorithm 4: SDP-QL (static + 2 patrol)")
    t0 = time.time()
    env4 = USVEnvironment(n_patrol=N_PATROL, n_ambush=0, seed=SEED)
    alg4 = Alg4_ProposedQL_Dynamic(env4, episodes=EPISODES, seed=SEED)
    alg4.train()
    t1 = time.time()

    prelim_path = alg4.get_path_sdp()
    print(f"  初步路径点数: {len(prelim_path)}, 到达目标: {np.hypot(prelim_path[-1][0] - env4.goal[0], prelim_path[-1][1] - env4.goal[1]) < CELL_SIZE * 2}")

    if len(prelim_path) > 3:
        aob = env4.place_ambush_on_path(prelim_path)
        if aob:
            print(f"  伏击障碍物已放置在路径上: ({aob.cx:.0f}, {aob.cy:.0f})")
        else:
            print("  [警告] 无法放置伏击障碍物")
    else:
        print("  [警告] 初步路径太短, 跳过伏击障碍物放置")

    path4 = alg4.get_path_sdp() if len(env4.ambush_obstacles) > 0 else alg4.get_path()
    success4 = len(path4) > 1 and np.hypot(path4[-1][0] - env4.goal[0],
                                             path4[-1][1] - env4.goal[1]) < CELL_SIZE * 2

    ambush_detected = any(aob.detected for aob in env4.ambush_obstacles)
    print(f"  伏击障碍物是否被探测到: {'是' if ambush_detected else '否'}")

    results['Alg4_ProposedQL_Dynamic'] = {
        'time': t1 - t0,
        'train_steps': alg4.episode_steps,
        'train_rewards': alg4.episode_rewards,
        'train_replans': alg4.episode_replans,
        'train_collisions': alg4.episode_collisions,
        'path': path4,
        'path_len': len(path4),
        'success': success4,
        'env': env4,
    }

    alg4_success_count = sum(1 for r in alg4.episode_rewards if r > 0)
    avg_reward4 = np.mean(alg4.episode_rewards[-500:]) if len(alg4.episode_rewards) >= 500 else np.mean(alg4.episode_rewards)
    total_replans4 = sum(alg4.episode_replans)
    total_collisions4 = sum(alg4.episode_collisions)
    print(f"  Train: {t1 - t0:.1f}s  Path pts: {len(path4)}  "
          f"GoalReached: {success4}  TrainSuccess: {alg4_success_count}/{EPISODES}")
    print(f"  AvgReward: {avg_reward4:.4f}  TotalReplans: {total_replans4}  TotalCollisions: {total_collisions4}")

    print("\n" + "=" * 70)
    print(" Evaluation Metrics")
    print("=" * 70)
    print(f" {'Algorithm':<30} {'Train(s)':>8} {'Steps':>8} {'Success':>8} {'AvgRew':>8} {'Replans':>8} {'Collis':>8}")
    print("-" * 78)
    for name in ['Alg2_OriginalQL_Dynamic', 'Alg4_ProposedQL_Dynamic']:
        r = results[name]
        rewards = r['train_rewards']
        avg_r = np.mean(rewards[-500:]) if len(rewards) >= 500 else np.mean(rewards)
        total_rp = sum(r['train_replans'])
        total_col = sum(r['train_collisions'])
        print(f" {name:<30} {r['time']:>8.1f} {r['path_len']:>8}  "
              f"{'Yes' if r['success'] else 'No':>8} {avg_r:>8.3f} {total_rp:>8} {total_col:>8}")

    print("\n>>> Generating figures...")
    os.makedirs(OUTDIR, exist_ok=True)
    w = max(1, EPISODES // 50)

    fig1, ax1 = plt.subplots(figsize=(10, 9))
    plot_env(ax1, env2, "Alg2: QL+epsilon-greedy (Static + Patrol + Ambush)", show_ambush=False)
    plot_path_with_arcs(ax1, path2, color='lime', label='Alg2 Path')
    ax1.set_title("Fig.1: Alg2 Path (Physical Model, Turning Radius)",
                  fontsize=12, fontweight='bold')
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=STATIC_COLOR,
                   markersize=10, label='Static'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=PATROL_COLOR,
                   markersize=10, label='Patrol'),
    ]
    ax1.legend(loc='upper left', fontsize=8)
    fig1.savefig(OUTDIR + 'Fig1_Alg2_physical_path.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  Saved: Fig1_Alg2_physical_path.png")

    fig2, ax2 = plt.subplots(figsize=(10, 9))
    plot_env(ax2, env4, "Alg4: SDP-QL (Static + Patrol + Ambush)", show_ambush=True)
    plot_path_with_arcs(ax2, path4, color='lime', label='Alg4 Path')
    ax2.set_title("Fig.2: Alg4 Path with Ambush Detection (Physical Model)",
                  fontsize=12, fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements2 = [
        Patch(facecolor=STATIC_COLOR, alpha=0.7, label='Static'),
        Patch(facecolor=PATROL_COLOR, alpha=0.5, label='Patrol'),
        Patch(facecolor=AMBUSH_DETECTED_COLOR, alpha=0.5, label='Ambush (detected)'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper left', fontsize=8)
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

    fig4, ax4 = plt.subplots(figsize=(9, 5))
    r2 = results['Alg2_OriginalQL_Dynamic']['train_rewards']
    ax4.plot(r2, alpha=0.15, color='steelblue', linewidth=0.5)
    if len(r2) >= w:
        sm = np.convolve(r2, np.ones(w) / w, mode='valid')
        ax4.plot(range(w - 1, len(r2)), sm, color='darkblue',
                 linewidth=2, label='Alg2 (eps-greedy)')
    r4 = results['Alg4_ProposedQL_Dynamic']['train_rewards']
    ax4.plot(r4, alpha=0.15, color='darkgreen', linewidth=0.5)
    if len(r4) >= w:
        sm = np.convolve(r4, np.ones(w) / w, mode='valid')
        ax4.plot(range(w - 1, len(r4)), sm, color='darkgreen',
                 linewidth=2, label='Alg4 (SDP-QL)')
    ax4.set_title("Fig.4: Cumulative Reward Convergence (avg reward)")
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Cumulative Reward')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    fig4.savefig(OUTDIR + 'Fig4_reward_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig4)
    print("  Saved: Fig4_reward_convergence.png")

    fig5, ax5 = plt.subplots(figsize=(9, 5))
    rp2 = results['Alg2_OriginalQL_Dynamic']['train_replans']
    ax5.plot(rp2, alpha=0.15, color='steelblue', linewidth=0.5)
    if len(rp2) >= w:
        sm = np.convolve(rp2, np.ones(w) / w, mode='valid')
        ax5.plot(range(w - 1, len(rp2)), sm, color='darkblue',
                 linewidth=2, label='Alg2 (eps-greedy)')
    rp4 = results['Alg4_ProposedQL_Dynamic']['train_replans']
    ax5.plot(rp4, alpha=0.15, color='darkgreen', linewidth=0.5)
    if len(rp4) >= w:
        sm = np.convolve(rp4, np.ones(w) / w, mode='valid')
        ax5.plot(range(w - 1, len(rp4)), sm, color='darkgreen',
                 linewidth=2, label='Alg4 (SDP-QL)')
    ax5.set_title("Fig.5: Replanning Count per Episode")
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Replanning Events')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    fig5.savefig(OUTDIR + 'Fig5_replanning_convergence.png', dpi=150, bbox_inches='tight')
    plt.close(fig5)
    print("  Saved: Fig5_replanning_convergence.png")

    print("\n>>> Generating GIF trajectory animation...")
    for algo_name, r in results.items():
        path = r['path']
        env = r['env']
        if len(path) > 1:
            gif_name = algo_name + '_trajectory.gif'
            generate_path_gif(env, path, OUTDIR + gif_name, fps=8)

    print("\nDone!")


if __name__ == '__main__':
    main()
