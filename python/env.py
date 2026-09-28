import sys
import os
import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quad_sim
from controller import QuadController


RAY_MAX_DIST = 8.0


class Obstacle:
    def __init__(self, shape, center, size):
        self.shape = shape
        self.center = np.array(center, dtype=np.float64)
        self.size = float(size)

    def check_collision(self, pos):
        if self.shape == "sphere":
            return np.linalg.norm(pos - self.center) < self.size
        return np.all(np.abs(pos - self.center) < self.size)

    def distance_to(self, pos):
        if self.shape == "sphere":
            return max(0.0, np.linalg.norm(pos - self.center) - self.size)
        outside = np.maximum(np.abs(pos - self.center) - self.size, 0.0)
        return np.linalg.norm(outside)

    def ray_intersect(self, origin, direction, max_dist):
        if self.shape == "sphere":
            return self._ray_sphere(origin, direction, max_dist)
        return self._ray_cube(origin, direction, max_dist)

    def _ray_sphere(self, origin, direction, max_dist):
        oc = origin - self.center
        b = np.dot(oc, direction)
        c = np.dot(oc, oc) - self.size ** 2
        disc = b * b - c
        if disc < 0:
            return max_dist
        s = np.sqrt(disc)
        t = -b - s
        if t < 0.01:
            t = -b + s
        return t if 0.01 < t < max_dist else max_dist

    def _ray_cube(self, origin, direction, max_dist):
        bmin = self.center - self.size
        bmax = self.center + self.size
        tmin, tmax = -1e9, 1e9
        for i in range(3):
            if abs(direction[i]) < 1e-12:
                if origin[i] < bmin[i] or origin[i] > bmax[i]:
                    return max_dist
            else:
                t1 = (bmin[i] - origin[i]) / direction[i]
                t2 = (bmax[i] - origin[i]) / direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return max_dist
        return tmin if 0.01 < tmin < max_dist else max_dist


def _build_ray_directions():
    dirs = []
    for deg in range(0, 360, 45):
        a = np.radians(deg)
        dirs.append(np.array([np.cos(a), np.sin(a), 0.0]))
    for deg in [0, 90, 180, 270]:
        a = np.radians(deg)
        d = np.array([0.7 * np.cos(a), 0.7 * np.sin(a), 0.7])
        dirs.append(d / np.linalg.norm(d))
    for deg in [0, 90, 180, 270]:
        a = np.radians(deg)
        d = np.array([0.7 * np.cos(a), 0.7 * np.sin(a), -0.7])
        dirs.append(d / np.linalg.norm(d))
    return [d.astype(np.float64) for d in dirs]


RAY_DIRS = _build_ray_directions()
NUM_RAYS = len(RAY_DIRS)


STATE_SCALE = np.array([
    5.0, 5.0, 10.0,
    3.0, 3.0, 3.0,
    1.0, 1.0, 3.14,
    5.0, 5.0, 5.0,
], dtype=np.float32)

DIR_SCALE = 8.0
WIND_SCALE = 0.5


def _seg_dist(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-12:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    return np.linalg.norm(p - (a + t * ab))


class QuadcopterEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=0.02,
        max_steps=500,
        enable_wind=False,
        enable_obstacles=False,
        num_obstacles_range=(0, 2),
        num_waypoints=1,
        arena_size=10.0,
        arena_height=10.0,
        wind_base_max=0.15,
        wind_gust_max=0.05,
        success_radius=0.7,
        corridor_clearance=1.5,
        spawn_radius=2.5,
        spawn_z_range=(3.0, 6.0),
        target_radius=4.0,
        target_z_range=(2.0, 8.0),
    ):
        super().__init__()

        self.dt = dt
        self.max_steps = max_steps
        self.enable_wind = enable_wind
        self.enable_obstacles = enable_obstacles
        self.num_obstacles_range = num_obstacles_range
        self.num_waypoints = num_waypoints
        self.arena_size = arena_size
        self.arena_height = arena_height
        self.wind_base_max = wind_base_max
        self.wind_gust_max = wind_gust_max
        self.success_radius = success_radius
        self.corridor_clearance = corridor_clearance
        self.spawn_radius = spawn_radius
        self.spawn_z_range = spawn_z_range
        self.target_radius = target_radius
        self.target_z_range = target_z_range

        self.current_step = 0
        self.start_pos = None
        self.waypoints = []
        self.current_wp_idx = 0
        self.obstacles = []
        self.wind_base = np.zeros(3)
        self.wind_gust_std = 0.0
        self.prev_dist = 0.0
        self._cached_rays = None

        self.drone = None
        self.controller = QuadController()

        obs_dim = 12 + 3 + 3 + NUM_RAYS

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _sample_start(self):
        r = self.np_random.uniform(0, self.spawn_radius)
        a = self.np_random.uniform(0, 2 * np.pi)
        z = self.np_random.uniform(*self.spawn_z_range)
        return np.array([r * np.cos(a), r * np.sin(a), z])

    def _sample_target(self, start):
        for _ in range(200):
            r = self.np_random.uniform(1.5, self.target_radius)
            a = self.np_random.uniform(0, 2 * np.pi)
            z = self.np_random.uniform(*self.target_z_range)
            t = np.array([r * np.cos(a), r * np.sin(a), z])
            if np.linalg.norm(t - start) > 2.0:
                return t
        return np.array([0.0, 0.0, 5.0])

    def _generate_waypoints(self, start):
        wps = []
        cur = start.copy()
        for _ in range(self.num_waypoints):
            wp = self._sample_target(cur)
            wps.append(wp)
            cur = wp
        return wps

    def _generate_wind(self):
        if not self.enable_wind:
            return np.zeros(3), 0.0
        base = self.np_random.uniform(
            -self.wind_base_max, self.wind_base_max, size=3
        )
        gust = float(self.np_random.uniform(0, self.wind_gust_max))
        return base.astype(np.float64), gust

    def _obstacle_ok(self, obs):
        pts = [self.start_pos] + self.waypoints
        for i in range(len(pts) - 1):
            if _seg_dist(obs.center, pts[i], pts[i+1]) \
                    < obs.size + self.corridor_clearance:
                return False
        for p in pts:
            if np.linalg.norm(obs.center - p) < obs.size + 1.5:
                return False
        return True

    def _generate_obstacles(self):
        if not self.enable_obstacles:
            return []
        n = int(self.np_random.integers(
            self.num_obstacles_range[0],
            self.num_obstacles_range[1] + 1))
        obs_list, att = [], 0
        while len(obs_list) < n and att < 500:
            att += 1
            shape = self.np_random.choice(["sphere", "cube"])
            c = np.array([
                self.np_random.uniform(-self.arena_size*0.4, self.arena_size*0.4),
                self.np_random.uniform(-self.arena_size*0.4, self.arena_size*0.4),
                self.np_random.uniform(1.5, self.arena_height*0.7),
            ])
            sz = float(self.np_random.uniform(
                0.3, 0.9 if shape == "sphere" else 0.7))
            o = Obstacle(shape, c, sz)
            if not self._obstacle_ok(o):
                continue
            if any(np.linalg.norm(e.center - o.center) < e.size + o.size + 0.8
                   for e in obs_list):
                continue
            obs_list.append(o)
        return obs_list

    def _pos(self):
        return np.array([self.drone.x, self.drone.y, self.drone.z])

    def _target(self):
        return self.waypoints[self.current_wp_idx]

    def _cast_rays(self):
        pos = self._pos()
        dists = np.full(NUM_RAYS, RAY_MAX_DIST)
        half = self.arena_size * 0.5

        for i, d in enumerate(RAY_DIRS):
            md = RAY_MAX_DIST

            if d[2] < -0.01:
                t = -pos[2] / d[2]
                if 0.01 < t < md:
                    md = t
            elif d[2] > 0.01:
                t = (self.arena_height - pos[2]) / d[2]
                if 0.01 < t < md:
                    md = t

            for axis in [0, 1]:
                if abs(d[axis]) > 1e-6:
                    for wall in (-half, half):
                        t = (wall - pos[axis]) / d[axis]
                        if 0.01 < t < md:
                            md = t

            for obs in self.obstacles:
                t = obs.ray_intersect(pos, d, md)
                if t < md:
                    md = t

            dists[i] = md

        return (dists / RAY_MAX_DIST).astype(np.float32)

    def _warm_start_hover(self, steps=25):
        hover_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.drone.set_wind(0.0, 0.0, 0.0)

        for _ in range(steps):
            motors = self.controller.compute(hover_action, self.drone)
            self.drone.set_motors_normalized(motors)
            self.drone.step(self.dt)

        self.drone.x = float(self.start_pos[0])
        self.drone.y = float(self.start_pos[1])
        self.drone.z = float(self.start_pos[2])

        self.drone.vx = 0.0
        self.drone.vy = 0.0
        self.drone.vz = 0.0

        self.drone.roll = 0.0
        self.drone.pitch = 0.0
        self.drone.yaw = 0.0

        self.drone.roll_rate = 0.0
        self.drone.pitch_rate = 0.0
        self.drone.yaw_rate = 0.0

        self.drone.sim_time = 0.0
        self._cached_rays = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        cfg = quad_sim.QuadConfig()
        cfg.max_angle = 1.55
        cfg.max_height = self.arena_height + 2.0
        cfg.max_distance = self.arena_size
        self.drone = quad_sim.Quadcopter(cfg)

        self.current_step = 0
        self.current_wp_idx = 0
        self._cached_rays = None

        if "start_pos" in options:
            self.start_pos = np.array(options["start_pos"], dtype=np.float64)
        else:
            self.start_pos = self._sample_start()

        if "target_pos" in options:
            self.waypoints = [np.array(options["target_pos"], dtype=np.float64)]
        else:
            self.waypoints = self._generate_waypoints(self.start_pos)

        if "wind_base" in options:
            self.wind_base = np.array(options["wind_base"], dtype=np.float64)
            self.wind_gust_std = float(options.get("wind_gust_std", 0.0))
        else:
            self.wind_base, self.wind_gust_std = self._generate_wind()

        if "obstacles" in options:
            self.obstacles = options["obstacles"]
        else:
            self.obstacles = self._generate_obstacles()

        self.drone.x = float(self.start_pos[0])
        self.drone.y = float(self.start_pos[1])
        self.drone.z = float(self.start_pos[2])

        self.controller.reset(initial_yaw=0.0)
        self._warm_start_hover()

        self.prev_dist = np.linalg.norm(self._pos() - self._target())

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        motors = self.controller.compute(action, self.drone)
        self.drone.set_motors_normalized(motors)

        gust = self.np_random.normal(0.0, self.wind_gust_std, size=3)
        wind = self.wind_base + gust
        self.drone.set_wind(float(wind[0]), float(wind[1]), float(wind[2]))

        self.drone.step(self.dt)
        self.current_step += 1
        self._cached_rays = None

        pos = self._pos()
        dist = np.linalg.norm(pos - self._target())

        collided = any(o.check_collision(pos) for o in self.obstacles)

        ri = self.drone.get_reward_info()
        crashed = bool(ri.crashed)
        oob = bool(ri.out_of_bounds)
        grounded = self.drone.z < 0.15

        reached_wp = False
        success = False
        if dist < self.success_radius:
            reached_wp = True
            if self.current_wp_idx < len(self.waypoints) - 1:
                self.current_wp_idx += 1
                self.prev_dist = np.linalg.norm(self._pos() - self._target())
            else:
                success = True

        reward = self._reward(dist, reached_wp, success,
                              collided, crashed, oob, grounded)

        if not reached_wp:
            self.prev_dist = dist

        terminated = success or collided or crashed or oob or grounded
        truncated = self.current_step >= self.max_steps

        if truncated and not success:
            reward -= 50.0 * dist

        return (self._get_obs(), reward, terminated, truncated,
                self._get_info(success=success, collided=collided))

    def _get_obs(self):
        state = np.array(self.drone.get_state(), dtype=np.float32)
        state_n = state / STATE_SCALE

        direction = (self._target() - self._pos()).astype(np.float32)
        dir_n = direction / DIR_SCALE

        wind_n = (self.wind_base / WIND_SCALE).astype(np.float32)

        if self._cached_rays is None:
            self._cached_rays = self._cast_rays()

        return np.concatenate([state_n, dir_n, wind_n, self._cached_rays])

    def _reward(self, dist, reached_wp, success,
                collided, crashed, oob, grounded):
        progress = self.prev_dist - dist
        speed = np.sqrt(self.drone.vx**2 + self.drone.vy**2 + self.drone.vz**2)

        r = 0.0

        r += 12.0 * progress
        r -= 0.08 * dist
        r -= 0.03

        if dist < 1.5:
            r += 0.1
        if dist < 0.8:
            r += 0.2
            r -= 0.10 * speed

        if self._cached_rays is None:
            self._cached_rays = self._cast_rays()
        min_ray_m = float(np.min(self._cached_rays)) * RAY_MAX_DIST
        if min_ray_m < 1.5:
            r -= 2.5 * (1.5 - min_ray_m) / 1.5

        if reached_wp:
            r += 50.0
        if success:
            r += 200.0

        if collided:
            r -= 120.0
        if crashed:
            r -= 60.0
        if oob:
            r -= 60.0
        if grounded:
            r -= 80.0

        return float(r)

    def _get_info(self, success=False, collided=False):
        pos = self._pos()
        return {
            "step": self.current_step,
            "distance": float(np.linalg.norm(pos - self._target())),
            "nearest_obstacle": float(
                min((o.distance_to(pos) for o in self.obstacles),
                    default=RAY_MAX_DIST)),
            "waypoint_idx": self.current_wp_idx,
            "total_waypoints": len(self.waypoints),
            "success": bool(success),
            "collided": bool(collided),
            "target": self._target().copy(),
            "wind": self.wind_base.copy(),
        }