import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO
from env import QuadcopterEnv


MODEL_CANDIDATES = [
    "ppo_hover",
    "ppo_finetuned_goal",
    "ppo_stage0",
    "best_stage3/best_model",
]


def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            print(f"Loading model: {path}")
            return PPO.load(path), path
    raise FileNotFoundError(
        "Не найдена обученная модель. Ожидается один из файлов: "
        + ", ".join(MODEL_CANDIDATES)
    )

def show_animation_from_data(
    positions,
    angle_data,
    target,
    obstacles=None,
    wind_base=None,
    info=None,
    title="Quadcopter flight visualization"
):
    positions = np.array(positions, dtype=np.float64)
    angle_data = np.array(angle_data, dtype=np.float64)
    target = np.array(target, dtype=np.float64)

    obstacles = obstacles or []
    wind_base = np.array(wind_base if wind_base is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    info = info or {}

    if len(positions) == 0:
        print("Нет данных траектории для отображения.")
        return

    final_dist = np.linalg.norm(positions[-1] - target)

    print(f"Steps: {len(positions)}")
    print(f"Start:  ({positions[0][0]:.2f}, {positions[0][1]:.2f}, {positions[0][2]:.2f})")
    print(f"End:    ({positions[-1][0]:.2f}, {positions[-1][1]:.2f}, {positions[-1][2]:.2f})")
    print(f"Target: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
    print(f"Dist:   {final_dist:.3f}")
    print(f"Wind:   {wind_base}")
    print(f"Obs:    {len(obstacles)}")
    print(f"Success: {info.get('success', False)}")

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    all_pts = np.vstack([positions, target.reshape(1, 3)])
    min_xyz = all_pts.min(axis=0) - 1.5
    max_xyz = all_pts.max(axis=0) + 1.5

    ax.set_xlim(min_xyz[0], max_xyz[0])
    ax.set_ylim(min_xyz[1], max_xyz[1])
    ax.set_zlim(max(0.0, min_xyz[2]), max_xyz[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)

    ax.scatter(*positions[0], color="green", s=80, label="Start")
    ax.scatter(*target, color="red", s=200, marker="*", label="Target")

    gx = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 8)
    gy = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 8)
    gx, gy = np.meshgrid(gx, gy)
    ax.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.08, color="green")

    for obs_obj in obstacles:
        c = obs_obj.center
        if obs_obj.shape == "sphere":
            u = np.linspace(0, 2 * np.pi, 16)
            v = np.linspace(0, np.pi, 12)
            r = obs_obj.size
            sx = c[0] + r * np.outer(np.cos(u), np.sin(v))
            sy = c[1] + r * np.outer(np.sin(u), np.sin(v))
            sz = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(sx, sy, sz, alpha=0.25, color="orange")
        else:
            s = obs_obj.size

            x = [c[0] - s, c[0] + s]
            y = [c[1] - s, c[1] + s]
            z = [c[2] - s, c[2] + s]

            X, Y = np.meshgrid(x, y)
            Z_top = np.full_like(X, z[1], dtype=np.float64)
            Z_bottom = np.full_like(X, z[0], dtype=np.float64)

            ax.plot_surface(X, Y, Z_top, color="orange", alpha=0.25, edgecolor="none")
            ax.plot_surface(X, Y, Z_bottom, color="orange", alpha=0.25, edgecolor="none")

            X, Z = np.meshgrid(x, z)
            Y_front = np.full_like(X, y[0], dtype=np.float64)
            Y_back = np.full_like(X, y[1], dtype=np.float64)

            ax.plot_surface(X, Y_front, Z, color="orange", alpha=0.25, edgecolor="none")
            ax.plot_surface(X, Y_back, Z, color="orange", alpha=0.25, edgecolor="none")

            Y, Z = np.meshgrid(y, z)
            X_left = np.full_like(Y, x[0], dtype=np.float64)
            X_right = np.full_like(Y, x[1], dtype=np.float64)

            ax.plot_surface(X_left, Y, Z, color="orange", alpha=0.25, edgecolor="none")
            ax.plot_surface(X_right, Y, Z, color="orange", alpha=0.25, edgecolor="none")

    wm = np.linalg.norm(wind_base)
    if wm > 1e-6:
        ax.quiver(
            positions[0][0], positions[0][1], positions[0][2],
            wind_base[0] * 8, wind_base[1] * 8, wind_base[2] * 8,
            color="cyan", arrow_length_ratio=0.2, linewidth=2,
            label=f"Wind ({wm:.2f})"
        )

    trail, = ax.plot([], [], [], "b-", alpha=0.4, linewidth=1)
    drone_pt = ax.scatter([], [], [], color="blue", s=80, zorder=5)

    arm_len = 0.3
    arm_lines = []
    for clr in ["red", "green", "red", "green"]:
        ln, = ax.plot([], [], [], color=clr, linewidth=3)
        arm_lines.append(ln)

    txt = ax.text2D(
        0.02, 0.95, "", transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    def get_arms(pos, r, p, y):
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])
        ds = [
            np.array([arm_len, 0, 0]),
            np.array([0, arm_len, 0]),
            np.array([-arm_len, 0, 0]),
            np.array([0, -arm_len, 0])
        ]
        return [(pos, pos + R @ d) for d in ds]

    def update(frame):
        pos = positions[frame]
        r, p, y = angle_data[frame]

        trail.set_data(positions[:frame + 1, 0], positions[:frame + 1, 1])
        trail.set_3d_properties(positions[:frame + 1, 2])

        drone_pt._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        arms = get_arms(pos, r, p, y)
        for i, (s, e) in enumerate(arms):
            arm_lines[i].set_data([s[0], e[0]], [s[1], e[1]])
            arm_lines[i].set_3d_properties([s[2], e[2]])

        d = np.linalg.norm(pos - target)
        txt.set_text(
            f"Step {frame}/{len(positions) - 1}\n"
            f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\n"
            f"Dist: {d:.3f} m"
        )

        return [trail, drone_pt] + arm_lines + [txt]

    anim = FuncAnimation(
        fig, update, frames=len(positions),
        interval=40, blit=False, repeat=True
    )

    fig._anim_ref = anim

    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def run_episode_with_model(
    model,
    env,
    reset_options=None,
    seed=None,
    deterministic=True
):
    obs, info = env.reset(seed=seed, options=reset_options)

    positions = []
    angles = []

    terminated = False
    truncated = False

    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        positions.append([env.drone.x, env.drone.y, env.drone.z])
        angles.append([env.drone.roll, env.drone.pitch, env.drone.yaw])

    return {
        "positions": np.array(positions, dtype=np.float64),
        "angles": np.array(angles, dtype=np.float64),
        "target": env.waypoints[0].copy(),
        "obstacles": env.obstacles,
        "wind_base": env.wind_base.copy(),
        "info": info,
    }


def create_animation():
    model, model_path = load_model()

    env = QuadcopterEnv(
        enable_wind=True,
        enable_obstacles=True,
        num_obstacles_range=(1, 3),
        num_waypoints=1,
        success_radius=0.7,
        corridor_clearance=0.8,
    )

    result = run_episode_with_model(model, env)

    show_animation_from_data(
        positions=result["positions"],
        angle_data=result["angles"],
        target=result["target"],
        obstacles=result["obstacles"],
        wind_base=result["wind_base"],
        info=result["info"],
        title=f"Quadcopter animation | model: {model_path}"
    )


if __name__ == "__main__":
    create_animation()