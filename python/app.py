import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from stable_baselines3 import PPO

from env import QuadcopterEnv

MODEL_CANDIDATES = [
    "ppo_finetuned_goal",
    "ppo_stage2",
    "ppo_stage1",
    "ppo_stage0",
]

def load_model(path=None):
    if path is not None:
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            return PPO.load(path), path
        raise FileNotFoundError(f"Model {path} not found")
    
    for path in MODEL_CANDIDATES:
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            return PPO.load(path), path
    raise FileNotFoundError("No trained model found.")

def get_available_models():
    available = []
    for path in MODEL_CANDIDATES:
        if os.path.exists(path) or os.path.exists(path + ".zip"):
            available.append(path)
    return available


class QuadcopterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Quadcopter RL Control Suite")
        self.root.geometry("1400x900")
        
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        style.configure("TButton", font=("Helvetica", 10, "bold"), padding=5)
        style.configure("Run.TButton", font=("Helvetica", 11, "bold"), foreground="white", background="#2c3e50")
        style.map("Run.TButton", background=[("active", "#34495e")])

        try:
            self.model, self.model_path = load_model()
        except FileNotFoundError as e:
            messagebox.showerror("Error", str(e))
            root.destroy()
            return

        self.last_result = None
        self.available_models = get_available_models()
        self.anim = None  # Ссылка на объект анимации

        self._build_ui()

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=50)
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(header, text="Quadcopter Autonomous Flight Simulator", 
                 font=("Helvetica", 14, "bold"), fg="white", bg="#2c3e50").pack(side="left", padx=15, pady=10)

        self.status_label = tk.Label(header, text=f"Active Model: {self.model_path}", 
                                     font=("Consolas", 11), fg="#2ecc71", bg="#2c3e50")
        self.status_label.pack(side="right", padx=15)

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # ЛЕВАЯ ПАНЕЛЬ
        left = ttk.Frame(main, width=320)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Выбор модели
        model_frame = ttk.LabelFrame(left, text=" Model Selection ")
        model_frame.pack(fill="x", pady=5)
        self.model_var = tk.StringVar(value=self.model_path)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, values=self.available_models, state="readonly")
        model_combo.pack(fill="x", padx=5, pady=5)
        model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        # Точки
        start_frame = ttk.LabelFrame(left, text=" Start Position ")
        start_frame.pack(fill="x", pady=5)
        self.start_x = tk.StringVar(value="0.0")
        self.start_y = tk.StringVar(value="0.0")
        self.start_z = tk.StringVar(value="5.0")
        self._add_entry(start_frame, "X:", self.start_x, 0)
        self._add_entry(start_frame, "Y:", self.start_y, 1)
        self._add_entry(start_frame, "Z:", self.start_z, 2)

        target_frame = ttk.LabelFrame(left, text=" Target Position ")
        target_frame.pack(fill="x", pady=5)
        self.target_x = tk.StringVar(value="3.0")
        self.target_y = tk.StringVar(value="-2.0")
        self.target_z = tk.StringVar(value="4.0")
        self._add_entry(target_frame, "X:", self.target_x, 0)
        self._add_entry(target_frame, "Y:", self.target_y, 1)
        self._add_entry(target_frame, "Z:", self.target_z, 2)

        # Среда
        env_frame = ttk.LabelFrame(left, text=" Environment ")
        env_frame.pack(fill="x", pady=5)

        self.enable_wind = tk.BooleanVar(value=True)
        self.enable_obstacles = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_frame, text="Wind", variable=self.enable_wind).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Checkbutton(env_frame, text="Obstacles", variable=self.enable_obstacles).grid(row=0, column=1, sticky="w", padx=5)

        self.wind_x = tk.StringVar(value="-0.03")
        self.wind_y = tk.StringVar(value="0.08")
        self.wind_z = tk.StringVar(value="-0.02")
        self.wind_gust = tk.StringVar(value="0.02")
        self.obstacle_count = tk.StringVar(value="2")
        self.goal_tolerance = tk.StringVar(value="0.7")
        self.seed_var = tk.StringVar(value="")

        self._add_entry(env_frame, "Wind X:", self.wind_x, 1)
        self._add_entry(env_frame, "Wind Y:", self.wind_y, 2)
        self._add_entry(env_frame, "Wind Z:", self.wind_z, 3)
        self._add_entry(env_frame, "Gust std:", self.wind_gust, 4)
        self._add_entry(env_frame, "Obstacles:", self.obstacle_count, 5)
        self._add_entry(env_frame, "Target Radius:", self.goal_tolerance, 6)
        self._add_entry(env_frame, "Seed:", self.seed_var, 7)

        # Кнопки
        btn_frame = ttk.LabelFrame(left, text=" Actions ")
        btn_frame.pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Random Scenario", command=self.fill_random).pack(fill="x", padx=5, pady=2)
        ttk.Button(btn_frame, text="Default Scenario", command=self.fill_demo).pack(fill="x", padx=5, pady=2)

        ttk.Button(btn_frame, text="RUN SIMULATION", style="Run.TButton", command=self.run_simulation).pack(fill="x", padx=5, pady=10)

        self.status_box = tk.Label(left, text="Ready.", font=("Helvetica", 10), bg="#ecf0f1", fg="#34495e", pady=10, wraplength=300)
        self.status_box.pack(fill="x", pady=(10, 5))

        # ПРАВАЯ ПАНЕЛЬ
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True)

        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)

        # Вкладка 3D
        tab_3d = ttk.Frame(notebook)
        notebook.add(tab_3d, text=" Live 3D View ")

        self.fig_3d = Figure(figsize=(8, 6), dpi=100)
        self.ax_3d = self.fig_3d.add_subplot(111, projection="3d")
        
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_3d, tab_3d).update()

        # Вкладка Телеметрия
        tab_telem = ttk.Frame(notebook)
        notebook.add(tab_telem, text=" Telemetry ")
        self.fig_telem = Figure(figsize=(8, 6), dpi=100)
        self.ax_dist = self.fig_telem.add_subplot(211)
        self.ax_height = self.fig_telem.add_subplot(212)
        self.canvas_telem = FigureCanvasTkAgg(self.fig_telem, master=tab_telem)
        self.canvas_telem.get_tk_widget().pack(fill="both", expand=True)

        # Вкладка Отчет
        tab_report = ttk.Frame(notebook)
        notebook.add(tab_report, text=" Report ")
        self.result_text = tk.Text(tab_report, wrap="word", font=("Consolas", 11), bg="#f8f8f8")
        self.result_text.pack(fill="both", expand=True, padx=8, pady=8)

    def _add_entry(self, parent, label, variable, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(parent, textvariable=variable, width=15).grid(row=row, column=1, sticky="ew", padx=5, pady=2)

    def on_model_change(self, event=None):
        try:
            self.model, self.model_path = load_model(self.model_var.get())
            self.status_label.config(text=f"Active Model: {self.model_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def fill_demo(self):
        self.start_x.set("0.0"); self.start_y.set("0.0"); self.start_z.set("5.0")
        self.target_x.set("3.0"); self.target_y.set("-2.0"); self.target_z.set("4.0")
        self.wind_x.set("-0.03"); self.wind_y.set("0.08"); self.wind_z.set("-0.02")
        self.obstacle_count.set("2")
        self.seed_var.set("")

    def fill_random(self):
        rng = np.random.default_rng()
        self.start_x.set(f"{rng.uniform(-2, 2):.2f}")
        self.start_y.set(f"{rng.uniform(-2, 2):.2f}")
        self.start_z.set(f"{rng.uniform(3.5, 6):.2f}")
        self.target_x.set(f"{rng.uniform(-4, 4):.2f}")
        self.target_y.set(f"{rng.uniform(-4, 4):.2f}")
        self.target_z.set(f"{rng.uniform(2, 6.5):.2f}")
        self.wind_x.set(f"{rng.uniform(-0.1, 0.1):.2f}")
        self.wind_y.set(f"{rng.uniform(-0.1, 0.1):.2f}")
        self.wind_z.set(f"{rng.uniform(-0.05, 0.02):.2f}")
        self.obstacle_count.set(str(int(rng.integers(1, 4))))
        self.seed_var.set(str(int(rng.integers(0, 100000))))

    def run_simulation(self):
        # 1. ЖЕСТКАЯ ОСТАНОВКА ПРОШЛОЙ АНИМАЦИИ
        if self.anim is not None:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self.anim = None

        try:
            start = np.array([float(self.start_x.get()), float(self.start_y.get()), float(self.start_z.get())])
            target = np.array([float(self.target_x.get()), float(self.target_y.get()), float(self.target_z.get())])
            wind = np.array([float(self.wind_x.get()), float(self.wind_y.get()), float(self.wind_z.get())])
            tol = float(self.goal_tolerance.get())
            seed = int(self.seed_var.get()) if self.seed_var.get().strip() else None
        except ValueError:
            messagebox.showerror("Error", "Invalid numeric input.")
            return

        self.status_box.config(text="Simulating...", fg="#2980b9", bg="#ecf0f1")
        self.root.update()

        env = QuadcopterEnv(
            enable_wind=self.enable_wind.get(),
            enable_obstacles=self.enable_obstacles.get(),
            num_obstacles_range=(int(self.obstacle_count.get()), int(self.obstacle_count.get())),
            max_steps=500, success_radius=tol, corridor_clearance=0.8
        )
        
        opts = {"start_pos": start.tolist(), "target_pos": target.tolist()}
        if self.enable_wind.get():
            opts["wind_base"] = wind.tolist()
            opts["wind_gust_std"] = float(self.wind_gust.get())

        obs, info = env.reset(seed=seed, options=opts)

        positions, angles, dists, heights = [], [], [], []
        
        while True:
            positions.append([env.drone.x, env.drone.y, env.drone.z])
            angles.append([env.drone.roll, env.drone.pitch, env.drone.yaw])
            pos = np.array(positions[-1])
            dists.append(float(np.linalg.norm(pos - target)))
            heights.append(float(env.drone.z))
            
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            
            if term or trunc:
                positions.append([env.drone.x, env.drone.y, env.drone.z])
                angles.append([env.drone.roll, env.drone.pitch, env.drone.yaw])
                dists.append(float(np.linalg.norm(np.array(positions[-1]) - target)))
                heights.append(float(env.drone.z))
                break

        self.last_result = {
            "positions": np.array(positions), "angles": np.array(angles),
            "dists": np.array(dists), "heights": np.array(heights),
            "start": start, "target": target, "dt": env.dt,
            "success": info.get("success", False), "collided": info.get("collided", False),
            "obstacles": env.obstacles, "wind": env.wind_base, "tol": tol
        }

        self._update_report(info)
        self._update_telemetry()
        self._start_live_animation()

    def _start_live_animation(self):
        r = self.last_result
        self.ax_3d.clear()
        
        pos, ang = r["positions"], r["angles"]
        target = r["target"]
        
        # Границы
        all_pts = np.vstack([pos, target])
        mins = all_pts.min(axis=0) - 1.0
        maxs = all_pts.max(axis=0) + 1.0
        self.ax_3d.set(xlim=[mins[0], maxs[0]], ylim=[mins[1], maxs[1]], zlim=[max(0, mins[2]), maxs[2]])
        self.ax_3d.set_xlabel("X (m)"); self.ax_3d.set_ylabel("Y (m)"); self.ax_3d.set_zlabel("Z (m)")

        # Сетка пола
        gx, gy = np.meshgrid(np.linspace(mins[0], maxs[0], 6), np.linspace(mins[1], maxs[1], 6))
        self.ax_3d.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.1, color="grey")

        # Цель и Сфера
        self.ax_3d.scatter(*target, color="red", s=100, marker="*")
        u, v = np.linspace(0, 2*np.pi, 15), np.linspace(0, np.pi, 15)
        bx = target[0] + r["tol"] * np.outer(np.cos(u), np.sin(v))
        by = target[1] + r["tol"] * np.outer(np.sin(u), np.sin(v))
        bz = target[2] + r["tol"] * np.outer(np.ones_like(u), np.cos(v))
        self.ax_3d.plot_wireframe(bx, by, bz, color="#2ecc71" if r["success"] else "gray", alpha=0.2, linewidth=0.5)

        # 2. ПОЛНОЦЕННЫЕ 3D КУБЫ И СФЕРЫ (ВСЕ 6 ГРАНЕЙ КУБА)
        for obs in r["obstacles"]:
            c, s = obs.center, obs.size
            if obs.shape == "sphere":
                sx = c[0] + s * np.outer(np.cos(u), np.sin(v))
                sy = c[1] + s * np.outer(np.sin(u), np.sin(v))
                sz = c[2] + s * np.outer(np.ones_like(u), np.cos(v))
                self.ax_3d.plot_surface(sx, sy, sz, alpha=0.3, color="#e67e22")
            else:
                x, y, z = [c[0]-s, c[0]+s], [c[1]-s, c[1]+s], [c[2]-s, c[2]+s]
                # Верх и Низ
                X, Y = np.meshgrid(x, y)
                self.ax_3d.plot_surface(X, Y, np.full_like(X, z[0]), color="#e67e22", alpha=0.25)
                self.ax_3d.plot_surface(X, Y, np.full_like(X, z[1]), color="#e67e22", alpha=0.25)
                # Перед и Зад
                X, Z = np.meshgrid(x, z)
                self.ax_3d.plot_surface(X, np.full_like(X, y[0]), Z, color="#e67e22", alpha=0.25)
                self.ax_3d.plot_surface(X, np.full_like(X, y[1]), Z, color="#e67e22", alpha=0.25)
                # Лево и Право
                Y, Z = np.meshgrid(y, z)
                self.ax_3d.plot_surface(np.full_like(Y, x[0]), Y, Z, color="#e67e22", alpha=0.25)
                self.ax_3d.plot_surface(np.full_like(Y, x[1]), Y, Z, color="#e67e22", alpha=0.25)

        # Линия следа и дрон
        self.trail_line, = self.ax_3d.plot([], [], [], color="#3498db", alpha=0.5, linewidth=1.5)
        self.drone_arms = [self.ax_3d.plot([], [], [], color=c, lw=3)[0] for c in ["red", "black", "red", "black"]]

        def update(frame):
            self.trail_line.set_data(pos[:frame+1, 0], pos[:frame+1, 1])
            self.trail_line.set_3d_properties(pos[:frame+1, 2])
            
            p = pos[frame]
            rl, pt, yw = ang[frame]
            cr, sr, cp, sp, cy, sy = np.cos(rl), np.sin(rl), np.cos(pt), np.sin(pt), np.cos(yw), np.sin(yw)
            
            R = np.array([
                [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                [-sp, cp*sr, cp*cr]
            ])
            
            arm = 0.2
            dirs = [np.array([arm,0,0]), np.array([0,arm,0]), np.array([-arm,0,0]), np.array([0,-arm,0])]
            
            for line, d in zip(self.drone_arms, dirs):
                end = p + R @ d
                line.set_data([p[0], end[0]], [p[1], end[1]])
                line.set_3d_properties([p[2], end[2]])
            
            self.status_box.config(text=f"Step: {frame}/{len(pos)-1}")
            return [self.trail_line] + self.drone_arms

        frames = list(range(0, len(pos), 2))
        if frames[-1] != len(pos)-1: frames.append(len(pos)-1)

        # Запускаем новую анимацию
        self.anim = FuncAnimation(self.fig_3d, update, frames=frames, interval=25, blit=False, repeat=False)
        self.canvas_3d.draw()

        def on_anim_end(*args):
            if r["success"]: self.status_box.config(text="MISSION SUCCESS", fg="white", bg="#27ae60")
            elif r["collided"]: self.status_box.config(text="COLLISION", fg="white", bg="#c0392b")
            else: self.status_box.config(text="TIMEOUT", fg="white", bg="#e67e22")
            
        self.anim._stop = on_anim_end

    def _update_telemetry(self):
        r = self.last_result
        t = np.arange(len(r["dists"])) * r["dt"]
        self.ax_dist.clear(); self.ax_height.clear()
        
        self.ax_dist.plot(t, r["dists"], color="#2980b9")
        self.ax_dist.axhline(r["tol"], color="#27ae60", ls="--")
        self.ax_dist.set(title="Distance to Target (m)", xlabel="Time (s)")
        self.ax_dist.grid(True, alpha=0.3)
        
        self.ax_height.plot(t, r["heights"], color="#8e44ad")
        self.ax_height.axhline(r["target"][2], color="#c0392b", ls="--")
        self.ax_height.set(title="Altitude (m)", xlabel="Time (s)")
        self.ax_height.grid(True, alpha=0.3)
        
        self.fig_telem.tight_layout()
        self.canvas_telem.draw()

    def _update_report(self, info):
        r = self.last_result
        res = "SUCCESS" if r["success"] else ("COLLISION" if r["collided"] else "TIMEOUT")
        text = f"--- FLIGHT REPORT ---\n\nStatus: {res}\n"
        text += f"Final Distance: {r['dists'][-1]:.3f} m\n"
        text += f"Target Radius: {r['tol']} m\n\n"
        text += f"Flight Time: {len(r['positions'])*r['dt']:.1f} s\n"
        
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = QuadcopterApp(root)
    root.mainloop()