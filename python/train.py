import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from env import QuadcopterEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed


STAGES = [
    {
        "name": "Stage 1: Hover (no wind, no obstacles)",
        "steps": 200_000,
        "env_kwargs": dict(
            enable_wind=False,
            enable_obstacles=False,
            num_waypoints=1,
            max_steps=500,
            success_radius=0.7,
        ),
    },
    {
        "name": "Stage 2: Wind + obstacles",
        "steps": 300_000,
        "env_kwargs": dict(
            enable_wind=True,
            enable_obstacles=True,
            num_obstacles_range=(1, 2),
            num_waypoints=1,
            wind_base_max=0.1,
            wind_gust_max=0.02,
            max_steps=500,
            success_radius=0.55,
        ),
    },
    {
        "name": "Stage 3: Full random, push into goal",
        "steps": 300_000,
        "env_kwargs": dict(
            enable_wind=True,
            enable_obstacles=True,
            num_obstacles_range=(1, 3),
            num_waypoints=1,
            wind_base_max=0.15,
            wind_gust_max=0.05,
            max_steps=500,
            success_radius=0.45,
        ),
    },
]

N_ENVS = 4


def make_env(rank, env_kwargs, seed=0):
    def _init():
        env = QuadcopterEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


class LogCallback(BaseCallback):
    def __init__(self, freq=20_000):
        super().__init__()
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0 and self.model.ep_info_buffer:
            buf = self.model.ep_info_buffer
            mr = sum(e["r"] for e in buf) / len(buf)
            ml = sum(e["l"] for e in buf) / len(buf)
            print(f"  step {self.num_timesteps:>9d} | reward={mr:+8.2f} | ep_len={ml:.0f}")
        return True


def train():
    model = None

    for stage_idx, stage in enumerate(STAGES):
        print(f"\n{'='*60}")
        print(f"  {stage['name']}")
        print(f"  timesteps: {stage['steps']:,}")
        print(f"{'='*60}\n")

        env_kwargs = stage["env_kwargs"]

        vec_env = SubprocVecEnv([make_env(i, env_kwargs) for i in range(N_ENVS)])
        vec_env = VecMonitor(vec_env)

        eval_env = SubprocVecEnv([make_env(100, env_kwargs)])
        eval_env = VecMonitor(eval_env)

        if model is None:
            model = PPO(
                "MlpPolicy", vec_env,
                n_steps=2048,
                batch_size=128,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                verbose=1,
            )
        else:
            model.set_env(vec_env)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=f"./best_stage{stage_idx}/",
            eval_freq=50_000 // N_ENVS,
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        )

        model.learn(
            total_timesteps=stage["steps"],
            callback=[LogCallback(), eval_cb],
            reset_num_timesteps=False,
        )

        model.save(f"ppo_stage{stage_idx}")
        print(f"  Saved: ppo_stage{stage_idx}.zip")

        vec_env.close()
        eval_env.close()

    model.save("ppo_finetuned_goal")
    print(f"\nFinal model saved: ppo_finetuned_goal.zip")


if __name__ == "__main__":
    train()