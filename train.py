#!/usr/bin/env python
# coding: utf-8

"""
Cleaned and fully runnable training script for DQN (CnnPolicy) on Pong-v5.
Runs experiment(s), saves each model, logs results, and plots metrics.
Best model is saved using the experiment's name.
"""

import os
import time
import datetime
import csv
import gc
from typing import Dict, Tuple

import gymnasium as gym
import ale_py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------
#  ENVIRONMENT
# ---------------------------------------------------------------------
ENV_ID = "ALE/Pong-v5"

def make_cnn_env(seed: int):
    env = make_atari_env(ENV_ID, n_envs=1, seed=seed)
    env = VecFrameStack(env, n_stack=4)
    return env


# ---------------------------------------------------------------------
#  CALLBACK LOGGER
# ---------------------------------------------------------------------
class EpisodeCSVLogger(BaseCallback):
    def __init__(self, run_name: str, csv_path: str, log_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.run_name = run_name
        self.csv_path = csv_path
        self.log_path = log_path
        self.rows = []

    @staticmethod
    def _ts():
        return datetime.datetime.now().strftime("%H:%M:%S")

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(f"[{self._ts()}] [RUN {self.run_name}] logging started\n")

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep = info["episode"]
                ep_len = ep["l"]
                ep_rew = ep["r"]
                self.rows.append((self.num_timesteps, ep_len, ep_rew))

                eps = getattr(self.model, "exploration_rate", None)
                line = (
                    f"[{self._ts()}] [RUN {self.run_name}] "
                    f"t={self.num_timesteps} | len={ep_len} | reward={ep_rew:.2f}"
                )
                if eps is not None:
                    line += f" | eps={eps:.4f}"

                print(line)
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")
        return True

    def _on_training_end(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "ep_length", "ep_reward"])
            writer.writerows(self.rows)

        print(f"Saved episode CSV to {self.csv_path}")
        with open(self.log_path, "a") as f:
            f.write(f"[{self._ts()}] [RUN {self.run_name}] episode CSV saved\n")


# ---------------------------------------------------------------------
#  TRAINING FUNCTION
# ---------------------------------------------------------------------
def train_experiment(name: str, hp: Dict, total_timesteps: int, seed: int, eval_episodes: int):
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env = make_cnn_env(seed)
    csv_log = f"logs/training_metrics_{name}.csv"
    run_log = f"logs/run_{name}.log"
    callback = EpisodeCSVLogger(name, csv_log, run_log)

    model = DQN(
        "CnnPolicy",
        env,
        seed=seed,
        tensorboard_log=f"logs/tensorboard/{name}",
        optimize_memory_usage=False,
        **hp,
    )

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] START {name} | hp={hp}")

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    minutes = (time.time() - t0) / 60

    mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    env.close()

    print(f"[END] {name}: reward={mean_r:.2f} ± {std_r:.2f} | time={minutes:.2f}m")

    metrics = {
        "name": name,
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
        "train_minutes": minutes,
        **hp,
    }
    return metrics, model


# ---------------------------------------------------------------------
#  EXPERIMENT CONFIG
# ---------------------------------------------------------------------
experiments = [
    {
        "name": "exp10_more_gradient_steps",
        "hp": dict(
            learning_rate=1e-4,
            gamma=0.99,
            batch_size=32,
            buffer_size=100_000,
            train_freq=4,
            gradient_steps=4,
            target_update_interval=10_000,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.01,
            verbose=0,
        ),
    },
]


# ---------------------------------------------------------------------
#  MAIN TRAIN + SAVE + PLOT
# ---------------------------------------------------------------------
def main():
    TOTAL_TIMESTEPS = 1_000
    SEED = 42
    EVAL_EPISODES = 3

    results = []
    best_mean = None
    best_model_path = None

    print("=== Running Experiments ===")

    for exp in experiments:
        name = exp["name"]
        hp = exp["hp"]

        try:
            metrics, model = train_experiment(
                name=name,
                hp=hp,
                total_timesteps=TOTAL_TIMESTEPS,
                seed=SEED,
                eval_episodes=EVAL_EPISODES,
            )

            # Save model for the experiment
            model_path = f"models/{name}.zip"
            model.save(model_path)
            print(f"[SAVE] {model_path}")

            # Save best model using experiment name
            if best_mean is None or metrics["mean_reward"] > best_mean:
                best_mean = metrics["mean_reward"]
                best_model_path = f"models/{name}_best.zip"
                model.save(best_model_path)
                print(f"[BEST] Updated best model -> {best_model_path}")

            metrics["model_path"] = model_path
            results.append(metrics)

        except MemoryError:
            print(f"[ERROR] {name} ran out of memory")

        finally:
            del model
            gc.collect()

    # Build results table
    df = pd.DataFrame(results).sort_values("mean_reward", ascending=False)

    # Use the best experiment's name for the CSV file
    best_exp_name = df.iloc[0]["name"] if not df.empty else "results"
    csv_file = f"logs/{best_exp_name}_results.csv"

    df.to_csv(csv_file, index=False)
    print(f"Saved results -> {csv_file}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(df["name"], df["mean_reward"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Reward")
    plt.title("DQN Mean Reward per Experiment")
    plt.tight_layout()
    plt.savefig("logs/plot_rewards.png")
    print("Saved plot: logs/plot_rewards.png")
    plt.show()

    print("\n=== DONE ===")
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
