from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------
# Hyperparameter scaffold (for teammates)
# ---------------------------
@dataclass
class DQNHyperParams:
	learning_rate: float = 1e-4
	gamma: float = 0.99
	batch_size: int = 32
	buffer_size: int = 100_000
	train_freq: int = 4
	gradient_steps: int = 1
	target_update_interval: int = 10_000
	exploration_fraction: float = 0.1  # fraction of total timesteps over which epsilon decays
	exploration_initial_eps: float = 1.0
	exploration_final_eps: float = 0.01
	verbose: int = 1

	def to_kwargs(self) -> Dict:
		return {
			"learning_rate": self.learning_rate,
			"gamma": self.gamma,
			"batch_size": self.batch_size,
			"buffer_size": self.buffer_size,
			"train_freq": self.train_freq,
			"gradient_steps": self.gradient_steps,
			"target_update_interval": self.target_update_interval,
			"exploration_fraction": self.exploration_fraction,
			"exploration_initial_eps": self.exploration_initial_eps,
			"exploration_final_eps": self.exploration_final_eps,
			"verbose": self.verbose,
		}


# Placeholder grid for later tuning (add / modify rows as needed)
HYPERPARAM_EXPERIMENTS: List[DQNHyperParams] = [
	DQNHyperParams(),
	DQNHyperParams(learning_rate=5e-5, gamma=0.98, batch_size=64, exploration_fraction=0.2),
]


def print_hyperparam_table():
	"""Prints a simple table of planned hyperparameter sets (not executed)."""
	header = [
		"learning_rate",
		"gamma",
		"batch_size",
		"buffer_size",
		"train_freq",
		"target_update_interval",
		"exploration_initial_eps",
		"exploration_final_eps",
		"exploration_fraction",
	]
	print("\nPlanned Hyperparameter Sets (for future tuning):")
	print(" | ".join(header))
	for hp in HYPERPARAM_EXPERIMENTS:
		row = [str(getattr(hp, h)) for h in header]
		print(" | ".join(row))
	print()


# ---------------------------
# Observation wrapper for MLP policy (flattened, downsampled grayscale)
# ---------------------------
class MLPObservationWrapper(gym.ObservationWrapper):
	"""Convert 210x160x3 Pong frames to a reduced flattened grayscale vector.

	Downsampling: simple stride (2,2) -> (105,80), grayscale by channel mean, normalize 0-1.
	Resulting shape: (105*80,) = 8400 features.
	"""

	def __init__(self, env: gym.Env):
		super().__init__(env)
		self.observation_space = gym.spaces.Box(
			low=0.0, high=1.0, shape=(105 * 80,), dtype=np.float32
		)

	def observation(self, obs):  
		gray = obs.mean(axis=2)  
		downsampled = gray[::2, ::2]  
		norm = (downsampled / 255.0).astype(np.float32)
		return norm.flatten()


# ---------------------------
# Episode statistics callback
# ---------------------------
class EpisodeStatsCallback(BaseCallback):
	def __init__(self, log_path: str, verbose: int = 0):
		super().__init__(verbose)
		self.log_path = log_path
		self.episode_rewards: List[float] = []
		self.episode_lengths: List[int] = []
		self._episode_start_timestep = 0
		os.makedirs(os.path.dirname(log_path), exist_ok=True)

	def _on_step(self) -> bool:
		infos = self.locals.get("infos", [])
		for info in infos:
			if "episode" in info:  # VecEnv provides this when an episode ends
				ep_info = info["episode"]
				self.episode_rewards.append(ep_info.get("r", np.nan))
				self.episode_lengths.append(ep_info.get("l", np.nan))
		return True

	def _on_training_end(self) -> None:
		# Write CSV: episode, reward, length, timesteps
		with open(self.log_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["episode", "reward", "length", "timestep"])
			timestep = 0
			for i, (r, l) in enumerate(zip(self.episode_rewards, self.episode_lengths), start=1):
				timestep += l
				writer.writerow([i, r, l, timestep])
		if self.verbose:
			print(f"Episode stats saved to {self.log_path}")


# ---------------------------
# Environment factory
# ---------------------------
def create_env(policy_type: str, seed: int) -> VecEnv:
	"""Create a vectorized Pong environment adapted for the given policy type.

	CnnPolicy: uses make_atari_env + VecFrameStack (4 frames), grayscale & resized via wrappers.
	MLPPolicy: uses raw env with a custom observation wrapper then vectorized.
	"""
	game_id = "ALE/Pong-v5"
	# Proactive check for ALE availability
	try:
		import ale_py  # noqa: F401
	except Exception:
		raise RuntimeError(
			"ALE Atari environments not available. Install ale-py and ROMs:\n"
			"  pip install ale-py\n  AutoROM --accept-license"
		)
	if policy_type.lower() == "cnn":
		env = make_atari_env(game_id, n_envs=1, seed=seed)
		# Stack 4 frames (channel-first expected by SB3's CnnPolicy automatically)
		env = VecFrameStack(env, n_stack=4)
		return env
	elif policy_type.lower() == "mlp":
		def _make():
			base_env = gym.make(game_id, render_mode=None)
			wrapped = MLPObservationWrapper(base_env)
			return wrapped

		env = DummyVecEnv([_make])
		return env
	else:
		raise ValueError(f"Unsupported policy_type '{policy_type}'. Use 'cnn' or 'mlp'.")


# ---------------------------
# Training & evaluation
# ---------------------------
def train_dqn(
	policy_type: str,
	total_timesteps: int,
	seed: int,
	hyperparams: DQNHyperParams | None = None,
	eval_episodes: int = 0,
) -> Tuple[DQN, Dict[str, float]]:
	"""Train a DQN model for Pong with selected policy.

	Returns the model and a dict with evaluation metrics (if eval_episodes > 0).
	"""
	hyperparams = hyperparams or DQNHyperParams()
	env = create_env(policy_type, seed)

	policy = "CnnPolicy" if policy_type.lower() == "cnn" else "MlpPolicy"

	log_csv = os.path.join("logs", f"training_metrics_{policy_type.lower()}.csv")
	callback = EpisodeStatsCallback(log_csv, verbose=1)

	model = DQN(
		policy,
		env,
		seed=seed,
		tensorboard_log="logs/tensorboard",
		**hyperparams.to_kwargs(),
	)

	print(f"\n[TRAIN] Starting DQN training for policy={policy} timesteps={total_timesteps}")
	start = time.time()
	model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
	duration_min = (time.time() - start) / 60.0
	print(f"[TRAIN] Finished training ({duration_min:.2f} min). Episode stats recorded.")

	os.makedirs("models", exist_ok=True)
	model_path = os.path.join("models", f"dqn_pong_{policy_type.lower()}.zip")
	model.save(model_path)
	print(f"[SAVE] Model saved to {model_path}")

	metrics: Dict[str, float] = {"train_time_min": duration_min}
	if eval_episodes > 0:
		print(f"[EVAL] Evaluating policy over {eval_episodes} episodes...")
		mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
		metrics.update({"eval_mean_reward": mean_reward, "eval_std_reward": std_reward})
		print(f"[EVAL] Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
	env.close()
	return model, metrics


def compare_policies(total_timesteps: int, seed: int, eval_episodes: int) -> str:
	"""Train both CNN and MLP variants (short runs) and return best policy type."""
	results = {}
	for p in ["cnn", "mlp"]:
		model, metrics = train_dqn(p, total_timesteps, seed, eval_episodes=eval_episodes)
		results[p] = metrics.get("eval_mean_reward", float("-inf"))
	best = max(results, key=results.get)
	print(f"\n[COMPARE] Results: {results}. Best policy: {best}")
	# Copy best model to canonical filename dqn_model.zip
	src = os.path.join("models", f"dqn_pong_{best}.zip")
	dst = os.path.join("models", "dqn_model.zip")
	try:
		import shutil
		shutil.copyfile(src, dst)
		print(f"[COPY] Best model copied to {dst}")
	except Exception as e:
		print(f"[WARN] Could not copy best model: {e}")
	return best


# ---------------------------
# CLI
# ---------------------------
def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train DQN on Pong with CNN or MLP policy.")
	parser.add_argument("--policy", choices=["cnn", "mlp"], default="cnn", help="Policy type to train (default: cnn).")
	parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps.")
	parser.add_argument("--seed", type=int, default=42, help="Random seed.")
	parser.add_argument("--eval-episodes", type=int, default=5, help="Episodes for post-training eval.")
	parser.add_argument("--compare", action="store_true", help="Train & evaluate both policies.")
	parser.add_argument("--show-hparams", action="store_true", help="Print planned hyperparameter sets and exit.")
	return parser.parse_args()


def main():
	args = parse_args()
	if args.show_hparams:
		print_hyperparam_table()
		return

	if args.compare:
		print("[MODE] Comparison run between CNN and MLP policies.")
		best = compare_policies(args.timesteps, args.seed, args.eval_episodes)
		print(f"Best policy after comparison: {best}")
		return

	# Single policy training
	model, metrics = train_dqn(
		args.policy, args.timesteps, args.seed, eval_episodes=args.eval_episodes
	)
	# Copy to canonical name for downstream usage
	src = os.path.join("models", f"dqn_pong_{args.policy}.zip")
	dst = os.path.join("models", "dqn_model.zip")
	try:
		import shutil
		shutil.copyfile(src, dst)
		print(f"[COPY] Model copied to {dst}")
	except Exception as e:
		print(f"[WARN] Copy to {dst} failed: {e}")

	print("\n[SUMMARY]")
	print(f"Policy: {args.policy}")
	for k, v in metrics.items():
		print(f"  {k}: {v}")


if __name__ == "__main__":
	main()

