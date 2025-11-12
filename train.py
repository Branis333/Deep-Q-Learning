import argparse
import os
import time
import gymnasium as gym
import ale_py  # ensure ALE namespace is registered
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View Atari Pong environment (no training).")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to render.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment.")
    parser.add_argument("--fps", type=float, default=60.0, help="Approximate render FPS throttle.")
    return parser.parse_args()


def main():
    args = parse_args()
    env_id = "ALE/Pong-v5"

    # Create a render-enabled env with frame stacking (same preprocessing as training)
    env = make_atari_env(env_id, n_envs=1, seed=args.seed, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)

    print(f"[INFO] Displaying {env_id} for {args.episodes} episode(s) with random actions.")
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            # Random action (VecEnv expects array-like)
            action = [env.action_space.sample()]
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])
            done = bool(dones[0])
            time.sleep(1.0 / max(args.fps, 1))
        print(f"Episode {ep+1} finished. Total random reward: {ep_reward:.2f}")

    env.close()
    print("[DONE] Finished viewing Pong.")


if __name__ == "__main__":
    main()

