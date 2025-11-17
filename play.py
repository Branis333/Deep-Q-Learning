# Load existing trained model (if present) and verify environment
# pip  install stable_baselines3
import os, time, sys
import argparse
import gymnasium as gym
import ale_py  
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.save_util import load_from_zip_file

ENV_ID = "ALE/Pong-v5"

try:
    env = make_atari_env(ENV_ID, n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)
except Exception as e:
    raise RuntimeError(
        f"Failed to build Atari VecEnv for {ENV_ID}. Ensure ale-py and ROMs are installed. Original error: {e}"
    )
env.close()

try:
    import numpy.core.numeric as _numeric
    if 'numpy._core.numeric' not in sys.modules:
        sys.modules['numpy._core.numeric'] = _numeric
except Exception:
    pass

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Play trained DQN model on Pong-v5')
parser.add_argument('--model', type=str, default=os.path.join('models', 'Branis_model', 'exp10_more_gradient_steps.zip'),
                    help='Path to trained model .zip file')
parser.add_argument('--episodes', type=int, default=1,
                    help='Number of episodes to play')
parser.add_argument('--save_video', type=str, default=None,
                    help='Path to save video file (e.g., videos/demo.mp4)')
parser.add_argument('--list', action='store_true',
                    help='List available models and exit')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')

args = parser.parse_args()

BEST_MODEL_PATH = args.model
MODEL_TO_PLAY = BEST_MODEL_PATH  
N_EPISODES = args.episodes
SEED = args.seed
SAVE_VIDEO = args.save_video

INFERENCE_BUFFER_SIZE = 1_000  


def list_models():
    """Recursively find all .zip model files"""
    models_found = []
    for root, dirs, files in os.walk('models'):
        for file in sorted(files):
            if file.endswith('.zip'):
                models_found.append(os.path.join(root, file))
    
    if not models_found:
        print('No model .zip files found in models/')
    else:
        print('Available models:')
        for model_path in models_found:
            print(' -', model_path)
    return models_found

if args.list:
    list_models()
    sys.exit(0)

if not os.path.isfile(MODEL_TO_PLAY):
    print('Model not found:', MODEL_TO_PLAY)
    list_models()
else:
    print(f'Loading model: {MODEL_TO_PLAY}')
    render_mode = 'human' if SAVE_VIDEO is None else 'rgb_array'
    env = make_atari_env(ENV_ID, n_envs=1, seed=SEED, env_kwargs={'render_mode': render_mode})
    env = VecFrameStack(env, n_stack=4)

    # Wrap with video recorder if saving video
    if SAVE_VIDEO:
        os.makedirs(os.path.dirname(SAVE_VIDEO) if os.path.dirname(SAVE_VIDEO) else '.', exist_ok=True)
        try:
            from gymnasium.wrappers import RecordVideo
            # Get the underlying environment to wrap with RecordVideo
            base_env = env.envs[0]
            base_env = RecordVideo(base_env, video_folder=os.path.dirname(SAVE_VIDEO) or '.', 
                                   name_prefix=os.path.basename(SAVE_VIDEO).replace('.mp4', ''),
                                   episode_trigger=lambda x: True)
            env.envs[0] = base_env
            print(f'Video will be saved to: {SAVE_VIDEO}')
        except ImportError:
            print('[WARN] RecordVideo not available; attempting to use opencv fallback...')
            try:
                import cv2
                video_writer = None
                frame_count = 0
            except ImportError:
                print('[ERROR] Neither RecordVideo nor opencv available for video saving')
                SAVE_VIDEO = None

    model = None
    try:
        model = DQN.load(
            MODEL_TO_PLAY,
            env=env,
            custom_objects={
                'buffer_size': INFERENCE_BUFFER_SIZE,
                'learning_starts': 0,
            },
        )
    except ModuleNotFoundError as e:
        print('[WARN] ModuleNotFoundError during load:', e)
        try:
            import numpy.core.numeric as _numeric
            sys.modules['numpy._core.numeric'] = _numeric
            print('[INFO] Patched numpy._core.numeric; retrying load...')
            model = DQN.load(
                MODEL_TO_PLAY,
                env=env,
                custom_objects={'buffer_size': INFERENCE_BUFFER_SIZE, 'learning_starts': 0},
            )
        except Exception as e2:
            print('[WARN] Patch retry failed:', e2)
    except (ValueError, MemoryError) as e:
        print('[WARN] Direct load failed (ValueError/MemoryError):', e)
    except Exception as e:
        print('[WARN] Unexpected error on direct load:', e)

    if model is None:
        print('[INFO] Falling back to parameter-only load (skipping metadata)...')
        try:
            _data, params, _vars = load_from_zip_file(MODEL_TO_PLAY, device='cpu', load_data=False, print_system_info=False)
            # Instantiating a fresh lightweight DQN for inference only (small buffer, no training planned)
            model = DQN(
                'CnnPolicy',
                env,
                seed=SEED,
                verbose=0,
                buffer_size=INFERENCE_BUFFER_SIZE,
                learning_starts=0,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.0,
                exploration_initial_eps=0.0,
                exploration_final_eps=0.0,
            )
            model.set_parameters(params, exact_match=False)
            print('[OK] Parameters loaded in fallback mode with tiny buffer_size =', INFERENCE_BUFFER_SIZE)
        except MemoryError as me:
            print('[FAIL] MemoryError even with reduced buffer_size:', me)
            print('Try lowering INFERENCE_BUFFER_SIZE further (e.g., 200) or close other apps to free RAM.')
            env.close()
            raise
        except Exception as e3:
            print('[FAIL] Fallback weight-only load failed:', e3)
            env.close()
            raise

    # Run deterministic episodes
    for ep in range(N_EPISODES):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_reward += float(rewards[0])
            done = bool(dones[0])
            time.sleep(1/60)  # ~60 FPS pacing
        print(f'Episode {ep+1} return: {ep_reward:.2f}')
    
    env.close()
    
    if SAVE_VIDEO:
        print(f'Video saved to: {SAVE_VIDEO}')

print('Done.')