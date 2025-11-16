# Cell 1: Load existing trained model (if present) and verify environment'
#pip  install stable_baselines3
import os, time, sys
import gymnasium as gym
import ale_py  # ensure ALE namespace is registered
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.save_util import load_from_zip_file

ENV_ID = "ALE/Pong-v5"

# Recreate the same preprocessing used for CNN training: make_atari_env + 4-frame stack
try:
    env = make_atari_env(ENV_ID, n_envs=1, seed=42)
    env = VecFrameStack(env, n_stack=4)
except Exception as e:
    raise RuntimeError(
        f"Failed to build Atari VecEnv for {ENV_ID}. Ensure ale-py and ROMs are installed. Original error: {e}"
    )
env.close()

# Attempt to patch older saved references expecting 'numpy._core.numeric'
try:
    import numpy.core.numeric as _numeric
    if 'numpy._core.numeric' not in sys.modules:
        sys.modules['numpy._core.numeric'] = _numeric
except Exception:
    pass

BEST_MODEL_PATH = os.path.join('models', 'Branis_model\\exp10_more_gradient_steps.zip')
MODEL_TO_PLAY = BEST_MODEL_PATH  # set to a specific experiment zip to test that model
N_EPISODES = 1
SEED = 42

# For pure inference we can use a small replay buffer to avoid huge RAM usage
INFERENCE_BUFFER_SIZE = 1_000  # reduce if memory is very tight (e.g., 200)


def list_models():
    if os.path.isdir('models'):
        zips = [f for f in sorted(os.listdir('models')) if f.endswith('.zip')]
        if not zips:
            print('No model .zip files found in models/')
        else:
            print('Available models:')
            for f in zips:
                print(' -', os.path.join('models', f))

if not os.path.isfile(MODEL_TO_PLAY):
    print('Model not found:', MODEL_TO_PLAY)
    list_models()
else:
    print(f'Loading model: {MODEL_TO_PLAY}')
    env = make_atari_env(ENV_ID, n_envs=1, seed=SEED, env_kwargs={'render_mode': 'human'})
    env = VecFrameStack(env, n_stack=4)

    model = None
    try:
        # First try to load while overriding large memory settings
        model = DQN.load(
            MODEL_TO_PLAY,
            env=env,
            custom_objects={
                'buffer_size': INFERENCE_BUFFER_SIZE,
                'learning_starts': 0,
                # keep other saved params; we only downsize memory-heavy ones
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
        # Handle numpy RNG pickle mismatch or large buffer memory error
        print('[WARN] Direct load failed (ValueError/MemoryError):', e)
    except Exception as e:
        print('[WARN] Unexpected error on direct load:', e)

    if model is None:
        print('[INFO] Falling back to parameter-only load (skipping metadata)...')
        try:
            # Avoid deserializing JSON data that can contain incompatible RNG objects
            _data, params, _vars = load_from_zip_file(MODEL_TO_PLAY, device='cpu', load_data=False, print_system_info=False)
            # Instantiate a fresh lightweight DQN for inference only (small buffer, no training planned)
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

print('Done.')