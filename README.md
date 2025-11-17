# Deep Q-Learning (DQN) on Atari Pong-v5
**Formative 3 Assignment: Deep Reinforcement Learning**

**Group 3 Members:**
- Branis Sumba
- Excel Asaph
- Ganza Owen Yhaan
- Roxanne Niteka

---

## Overview

This project implements a Deep Q-Network (DQN) agent trained to play the Atari Pong-v5 environment using Stable Baselines3 and Gymnasium. This requires training and comparing multiple DQN configurations with different hyperparameters and policy architectures (CNNPolicy vs MLPPolicy).

**Key Deliverables:**
- `train.py`: Training script that runs DQN experiments and saves trained models
- `play.py`: Inference script that loads trained models and demonstrates agent gameplay

---

## Project Structure

```
Deep-Q-Learning/
├── README.md                          
├── requirements.txt                   
├── train.py                           # Training script
├── play.py                            # Inference/gameplay script
├── notebooks/
│   ├── branis.ipynb                   
│   ├── excel1.ipynb                  
│   └── excel2.ipynb                   
├── models/
│   ├── Branis_model/                  
│   ├── Excel_model/                   
│   └── Owen_model/                    
│   └── Roxanne_model/                 
├── logs/
│   ├── training-metrics/              # Episode-level training logs (CSV)
│   ├── Excel_training_metrics/
│   ├── Owen_training_metrics/         
│   ├── Roxanne_training_metrics/      
│   └── tensorboard/                   # TensorBoard event files
└── videos/                            # Demo videos of trained agents
```

---

## Environment & Setup

### Atari Environment
- **Environment ID:** `ALE/Pong-v5`
- **Description:** Classic Atari Pong game where the agent learns to play as a paddle opponent
- **Action Space:** 6 discrete actions (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
- **Observation Space:** 84×84 grayscale images (stacked 4-frame history)
- **Reward Structure:** +1 for scoring, -1 for each step taken

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# On some systems, may need to install ROMs separately
AutoROM --accept-license
```

### Key Libraries
- **stable-baselines3** (≥2.3.2): DQN algorithm and policy implementations
- **gymnasium[atari]** (≥1.0.0): Atari environment wrapper
- **ale-py** (≥0.10.1): Atari Learning Environment interface
- **torch**: Neural network backend
- **pandas, matplotlib**: Data analysis and visualization

---

## Group Member Contributions

### 1. Branis Sumba

**Experiments (10 configurations):**

| # | Configuration | Learning Rate | Gamma | Batch Size | Epsilon Decay | Timesteps | Mean Reward | Noted Behavior |
|---|---------------|---------------|-------|------------|---------------|-----------|-------------|---|
| 1 | Baseline | 1e-4 | 0.99 | 32 | 0.10 | 50,000 | -20.33 | Standard epsilon decay; limited exploration time |
| 2 | Small LR Larger Batch | 5e-5 | 0.99 | 54 | 0.15 | 50,000 | -20.67 | Conservative learning + larger batch; performance degradation |
| 3 | Higher Gamma | 1.5e-4 | 0.995 | 32 | 0.10 | 50,000 | -21.00 | Higher gamma led to over-optimism; slower effective learning |
| 4 | Fast Epsilon Decay Extended | 1e-4 | 0.99 | 32 | 0.05 | 500,000 | -11.33 | Faster epsilon decay over extended training; significant improvement |
| 5 | Higher LR | 2e-4 | 0.99 | 32 | 0.12 | 50,000 | -20.67 | Higher learning rate did not improve outcomes; stays near baseline |
| 6 | Faster Updates | 1e-4 | 0.99 | 32 | 0.20 | 50,000 | -20.00 | Faster epsilon updates; marginal stability gains only |
| 7 | Large Batch High LR | 3e-4 | 0.99 | 64 | 0.18 | 50,000 | -20.00 | Combined changes; no improvement over baseline |
| 8 | Very High Gamma | 1e-4 | 0.997 | 32 | 0.10 | 150,000 | -19.33 | Very high gamma with extended training; modest improvement |
| 9 | Small Batch Extended | 1e-4 | 0.99 | 16 | 0.12 | 700,000 | 4.66 | Small batch + very long training → positive reward (but unstable in test) |
| 10 | More Gradient Steps Extended | 1e-4 | 0.99 | 32 | 0.10 | 500,000 | **6.7** | ✓ BEST: Gradient accumulation + extended training → strong positive reward |

**Best Model:** `exp10_more_gradient_steps` (saved as `models/Branis_model/exp10_more_gradient_steps.zip`) with mean_reward = **6.7**

**Files:**
- `notebooks/branis.ipynb`: Full training pipeline with callbacks and visualization
- `logs/branis_models.csv`: Summary results table

---

### 2. Excel Asaph 

**Experiments (10 configurations):**

| # | Configuration | Policy | Learning Rate | Gamma | Batch Size | Buffer Size | Train Freq | Gradient Steps | Target Update | Noted Behavior |
|---|---------------|--------|---------------|-------|------------|-------------|------------|----------------|----------------|---|
| 1 | Baseline | CNN | 1e-4 | 0.99 | 32 | 100k | 4 | 1 | 10k | Stable baseline; mean_reward = **-10.85** ✓ BEST |
| 2 | Large Batch | CNN | 7e-5 | 0.99 | 64 | 200k | 4 | 1 | 8k | Large batch improves stability; mean_reward = -12.38 |
| 3 | Freq1 Small Batch | CNN | 1e-4 | 0.99 | 16 | 100k | 1 | 1 | 5k | Frequent updates accelerate learning; mean_reward = -11.78 |
| 4 | More Gradient Steps | CNN | 8e-5 | 0.99 | 32 | 150k | 4 | 4 | 8k | Multiple gradient steps stabilize but slow convergence; mean_reward = -12.48 |
| 5 | High Gamma | CNN | 1e-4 | 0.997 | 32 | 120k | 4 | 1 | 7k | Higher discount factor improves long-term planning; mean_reward = -11.52 |
| 6 | Small Buffer Fast Target | CNN | 1.2e-4 | 0.99 | 32 | 50k | 4 | 1 | 4k | Small buffer ↓ memory; fast updates ↑ variance; mean_reward = -13.21 |
| 7 | MLP Small | MLP | 5e-4 | 0.99 | 64 | 100k | 4 | 1 | 10k | MLPPolicy struggles with high-dim input; mean_reward = -14.05 |
| 8 | MLP Deep | MLP | 3e-4 | 0.99 | 64 | 150k | 4 | 2 | 8k | Deeper MLP slightly better but still ≈ 2-3 points worse than CNN; mean_reward = -13.48 |
| 9 | Quick Decay | CNN | 1e-4 | 0.99 | 32 | 120k | 4 | 1 | 8k | Fast ε-decay forces exploitation too early; mean_reward = -12.81 |
| 10 | Slow LR + Clip | CNN | 5e-5 | 0.99 | 32 | 150k | 4 | 1 | 8k | Conservative approach with gradient clipping; mean_reward = -11.28 |

**Best Model:** `excel_exp1_baseline` (saved as `models/Excel_model/excel_best_dqn.zip`)

**Files:**
- `notebooks/excel1.ipynb`: Local training experiments (Windows)
- `notebooks/excel2.ipynb`: Google Colab extended training (1.5M timesteps)
- `logs/training-metrics/Excel_training_metrics/excel_models.csv`: Results summary
- `logs/training-metrics/Excel_training_metrics/training_metrics_excel_exp[1-10].csv`: Per-experiment logs

---

### 3. Ganza Owen Yhaan 

**Experiments (10 configurations):**

| # | Configuration | Learning Rate | Gamma | Batch Size | Buffer Size | Train Freq | Gradient Steps | Target Update | Mean Reward | Noted Behavior |
|---|---------------|---------------|-------|------------|-------------|------------|----------------|----------------|-------------|---|
| 1 | Multi Gradient Steps | 1e-4 | 0.99 | 32 | 100k | 4 | 8 | 10k | **-6.00** | BEST: Multiple gradient steps improve stability and convergence |
| 2 | Low Gamma Fast Updates | 2e-4 | 0.97 | 32 | 100k | 2 | 1 | 4k | -7.33 | Lower gamma discounts future less; frequent training updates |
| 3 | Large Batch Moderate LR | 2e-4 | 0.99 | 128 | 150k | 4 | 1 | 10k | -14.33 | Large batch smooths gradients; moderate learning rate |
| 4 | High Gamma Slow Decay | 1e-4 | 0.999 | 32 | 200k | 4 | 1 | 12k | -16.00 | High gamma emphasizes long-term planning; slow exploration decay |
| 5 | Train Every Step | 1e-4 | 0.99 | 16 | 80k | 1 | 1 | 10k | -16.67 | Training every step maximizes update frequency |
| 6 | Aggressive Decay | 1e-4 | 0.99 | 32 | 100k | 4 | 1 | 10k | -18.33 | Aggressive exploration decay forces fast exploitation |
| 7 | Slow Stable | 5e-5 | 0.995 | 64 | 250k | 4 | 1 | 15k | -20.00 | Conservative setup; slower convergence but very stable |
| 8 | New Baseline | 1e-4 | 0.99 | 32 | 100k | 4 | 1 | 8k | -20.67 | New baseline; standard hyperparameters |
| 9 | Fast Target High Gamma | 1e-4 | 0.997 | 32 | 120k | 4 | 1 | 3k | -21.00 | Very fast target updates destabilize learning; high gamma |
| 10 | Fast Adaptation | 4e-4 | 0.98 | 32 | 50k | 4 | 1 | 5k | -21.00 | Aggressive learning rate with small buffer; high variance |

**Best Model:** `exp8_multi_gradient_steps` (saved as `models/Owen_model/exp8_multi_gradient_steps_best.zip`) with mean_reward = **-6.00**

**Files:**
- `notebooks/owen.ipynb`: Full training pipeline with gradient step variations
- `logs/Owen_training_metrics/owen_models.csv`: Summary results table
- `logs/Owen_training_metrics/exp[1-10]_*.csv`: Per-experiment episode-level logs

---

### 4. Roxanne Niteka 

**Experiments (10 configurations):**

| # | Configuration | Policy | Learning Rate | Gamma | Batch Size | Buffer Size | Train Freq | Gradient Steps | Target Update | Noted Behavior |
|---|---------------|--------|---------------|-------|------------|-------------|------------|----------------|----------------|---|
| 1 | [Exp Name] | [CNN/MLP] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [To be documented] |
| 2 | [Exp Name] | [CNN/MLP] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [To be documented] |
| 3-10 | ... | ... | ... | ... | ... | ... | ... | ... | ... | [To be documented] |

**Best Model:** `models/Roxanne_model/roxanne_best_dqn.zip`

**Files:**
- `notebooks/roxanne.ipynb`: [To be created]
- `logs/Roxanne_training_metrics/roxanne_models.csv`: [To be created]

---

## Training Pipeline

### train.py Usage

```bash
# Train with default settings
python train.py

# Train with custom hyperparameters (example)
python train.py --learning_rate 1e-4 --gamma 0.99 --batch_size 32 --buffer_size 100000
```

**Output Files:**
- `models/dqn_exp_{n}.zip`: Individual experiment models
- `logs/training_metrics_{exp_name}.csv`: Episode-by-episode metrics
- `logs/{member}_models.csv`: Summary table (mean_reward, std_reward, hyperparams)

---

## Inference Pipeline

### play.py Usage

```bash
# Play with best model (Branis's baseline) with real-time rendering
python play.py

# Play specific experiment model for multiple episodes
python play.py --model models/Branis_model/exp10_more_gradient_steps.zip --episodes 5

# Save gameplay as video file
python play.py --model models/Owen_model/exp8_multi_gradient_steps_best.zip --episodes 10 --save_video videos/owen_demo.mp4

# Display available models
python play.py --list

# Display help
python play.py --help
```

**Command-line Arguments:**
- `--model PATH`: Path to trained model .zip file (default: `models/Branis_model/exp10_more_gradient_steps.zip`)
- `--episodes N`: Number of episodes to play (default: 1)
- `--save_video PATH`: Path to save video file (e.g., `videos/demo.mp4`); if not specified, renders to screen
- `--list`: List all available model files and exit
- `--seed N`: Random seed for reproducibility (default: 42)

**Output:**
- Real-time Pong gameplay visualization (when not saving video)
- Episode return (total discounted reward) printed to console for each episode
- Video file saved to specified location (when `--save_video` is provided)

---

## Performance Summary & Hyperparameter Analysis

### All Experiments Ranked by Mean Reward

| Rank | Member | Experiment | Mean Reward | Timesteps | Learning Rate | Gamma | Batch Size | Epsilon Decay | Key Observation |
|------|--------|-----------|-------------|-----------|---------------|-------|------------|---------------|---|
| 🥇 1 | Branis | exp10_more_gradient_steps | **6.7** | 500,000 | 1e-4 | 0.99 | 32 | 0.10 | Best overall; extended training + proper epsilon decay |
| 🥈 2 | Branis | exp9_small_batch_extended | 4.66 | 700,000 | 1e-4 | 0.99 | 16 | 0.12 | Very long training yields positive return but unstable |
| 🥉 3 | Owen | exp8_multi_gradient_steps | -6.00 | 325 min | 1e-4 | 0.99 | 32 | 8 | Multi-gradient optimization outperforms single-step |
| 4 | Owen | exp10_low_gamma_fast_updates | -7.33 | 87 min | 2e-4 | 0.97 | 32 | 1 | Lower gamma enables faster exploitation |
| 5 | Excel | exp1_baseline | -10.85 | 2,041 | 1e-4 | 0.99 | 32 | N/A | Stable CNN baseline; best among shorter runs |
| 6 | Excel | exp10_slow_lr_clip | -11.28 | 2,000 | 5e-5 | 0.99 | 32 | N/A | Conservative LR + gradient clipping |
| 7 | Excel | exp5_high_gamma | -11.52 | 2,000 | 1e-4 | 0.997 | 32 | N/A | Higher discount factor improves planning |
| 8 | Excel | exp3_freq1_small_batch | -11.78 | 2,000 | 1e-4 | 0.99 | 16 | N/A | Frequent updates help; train time increases |
| 9 | Branis | exp1_baseline | -20.33 | 50,000 | 1e-4 | 0.99 | 32 | 0.10 | Short training; baseline for comparison |
| 10 | Branis | exp8_very_high_gamma | -19.33 | 150,000 | 1e-4 | 0.997 | 32 | 0.10 | Very high gamma; modest improvement over short runs |
| 11 | Excel | exp2_large_batch | -12.38 | 2,000 | 7e-5 | 0.99 | 64 | N/A | Larger batch = smoother but slower |
| 12 | Excel | exp4_more_gradsteps | -12.48 | 2,000 | 8e-5 | 0.99 | 32 | N/A | Multiple gradient steps: stable but slower |
| 13 | Excel | exp9_quick_decay | -12.81 | 2,000 | 1e-4 | 0.99 | 32 | N/A | Fast ε-decay forces premature exploitation |
| 14 | Excel | exp6_small_buffer | -13.21 | 2,000 | 1.2e-4 | 0.99 | 32 | N/A | Small buffer ↑ variance |
| 15 | Owen | exp6_large_batch_moderate_lr | -14.33 | 87 min | 2e-4 | 0.99 | 128 | 1 | Large batch smooths gradients |
| 16 | Owen | exp9_high_gamma_slow_decay | -16.00 | 84 min | 1e-4 | 0.999 | 32 | 1 | High gamma emphasizes long-term planning |
| 17 | Owen | exp7_train_every_step | -16.67 | 131 min | 1e-4 | 0.99 | 16 | 1 | Training every step causes instability |
| 18 | Owen | exp5_aggressive_decay | -18.33 | 26 min | 1e-4 | 0.99 | 32 | 1 | Aggressive decay forces premature exploitation |
| 19 | Owen | exp3_slow_stable | -20.00 | 18 min | 5e-5 | 0.995 | 64 | 1 | Conservative setup; stable but slow |
| 20 | Owen | exp1_new_baseline | -20.67 | 15 min | 1e-4 | 0.99 | 32 | 1 | New baseline configuration |
| 21 | Owen | exp4_fast_target_high_gamma | -21.00 | 20 min | 1e-4 | 0.997 | 32 | 1 | Very fast target updates destabilize |
| 22 | Owen | exp2_fast_adaptation | -21.00 | 9 min | 4e-4 | 0.98 | 32 | 1 | Aggressive learning rate high variance |

---

## Video Demonstration

### Branis's Best Model (exp10_more_gradient_steps)
**Mean Reward: 6.7 (BEST OVERALL)** | Extended training (500k timesteps)

**Episode 7 Return: 15.00** (Best performing episode)

[▶️ Watch Branis Demo Video](videos/branis_demo-episode-6.mp4)

---

### Owen's Best Model (exp8_multi_gradient_steps)
**Mean Reward: -6.0 (2nd best)** | Multi-gradient optimization

**Episode 1 Return: 1.00** (Best performing episode)

[▶️ Watch Owen Demo Video](videos/owen_demo-episode-0.mp4)

---

### Excel's Best Model (excel_best_dqn)
**Mean Reward: -10.85 (3rd best)** | CNN baseline with stability tuning

**Episode 3 Return: 15.00** (Best performing episode)

[▶️ Watch Excel Demo Video](videos/excel_demo-episode-2.mp4)

---

### Roxanne's Best Model
*To be recorded*

---

**To generate videos:**
```bash
# Record Branis's best model
python play.py --model models/Branis_model/exp10_more_gradient_steps.zip --episodes 10 --save_video videos/branis_demo.mp4

# Record Owen's best model
python play.py --model models/Owen_model/exp8_multi_gradient_steps_best.zip --episodes 10 --save_video videos/owen_demo.mp4

# Record Excel's best model
python play.py --model models/Excel_model/excel_best_dqn.zip --episodes 10 --save_video videos/excel_demo.mp4
```
---

## Challenges & Lessons Learned

### 1. Memory Management
- **Issue:** Large replay buffers (200k+) consume >4GB RAM
- **Solution:** Use `buffer_size=100k` and `optimize_memory_usage=False` for inference
- **Lesson:** Trade-off between sample efficiency and computational resources

### 2. Numpy Compatibility
- **Issue:** Older trained models reference deprecated numpy modules (`numpy._core.numeric`)
- **Solution:** Patch sys.modules in play.py; load parameters separately if needed
- **Lesson:** Model compatibility across library versions is important

### 3. Sparse Rewards
- **Issue:** Pong provides -1 per step; only +1 on score (rare event)
- **Solution:** Careful exploration schedule essential; CNN policy helps extract features
- **Lesson:** Reward shaping or curriculum learning needed for very sparse environments

### 4. Policy Architecture Selection
- **Issue:** Chose MLPPolicy without considering image input dimensionality
- **Solution:** Use CNN for image inputs; MLP for low-dim state spaces
- **Lesson:** Architecture choice critically impacts sample efficiency

---

## How to Run the Full Pipeline

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/Branis333/Deep-Q-Learning.git
cd Deep-Q-Learning
pip install -r requirements.txt
AutoROM --accept-license

# 2. Train all experiments (takes ~6-8 hours total)
python train.py

# 3. Evaluate best model
python play.py --model models/Excel_model/excel_best_dqn.zip --episodes 5

# 4. Analyze results
python -c "import pandas as pd; print(pd.read_csv('logs/excel_models.csv').sort_values('mean_reward'))"
```

---

