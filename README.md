# Deep Q-Learning (DQN) on Atari Pong-v5
**Formative 3 Assignment: Deep Reinforcement Learning**

**Group 3 Members:**
- Branis Sumba
- Excel Asaph
- Ganza Owen Yhaan
- Roxanne Niteka

**Best Model Playing Video**

[▶️ Watch Branis Demo Video](https://drive.google.com/file/d/1MsPk1w5dAUm3NkBr0MdBTS0qG_NyGfPe/view)

![Branis Gameplay Demo](videos/branis_demo-episode-12.gif)

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

| Name | Hyperparameter set | Time steps | Noted Behavior | Mean Reward |
|------|-------------------|-----------|---|---|
| exp1 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 50,000 | Standard epsilon decay; limited exploration time | -20.33 |
| exp2 | lr=5e-5, gamma=0.99, batch=54, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.15 | 50,000 | Increasing batch size and lowering the learning rate slightly worsened performance | -20.67 |
| exp3 | lr=1.5e-4, gamma=0.995, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 50,000 | A higher gamma led to even lower returns, suggesting slower learning and over-optimism | -21.00 |
| exp4 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.05 | 500,000 | Faster epsilon decay over many steps significantly improved performance | -11.33 |
| exp5 | lr=2e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.12 | 50,000 | A higher learning rate with the same setup did not improve reward outcomes | -20.67 |
| exp6 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.2 | 50,000 | Faster updates with a small buffer slightly improved stability but not overall performance | -20.00 |
| exp7 | lr=3e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.18 | 50,000 | A large batch with a higher learning rate produced average but not improved rewards | -20.00 |
| exp8 | lr=1e-4, gamma=0.997, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 150,000 | Very high gamma slightly improved reward but still kept performance low | -19.33 |
| exp9 | lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.12 | 700,000 | Small batch size with long training time yielded strong positive reward. And yet it failed in the test | 4.66 |
| exp10 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 500,000 | More gradient updates helped the model reach a positive reward | **6.7** |

**Best Model:** `exp10_more_gradient_steps` (saved as `models/Branis_model/exp10_more_gradient_steps.zip`) with mean_reward = **6.7 (BEST OVERALL)**

**Files:**
- `notebooks/branis.ipynb`: Full training pipeline with callbacks and visualization
- `logs/branis_models.csv`: Summary results table

**Video Demonstration:**

**Episode 7 Return: 15.00** (Best performing episode)

[▶️ Watch Branis Demo Video](https://drive.google.com/file/d/1MsPk1w5dAUm3NkBr0MdBTS0qG_NyGfPe/view)

![Branis Gameplay Demo](videos/branis_demo-episode-12.gif)

---

### 2. Excel Asaph 

**Experiments (10 configurations):**

| Name | Hyperparameter set | Time steps | Noted Behavior | Mean Reward |
|------|-------------------|-----------|---|---|
| exp1 | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 1,500,000 | Extended training (1.5M steps) with CNN baseline yielded stable convergence. Best among all experiments | **-12.24** |
| exp2 | lr=7e-05, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.12 | 1,500,000 | Doubled batch size and larger buffer with lower learning rate slightly worsened performance | -12.38 |
| exp3 | lr=0.0001, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.12 | 1,500,000 | Frequent training (train_freq=1) with small batch caused severe instability; worst CNN result | -13.78 |
| exp4 | lr=8e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 1,500,000 | Four gradient steps per update stabilized training but converged more slowly than baseline | -12.48 |
| exp5 | lr=0.0001, gamma=0.997, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 1,000,000 | Higher gamma (0.99→0.997) provided only marginal improvement | -12.52 |
| exp6 | lr=0.00012, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.2 | 1,000,000 | Small buffer with aggressive exploration decay increased instability; fast epsilon decay forced premature exploitation | -13.21 |
| exp7 | lr=0.0005, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.15 | 1,000,000 | MLP policy significantly underperformed CNN | -14.05 |
| exp8 | lr=0.0003, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.15 | 500,000 | Deeper MLP helped slightly but remains fundamentally limited for high-dimensional image inputs | -13.48 |
| exp9 | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.03 | 500,000 | Very fast exploration decay forced premature exploitation | -12.81 |
| exp10 | lr=5e-05, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.12 | 500,000 | Conservative low learning rate with gradient clipping provided marginal benefit; insufficient training steps hindered performance | -13.28 |

**Best Model:** `excel_exp1_baseline` (saved as `models/Excel_model/excel_best_dqn.zip`) with mean_reward = **-12.24**

**Files:**
- `notebooks/excel1.ipynb`: Local training experiments (Windows)
- `notebooks/excel2.ipynb`: Google Colab extended training (1.5M timesteps)
- `logs/training-metrics/Excel_training_metrics/excel_models.csv`: Results summary
- `logs/training-metrics/Excel_training_metrics/training_metrics_excel_exp[1-10].csv`: Per-experiment logs

**Video Demonstration:**

**Episode 3 Return: 15.00** (Best performing episode)

[▶️ Watch Excel Demo Video](videos/excel_demo-episode-4.mp4)

![Excel Gameplay Demo](videos/excel_demo-episode-4.gif)

---

### 3. Ganza Owen Yhaan 

**Experiments (10 configurations):**

| Name | Hyperparameter set | Time steps | Noted Behavior |
|------|-------------------|-----------|---|
| exp1 | lr=0.000125, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.1 | 50,000 | Mean reward = -20.67. Baseline at only 50k steps. Agent barely moves paddle and loses every game quickly. |
| exp2 | lr=0.0004, gamma=0.98, batch=32, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.15 | 50,000 | Mean reward = -21.0. Worst result. High learning rate and low gamma made training completely unstable. |
| exp3 | lr=5e-05, gamma=0.995, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 50,000 | Mean reward = -20.0. Very low learning rate caused almost no learning in just 50k steps. |
| exp4 | lr=0.00015, gamma=0.997, batch=32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.12 | 50,000 | Mean reward = -21.0. High gamma and fast target updates caused severe overestimation and total failure. |
| exp5 | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.05 | 150,000 | Mean reward = -18.33. Fast exploration decay gave slight improvement with occasional ball returns. |
| exp6 | lr=0.0002, gamma=0.99, batch=128, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.1 | 150,000 | Mean reward = -14.33. Large batch and higher learning rate gave clear progress in ball tracking. |
| exp7 | lr=0.0001, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.15 | 200,000 | Mean reward = -16.67. Frequent updates helped but tiny batches kept gradients noisy. |
| exp8 | lr=0.0001, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 200,000 | Mean reward = **-6.0**. Best result. Eight gradient steps per update greatly improved learning efficiency. |
| exp9 | lr=0.0001, gamma=0.999, batch=32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.2 | 450,000 | Mean reward = -16.0. Very high gamma hindered short-term decision making even after 450k steps. |
| exp10 | lr=0.0002, gamma=0.97, batch=32, epsilon_start=1.0, epsilon_end=0.03, epsilon_decay=0.1 | 450,000 | Mean reward = -7.33. Second-best result. Lower gamma and fast updates produced strong short-term play. |

**Best Model:** `exp8_multi_gradient_steps` (saved as `models/Owen_model/exp8_multi_gradient_steps_best.zip`) with mean_reward = **-6.00**

**Files:**
- `notebooks/owen.ipynb`: Full training pipeline with gradient step variations
- `logs/Owen_training_metrics/owen_models.csv`: Summary results table
- `logs/Owen_training_metrics/exp[1-10]_*.csv`: Per-experiment episode-level logs

**Video Demonstration:**

**Episode 1 Return: 1.00** (Best performing episode)

[▶️ Watch Owen Demo Video](videos/owen_demo-episode-0.mp4)

![Owen Gameplay Demo](videos/owen_demo-episode-0.gif)

---

### 4. Roxanne Niteka 

**Experiments (10 configurations):**

| Name      | Hyperparameter set                                                                                              | Time steps | Noted Behavior                                                                                                      |
|-----------|-----------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------|
| exp1      | lr=0.0002, gamma=0.98, batch=40, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.15                       | 10 000     | Mean reward equals -21.0. Baseline performance. Agent loses every game with no meaningful learning.                 |
| exp2      | lr=0.00015, gamma=0.97, batch=24, epsilon_start=1.0, epsilon_end=0.03, epsilon_decay=0.2                        | 20 000    | Mean reward equals -20.0. Small batch and fast updates gave slight improvement but still very poor performance.    |
| exp3      | lr=8e-05, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.3                          | 30 000    | Mean reward equals -21.0. Large buffer and slow decay did not help. Agent showed no paddle control at all.          |
| exp4      | lr=0.0003, gamma=0.96, batch=48, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.12                       | 50 000    | Mean reward equals -21.0. Fast learning rate caused complete instability and total failure.                        |
| exp5      | lr=0.0001, gamma=0.999, batch=40, epsilon_start=1.0, epsilon_end=0.015, epsilon_decay=0.1                      | 50 000    | Mean reward equals -20.33. Extremely high gamma led to overestimation of future rewards and poor short-term play.  |
| exp6      | lr=0.00018, gamma=0.95, batch=32, epsilon_start=1.0, epsilon_end=0.04, epsilon_decay=0.25                      | 10 000     | Mean reward equals -20.67. Small buffer and frequent gradient steps finished very fast but learned almost nothing. |
| exp7      | lr=0.0007, gamma=0.97, batch=48, epsilon_start=1.0, epsilon_end=0.025, epsilon_decay=0.18, MLP [128,128]        | 10 000    | Mean reward equals -21.0. Using MLP policy instead of CNN gave no learning at all on visual Pong environment.      |
| exp8      | lr=0.0004, gamma=0.98, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.22, MLP [384,384]        | 10 000    | Mean reward equals -21.0. Deeper MLP still completely failed. Vision-based task requires CNN policy.                |
| exp9      | lr=0.00012, gamma=0.99, batch=36, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.05                      | 10 000     | Mean reward equals -21.0. Very aggressive exploration decay forced early exploitation with underdeveloped policy. |
| exp10     | lr=6e-05, gamma=0.99, batch=36, epsilon_start=1.0, epsilon_end=0.015, epsilon_decay=0.15                       | 10 000     | Mean reward equals -21.0. Very slow learning rate and gradient clipping resulted in almost no progress.            |

**Best Model:** `models/Roxanne_model/roxanne_best_dqn.zip`

**Files:**
- `notebooks/roxanne.ipynb`: 
- `logs/Roxanne_training_metrics/roxanne_models.csv`

**Video Demonstration:**

**Episode 4 Return: -18.00** (Best performing episode)

[▶️ Watch Roxanne Demo Video](videos/roxanne_demo-episode-4.mp4)

![Roxanne Gameplay Demo](videos/roxanne_demo-episode-4.gif)

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
| 🥉 3 | Owen | exp8_multi_gradient_steps | -6.0 | 200,000 | 1e-4 | 0.99 | 32 | 0.10 | Best result; eight gradient steps per update greatly improved efficiency |
| 4 | Owen | exp10_lower_gamma_fast_updates | -7.33 | 450,000 | 2e-4 | 0.97 | 32 | 0.10 | Second-best; lower gamma and fast updates produced strong short-term play |
| 5 | Excel | exp1_baseline | -12.24 | 2,041 | 1e-4 | 0.99 | 32 | N/A | Stable CNN baseline; best among shorter runs |
| 6 | Excel | exp10_slow_lr_clip | -13.28 | 2,000 | 5e-5 | 0.99 | 32 | N/A | Conservative LR + gradient clipping |
| 7 | Excel | exp5_high_gamma | -12.52 | 2,000 | 1e-4 | 0.997 | 32 | N/A | Higher discount factor improves planning |
| 8 | Excel | exp3_freq1_small_batch | -13.78 | 2,000 | 1e-4 | 0.99 | 16 | N/A | Frequent updates help; train time increases |
| 9 | Owen | exp9_very_high_gamma | -16.0 | 450,000 | 1e-4 | 0.999 | 32 | 0.20 | Very high gamma hindered short-term decision making |
| 10 | Owen | exp5_fast_exploration_decay | -18.33 | 150,000 | 1e-4 | 0.99 | 32 | 0.05 | Fast exploration decay gave slight improvement |
| 11 | Owen | exp7_small_batch_frequent_updates | -16.67 | 200,000 | 1e-4 | 0.99 | 16 | 0.15 | Frequent updates helped but tiny batches kept gradients noisy |
| 12 | Owen | exp6_large_batch_higher_lr | -14.33 | 150,000 | 2e-4 | 0.99 | 128 | 0.10 | Large batch and higher learning rate gave clear progress in ball tracking |
| 13 | Owen | exp3_very_low_lr | -20.0 | 50,000 | 5e-5 | 0.995 | 64 | 0.10 | Very low learning rate caused almost no learning in 50k steps |
| 14 | Branis | exp1_baseline | -20.33 | 50,000 | 1e-4 | 0.99 | 32 | 0.10 | Short training; baseline for comparison |
| 15 | Branis | exp8_very_high_gamma | -19.33 | 150,000 | 1e-4 | 0.997 | 32 | 0.10 | Very high gamma; modest improvement over short runs |
| 16 | Owen | exp1_baseline_short | -20.67 | 50,000 | 1.25e-4 | 0.99 | 32 | 0.10 | Baseline at only 50k steps; agent barely moves paddle |
| 17 | Owen | exp2_worst_unstable | -21.0 | 50,000 | 4e-4 | 0.98 | 32 | 0.15 | Worst result; high LR and low gamma made training unstable |
| 18 | Owen | exp4_overestimation_failure | -21.0 | 50,000 | 1.5e-4 | 0.997 | 32 | 0.12 | High gamma and fast targets caused severe overestimation |
| 19 | Excel | exp2_large_batch | -12.38 | 2,000 | 7e-5 | 0.99 | 64 | N/A | Larger batch = smoother but slower |
| 20 | Excel | exp4_more_gradsteps | -12.48 | 2,000 | 8e-5 | 0.99 | 32 | N/A | Multiple gradient steps: stable but slower |
| 21 | Excel | exp9_quick_decay | -12.81 | 2,000 | 1e-4 | 0.99 | 32 | N/A | Fast ε-decay forces premature exploitation |
| 22 | Excel | exp6_small_buffer | -13.21 | 2,000 | 1.2e-4 | 0.99 | 32 | N/A | Small buffer ↑ variance |

---

## Video Demonstration

### Branis's Best Model (exp10_more_gradient_steps)
**Mean Reward: 6.7 (BEST OVERALL)** | Extended training (500k timesteps)

**Episode 7 Return: 15.00** (Best performing episode)

[▶️ Watch Branis Demo Video](https://drive.google.com/file/d/1MsPk1w5dAUm3NkBr0MdBTS0qG_NyGfPe/view)

![Branis Gameplay Demo](videos/branis_demo-episode-12.gif)

---

### Owen's Best Model (exp8_multi_gradient_steps)
**Mean Reward: -6.0 (2nd best)** | Eight gradient steps per update greatly improved efficiency

**Episode 1 Return: 1.00** (Best performing episode)

[▶️ Watch Owen Demo Video](videos/owen_demo-episode-0.mp4)

![Owen Gameplay Demo](videos/owen_demo-episode-0.gif)

---

### Excel's Best Model (excel_best_dqn)
**Mean Reward: -12.24 (3rd best)** | CNN baseline with stability tuning

**Episode 3 Return: 15.00** (Best performing episode)

[▶️ Watch Excel Demo Video](videos/excel_demo-episode-4.mp4)

![Excel Gameplay Demo](videos/excel_demo-episode-4.gif)
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

# Record Roxanne's best model
python play.py --model models/Roxanne_model/roxanne_exp5_high_gamma.zip --episodes 10 --save_video videos/roxanne_demo.mp4
```
---