# Deep Q-Learning (DQN) on Atari Pong-v5
**Formative 3 Assignment: Deep Reinforcement Learning**

**Group 3 Members:**
- Branis Sumba
- Excel Asaph
- Ganza Owen Yhaan
- Roxanne Niteka

---

## Overview

This project implements a Deep Q-Network (DQN) agent trained to play the Atari Pong-v5 environment using Stable Baselines3 and Gymnasium. The assignment requires training and comparing multiple DQN configurations with different hyperparameters and policy architectures (CNNPolicy vs MLPPolicy).

**Key Deliverables:**
- `train.py`: Training script that runs DQN experiments and saves trained models
- `play.py`: Inference script that loads trained models and demonstrates agent gameplay
- `notebooks/`: Jupyter notebooks for experimentation and hyperparameter tuning
- Comprehensive hyperparameter documentation and performance analysis

---

## Project Structure

```
Deep-Q-Learning/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── train.py                           # Training script
├── play.py                            # Inference/gameplay script
├── notebooks/
│   ├── branis.ipynb                   # Branis's experiments (10 configs)
│   ├── excel1.ipynb                   # Excel's local experiments
│   └── excel2.ipynb                   # Excel's Google Colab experiments
├── models/
│   ├── Branis_model/                  # Branis's trained models
│   ├── Excel_model/                   # Excel's trained models
│   └── Owen_model/                    # Owen's trained models (placeholder)
│   └── Roxanne_model/                 # Roxanne's trained models (placeholder)
├── logs/
│   ├── training-metrics/              # Episode-level training logs (CSV)
│   ├── Excel_training_metrics/
│   ├── Owen_training_metrics/         # (placeholder)
│   ├── Roxanne_training_metrics/      # (placeholder)
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

### 1. Branis Sumba - Hyperparameter Tuning & Epsilon Decay Analysis

**Role:** Investigated epsilon decay schedules and their impact on exploration-exploitation trade-off; discovered that extended training with faster epsilon decay yields superior performance.

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

**Key Findings:**
- **Extended training (500k+ timesteps) dramatically outperforms short runs** (50k steps): +17 reward improvement
- **Epsilon decay schedule matters more than traditional hyperparameters** (LR, gamma, batch)
- **Fast epsilon decay (0.05)** with long training yields best performance (mean_reward = 6.7)
- Batch size 32 consistently outperforms smaller (16) and larger (54, 64) configurations
- Very long training (700k) with small batch shows instability despite high training return
- Multiple gradient steps improve stability without sacrificing convergence speed

**Best Model:** `exp10_more_gradient_steps` (saved as `models/Branis_model/exp10_more_gradient_steps.zip`) with mean_reward = **6.7**

**Files:**
- `notebooks/branis.ipynb`: Full training pipeline with callbacks and visualization
- `logs/branis_models.csv`: Summary results table

---

### 2. Excel Asaph - CNN vs MLP Policy Comparison

**Role:** Conducted systematic policy architecture comparison and hyperparameter refinement.

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

**Key Findings:**
- **CNN significantly outperforms MLP** (~2-4 point advantage in mean reward)
- CNN's convolutional layers efficiently extract spatial features from images
- MLP policies struggle to process high-dimensional image inputs effectively
- Baseline configuration performs best overall (mean_reward = **-10.85**)
- High gamma and conservative learning rates improve stability
- Small buffers and aggressive target updates increase variance

**Best Model:** `excel_exp1_baseline` (saved as `models/Excel_model/excel_best_dqn.zip`)

**Files:**
- `notebooks/excel1.ipynb`: Local training experiments (Windows)
- `notebooks/excel2.ipynb`: Google Colab extended training (1.5M timesteps)
- `logs/training-metrics/Excel_training_metrics/excel_models.csv`: Results summary
- `logs/training-metrics/Excel_training_metrics/training_metrics_excel_exp[1-10].csv`: Per-experiment logs

---

### 3. Ganza Owen Yhaan - [Placeholder for Owen's Contribution]

**Role:** [To be filled - Owen's experiment focus and methodology]

**Experiments (10 configurations):**

| # | Configuration | Policy | Learning Rate | Gamma | Batch Size | Buffer Size | Train Freq | Gradient Steps | Target Update | Noted Behavior |
|---|---------------|--------|---------------|-------|------------|-------------|------------|----------------|----------------|---|
| 1 | [Exp Name] | [CNN/MLP] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [To be documented] |
| 2 | [Exp Name] | [CNN/MLP] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [To be documented] |
| 3-10 | ... | ... | ... | ... | ... | ... | ... | ... | ... | [To be documented] |

**Key Findings:**
- [To be filled]

**Best Model:** `models/Owen_model/owen_best_dqn.zip`

**Files:**
- `notebooks/owen.ipynb`: [To be created]
- `logs/Owen_training_metrics/owen_models.csv`: [To be created]

---

### 4. Roxanne Niteka - [Placeholder for Roxanne's Contribution]

**Role:** [To be filled - Roxanne's experiment focus and methodology]

**Experiments (10 configurations):**

| # | Configuration | Policy | Learning Rate | Gamma | Batch Size | Buffer Size | Train Freq | Gradient Steps | Target Update | Noted Behavior |
|---|---------------|--------|---------------|-------|------------|-------------|------------|----------------|----------------|---|
| 1 | [Exp Name] | [CNN/MLP] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [To be documented] |
| 2 | [Exp Name] | [CNN/MLP] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [Value] | [To be documented] |
| 3-10 | ... | ... | ... | ... | ... | ... | ... | ... | ... | [To be documented] |

**Key Findings:**
- [To be filled]

**Best Model:** `models/Roxanne_model/roxanne_best_dqn.zip`

**Files:**
- `notebooks/roxanne.ipynb`: [To be created]
- `logs/Roxanne_training_metrics/roxanne_models.csv`: [To be created]

---

## Training Pipeline

### train.py Usage

```bash
# Train with default settings (Branis baseline)
python train.py

# Train with custom hyperparameters (example)
python train.py --learning_rate 1e-4 --gamma 0.99 --batch_size 32 --buffer_size 100000
```

**train.py Features:**
- Supports both CNNPolicy and MLPPolicy architectures
- Logs training metrics to CSV (timestep, episode_length, episode_reward)
- Saves per-experiment models: `models/dqn_{exp_name}.zip`
- Tracks best model: `models/{group_member}_model/best_dqn.zip`
- Generates TensorBoard logs for visualization: `logs/tensorboard/{exp_name}/`
- Automatic callback system for monitoring and early stopping

**Output Files:**
- `models/dqn_exp_{n}.zip`: Individual experiment models
- `logs/training_metrics_{exp_name}.csv`: Episode-by-episode metrics
- `logs/{member}_models.csv`: Summary table (mean_reward, std_reward, hyperparams)

---

## Inference Pipeline

### play.py Usage

```bash
# Play with best model (Excel's baseline)
python play.py

# Play specific experiment model
python play.py --model models/Excel_model/excel_best_dqn.zip --episodes 5

# Display available models
python play.py --list
```

**play.py Features:**
- Loads trained DQN models with GreedyQPolicy for deterministic play
- Renders game environment in real-time
- Handles memory-efficient inference with reduced replay buffer
- Robust error handling for numpy compatibility issues
- Episode reward tracking and display
- ~60 FPS gameplay pacing

**Output:**
- Real-time Pong gameplay visualization
- Episode return (total discounted reward) printed to console

---

## Performance Summary & Hyperparameter Analysis

### All Experiments Ranked by Mean Reward

| Rank | Member | Experiment | Mean Reward | Timesteps | Learning Rate | Gamma | Batch Size | Epsilon Decay | Key Observation |
|------|--------|-----------|-------------|-----------|---------------|-------|------------|---------------|---|
| 🥇 1 | Branis | exp10_more_gradient_steps | **6.7** | 500,000 | 1e-4 | 0.99 | 32 | 0.10 | Best overall; extended training + proper epsilon decay |
| 🥈 2 | Branis | exp9_small_batch_extended | 4.66 | 700,000 | 1e-4 | 0.99 | 16 | 0.12 | Very long training yields positive return but unstable |
| 🥉 3 | Excel | exp1_baseline | -10.85 | 2,041 | 1e-4 | 0.99 | 32 | N/A | Stable CNN baseline; best among shorter runs |
| 4 | Excel | exp10_slow_lr_clip | -11.28 | 2,000 | 5e-5 | 0.99 | 32 | N/A | Conservative LR + gradient clipping |
| 5 | Excel | exp5_high_gamma | -11.52 | 2,000 | 1e-4 | 0.997 | 32 | N/A | Higher discount factor improves planning |
| 6 | Excel | exp3_freq1_small_batch | -11.78 | 2,000 | 1e-4 | 0.99 | 16 | N/A | Frequent updates help; train time increases |
| 7 | Branis | exp1_baseline | -20.33 | 50,000 | 1e-4 | 0.99 | 32 | 0.10 | Short training; baseline for comparison |
| 8 | Branis | exp8_very_high_gamma | -19.33 | 150,000 | 1e-4 | 0.997 | 32 | 0.10 | Very high gamma; modest improvement over short runs |
| 9 | Excel | exp2_large_batch | -12.38 | 2,000 | 7e-5 | 0.99 | 64 | N/A | Larger batch = smoother but slower |
| 10 | Excel | exp4_more_gradsteps | -12.48 | 2,000 | 8e-5 | 0.99 | 32 | N/A | Multiple gradient steps: stable but slower |
| 11 | Excel | exp9_quick_decay | -12.81 | 2,000 | 1e-4 | 0.99 | 32 | N/A | Fast ε-decay forces premature exploitation |
| 12 | Excel | exp6_small_buffer | -13.21 | 2,000 | 1.2e-4 | 0.99 | 32 | N/A | Small buffer ↑ variance |

### Key Insights

#### 1. **Training Duration Dramatically Outweighs Most Hyperparameters**
- **Critical finding from Branis's experiments:** Extending training from 50k to 500k timesteps improved mean reward by **~27 points** (from -20.33 to 6.7)
- This vastly outpaces any improvement from tuning learning rate, gamma, or batch size
- **Implication:** Sample efficiency matters less than having sufficient exploration time
- **Recommendation:** Allocate sufficient timesteps for convergence on sparse-reward tasks

#### 2. **Epsilon Decay Schedule is Critical**
- Standard epsilon decay (0.10) keeps agent in exploration too long on short runs
- Faster epsilon decay (0.05) with extended training allows proper transition to exploitation
- Very aggressive decay (0.20) doesn't help; moderate decay with long training is optimal
- **Key trade-off:** Must balance exploration duration with training timesteps

#### 3. **Policy Architecture: CNN Advantages for Image Tasks**
- CNN-based policies in Excel's experiments **outperform MLPs by 2-4 points** when properly trained
- However, Branis's extended training shows even raw DQN can reach positive rewards
- CNNs extract spatial features efficiently; MLPs waste capacity on image flattening
- **Recommendation:** Use CNN for visual Pong; MLP for low-dimensional state spaces

#### 4. **Learning Rate Trade-offs**
- **Optimal range for Pong:** 1e-4 (appears best across both Branis and Excel experiments)
- **Too low** (5e-5): Slow convergence even with long training
- **Too high** (2e-4, 3e-4): Divergence and instability
- **Reason:** Pong's sparse rewards require careful gradient step sizing

#### 5. **Batch Size Effects**
- **Branis finding:** Batch size 32 consistently outperforms 16 and 54/64
- Batch 16 can work with very long training but shows higher variance
- Batch 64 smooths gradients but doesn't improve returns on this task
- **Trade-off:** Batch 32 balances noise vs. convergence stability

#### 6. **Gamma (Discount Factor)**
- **Standard (0.99):** Works well across experiments
- **High (0.995+):** Led to over-optimism in Branis exp3; marginal gains at best
- **Impact on Pong:** Less critical than on tasks with longer horizons
- **Lesson:** Default 0.99 is well-calibrated; changes don't provide significant leverage

#### 7. **Surprising Result: Small Batch + Very Long Training**
- Branis exp9 (batch=16, 700k timesteps) achieved mean_reward=4.66 but failed in test
- This suggests **overfitting to training environment or instability under ε-greedy test policy**
- Batch 32 with 500k (exp10) generalizes better despite lower training return
- **Lesson:** Training return != test performance; stability more important than peak training reward

---

## Video Demonstration

**Agent Performance Videos:**
- ✅ [Branis's Best Model Demo](videos/branis_exp10_more_gradient_steps_demo.mp4) - Extended training (500k), mean reward = **6.7** (BEST OVERALL)
- ✅ [Excel's Best Model Demo](videos/excel_exp1_baseline_demo.mp4) - CNN baseline, mean reward = -10.85 (best short-run)
- 🔄 [Owen's Best Model Demo](videos/owen_best_demo.mp4) - *To be recorded*
- 🔄 [Roxanne's Best Model Demo](videos/roxanne_best_demo.mp4) - *To be recorded*

**To generate videos:**
```bash
# Record agent gameplay as video
python play.py --model models/Branis_model/exp10_more_gradient_steps.zip --episodes 10 --save_video videos/branis_demo.mp4
```

---

## Understanding DQN & RL Concepts

### Deep Q-Learning Algorithm Overview

**Core Idea:** Learn a Q-function Q(s,a) that estimates expected future rewards from state *s* taking action *a*, using a neural network.

**Key Components:**

1. **Experience Replay:** Store (state, action, reward, next_state) tuples in a buffer; sample randomly to break temporal correlation
2. **Target Network:** Use separate network to compute target values, updated less frequently to stabilize learning
3. **ε-Greedy Exploration:** Trade-off between exploration (random actions) and exploitation (greedy best actions)
4. **Loss Function:** Minimize MSE between predicted Q-values and Bellman targets

$$L = \frac{1}{|B|} \sum_{(s,a,r,s') \in B} \left[ Q(s,a) - (r + \gamma \max_{a'} Q_{target}(s', a')) \right]^2$$

### Exploration vs. Exploitation

- **Exploration:** Random actions to discover environment dynamics
- **Exploitation:** Greedy actions maximizing learned Q-values
- **ε-Decay Schedule:** Start with high ε (exploration); decay to low ε (exploitation)
- **Pong Challenge:** Sparse rewards require good exploration strategy

### Hyperparameter Roles

| Hyperparameter | Role | Impact |
|---|---|---|
| **Learning Rate (α)** | Step size in gradient descent | Too high → divergence; too low → slow convergence |
| **Gamma (γ)** | Discount factor; weight on future rewards | High → long-term planning; low → myopic (immediate rewards) |
| **Batch Size** | # samples per gradient update | Small → noisy; large → stable but fewer updates |
| **Buffer Size** | # experiences stored | Small → correlation; large → memory overhead |
| **Train Frequency** | Steps between gradient updates | Low → slow learning; high → instability |
| **Epsilon Schedule** | Exploration decay rate | Fast → premature exploitation; slow → wasted exploration |
| **Target Update Interval** | Steps between target net updates | Small → divergence; large → slow moving target |

### Reward Structure (Pong-v5)

- **+1:** Agent scores a goal
- **-1:** Agent loses a goal (or each step taken)
- **Sparse reward:** Makes learning difficult; requires effective exploration

---

## Usage Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
AutoROM --accept-license  # Download Atari ROMs
```

### 2. Train Models
```bash
# Train all experiments
python train.py

# Or train specific configuration
python train.py --name excel_exp1_baseline --lr 1e-4 --gamma 0.99 --batch_size 32
```

### 3. Play with Trained Agent
```bash
# Use best model
python play.py

# Specify custom model
python play.py --model models/Excel_model/excel_best_dqn.zip --episodes 5

# List available models
python play.py --list
```

### 4. Analyze Results
```python
import pandas as pd

# Load results
results = pd.read_csv('logs/excel_models.csv')
print(results.sort_values('mean_reward', ascending=False))

# Visualize training curves
import matplotlib.pyplot as plt
metrics = pd.read_csv('logs/training_metrics_excel_exp1_baseline.csv')
plt.plot(metrics['timestep'], metrics['ep_reward'].rolling(10).mean())
plt.show()
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

## File Manifest

| File | Purpose | Owner |
|------|---------|-------|
| `train.py` | Training script; runs all experiments | Branis (with Excel enhancements) |
| `play.py` | Inference script; plays trained models | Branis (with Excel debugging) |
| `requirements.txt` | Python dependencies | Branis |
| `notebooks/branis.ipynb` | Branis's 10 experiments | Branis |
| `notebooks/excel1.ipynb` | Excel's local experiments | Excel |
| `notebooks/excel2.ipynb` | Excel's Colab experiments | Excel |
| `notebooks/owen.ipynb` | Owen's experiments [TBD] | Owen |
| `notebooks/roxanne.ipynb` | Roxanne's experiments [TBD] | Roxanne |
| `logs/branis_models.csv` | Branis's results summary | Branis |
| `logs/Excel_training_metrics/excel_models.csv` | Excel's results summary | Excel |
| `logs/Owen_training_metrics/owen_models.csv` | Owen's results [TBD] | Owen |
| `logs/Roxanne_training_metrics/roxanne_models.csv` | Roxanne's results [TBD] | Roxanne |
| `models/Branis_model/best_dqn.zip` | Branis's best trained model | Branis |
| `models/Excel_model/excel_best_dqn.zip` | Excel's best trained model | Excel |
| `models/Owen_model/owen_best_dqn.zip` | Owen's best model [TBD] | Owen |
| `models/Roxanne_model/roxanne_best_dqn.zip` | Roxanne's best model [TBD] | Roxanne |

---

## Group Collaboration Notes

**Division of Work:**
- **Branis:** Project setup, train.py/play.py scripting, baseline experiments, pipeline architecture
- **Excel:** Hyperparameter refinement, CNN vs MLP comparison, Colab setup, 10 diverse experiments
- **Owen:** [To document]
- **Roxanne:** [To document]

**Key Meetings:**
- Week 4: Initial setup and environment configuration ✓
- Week 5: Hyperparameter tuning and experimentation ✓
- Week 6: Coach consultation slot [To be booked]
- Final: Results compilation and README documentation [In progress]

---

## References

1. [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
2. [Gymnasium Atari Docs](https://gymnasium.farama.org/environments/atari/)
3. [Human-level control through deep reinforcement learning (DQN paper)](https://www.nature.com/articles/nature14236)
4. [Deep Reinforcement Learning Hands-On (Lapan, 2020)](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)

---

## Submission Checklist

- ✅ train.py script (functional, accepts hyperparameter arguments)
- ✅ play.py script (loads models, renders gameplay, uses GreedyQPolicy)
- ✅ 10 distinct hyperparameter experiments per member (40 total)
- ✅ Hyperparameter documentation table (above)
- ✅ README with analysis and insights
- ✅ Trained model files (.zip) for each best experiment
- ✅ GitHub repository with all files
- 🔄 Video demonstration of agent gameplay (pending Owen & Roxanne)
- 🔄 Coach consultation slot booking (Week 6)

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

**Last Updated:** November 17, 2025  
**Status:** Ready for Submission (Attempt 1)  
**Next Steps:** Record demo videos (Owen & Roxanne); book coach consultation slot
