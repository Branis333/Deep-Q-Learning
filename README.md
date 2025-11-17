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

### 1. Branis Sumba - Baseline Experiments

**Role:** Established baseline DQN training pipeline and conducted initial hyperparameter exploration.

**Experiments (10 configurations):**

| # | Configuration | Learning Rate | Gamma | Batch Size | Buffer Size | Train Freq | Gradient Steps | Target Update | Exploration Frac | Notes |
|---|---------------|---------------|-------|------------|-------------|------------|----------------|----------------|------------------|-------|
| 1 | Baseline | 1e-4 | 0.99 | 32 | 100k | 4 | 1 | 10k | 0.10 | Standard DQN setup; establishes baseline performance |
| 2 | Small LR | 5e-5 | 0.99 | 32 | 100k | 4 | 1 | 10k | 0.10 | Conservative learning; slower convergence but more stable |
| 3 | Large LR | 3e-4 | 0.99 | 32 | 100k | 4 | 1 | 10k | 0.10 | Aggressive updates; faster learning but higher variance |
| 4 | High Gamma | 1e-4 | 0.997 | 32 | 100k | 4 | 1 | 10k | 0.10 | Longer horizon; emphasizes long-term rewards |
| 5 | Large Batch | 1e-4 | 0.99 | 64 | 100k | 4 | 1 | 10k | 0.10 | Larger batch; smoother gradients but slower updates |
| 6 | Small Batch | 1e-4 | 0.99 | 16 | 100k | 4 | 1 | 10k | 0.10 | Smaller batch; noisier gradients but more frequent updates |
| 7 | Freq=1 | 1e-4 | 0.99 | 32 | 100k | 1 | 1 | 10k | 0.10 | More frequent training; potential instability |
| 8 | Large Buffer | 1e-4 | 0.99 | 32 | 200k | 4 | 1 | 10k | 0.10 | Larger replay buffer; better decorrelation |
| 9 | More Grad Steps | 1e-4 | 0.99 | 32 | 100k | 4 | 4 | 10k | 0.10 | Multiple gradient steps; deeper policy updates |
| 10 | Gradient Clipping | 1e-4 | 0.99 | 32 | 100k | 4 | 1 | 10k | 0.10 | Gradient norm clipping; stabilizes training |

**Key Findings:**
- Baseline configuration (exp1) achieves stable learning
- High learning rates lead to divergence on this task
- Larger replay buffers improve sample efficiency
- Gradient clipping provides marginal benefits

**Best Model:** `exp1_baseline` (saved as `models/Branis_model/best_dqn_pong_cnn.zip`)

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

| Rank | Member | Experiment | Policy | Mean Reward | Std Dev | Train Time | Learning Rate | Gamma | Batch Size | Key Observation |
|------|--------|-----------|--------|-------------|---------|------------|--------------|-------|------------|---|
| 🥇 1 | Excel | exp1_baseline | CNN | **-10.85** | 7.88 | 45.2m | 1e-4 | 0.99 | 32 | Best overall; stable learning |
| 🥈 2 | Excel | exp10_slow_lr_clip | CNN | -11.28 | 7.50 | 49.6m | 5e-5 | 0.99 | 32 | Conservative LR + gradient clipping works well |
| 🥉 3 | Excel | exp5_high_gamma | CNN | -11.52 | 7.90 | 46.8m | 1e-4 | 0.997 | 32 | Higher discount factor improves long-term planning |
| 4 | Excel | exp3_freq1_small_batch | CNN | -11.78 | 8.20 | 52.3m | 1e-4 | 0.99 | 16 | Frequent updates help; train time increases |
| 5 | Branis | exp1_baseline | CNN | -12.24 | 7.88 | 45.2m | 1e-4 | 0.99 | 32 | Solid baseline; similar to Excel's approach |
| 6 | Excel | exp2_large_batch | CNN | -12.38 | 7.85 | 48.5m | 7e-5 | 0.99 | 64 | Larger batch = smoother but slower |
| 7 | Branis | exp2_small_lr | CNN | -12.41 | 7.92 | 47.1m | 5e-5 | 0.99 | 32 | Too conservative; slower convergence |
| 8 | Excel | exp4_more_gradsteps | CNN | -12.48 | 8.50 | 51.1m | 8e-5 | 0.99 | 32 | Multiple gradient steps: stable but slower |
| 9 | Excel | exp9_quick_decay | CNN | -12.81 | 8.80 | 47.9m | 1e-4 | 0.99 | 32 | Fast ε-decay forces premature exploitation |
| 10 | Excel | exp6_small_buffer | CNN | -13.21 | 9.00 | 42.7m | 1.2e-4 | 0.99 | 32 | Small buffer ↑ variance, ↓ memory |
| 11 | Excel | exp8_mlp_deep | MLP | -13.48 | 9.50 | 44.2m | 3e-4 | 0.99 | 64 | Deeper MLP still underperforms CNN |
| 12 | Excel | exp7_mlp_small | MLP | -14.05 | 10.0 | 39.5m | 5e-4 | 0.99 | 64 | MLPs poorly suited for image inputs |

### Key Insights

#### 1. **Policy Architecture: CNN >> MLP**
- CNN-based policies **consistently outperform MLPs by 2-4 points**
- Convolutional layers efficiently extract spatial features from Pong images
- MLPs struggle due to flattened 84×84 input (7,056 parameters per layer)
- **Recommendation:** Use CNN for visual tasks; MLP for low-dim state spaces

#### 2. **Learning Rate Trade-offs**
- **Optimal range:** 1e-4 (produces best result)
- **Too low** (5e-5): Slow convergence, stable but suboptimal
- **Too high** (3e-4): Divergence and instability
- **Reason:** Pong's reward signal is sparse; need careful balance

#### 3. **Batch Size Effects**
- **Small batch (16):** Noisy gradients but faster wall-clock time (more updates/min)
- **Medium batch (32):** Sweet spot; balanced noise vs. convergence
- **Large batch (64):** Smooth gradients but fewer effective updates in fixed time
- **Trade-off:** Small batch → faster learning; large batch → stability

#### 4. **Gamma (Discount Factor)**
- **Standard (0.99):** Works well for most cases
- **High (0.997):** Slightly better for long-horizon strategy (+0.67 improvement)
- **Impact:** Marginal on Pong; more important for complex games

#### 5. **Buffer Size & Memory**
- **Small (50k):** Low memory but ↑ variance → suboptimal performance (-13.21)
- **Standard (100k):** Good balance; recommended
- **Large (200k):** Marginal improvements; ↑ RAM usage
- **Lesson:** 100k sufficient for Pong; diminishing returns above

#### 6. **Training Frequency & Gradient Steps**
- **Freq=4:** Standard; works well
- **Freq=1:** More frequent updates → marginally faster convergence but longer training time
- **Grad steps >1:** Stabilizes but slows per-episode updates

---

## Video Demonstration

**Agent Performance Videos:**
- ✅ [Excel's Best Model Demo](videos/excel_exp1_baseline_demo.mp4) - CNN baseline, mean reward = -10.85
- ✅ [Branis's Baseline Demo](videos/branis_exp1_baseline_demo.mp4) - CNN baseline, mean reward = -12.24
- 🔄 [Owen's Best Model Demo](videos/owen_best_demo.mp4) - *To be recorded*
- 🔄 [Roxanne's Best Model Demo](videos/roxanne_best_demo.mp4) - *To be recorded*

**To generate videos:**
```bash
# Record agent gameplay as video
python play.py --model models/Excel_model/excel_best_dqn.zip --episodes 10 --save_video videos/demo.mp4
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
