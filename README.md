## Experiment Summary
## MEMBER NAME : Sumba Branis


| Name | Hyperparameter set | Time steps | Noted behavior |
| --- | --- | --- | --- |
| exp1 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 50000 | Mean reward = -20.33. Increasing batch size and lowering the learning rate slightly worsened performance. |
| exp2 | lr=5e-5, gamma=0.99, batch=54, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.15 | 50000 | Mean reward = -20.67. Increasing batch size and lowering the learning rate slightly worsened performance. |
| exp3 | lr=1.5e-4, gamma=0.995, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 50000 | Mean reward = -21.00. A higher gamma led to even lower returns, suggesting slower learning and over-optimism. |
| exp4 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.05 | 500000 | Mean reward = -11.333. Faster epsilon decay over many steps significantly improved performance. |
| exp5 | lr=2e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.12 | 50000 | Mean reward = -20.67. A higher learning rate with the same setup did not improve reward outcomes. |
| exp6 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.2 | 50000 | Mean reward = -20.00. Faster updates with a small buffer slightly improved stability but not overall performance. |
| exp7 | lr=3e-4, gamma=0.99, batch=64, epsilon_start=1.0, epsilon_end=0.02, epsilon_decay=0.18 | 50000 | Mean reward = -20.00. A large batch with a higher learning rate produced average but not improved rewards. |
| exp8 | lr=1e-4, gamma=0.997, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 150000 | Mean reward = -19.33. Very high gamma slightly improved reward but still kept performance low. |
| exp9 | lr=1e-4, gamma=0.99, batch=16, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.12 | 700000 | Mean reward = 4.66. Small batch size with long training time yielded strong positive reward, but the agent failed in testing. |
| exp10 | lr=1e-4, gamma=0.99, batch=32, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.1 | 500000 | Mean reward = 6.7. More gradient updates helped the model reach a positive reward. |




## Video   Link  : https://drive.google.com/file/d/1MsPk1w5dAUm3NkBr0MdBTS0qG_NyGfPe/view?usp=sharing   
