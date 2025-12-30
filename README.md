

Directory Structure:
```graphql
final_assignemnt/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── env.yaml
│   ├── ppo.yaml
│   └── training.yaml
│
├── envs/               # Defines the RL environment:
│   ├── __init__.py
│   ├── unicycle_env.py     # robot kinematics
│   ├── obstacles.py        # obstacle handling
│   └── rewards.py          # reward function
│
├── controllers/            #Implementation of classic and RL:
│   ├── __init__.py
│   ├── lyapunov_controller.py
│   └── rl_controller.py
│
├── models/                 # Neural networks only:
│   ├── __init__.py
│   ├── actor.py
│   ├── critic.py
│   └── networks.py
│
├── algorithms/     # RL algorithm implementation:
│   ├── __init__.py
│   └── ppo.py
│
├── training/      # Scripts to train the policy and evaluate it
│   ├── train_ppo.py
│   └── evaluate.py
│
├── utils/
│   ├── math_utils.py
│   ├── logger.py
│   ├── normalization.py
│   └── visualization.py
│
├── logs/
│   └── runs/
│
├── checkpoints/
│   └── ppo/
│
└── scripts/
    ├── run_training.sh
    └── run_eval.sh
```