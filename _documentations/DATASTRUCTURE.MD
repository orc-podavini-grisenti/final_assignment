

Directory Structure:
```graphql
final_assignemnt/
│
├── README.md
├── requirements.txt
│
├── configs/        # Configuration files:
│   ├── env.yaml        # environment configuration
│   ├── ppo.yaml
│   └── training.yaml
│
├── envs/           # Defines the RL environment:
│   ├── __init__.py
│   ├── obstacles.py        # obstacle handling
|   ├── test_env.py         # test environment script
|   ├── unicycle_env.py     # global environment definition
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
│   ├── reinforcment.ipynb      # noteboock for training the TT trajectory
│   ├── train_ppo.py
│   └── evaluate.py
│
├── utils/
│   ├── math_utils.py
│   ├── logger.py
│   ├── normalization.py    # normalize the input of the networks
│   ├── simulation.py       # run a single simulation on the enviroment
│   └── visualization.py
│
├── logs/
│   └── runs/
│
├── checkpoints/
│   └── ppo/
│
└── scripts/
    ├── run_controller_benchmark.py  # benchmark between RL Policy - Lyapunov Controller    
    ├── run_policy_evaluation.sh              
    └── run_trajectory_tracking.sh
```