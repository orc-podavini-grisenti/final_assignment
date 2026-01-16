# 📂​ Directory Structure:
```graphql
final_assignemnt/
│
├── README.md
├── requirements.txt
│
├── _documentations/    
│   ├── DATASTRUCTURE.md
│   ├── navigation_videos/ 
│   └── tracking_video/     
│
├── configs/        # Enviroment Configuration Files:
│   ├── env.yaml            # randome environment with 3 obstacle
│   ├── empty_env.yaml      # enviroment without obstacle and random goal   
│   ├── fixed_env.yaml      # fixed enviroment    
│   └── env_nav_ev.yaml     # randome environment with 10 obstacle
│
├── controllers/    # Implementation of classic and RL:
│   ├── __init__.py
│   ├── lyapunov_controller.py
│   └── rl_controller.py    # wrapper that call the policy
│
├── envs/           # Defines the RL environment:
│   ├── __init__.py
│   ├── obstacles.py        # obstacle handling
|   ├── test_env.py         # test environment script
|   ├── unicycle_env.py     # global environment definition
│   ├── rewards.py          # reward function
│   └── README.md          
│
├── evaluation/      # Contain the evaluation scripts:
│   ├── output/             # store all the evaluation results and outputs
│   │   └── ...
│   ├── controller_evaluation.py        
|   ├── navigation_evaluation.py         
|   ├── README.md           # explain how to run all the evaluation process
│   ├── statistics,py       # implement statistical analysis methods 
│   └── utils.py            
│
├── models/         # Neural networks only:
│   ├── trajectory_tracking_network.py      
│   ├── value_network.py                # network used to estimate the value function 
│   └── navigation_network.py           # Actor and Critic networks for PPO
│ 
├── planner/         # Dubins Planner:
│   ├── __init__.py    
│   ├── dubins_planner.py        # Implementation of the Dubins Planner class
│   └── dubins_test.py           # simple dubins planner testing
│
├── scripts/        # Launch Scripts
│   ├── __init__.py
│   ├── run_nav_evaluation.py       # launch script for all the navigation evaluation    
│   ├── run_navigation.py           # launch script for a navigation episode              
│   ├── run_tracking_evaluation.py  # launch script for all the tracking evaluation    
│   └── run_trajectory_tracking.py   # launch script fro a tracking episode
│
├── training/      # Scripts to train the policy and evaluate it
│   ├── experimets/             # locla storage for trainign ( gitignore )
│   │   └── ... 
│   ├── nav_1/                  # contain the navigation policy
│   │   └── ... 
│   ├── v1_no_baseline/         # contain the tracking policy trained without baseline
│   │   └── ...
│   ├── v2_baseline/            # contain the tracking policy trained with baseline
│   │   └── ... 
│   ├── navigation.ipynb                # training noteboock for PPO navigation policy 
│   ├── reinforcment_baseline.ipynb     # training noteboock for Reinfocment with baseline 
│   └── reinforcment.ipynb              # training noteboock for Reinfocment without baseline
│ 
├── utils/
    ├── normalization.py    # normalize the input of the networks
    ├── simulation.py       # run a single simulation on the enviroment
    └── reward.py           # reward functions implementation

```