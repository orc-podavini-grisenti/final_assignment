# Unicycle Simulation Environment
A modular 2D simulation environment built with OpenAI Gymnasium and OpenCV for training and evaluating reinforcement learning agents on non-holonomic robots.

## 🚀 Overview
The UnicycleEnv provides a standardized interface for a robot navigating a 2D workspace.
### Key Features:
- Gymnasium API: Fully compatible with standard RL libraries (Stable Baselines3, Ray RLLib, etc.).
- Modular Configuration: Control environment parameters, obstacle density, and spawning logic via YAML files.
- Ego-centric Observations: Optimized for policy generalization by focusing on relative coordinates.- Visualization: OpenCV-based rendering for debugging and monitoring agent behavior.

## 🛠 Technical Specifications
### Unicycle Dynamics
The robot follows non-holonomic dynamics, integrated over a discrete time step $\Delta t$ based on the following state transition:$$\begin{cases} 
x_{t+1} = x_t + v \cdot \cos(\theta_t) \cdot \Delta t \\
y_{t+1} = y_t + v \cdot \sin(\theta_t) \cdot \Delta t \\
\theta_{t+1} = \theta_t + \omega \cdot \Delta t 
\end{cases}$$

### Observation & Action Spaces
* Action Space: Represents continuous $[v, \omega]$ (linear and angular velocity).
* Observation Space: Ego-centric state including:
    * Goal: Distance and heading error to the target.
    * Lidar/Obstacles: Relative distances and angles to the nearest detected obstacles.
## ⚙️ ConfigurationThe 
Environment behavior is controlled via the */configs* directory. This allows for seamless switching between different training and benchmarking scenarios. 

**Initialize with a specific config**:
```python
env = UnicycleEnv(config_path="configs/env.yaml")
```

## 📁 Enviroment Structureenvs
- *envs/unicycle_env.pyv*: Core Gymnasium environment implementation.
- *envs/obstacle.py*: Implementation of the obstacles managment. 
- *envs/test_env.py*: A simple test script.

- *configs/*: YAML files defining workspace dimensions, obstacle counts, and spawn logic.

