# Implementation Workflow: RL-Based Navigation Controller

## 1. Goal
Implement a reinforcement-learning-based feedback controller for a mobile robot (unicycle model) that directly outputs continuous velocity commands $(v, \omega)$, given the robot state, a target pose, and local obstacle information. The RL controller will be benchmarked against a classical Lyapunov-based controller.

---

## 2. Overall Architecture
The system follows a layered robotics architecture:

* **Planning Layer (Fixed/Classical):** Uses a roadmap or planner (e.g., `loco_nav`) to output a global path or local target (waypoint).
* **Control Layer (Learning-Based):** Receives the robot state, target pose, and obstacle information; outputs control commands $(v, \omega)$.
* **Robot Model (Fixed):** Unicycle/Dubins kinematics used inside the simulation environment.



---

## 3. Step-by-Step Implementation

### Step 1 – Define the Simulation Environment
**Module:** `envs/unicycle_env.py`
Implement the unicycle kinematics equations:
$$\dot{x} = v \cos \theta$$
$$\dot{y} = v \sin \theta$$
$$\dot{\theta} = \omega$$

The environment should follow the OpenAI Gym/Gymnasium interface:
* `step(action)`: Transitions state, returns `obs, reward, terminated, info`.
* `reset()`: Resets robot/goal to initial positions.
* **Termination Conditions:** Target reached, collision, or max episode length.

### Step 2 – Define the State Representation
**Observation Vector ($S$):** Low-dimensional and ego-centric.
* Robot pose error relative to target: $[dx_{goal}, dy_{goal}, d\theta_{goal}]$.
* Local obstacle information: $[d_{obs, 1}, \phi_{obs, 1}, \dots, d_{obs, N}, \phi_{obs, N}]$ for the $N$ closest obstacles.

### Step 3 – Define the Action Space
**Action ($a$):** Continuous velocity commands.
* $a = (v, \omega)$
* **Constraints:** Apply physical limits $v \in [v_{min}, v_{max}]$ and $\omega \in [\omega_{min}, \omega_{max}]$.

### Step 4 – Implement the Reward Function
**Module:** `envs/rewards.py`
The reward $R$ encourages efficient and safe navigation:
* **Positive:** Progress toward the goal (decrease in Euclidean distance).
* **Negative:** Collision penalty (large), proximity penalty (exponential), and control effort penalty ($|\omega|$).

### Step 5 – Implement the Baseline Controller
**Module:** `controllers/lyapunov_controller.py`
* Integrate the existing Lyapunov-based controller from your professor's framework.
* Used to provide a quantitative baseline for success rate and smoothness.

### Step 6 – Implement the RL Controller Wrapper
**Module:** `controllers/rl_controller.py`
* A wrapper that loads the trained policy.
* Ensures the interface is identical to the Lyapunov controller for easy switching.

### Step 7 – Implement the Neural Networks
**Module:** `models/`
* **Actor Network:** Maps state to the mean/std of $(v, \omega)$.
* **Critic Network:** Maps state to the estimated Value $V(s)$.
* **Architecture:** Small MLPs (2–3 hidden layers, 64–128 units).

### Step 8 – Implement the RL Algorithm
**Module:** `algorithms/ppo.py`
* Use **Proximal Policy Optimization (PPO)**.
* Manage rollout collection, advantage estimation (GAE), and policy updates.

### Step 9 – Training Pipeline
**Module:** `training/train_ppo.py`
* **Incremental Training:** Start in obstacle-free zones, then move to cluttered environments.
* Log metrics using TensorBoard/WandB.

### Step 10 – Evaluation and Comparison
**Module:** `training/evaluate.py`
Compare the two controllers across:
1.  **Success Rate** (Reached goal vs. Collided).
2.  **Path Efficiency** (Actual distance vs. Straight-line distance).
3.  **Control Smoothness** (Variance in $\omega$).

---

## 4. Development Strategy
1.  **Verify Physics:** Ensure the `step` function correctly moves the robot.
2.  **Baseline Test:** Ensure the Lyapunov controller works in the new Gym environment.
3.  **RL Training:** Train the agent and monitor the reward curve.
4.  **Reporting:** Compare performance in diverse scenarios.