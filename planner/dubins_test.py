import cv2
import time
import numpy as np

# Classic absolute imports
# (This works because we will set PYTHONPATH before running)
from envs.unicycle_env import UnicycleEnv
from planner.dubins_planner import DubinsPlanner

# 1. Setup
env = UnicycleEnv()
env.render_mode = "human"  # Enable window
planner = DubinsPlanner(curvature_max=1.5)

obs, _ = env.reset()

# 2. Calculate Dubins Path
start_pose = env.state  # [x, y, theta]
goal_pose = env.goal    # [x, y, theta]

print(f"Planning from {start_pose} to {goal_pose}...")
path = planner.get_path(start_pose, goal_pose)

if path is None:
    print("No path found!")
else:
    print(f"Path found with {len(path)} waypoints.")
    # 3. Inject into environment
    env.set_render_trajectory(path)

# 4. Wait loop to visualize
print("Press SPACE to start stepping...")
while True:
    env.render()
    
    # Wait for space key (32) to step, or Esc (27) to exit
    key = cv2.waitKey(10)
    if key == 27: # ESC
        break