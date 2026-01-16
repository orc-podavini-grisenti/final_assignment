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
planner = DubinsPlanner(curvature=1.5)

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
    env.render()

# 4. Wait loop to visualize
print("Press any key to close window...")

cv2.waitKey(0)
cv2.destroyAllWindows()
