import cv2
import time
import sys
import os

# Add the parent directory to path so we can import from 'controllers'
# (Only needed if you are running this script from inside the 'envs' folder)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unicycle_env import UnicycleEnv

# 1. Setup
randome_env = 'env'
empty_env = 'empty_env'
fixed_env = 'fixed_env'
nav_env = 'env_nav_ev'
env = UnicycleEnv(nav_env)
env.render_mode = "human" # Enable the window

seed = 0
obs, info = env.reset(seed)
print("Environment Reset.")


# 2. Render the first frame and Wait for Space
first_frame = env.render()
cv2.imshow("Unicycle Nav", first_frame) # Ensure window is created

print("paused... Press SPACE to start!")
while True:
    # 0 means wait indefinitely for a key press
    key = cv2.waitKey(0) 
    if key == 32:  # 32 is the ASCII code for Space bar
        print("Starting Simulation...")
        break


# 3. Main Loop
for _ in range(10):
    action = env.action_space.sample()  # Random v, omega
    obs, terminated, truncated, info = env.step(action)
    
    # Render
    env.render()
    
    # Slow down the loop so you can watch it (50ms delay)
    # This also allows the window to refresh events
    cv2.waitKey(50)

    print(f"Obs: {obs[:3]}... | Done: {terminated}")
    
    if terminated or truncated:
        if info.get('is_success', False):
            print("🎉 GOAL REACHED!")
        else:
            print("❌ FAILED (Collision or Timeout)")
        
# Optional: Wait again at the end before closing
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()