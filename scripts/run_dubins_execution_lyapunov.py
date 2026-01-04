from simulation import run_simulation

def run_single_demo():
    from envs.unicycle_env import UnicycleEnv
    from planner.dubins_planner import DubinsPlanner
    from controllers.lyapunov_controller import LyapunovController, LyapunovParams

    env = UnicycleEnv()
    planner = DubinsPlanner(curvature_max=1.5, step_size=0.05)
    controller = LyapunovController(LyapunovParams(K_P=2.0, K_THETA=5.0))

    print("Starting visual demonstration...")
    result = run_simulation(env, planner, controller, render=True)
    
    if result:
        status = "SUCCESS" if result["is_success"] else "FAILED"
        print(f"Result: {status} | Final Error: {result['distance_error']:.4f}m")

if __name__ == "__main__":
    run_single_demo()
