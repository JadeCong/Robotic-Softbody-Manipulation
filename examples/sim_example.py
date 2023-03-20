import numpy as np
import robosuite as suite
from robosuite.environments.base import register_env


if __name__ == "__main__":
    # Register custom environment
    # register_env(UltrasoundScanning)
    
    # Create environment instance
    env = suite.make(
        env_name="PickPlace",  # try with other tasks like "Stack" and "Door", "PickPlace", "Lift", "NutAssembly", "TwoArmLift", "TwoArmPegInHole", "TwoArmHandover", "UltrasoundScan"
        robots="Panda",  # try with other robots like "Sawyer" and "Jaco", "Baxter", "IIWA", "Kinova3", "Panda", "UR5e", "Aubo", "Diana", "Realman", "xArm", "xMate"
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    
    # Reset the environment
    env.reset()
    
    # Run the simulation
    for i in range(1000):
        action = np.random.randn(env.robots[0].dof)  # sample random action
        obs, reward, done, info = env.step(action)  # take action in the environment
        env.render()  # render on display
