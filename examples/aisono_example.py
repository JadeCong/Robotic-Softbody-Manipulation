import sys
sys.path.append("/home/jade/Documents/JadeCloud/Works/Aisono/Projects/Workflows/Doing/PycharmProjects/aisono-robosuite")

import numpy as np
import robosuite as suite
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.ultrasound_scanning import UltrasoundScanning

# Register custom environment
register_env(UltrasoundScanning)

# Create environment instance
env = suite.make(
    env_name="PickPlace",  # try with other tasks like "Stack" and "Door", "PickPlace", "Lift", "NutAssembly", "UltrasoundScanning"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco", "Aubo", "Baxter", "IIWA", "Kinova3", "Panda", "UR5e"
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
