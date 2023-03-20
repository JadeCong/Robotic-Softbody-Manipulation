import sys, os, time
import pytest
import math
import numpy as np
import traceback
import glfw
import copy

from mujoco_py import ignore_mujoco_warnings as mj_ignore_warnings
from mujoco_py import MujocoException
from mujoco_py import const as mj_const
from mujoco_py import functions as mj_fn
from mujoco_py import load_model_from_path, load_model_from_mjb, load_model_from_xml
from mujoco_py import MjRenderContext, MjRenderContextOffscreen, MjRenderContextWindow, MjBatchRenderer, GlfwContext
from mujoco_py import MjSim, MjSimState
from mujoco_py import MjViewer, MjViewerBasic

from mujoco_py.cymj import PyMjModel, PyMjData
from mujoco_py.cymj import PyMjvScene, PyMjvCamera, PyMjvOption, PyMjvPerturb
from mujoco_py.cymj import PyMjUI, PyMjuiState, PyMjrContext, PyMjrRect
from mujoco_py.modder import LightModder, CameraModder, MaterialModder, TextureModder

from scipy.spatial.transform import Rotation as R


class HMD():  # anonymous object we can set fields on
    """
    Normally global variables aren't used like this in python,
    but we want to be as close as possible to the original file.
    """
    pass


model = None
sim = None
modder = None
hmd = HMD()


def render_callback(sim, viewer):
    """
    Render callback function for update texture of scene objects.
    """
    # global modder
    # if modder is None:
    #     modder = TextureModder(sim)
    # for name in sim.model.geom_names:
    #     modder.rand_all(name)
    renderer = MjRenderContext(sim, offscreen=False)
    renderer.add_marker(type=mj_const.GEOM_SPHERE,
                        size=np.ones(3) * 0.1,
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.ones(4),
                        label="marker")


@pytest.mark.requires_rendering
@pytest.mark.requires_glfw
def mujoco_sim():
    """
    Simulation test function.
    """
    # Statement of global variables
    global model, sim
    
    # Load the simulation model
    # model = load_model_from_path("/home/jade/Documents/JadeCloud/Works/Aisono/Projects/Workflows/Doing/PycharmProjects/"
    #                              "aisono-mujoco-py/apps/resources/scenes/aisono_simulation.xml")
    model = load_model_from_path("./resources/scenes/softcylinder.xml")
    # model = load_model_from_mjb("./resources/scenes/mjmodel.mjb")
    
    # Construct the simulation and viewer
    # sim = MjSim(model, render_callback=render_callback)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    
    # Simulation step update and viewer scene render
    step = 0
    while True:
        # TODO: Update the simulation scene states(robots and objects)
        # t = time.time()
        # x, y = math.cos(t), math.sin(t)
        # viewer.add_marker(type=4, pos=np.array([y, x, 0.2]), label="sphere")
        # sim.data.ctrl[0] = math.cos(t / 10.) * 0.01
        # sim.data.ctrl[1] = math.sin(t / 10.) * 0.01
        
        # TODO: Handle events (calls all callbacks)
        
        # Simulation forward and render the scene
        # sim.forward()
        sim.step()
        add_waypoint_site(viewer, mj_const, "wp1", pos=[0.2, 0.2, 0.2], quat=np.array([1, 0, 0, 0]))
        add_waypoint_site(viewer, mj_const, "wp2", pos=[0.3, 0.2, 0.3], quat=np.array([1, 0, 0, 0]))
        add_waypoint_line(viewer, mj_const, "line", pos_start=np.array([0.2, 0.2, 0.2]), pos_end=np.array([0.3, 0.2, 0.3]))
        
        viewer.render()
        time.sleep(0.01)
        
        # Check the step whether need to stop
        step += 1
        if step > 100000 and os.getenv('TESTING') is not None:
            break
    
    # Delete everything we allocated
    mj_fn.mj_deleteModel(model)
    
    # Deactivate MuJoCo
    mj_fn.mj_deactivate()

def add_waypoint_line(viewer, const, name="line", rgba=[1, 1, 0, 1], pos_start=[0, 0, 0], pos_end=[1, 1, 1]):
    # Import the dependencies
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Function for getting the rotation vector between two vectors
    def get_rotation_vector(vector1, vector2):  # rotation from vector1 to vector2
        # get the rotation axis
        rot_axis = np.cross(vector1, vector2)
        # get the rotation angle
        rot_angle = np.arccos(vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))
        # get the rotation vector
        rot_vector = rot_axis / np.linalg.norm(rot_axis) * rot_angle
        
        return rot_vector
    
    # Add the line between two waypoints
    viewer.add_marker(type=103,  # type=const.GEOM_LINE,  # 103 for line
                      size=[1, 1, np.linalg.norm(pos_end - pos_start)],
                      pos=pos_start,  # the pos of start waypoint
                      mat=R.from_rotvec(get_rotation_vector(np.array([0, 0, 1]), (pos_end - pos_start))).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                      rgba=rgba,
                      label=name)

def add_waypoint_site(viewer, const, name="wp", origin_size=[0.006, 0.006, 0.006], axis_size=[0.002, 0.002, 0.2],
                      pos=[0, 0, 0], quat=[1, 0, 0, 0]):
    # Import the dependencies
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    # Add the origin of the site
    viewer.add_marker(type=2,
                      # type=const.GEOM_SPHERE,  # 2 for sphere
                      size=origin_size,
                      pos=pos,
                      mat=R.from_quat(quat[[1,2,3,0]]).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                      rgba=[1, 1, 0, 1],
                      label=name)
    
    # Add the XYZ axes of the site
    viewer.add_marker(type=100,
                      # type=const.GEOM_ARROW,  # 5 for cylinder and 100 for arrow
                      size=axis_size,
                      pos=np.array(pos),
                      mat=(R.from_quat(quat[[1, 2, 3, 0]]) * R.from_euler('xyz', [0, 90, 0], degrees=True)).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                      rgba=[1, 0, 0, 1],
                      label="X")
    viewer.add_marker(type=100,
                      # type=const.GEOM_ARROW,  # 5 for cylinder and 100 for arrow
                      size=axis_size,
                      pos=np.array(pos),
                      mat=(R.from_quat(quat[[1, 2, 3, 0]]) * R.from_euler('xyz', [-90, 0, 0], degrees=True)).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                      rgba=[0, 1, 0, 1],
                      label="Y")
    viewer.add_marker(type=100,
                      # type=const.GEOM_ARROW,  # 5 for cylinder and 100 for arrow
                      size=axis_size,
                      pos=np.array(pos),
                      mat=(R.from_quat(quat[[1, 2, 3, 0]]) * R.from_euler('xyz', [0, 0, 0], degrees=True)).as_matrix().flatten(),  # transfer the index from wxyz to xyzw
                      rgba=[0, 0, 1, 1],
                      label="Z")


if __name__ == "__main__":
    print("Simulation Starting...")
    mujoco_sim()
    print("Simulation Terminated!")
