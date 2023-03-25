"""
Probe Gripper.
"""
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel


class ProbeGripper(GripperModel):
    """
    Ultrasound Probe Gripper.
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """
    
    def __init__(self, idn=0):
        super().__init__("scan_models/assets/grippers/probe_gripper.xml", idn=idn)
    
    def format_action(self, action):
        return action
    
    @property
    def init_qpos(self):
        return None
    
    @property
    def _important_geoms(self):
        return {
            "probe": ["probe_collision"]
        }
