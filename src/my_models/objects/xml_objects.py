import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.transform_utils import *  # (by JadeCong)

class SoftTorsoObject(MujocoXMLObject):
    """
    Soft torso object
    """

    def __init__(self, name, joints="default", damping=None, stiffness=None):
        # super().__init__("my_models/assets/objects/soft_human_torso.xml", name=name, duplicate_collision_geoms=False)
        super().__init__("my_models/assets/objects/soft_human_torso.xml", name=name, joints=joints, duplicate_collision_geoms=False)  # by JadeCong

        self.damping = damping
        self.stiffness = stiffness

        if self.damping is not None:
            self.set_damping(damping)
        if self.stiffness is not None:
            self.set_stiffness(stiffness)

        # Define the body_pos and body_quat  # (by JadeCong)
        self.body_num = self._get_body_num()
        self.body_local_pos_array = self._get_body_local_pos_array()
        self.body_local_quat_array = self._get_body_local_quat_array()

    def _get_composite_element(self):
        return self._obj.find("./composite")

    def set_damping(self, damping):
        """
        Helper function to override the soft body's damping directly in the XML
        Args:
            damping (float, must be greater than zero): damping parameter to override the ones specified in the XML
        """
        assert damping > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        stiffness = float(solref_str[0])

        solref = np.array([stiffness, -damping])
        composite.set('solrefsmooth', array_to_string(solref))

    def set_stiffness(self, stiffness):
        """
        Helper function to override the soft body's stiffness directly in the XML
        Args:
            stiffness (float, must be greater than zero): stiffness parameter to override the ones specified in the XML
        """
        assert stiffness > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        damping = float(solref_str[1])

        solref = np.array([-stiffness, damping])
        composite.set('solrefsmooth', array_to_string(solref))

    def _get_body_num(self):  # (by JadeCong)
        """
        Get the number of bodies for this xml model.

        Returns: int
        """
        return self.get_model().nbody

    def _get_body_local_pos_array(self):  # (by JadeCong)
        """
        Get the body_pos array(position offset rel. to parent body)

        Returns: numpy.array
        """
        return np.array(self.get_model().body_pos)

    def _get_body_local_quat_array(self):  # (by JadeCong)
        """
        Get the body_quat array(orientation offset rel. to parent body)

        Returns: numpy.array
        """
        return np.array(self.get_model().body_quat)

    def get_body_local_pos(self, index):  # (by JadeCong)
        """
        Get the position relative to parent body given body index.

        Returns: numpy.array
        """
        if index > self.body_num:
            raise IndexError

        return np.array(self.body_local_pos_array[index])

    def get_body_local_quat(self, index):  # (by JadeCong)
        """
        Get the orientation relative to parent body given body index.

        Returns: numpy.array
        """
        if index > self.body_num:
            raise IndexError

        return np.array(self.body_local_quat_array[index])

    def get_body_world_pose_array(self, parent_pos, parent_quat):  # (by JadeCong)
        """
        Get the body_pos and body_quat(wxyz) arraies rel. to world.

        Returns: numpy.array, numpy.array
        """
        print("The body number of object: ", self.body_num)
        body_world_pos_array = np.zeros((self.body_num, 3))
        body_world_quat_array = np.zeros((self.body_num, 4))

        for idx in iter(range(self.body_num)):
            body_world_pos_array[idx], body_world_quat_array[idx] = self.get_body_world_pose(parent_pos, parent_quat, idx)

        return body_world_pos_array, body_world_quat_array

    def get_body_world_pose(self, parent_pos, parent_quat, index):  # (by JadeCong)
        """
        Get the body_pos and body_quat(wxyz) array rel. to world given the body index.

        Returns: numpy.array, numpy.array
        """
        body_mat_in_parent = pose2mat(tuple((tuple(self.body_local_pos_array[index]), tuple(convert_quat(self.body_local_quat_array[index])))))
        parent_mat_in_world = pose2mat(tuple((tuple(parent_pos), tuple(convert_quat(parent_quat)))))

        body_mat_in_world = pose_in_A_to_pose_in_B(body_mat_in_parent, parent_mat_in_world)
        pos, quat = mat2pose(body_mat_in_world)

        return np.array(pos), np.array(convert_quat(quat, to="wxyz"))

    def fit_body_world_quat(self, target_point, side_point, back_point, part="right"):  # (by JadeCong)
        """
        Fit the body_quat of torso up surface given torso_up_surface_index according to forward direction.

        Returns: numpy.array
        """
        # get the y axis direction
        y_axis = np.array(back_point - target_point)

        if part != "middle":
            # get the x axis direction
            if part == "right":
                x_axis = np.array(target_point - side_point)
            elif part == "left":
                x_axis = np.array(side_point - target_point)
            # get the z axis direction(based on right-hand rule)
            z_axis = np.cross(x_axis, y_axis)
        elif part == "middle":
            # get the z axis direction
            z_axis = np.array([0.000001, 0.000001, -1])  # right down direction for middle line of scanning trajectory:140-150
            # get the x axis direction
            x_axis = np.cross(y_axis, z_axis)

        # get the rotation matrix of target_point relative the world coordinate system
        rot_mat = np.array([[np.dot(np.array([1, 0, 0]), x_axis), np.dot(np.array([0, 1, 0]), x_axis), np.dot(np.array([0, 0, 1]), x_axis)],
                            [np.dot(np.array([1, 0, 0]), y_axis), np.dot(np.array([0, 1, 0]), y_axis), np.dot(np.array([0, 0, 1]), y_axis)],
                            [np.dot(np.array([1, 0, 0]), z_axis), np.dot(np.array([0, 1, 0]), z_axis), np.dot(np.array([0, 0, 1]), z_axis)]])

        # get the quaternion of target_point relative the world coordinate system(xyzw)
        target_quat = mat2quat(rot_mat)

        return target_quat

    def fit_body_world_quat_array(self, torso_start_index, torso_up_surface_index, torso_mesh_body_pos_world_array):  # (by JadeCong)
        """
        Fit the body_quat_array of torso up surface given torso_up_surface_index.

        Returns: numpy.array(xyzw)
        """
        # get the point array for every target point
        point_list_array = list()
        for idx in torso_up_surface_index:
            if (idx > torso_start_index[int(len(torso_start_index) / 2)]) or (idx < torso_start_index[int(len(torso_start_index) / 2)] - 10):
                if idx > torso_start_index[int(len(torso_start_index) / 2)]:  # for the torso right part(from start to end viewpoint:176-272)
                    if len(torso_start_index) > 8:  # for only 218-228 line
                        point_list_array.append([torso_mesh_body_pos_world_array[idx], torso_mesh_body_pos_world_array[idx - 44], torso_mesh_body_pos_world_array[idx + 1], "right"])
                    else:
                        point_list_array.append([torso_mesh_body_pos_world_array[idx], torso_mesh_body_pos_world_array[idx - 26], torso_mesh_body_pos_world_array[idx + 1], "right"])
                elif idx < torso_start_index[int(len(torso_start_index) / 2)] - 10:  # for the torso left part(from start to end viewpoint:46-124)
                    point_list_array.append([torso_mesh_body_pos_world_array[idx], torso_mesh_body_pos_world_array[idx + 26], torso_mesh_body_pos_world_array[idx + 1], "left"])
            elif (idx <= torso_start_index[int(len(torso_start_index) / 2)]) and (idx >= torso_start_index[int(len(torso_start_index) / 2)] - 10):  # for the torso middle part(from start to end viewpoint:150)
                point_list_array.append([torso_mesh_body_pos_world_array[idx], torso_mesh_body_pos_world_array[idx], torso_mesh_body_pos_world_array[idx + 1], "middle"])

        # get the body world quat array(for order trajectory)
        fit_traj_quat_array = np.zeros((len(point_list_array), 4))
        for i in iter(range(len(point_list_array))):
            fit_traj_quat_array[i] = self.fit_body_world_quat(point_list_array[i][0], point_list_array[i][1], point_list_array[i][2], point_list_array[i][3])

        return fit_traj_quat_array


class SoftBoxObject(MujocoXMLObject):
    """
    Soft box object
    """

    def __init__(self, name, damping=None, stiffness=None):
        super().__init__("my_models/assets/objects/soft_box.xml", name=name, duplicate_collision_geoms=False)

        self.damping = damping
        self.stiffness = stiffness

        if self.damping is not None:
            self.set_damping(damping)
        if self.stiffness is not None:
            self.set_stiffness(stiffness)


    def _get_composite_element(self):
        return self._obj.find("./composite")


    def set_damping(self, damping):
        """
        Helper function to override the soft body's damping directly in the XML
        Args:
            damping (float, must be greater than zero): damping parameter to override the ones specified in the XML
        """
        assert damping > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        stiffness = float(solref_str[0])

        solref = np.array([stiffness, -damping])
        composite.set('solrefsmooth', array_to_string(solref))


    def set_stiffness(self, stiffness):
        """
        Helper function to override the soft body's stiffness directly in the XML
        Args:
            stiffness (float, must be greater than zero): stiffness parameter to override the ones specified in the XML
        """
        assert stiffness > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        damping = float(solref_str[1])

        solref = np.array([-stiffness, damping])
        composite.set('solrefsmooth', array_to_string(solref))


class BoxObject(MujocoXMLObject):
    """
    Box object
    """

    def __init__(self, name):
        super().__init__("my_models/assets/objects/box.xml", name=name)