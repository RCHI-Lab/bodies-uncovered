import os
from gym import spaces
import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from .agents import human
from .agents.human import Human

human_controllable_joint_indices = human.test_joints # human.right_arm_joints# + human.left_arm_joints

class HumanSMPLXTestingEnv(AssistiveEnv):
    def __init__(self):
        super(HumanSMPLXTestingEnv, self).__init__(robot=None, human=Human(human_controllable_joint_indices, controllable=True), task='human_smplx_testing', obs_robot_len=0, obs_human_len=0)

    def step(self, action):
        # action = np.zeros(np.shape(action))
        self.take_step(action, gains=0.05, forces=1.0)
        # NOTE: camera rotation
        # speed = 6.0
        # p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=self.iteration*speed, cameraPitch=-15, cameraTargetPosition=[0, 0, self.human.get_heights()[0]/2.0], physicsClientId=self.id)
        return [], 0, False, {}

    def _get_obs(self, agent=None):
        return []

    def reset(self):
        super(HumanSMPLXTestingEnv, self).reset()
        self.build_assistive_env(furniture_type=None, human_impairment='no_tremor')

        # NOTE: set joint angles for human joints (in degrees)
        joints_positions = [(self.human.j_left_shoulder_x, -85), (self.human.j_right_shoulder_x, 85)]
        # joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)

        joint_angles = self.human.get_joint_angles_dict(self.human.all_joint_indices)
        for j in human.right_leg_joints + human.left_leg_joints:
            # Make all non controllable joints on the person static by setting mass of each link (joint) to 0
            p.changeDynamics(self.human.body, j, mass=0, physicsClientId=self.id)
            # Set velocities to 0
            self.human.set_joint_angles([j], [joint_angles[j]])

        # self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None)
        human_height, human_base_height = self.human.get_heights(set_on_ground=True)

        # NOTE: show mesh model
        self.human_mesh = HumanMesh()
        self.human_mesh.init(self.directory, self.id, self.np_random, out_mesh=self.human.out_mesh, vertices=self.human.vertices, joints=self.human.joints)
        mesh_height = (self.human.get_pos_orient(self.human.waist)[0] - self.human_mesh.get_joint_positions([0]))[0]
        self.human_mesh.set_base_pos_orient(mesh_height, [0, 0, 0, 1])
        spheres = self.create_spheres(radius=0.02, mass=0, batch_positions=self.human_mesh.get_joint_positions(list(range(22))), visual=True, collision=False, rgba=[0, 1, 0, 1])

        spheres = self.create_spheres(radius=0.02, mass=0, batch_positions=self.human_mesh.get_vertex_positions([6832, 4088]), visual=True, collision=False, rgba=[1, 1, 1, 1])
        # indices = self.human.vertices[:, 0] < 0.2
        # indices = np.logical_and(indices, self.human.vertices[:, 2] < -0.4)
        # indices = np.logical_and(indices, self.human.vertices[:, 2] > -0.45)
        # indices = np.logical_and(indices, self.human.vertices[:, 1] < 0.0)
        # indices = np.logical_and(indices, self.human.vertices[:, 1] > -0.05)
        # print('Indices:', [i for i, idx in enumerate(indices) if idx])
        # positions = self.human_mesh.get_vertex_positions(indices)
        # for i, v in enumerate(positions):
        #     self.point = self.create_sphere(radius=0.02, mass=0.0, pos=v, visual=True, collision=False, rgba=[(i+1)/len(positions), 0, 0, 1])

        # print('Human Abstract height:', human_height, 'm')
        # self.human.set_base_pos_orient([0, -0.09, human_base_height + 0.12], [0, 0, 0, 1])

        # NOTE: visualizing abstract model joints
        #rest
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.head)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.neck)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.chest)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.waist)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.upper_chest)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.j_upper_chest_x)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        #arms
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.right_upperarm)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.left_upperarm)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.right_forearm)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.left_forearm)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.j_right_wrist_x)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.j_left_wrist_x)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.right_pecs)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.left_pecs)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        #legs
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.right_shin)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.left_shin)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.right_thigh)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.left_thigh)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.right_foot)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.point = self.create_sphere(radius=0.02, mass=0.0, pos=self.human.get_pos_orient(self.human.left_foot)[0], visual=True, collision=False, rgba=[0, 1, 1, 1])

        p.setGravity(0, 0, 0, physicsClientId=self.id)

        p.resetDebugVisualizerCamera(cameraDistance=1.6, cameraYaw=0, cameraPitch=-15, cameraTargetPosition=[0, 0, 1.0], physicsClientId=self.id)
        # p.resetDebugVisualizerCamera(cameraDistance=2.0, cameraYaw=-87, cameraPitch=0, cameraTargetPosition=[0, 0, 0.6], physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

