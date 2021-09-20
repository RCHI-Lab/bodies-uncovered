import os, time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

class BedPoseEnv(AssistiveEnv):
    def __init__(self, robot, human, use_mesh=False):
        super(BedPoseEnv, self).__init__(robot=robot, human=human, task='bed_bathing', obs_robot_len=(16 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)), frame_skip=5, time_step=0.02, deformable=True)
        self.use_mesh = use_mesh

    def step(self, action):
        self.take_step(action, action_multiplier=0.003)
        img, depth = self.get_camera_image_depth()
        print(depth.shape)

        return np.zeros(1), 0, False, {}

    def _get_obs(self, agent=None):
        return np.zeros(1)

    def reset(self):
        super(BedPoseEnv, self).reset()
        self.build_assistive_env(None, fixed_human_base=False, gender='female', human_impairment='none')

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # Update robot and human motor gains
        self.robot.motor_gains = 0.05
        self.robot.motor_forces = 100.0

        # Create bed
        self.bed = self.create_box(half_extents=[0.965/2, 1.905/2, 0.305/2], mass=0.0, pos=[0, 0, 0.305/2], orientation=[0, 0, 0, 1], visual=True, collision=True, rgba=[1, 1, 1, 1])

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([0, 0.2, 0.85], [-np.pi/2.0, 0, 0])
        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        if self.use_mesh:
            # Replace the capsulized human with a human mesh
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -90), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])


        shoulder_pos = self.human.get_pos_orient(self.human.right_upperarm)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_forearm)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_hand)[0]

        # Perform base pose optimization
        target_ee_pos = np.array([-0.6, 0.2, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        pos, orient = self.robot.get_base_pos_orient()
        self.robot.set_base_pos_orient(pos + np.array([-2, -2, 0]), orient)
        # base_position = self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[], collision_objects=[self.human], wheelchair_enabled=False)

        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for mounted arms
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-0.9, 0.7, 0]) + base_position, [0, 0, 0, 1])

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, 0)

        # Create a blanket above the person
        # self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_1061v.obj'), scale=0.8, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        # self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_1061v.obj'), scale=0.8, mass=0.15, useBendingSprings=0, useMassSpring=1, springElasticStiffness=5, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.8, mass=0.15, useBendingSprings=0, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.01, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.0001, frictionCoeff=0.1, useFaceContact=1, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 1], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.blanket, [0, 0, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)

        # Drop the blanket on the person
        for _ in range(30):
            p.stepSimulation(physicsClientId=self.id)

        # Setup camera for taking depth images
        # self.setup_camera(camera_eye=[0, 0, 0.305+2.101], camera_target=[0, 0, 0.305], fov=60, camera_width=1920//4, camera_height=1080//4)
        # self.setup_camera(camera_eye=[0, 0, 0.305+0.101], camera_target=[0, 0, 0.305], fov=60, camera_width=1920//4, camera_height=1080//4)
        self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 0], fov=60, camera_width=468//2, camera_height=398)
        # 468 x 398
        # self.setup_camera(camera_eye=[0.5, 0.75, 1.5], camera_target=[-0.2, 0, 0.75])
        img, depth = self.get_camera_image_depth()
        depth = (depth - np.amin(depth)) / (np.amax(depth) - np.amin(depth))
        depth = (depth * 255).astype(np.uint8)
        print(depth)
        # cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_VIRIDIS)
        cv2.imshow('image', depth_colormap)
        # plt.imshow(img)
        # plt.show()
        # Image.fromarray(img[:, :, :3], 'RGB').show()

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()
