from logging import captureWarnings
import os, time
import numpy as np
from numpy.lib.function_base import append
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from trimesh.permutate import transform

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

#TODO: NEED TO FIX THE MISALIGNMENT BETWEEN THE CAPSULE HUMAN AND THE SMPLX MESH

class RemoveContactEnv(AssistiveEnv):
    def __init__(self, robot, human, use_mesh=False):
        if robot is None:
            super(RemoveContactEnv, self).__init__(robot=robot, human=human, task='bed_bathing', obs_robot_len=4, obs_human_len=(16 + (len(human.controllable_joint_indices) if human is not None else 0)), frame_skip=1, time_step=0.01, deformable=True)
            self.use_mesh = use_mesh

    def step(self, action):
        obs = self._get_obs()

        grasp_loc = action

        # move sphere to 2D grasp location, some arbitrary distance z = 1 in the air
        #! don't technically need to do this, remove later
        self.sphere_ee.set_base_pos_orient(np.append(grasp_loc, 1), np.array([0,0,0]))

        # get points on the blanket, initial state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        print("got blanket data")

        #check how many points on the blanket are initally in contact with the blanket
        points = self.points_in_contact(data)
        print("points in contact with the blanket: ", points)

        # calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist = []
        for i, v in enumerate(data[1]):
            v = np.array(v)
            d = np.linalg.norm(v[0:2] - grasp_loc)
            dist.append(d)
        anchor_idx = np.argpartition(np.array(dist), 4)[:4]
        for a in anchor_idx:
            print("anchor loc: ", data[1][a])

        # update grasp_loc var with the location of the central anchor point on the cloth
        grasp_loc = np.array(data[1][anchor_idx[0]][0:2])
        print("GRASP LOC =", grasp_loc)

        # move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
        self.sphere_ee.set_base_pos_orient(data[1][anchor_idx[0]], np.array([0,0,0]))
        constraint_ids = []
        constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))

        for i in anchor_idx[1:]:
            pos_diff = np.array(data[1][i]) - np.array(data[1][anchor_idx[0]])
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, [0, 0, 0]))
        print("sphere moved to grasp loc, anchored")


        # move sphere up to the arbitrary z position z = 1
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        # print(current_pos[2])
        delta_z = 0.3                           # distance to move up
        final_z = delta_z + current_pos[2]      # global z position after moving up delta z
        while current_pos[2] <= final_z:
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
            # print(current_pos[2])

        print(f"sphere moved {delta_z}, current z pos {current_pos[2]}")

        # get points on the blanket, final state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        points = self.points_in_contact(data)
        print("points in contact with the blanket: ", points)



        time.sleep(600)

        self.iteration += 1
        done = self.iteration >= 1

        return obs, 0, done, {}
    
    def change_point_color(self, limb, ind, rgb = [0, 1, 0.5, 1]):
        p.changeVisualShape(self.points_target_limb[limb][0][ind].body, -1, rgbaColor=rgb, flags=0, physicsClientId=self.id)

    def get_contact_state(self, limb, ind):
        return self.points_target_limb[limb][1][ind]

    def set_contact_state(self, limb, ind, state=False):
        self.points_target_limb[limb][1][ind] = state


    def points_in_contact(self, blanket_state):
        contact_points = 0
        contact_rgb = [0, 1, 0.5, 1]
        no_contact_rgb = [1, 1, 1, 1]
        threshold = 0.03

        # count number of target points covered by the blanket
        for limb, points_pos_target_limb_world in self.points_pos_target_limb_world.items():
            for point in range(len(points_pos_target_limb_world)):
                for i, v in enumerate(blanket_state[1]):
                    # target_foot = np.array(target_foot)
                    # v = np.array(v)
                    if abs(np.linalg.norm(v-points_pos_target_limb_world[point])) < threshold:
                        contact_points += 1
                        self.set_contact_state(limb, point, True)
                        break
                rgb = contact_rgb if self.get_contact_state(limb, point) else no_contact_rgb
                self.change_point_color(limb, point, rgb = rgb)

        return contact_points

    def points_in_contact_smplx(self, blanket_state):
        contact_points = 0
        contact_rgb = [0, 1, 0.5, 1]
        no_contact_rgb = [1, 1, 1, 1]
        threshold = 0.03
        pos, orient = self.human.get_base_pos_orient()
        print(pos, orient)
        human_vertices = self.human.vertex_positions
        print("entered")

        human_vertices_world = []
        for point in range(0,len(human_vertices)-1,10):
            # print(point)
            point_pos = p.multiplyTransforms(pos, orient, human_vertices[point], [0, 0, 0, 1], physicsClientId=self.id)[0]
            human_vertices_world.append(np.array(point_pos))

        # count number of target points covered by the blanket
        for point in human_vertices_world:
            self.sphere = self.create_sphere(radius=0.005, mass=0.0, pos = point, visual=True, collision=True, rgba=no_contact_rgb)
            for i, v in enumerate(blanket_state[1]):
                # target_foot = np.array(target_foot)
                # v = np.array(v)
                if abs(np.linalg.norm(v-point)) < threshold:
                    contact_points += 1
                    p.changeVisualShape(self.sphere.body, -1, rgbaColor=contact_rgb, flags=0, physicsClientId=self.id)
                    break
        print("total points: ", len(human_vertices_world))
        print("contacts: ", contact_points)
        return contact_points

    def _get_obs(self, agent=None):
        return np.zeros(1)

    def reset(self):
        super(RemoveContactEnv, self).reset()
        self.build_assistive_env(fixed_human_base=False, gender='female', human_impairment='none', furniture_type='hospital_bed')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        
        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([0, -0.4, 0.8], [-np.pi/2.0, 0, np.pi])


        # Seperate the human's legs so that it's easier to uncover a single shin
        current_l = self.human.get_joint_angles(self.human.left_leg_joints)
        current_l[1] = -0.2
        current_r = self.human.get_joint_angles(self.human.right_leg_joints)
        current_r[1] = 0.2
        self.human.set_joint_angles(self.human.left_leg_joints, current_l, use_limits=True, velocities=0)
        self.human.set_joint_angles(self.human.right_leg_joints, current_r, use_limits=True, velocities=0)


        # Let the person settle on the bed
        p.setGravity(0, 0, -1, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)


        # Lock the person in place
        # self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        # self.human.set_mass(self.human.base, mass=0)
        # self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        if self.use_mesh:
            self.capsule_to_mesh_human()
            # print("done")

        # time.sleep(3600)

        # self.generate_points_along_body()

        # time.sleep(600)

        # spawn blanket
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        #TODO Adjust friction, should be lower so that the cloth can slide over the limbs
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=.2, useFaceContact=1, physicsClientId=self.id)


        # change alpha value so that it is a little more translucent, easier to see the relationship to the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.5], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.blanket, [0, 0, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)

        # Drop the blanket on the person, allow to settle
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)
    
        if self.robot is None:
            position = np.array([-0.3, -0.86, 0.8])
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos = position, visual=True, collision=True, rgba=[0, 0, 0, 1])

        # Initialize enviornment variables
        self.time = time.time()
        if self.robot is None:      # Sphere manipulator
            from gym import spaces
            # modified version of init_env_variables
            # update observation and action spaces
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            action_len = 4
            self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)

            # Define action/obs lengths
            self.action_robot_len = 4
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))    # 0
            self.obs_human_len = len(self._get_obs('human'))    # 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)
        else:
            self.init_env_variables()

        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        self.points_in_contact_smplx(data)

        time.sleep(600)
            
        return self._get_obs()

    def capsule_to_mesh_human(self):
        pos, orient = self.human.get_base_pos_orient()
        cap_joint_angles = np.degrees(self.human.get_joint_angles())
        waist_joints = cap_joint_angles[0:3]
        chest_joints = cap_joint_angles[3:6]
        upperchest_joints = cap_joint_angles[6:9]
        right_pecs_joints = cap_joint_angles[9:12]
        right_shoulder_joints = cap_joint_angles[12:15]
        right_elbow_joints = cap_joint_angles[15]
        right_forearm_joints = cap_joint_angles[16]
        right_wrist_joints = cap_joint_angles[17:19]
        left_pecs_joints = cap_joint_angles[19:22]
        left_shoulder_joints = cap_joint_angles[22:25]
        left_elbow_joints = cap_joint_angles[25]
        left_forearm_joints = cap_joint_angles[26]
        left_wrist_joints = cap_joint_angles[27:29]
        neck_joints = cap_joint_angles[29]
        head_joints = cap_joint_angles[30:33]
        right_hip_joints = cap_joint_angles[33:36]
        right_knee_joints = cap_joint_angles[36]
        right_ankle_joints = cap_joint_angles[37:40]
        left_hip_joints = cap_joint_angles[40:43]
        left_knee_joints = cap_joint_angles[43]
        left_ankle_joints = cap_joint_angles[44:]
        
        # print(waist_joints, chest_joints, upperchest_joints, right_pecs_joints, right_shoulder_joints, right_elbow_joints, right_forearm_joints, right_wrist_joints, left_pecs_joints, left_shoulder_joints, left_elbow_joints, left_forearm_joints, left_wrist_joints, neck_joints, head_joints, right_hip_joints, right_knee_joints, right_ankle_joints, left_hip_joints, left_knee_joints, left_ankle_joints)
        # print(cap_joint_angles)
        
        # p.removeBody(self.human.body)

        self.human = HumanMesh()

        joints_positions = [(self.human.j_left_hip_x, left_hip_joints[0]),    #! OFF
                            (self.human.j_left_hip_y, left_hip_joints[1]),
                            (self.human.j_left_hip_z, left_hip_joints[2]),
                            (self.human.j_right_hip_x, right_hip_joints[0]),   #! OFF
                            (self.human.j_right_hip_y, right_hip_joints[1]),
                            (self.human.j_right_hip_z, right_hip_joints[2]),
                            (self.human.j_waist_x, waist_joints[0]),
                            (self.human.j_waist_y, waist_joints[1]),
                            (self.human.j_waist_z, waist_joints[2]),
                            (self.human.j_left_knee_x, left_knee_joints),
                            (self.human.j_right_knee_x, right_knee_joints),
                            (self.human.j_chest_x, chest_joints[0]),
                            (self.human.j_chest_y, chest_joints[1]),
                            (self.human.j_chest_z, chest_joints[2]),
                            (self.human.j_left_ankle_x, left_ankle_joints[0]),
                            (self.human.j_left_ankle_y, left_ankle_joints[1]),
                            (self.human.j_left_ankle_z, left_ankle_joints[2]),
                            (self.human.j_right_ankle_x, right_ankle_joints[0]),
                            (self.human.j_right_ankle_y, right_ankle_joints[1]),
                            (self.human.j_right_ankle_z, right_ankle_joints[2]),
                            (self.human.j_upper_chest_x, upperchest_joints[0]),
                            (self.human.j_upper_chest_y, upperchest_joints[1]),
                            (self.human.j_upper_chest_z, upperchest_joints[2]),
                            (self.human.j_lower_neck_y, neck_joints),
                            (self.human.j_left_pecs_x, left_pecs_joints[0]),
                            (self.human.j_left_pecs_y, left_pecs_joints[1]),
                            (self.human.j_left_pecs_z, left_pecs_joints[2]),
                            (self.human.j_right_pecs_x, right_pecs_joints[0]),
                            (self.human.j_right_pecs_y, right_pecs_joints[1]),
                            (self.human.j_right_pecs_z, right_pecs_joints[2]),
                            (self.human.j_upper_neck_x, head_joints[0]),
                            (self.human.j_upper_neck_y, head_joints[1]),
                            (self.human.j_upper_neck_z, head_joints[2]),
                            (self.human.j_left_shoulder_x, left_shoulder_joints[0]+85),
                            (self.human.j_left_shoulder_y, left_shoulder_joints[1]),
                            (self.human.j_left_shoulder_z, left_shoulder_joints[2]),  #! OFF
                            (self.human.j_right_shoulder_x, right_shoulder_joints[0]-85),
                            (self.human.j_right_shoulder_y, right_shoulder_joints[1]),
                            (self.human.j_right_shoulder_z, right_shoulder_joints[2]),  #! OFF
                            (self.human.j_left_elbow_y, left_elbow_joints),
                            (self.human.j_right_elbow_y, right_elbow_joints),
                            (self.human.j_left_wrist_x, left_wrist_joints[0]),
                            (self.human.j_left_wrist_y, left_wrist_joints[1]),
                            (self.human.j_right_wrist_x, right_wrist_joints[0]),
                            (self.human.j_right_wrist_y, right_wrist_joints[1])]
        # joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -60), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
        body_shape = np.zeros((1, 10))
        gender = 'female' # 'random'
        self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions)
        # self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=[])

        # self.human.set_base_pos_orient(pos + np.array([2,0.4,0]), orient)
        # self.human.set_base_pos_orient(pos+ np.array([0,-0.35,0.05]), orient)

    #TODO Consolidate information in self.points_target_limb and self.points_pos_target_limb_world <-- too many of these variables, get rid of some
    def generate_points_along_body(self):
        self.point_indices_to_ignore = []

        # list of all the target limbs to uncover
        self.target_limbs = [self.human.right_shin, self.human.right_foot, self.human.right_thigh, self.human.left_shin, self.human.left_foot, self.human.left_thigh]

        # initialize dictionaries for the positions of the points on the target limb and for the ids of those points
        self.points_pos_on_target_limb = {}
        self.points_target_limb = {}

        # initialize count of all the points created
        self.total_target_point_count = 0
        
        for limb in self.target_limbs:
            length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]
            self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
            num_points_on_target_limb = len(self.points_pos_on_target_limb[limb])
            point_ids = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*num_points_on_target_limb, visual=True, collision=False, rgba=[1, 1, 1, 1])
            contact_states = [False]*num_points_on_target_limb
            self.points_target_limb[limb] = [point_ids, contact_states]
            self.total_target_point_count += num_points_on_target_limb

        self.update_points_along_body()

    
    def update_points_along_body(self):
        self.points_pos_target_limb_world = {}
        for limb in self.target_limbs:
            limb_pos, limb_orient = self.human.get_pos_orient(limb)
            if limb in self.human.limbs_need_corrections:
                limb_pos = limb_pos + self.human.body_info[limb][1]
                limb_orient = self.get_quaternion(self.get_euler(limb_orient) + self.human.body_info[limb][2])
            points_pos_limb_world = []
            for points_pos_on_target_limb, point in zip(self.points_pos_on_target_limb[limb], self.points_target_limb[limb][0]):
                point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, points_pos_on_target_limb, [0, 0, 0, 1], physicsClientId=self.id)[0])
                points_pos_limb_world.append(point_pos)
                point.set_base_pos_orient(point_pos, [0, 0, 0, 1])
            self.points_pos_target_limb_world[limb] = points_pos_limb_world