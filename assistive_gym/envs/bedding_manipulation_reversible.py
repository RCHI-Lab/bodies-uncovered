import os, time
import numpy as np
from numpy.lib.function_base import append
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch import set_default_tensor_type

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh

class BeddingManipulationEnv(AssistiveEnv):
    def __init__(self, robot, human, use_mesh=False):
        if robot is None:
            super(BeddingManipulationEnv, self).__init__(robot=None, human=human, task='bedding_manipulation', obs_robot_len=14, obs_human_len=0, frame_skip=1, time_step=0.01, deformable=True)
            self.use_mesh = use_mesh
        
        self.take_pictures = False
        self.rendering = False
        self.fixed_target = True
        self.target_limb_code = 4
        self.fixed_pose = True
        self.reverse = True

    def step(self, action):
        obs = self._get_obs()
        # return 0, 0, 1, {}
        if self.rendering:
            print(obs)

        # * scale bounds the 2D grasp and release locations to the area over the mattress (action nums only in range [-1, 1])
        scale = [0.44, 1.05]
        grasp_loc = action[0:2]*scale
        release_loc = action[2:4]*scale

        # * get points on the blanket, initial state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        # * get rid of any nontarget points that are not covered by the initial state of the blanket (will not be considered in reward calculation at the end of the step)
        self.non_target_initially_uncovered(data)

        # initial_uncovered_target = self.uncover_target_reward(data)
        # initial_uncovered_nontarget = self.uncover_nontarget_reward(data)
        # reward_head_kept_uncovered = self.keep_head_uncovered_reward(data)

        # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
        dist = []
        for i, v in enumerate(data[1]):
            v = np.array(v)
            d = np.linalg.norm(v[0:2] - grasp_loc)
            dist.append(d)
        anchor_idx = np.argpartition(np.array(dist), 4)[:4]
        # for a in anchor_idx:
            # print("anchor loc: ", data[1][a])

        # * anchor sphere to the calculated points on the blanket
        constraint_ids = self.grasp_blanket(data, anchor_idx)
        # * update grasp_loc var with the location of the central anchor point on the cloth
        grasp_loc = self.sphere_ee.get_base_pos_orient()[0][0:2]
        # * move sphere up by some delta z, then to the release location
        self.move_from_start_to_end_loc(grasp_loc, release_loc)
        # * release the cloth at the release point
        self.release_blanket(constraint_ids)

        # * get points on the blanket, final state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        # * compute uncover rewards
        reward_uncover_target = self.target_reward(data, reward_on="uncover")
        reward_uncover_nontarget = self.uncover_nontarget_reward(data)
        reward_distance_btw_grasp_release = -150 if np.linalg.norm(grasp_loc - release_loc) >= 1.5 else 0
        reward_head_kept_uncovered = self.keep_head_uncovered_reward(data)

        # * if attempting to find reversible solutions, repeat the process
        # *    grasp the same points on the blanket that were grasped earlier
        # ?    should we grasp the same points as before or should we grasp the points directly underneath the previous release location?
        # ?    cloth could move while settling such that there is no cloth underneath the previous release points
        if self.reverse:
            # * release location is the previous grasp location, grasp location is the previous release location
            release_loc = grasp_loc
            # * anchor sphere to the calculated points on the blanket
            #! currently anchoring to the same points as when the blanket was initially grasped. is this an appropriate approach?
            constraint_ids = self.grasp_blanket(data, anchor_idx)
            # * update grasp_loc var with the location of the central anchor point on the cloth
            grasp_loc = self.sphere_ee.get_base_pos_orient()[0][0:2]
            # * move sphere up by some delta z, then to the release location
            self.move_from_start_to_end_loc(grasp_loc, release_loc)
            # * release the cloth at the release point
            self.release_blanket(constraint_ids)

            #! NEEDS TESTING
            reward_recover_target = self.target_reward(data, reward_on="cover")
            # ? how should the reward consider the nontarget points when reversing the grasp/release


        # * sum and weight rewards from individual functions to get overall reward
        # TODO: Need to consider all the rewards for reversing (recovering target) in this sum as well!
        reward = self.config('uncover_target_weight')*reward_uncover_target + self.config('uncover_nontarget_weight')*reward_uncover_nontarget + self.config('grasp_release_distance_max_weight')*reward_distance_btw_grasp_release + self.config('keep_head_uncovered_weight')*reward_head_kept_uncovered
        
        if self.rendering:
            print("rewards for each measure:", reward_uncover_target, reward_uncover_nontarget, reward_distance_btw_grasp_release, reward_head_kept_uncovered)
            print("overall reward: ", reward)

        info = {}
        self.iteration += 1
        done = self.iteration >= 1

        # return 0, 0, 1, {}
        return obs, reward, done, info


    def grasp_blanket(self, blanket_state, anchor_idx):

        # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
        self.sphere_ee.set_base_pos_orient(blanket_state[1][anchor_idx[0]], np.array([0,0,0]))
        constraint_ids = []
        constraint_ids.append(p.createSoftBodyAnchor(self.blanket, anchor_idx[0], self.sphere_ee.body, -1, [0, 0, 0]))
        for i in anchor_idx[1:]:
            pos_diff = np.array(blanket_state[1][i]) - np.array(blanket_state[1][anchor_idx[0]])
            constraint_ids.append(p.createSoftBodyAnchor(self.blanket, i, self.sphere_ee.body, -1, pos_diff))

        return constraint_ids

    def release_blanket(self, constraint_ids):
        # * release the cloth at the release point, sphere is at the same arbitrary z position in the air
        for i in constraint_ids:
            p.removeConstraint(i, physicsClientId=self.id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

    def move_from_start_to_end_loc(self, start_loc, end_loc, delta_z=0.5):

        # * move sphere up by some delta z
        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        delta_z = 0.5                           # distance to move up
        final_z = delta_z + current_pos[2]      # global z position after moving up delta z
        while current_pos[2] <= final_z:
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([0, 0, 0.005]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]
        
        # * move sphere to the release location, release the blanket
        # * determine delta x and y, make sure it is, at max, close to 0.005
        travel_dist = end_loc - start_loc
        num_steps = np.abs(travel_dist//0.005).max()
        delta_x, delta_y = travel_dist/num_steps

        current_pos = self.sphere_ee.get_base_pos_orient()[0]
        for _ in range(int(num_steps)):
            self.sphere_ee.set_base_pos_orient(current_pos + np.array([delta_x, delta_y, 0]), np.array([0,0,0]))
            p.stepSimulation(physicsClientId=self.id)
            current_pos = self.sphere_ee.get_base_pos_orient()[0]

        # * continue stepping simulation to allow the cloth to settle
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.id)

    def change_point_color(self, points_target_limb, limb, ind, rgb = [0, 1, 0.5, 1]):
        p.changeVisualShape(points_target_limb[limb][ind].body, -1, rgbaColor=rgb, flags=0, physicsClientId=self.id)

    def target_reward(self, blanket_state, reward_on):
        '''
        give the robot a reward for either covering or uncovering (controlled by reward_on arg) the target limb
        '''
        points_covered = 0
        uncovered_rgb = [0, 1, 0.5, 1]
        covered_rgb = [1, 1, 1, 1]
        threshold = 0.028
        total_points = self.total_target_point_count

        # * count # of target points covered by the blanket, subtract from # total points to get the # point uncovered
        for limb, points_pos_target_limb_world in self.points_pos_target_limb_world.items():
            for point in range(len(points_pos_target_limb_world)):
                covered = False
                for i, v in enumerate(blanket_state[1]):
                    if abs(np.linalg.norm(v[0:2]-points_pos_target_limb_world[point][0:2])) < threshold:
                        covered = True
                        points_covered += 1
                        break
                if self.rendering:
                    rgb = covered_rgb if covered else uncovered_rgb
                    self.change_point_color(self.points_target_limb, limb, point, rgb = rgb)

        if reward_on == "cover":
            return (points_covered/total_points)*100
        if reward_on == "uncover":
            points_uncovered = total_points - points_covered

            if self.rendering:
                print("total_target points:", total_points)
                print("target uncovered:", points_uncovered)

            return (points_uncovered/total_points)*100


    # ? when looking to find reversible solutions, what are we concerned about with the nontarget points
    # ? are we just looking at if there are new nontarget exposures?
    def uncover_nontarget_reward(self, blanket_state):
        '''
        discourage the robot from learning policies that uncover nontarget body parts
        '''
        points_covered = 0
        uncovered_rgb = [1, 0, 0, 1]
        covered_rgb = [0, 0, 1, 1]
        threshold = 0.028
        total_points = self.total_nontarget_point_count - len(self.points_pos_nontarget_limb_world[self.human.head])

        # account for case where all nontarget points were initially uncovered
        if total_points == 0:
            return 0

        # count number of target points covered by the blanket
        for limb, points_pos_nontarget_limb_world in self.points_pos_nontarget_limb_world.items():
            for point in range(len(points_pos_nontarget_limb_world)):
                covered = False
                for i, v in enumerate(blanket_state[1]):
                    if abs(np.linalg.norm(v[0:2]-points_pos_nontarget_limb_world[point][0:2])) < threshold:
                        covered = True
                        points_covered += 1
                        break
                if self.rendering:
                    rgb = covered_rgb if covered else uncovered_rgb
                    self.change_point_color(self.points_nontarget_limb, limb, point, rgb = rgb)
        points_uncovered = total_points - points_covered
        
        if self.rendering:
            print("total nontarget points:", total_points)
            print("nontarget uncovered:", points_uncovered)

        # 100 when all points uncovered, 0 when all still covered
        return (points_uncovered/total_points)*-100
        
    def keep_head_uncovered_reward(self, blanket_state):
        '''
        discourage the robot from learning policies that cover the head
        '''
        points_covered = 0
        uncovered_rgb = [0, 0, 1, 1]
        covered_rgb = [1, 0, 0, 1]
        threshold = 0.028
        points_pos_head_world = self.points_pos_nontarget_limb_world[self.human.head]
        total_points = len(points_pos_head_world)

        # count number of target points covered by the blanket
        for point in range(len(self.points_pos_nontarget_limb_world[self.human.head])):
            covered = False
            for i, v in enumerate(blanket_state[1]):
                if abs(np.linalg.norm(v[0:2]-points_pos_head_world[point][0:2])) < threshold:
                    covered = True
                    points_covered += 1
                    break
            if self.rendering:
                rgb = covered_rgb if covered else uncovered_rgb
                self.change_point_color(self.points_nontarget_limb, self.human.head, point, rgb = rgb)

        if self.rendering:
            print("total points on head:", total_points)
            print("points on head covered", points_covered)

        # 100 when all points uncovered, 0 when all still covered
        return (points_covered/total_points)*-100

    def non_target_initially_uncovered(self, blanket_state):
        '''
        removes nontarget points on the body that are uncovered when the blanket is in it's initial state from the nontarget point set
        '''
        points_covered = 0
        threshold = 0.028
        points_to_remove = {}
        points_to_remove_count = 0

        # * create a list of the nontarget points not covered by the blanket
        for limb, points_pos_nontarget_limb_world in self.points_pos_nontarget_limb_world.items():
            if limb != self.human.head:
                points_to_remove[limb] = []
                for point in range(len(points_pos_nontarget_limb_world)):
                    covered = False
                    for i, v in enumerate(blanket_state[1]):
                        if abs(np.linalg.norm(v[0:2]-points_pos_nontarget_limb_world[point][0:2])) < threshold:
                            covered = True
                            points_covered += 1
                            break
                    if covered == False:
                        points_to_remove[limb].append(point)
        
        # * remove the identified points from the list of all nontarget points for each limb
        # *     points removed in reverse order so that indexs of identified points don't shift
        for limb in self.points_pos_nontarget_limb_world.keys():
            if limb != self.human.head:
                points_to_remove_count += len(points_to_remove[limb])
                # print("count of points on nontarget initially:", len(self.points_pos_nontarget_limb_world[limb]), len(self.points_nontarget_limb[limb]))
                for point in reversed(points_to_remove[limb]):
                    self.points_pos_nontarget_limb_world[limb].pop(point)
                    if self.rendering:
                        p.removeBody(self.points_nontarget_limb[limb][point].body)
                        self.points_nontarget_limb[limb].pop(point)
                # print("count of points on nontarget now:", len(self.points_pos_nontarget_limb_world[limb]), len(self.points_nontarget_limb[limb]))

        # print(self.total_nontarget_point_count)
        self.total_nontarget_point_count -= points_to_remove_count

    def _get_obs(self, agent=None):
        if self.fixed_target:
            return np.concatenate([np.concatenate(self.human.get_pos_orient(limb)) for limb in self.target_limb], axis = None)
        else:
            #! REVISIT THIS
            selected_target_limb = np.zeros(12)
            selected_target_limb[self.target_limb_code] = 1
            human_limb_pos = np.concatenate([self.human.get_pos_orient(joint) for joint in self.human.all_joint_indices]).ravel()
            return selected_target_limb + human_limb_pos

    def reset(self):
        super(BeddingManipulationEnv, self).reset()
        body_shape = body_shape=np.zeros((1, 10)) if self.fixed_pose else None
        self.build_assistive_env(fixed_human_base=False, gender='female', human_impairment='none', furniture_type='hospital_bed', body_shape=body_shape)

        # * enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        # * Setup human in the air, with legs and arms slightly seperated
        joints_positions = [(self.human.j_left_hip_y, -10), (self.human.j_right_hip_y, 10), (self.human.j_left_shoulder_x, -20), (self.human.j_right_shoulder_x, 20)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([0, -0.2, 1.1], [-np.pi/2.0, 0, np.pi])

        if not self.fixed_pose:
            # * Add small variation to the body pose
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
            # print(motor_positions)
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(-0.2, 0.2, size=len(motor_indices)))
            # self.increase_pose_variation()
            # * Increase friction of joints so human doesn't fail around exessively as they settle
            # print([p.getDynamicsInfo(self.human.body, joint)[1] for joint in self.human.all_joint_indices])
            self.human.set_whole_body_frictions(spinning_friction=2)

        # * Let the person settle on the bed
        p.setGravity(0, 0, -1, physicsClientId=self.id)
        # * step the simulation a few times so that the human has some initial velocity greater than the at rest threshold
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.id)
        # * continue stepping the simulation until the human joint velocities are under the threshold
        threshold = 1e-2
        settling = True
        numsteps = 0
        while settling:
            settling = False
            for i in self.human.all_joint_indices:
                if np.any(np.abs(self.human.get_velocity(i)) >= threshold):
                    p.stepSimulation(physicsClientId=self.id)
                    numsteps += 1
                    settling = True
                    break
            if numsteps > 400:
                break
        # print("steps to rest:", numsteps)

        # * Lock the person in place
        self.human.control(self.human.all_joint_indices, self.human.get_joint_angles(), 0.05, 100)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        
        if self.use_mesh:
            # Replace the capsulized human with a human mesh
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -60), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])
 
        # * select a target limb to uncover (may be fixed or random)
        self.target_limb_code = self.np_random.random_integers(0,11) if not self.fixed_target else self.target_limb_code
        self.target_limb = self.human.all_possible_target_limbs[self.target_limb_code]
        self.generate_points_along_body()
       
        # * spawn blanket
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # * change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)


        # * Drop the blanket on the person, allow to settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)


        # data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)
        # self.non_target_initially_uncovered(data)
        # self.uncover_nontarget_reward(data)

    
        # * Initialize enviornment variables
        # *     if using the sphere manipulator, spawn the sphere and run a modified version of init_env_variables()
        # self.time = time.time()
        if self.robot is None:
            # * spawn sphere manipulator
            position = np.array([-0.3, -0.86, 0.8])
            self.sphere_ee = self.create_sphere(radius=0.01, mass=0.0, pos = position, visual=True, collision=True, rgba=[0, 0, 0, 1])
            
            # * initialize env variables
            from gym import spaces
            # * update observation and action spaces
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            action_len = 4
            self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)
            # * Define action/obs lengths
            self.action_robot_len = 4
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))    # 1
            self.obs_human_len = 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)
        else:
            self.init_env_variables()
        
        
        # * Setup camera for taking images
        # *     Currently saves color images only to specified directory
        if self.take_pictures == True:
            self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=398)
            img, depth = self.get_camera_image_depth()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filename = time.strftime("%Y%m%d-%H%M%S") + '.png'
            cv2.imwrite(os.path.join('/home/mycroft/git/vBMdev/pose_variation_images/lower_var2', filename), img)
            
        return self._get_obs()


    def generate_points_along_body(self):
        '''
        generate all the target/nontarget posistions necessary to uniformly cover the body parts with points
        if rendering, generates sphere bodies as well
        '''

        self.points_pos_on_target_limb = {}
        self.points_target_limb = {}
        self.total_target_point_count = 0

        self.points_pos_on_nontarget_limb = {}
        self.points_nontarget_limb = {}
        self.total_nontarget_point_count = 0

        #! just for tuning points on torso
        # self.human.all_body_parts = [self.human.waist]

        # * create points on all the body parts
        for limb in self.human.all_body_parts:

            # * get the length and radius of the given body part
            length, radius = self.human.body_info[limb] if limb not in self.human.limbs_need_corrections else self.human.body_info[limb][0]

            # * create points seperately depending on whether or not the body part is/is a part of the target limb
            # *     generates list of point positions around the body part capsule (sphere if the hands)
            # *     creates all the spheres necessary to uniformly cover the body part (spheres created at some arbitrary position (transformed to correct location in update_points_along_body())
            # *     add to running total of target/nontarget points
            # *     only generate sphere bodies if self.rendering == True
            if limb in self.target_limb:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_target_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                if self.rendering:
                    self.points_target_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_target_limb[limb]), visual=True, collision=False, rgba=[1, 1, 1, 1])
                self.total_target_point_count += len(self.points_pos_on_target_limb[limb])
            else:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_nontarget_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_nontarget_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.04)
                if self.rendering:
                    self.points_nontarget_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_nontarget_limb[limb]), visual=True, collision=False, rgba=[0, 0, 1, 1])
                self.total_nontarget_point_count += len(self.points_pos_on_nontarget_limb[limb])

        # * transforms the generated spheres to the correct coordinate space (aligns points to the limbs)
        self.update_points_along_body()
    
    def update_points_along_body(self):
        '''
        transforms the target/nontarget points created in generate_points_along_body() to the correct coordinate space so that they are aligned with their respective body part
        if rendering, transforms the sphere bodies as well
        '''

        # * positions of the points on the target/nontarget limbs in world coordinates
        self.points_pos_target_limb_world = {}
        self.points_pos_nontarget_limb_world = {}

        # * transform all spheres for all the body parts
        for limb in self.human.all_body_parts:

            # * get current position and orientation of the limbs, apply a correction to the pos, orient if necessary
            limb_pos, limb_orient = self.human.get_pos_orient(limb)
            if limb in self.human.limbs_need_corrections:
                limb_pos = limb_pos + self.human.body_info[limb][1]
                limb_orient = self.get_quaternion(self.get_euler(limb_orient) + self.human.body_info[limb][2])
            
            # * transform target/nontarget point positions to the world coordinate system so they align with the body parts
            points_pos_limb_world = []

            if limb in self.target_limb:
                for i in range(len(self.points_pos_on_target_limb[limb])):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_target_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                    points_pos_limb_world.append(point_pos)
                    if self.rendering:
                        self.points_target_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_target_limb_world[limb] = points_pos_limb_world
            else:
                for i in range(len(self.points_pos_on_nontarget_limb[limb])):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_nontarget_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                    points_pos_limb_world.append(point_pos)
                    if self.rendering:
                        self.points_nontarget_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_nontarget_limb_world[limb] = points_pos_limb_world

    def increase_pose_variation(self):
        '''
        Allow more variation in the knee and elbow angles
          can be some random position within the lower and upper limits of the joint movement (range is made a little smaller than the limits of the joint to prevent angles that are too extreme)
        '''
        for joint in (self.human.j_left_knee, self.human.j_right_knee, self.human.j_left_elbow, self.human.j_right_elbow):
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states([joint])
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(self.human.lower_limits[joint]+0.1, self.human.upper_limits[joint]-0.1, 1))
