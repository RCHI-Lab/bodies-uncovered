import os, time, argparse
from inspect import Traceback
import numpy as np
from numpy.lib.function_base import append
import pybullet as p
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torch import set_default_tensor_type

from .env import AssistiveEnv
from .agents.human_mesh import HumanMesh
from gym.utils import seeding

class BeddingManipulationParallelEnv(AssistiveEnv):
    def __init__(self, robot, human, use_mesh=False):
        if robot is None:
            super(BeddingManipulationParallelEnv, self).__init__(robot=None, human=human, task='bedding_manipulation', obs_robot_len=12, obs_human_len=0, frame_skip=1, time_step=0.01, deformable=True, opts='2_interactions')
            self.use_mesh = use_mesh

        self.bm_config = self.configp[self.task]

        if self.bm_config['target_limb_code'] == 'random':
            self.fixed_target_limb = False
        else:
            self.fixed_target_limb = True
            self.target_limb_code = int(self.bm_config['target_limb_code'])

        self.body_shape = None if self.bm_config.getboolean('vary_body_shape') else np.zeros((1, 10))

        if self.bm_config.getboolean('take_images'):
            self.take_images = True
            self.save_image_dir = self.bm_config['save_image_dir']

        # * all the parameters below are False unless spepcified otherwise by args
        self.render_body_points = self.bm_config.getboolean('render_body_points')
        self.fixed_pose = self.bm_config.getboolean('fixed_human_pose')
        self.verbose = self.bm_config.getboolean('bm_verbose')
        self.blanket_pose_var = self.bm_config.getboolean('vary_blanket_pose')
        self.take_images = self.bm_config.getboolean('take_images')
        self.cmaes_dc = self.bm_config.getboolean('cmaes_data_collect')

        # * these parameters don't have cmd args to modify them
        self.seed_val = 1001
        self.save_pstate = False
        self.pstate_file = None
    
    def step(self, action):
        obs = self._get_obs()

        if self.verbose:
            print("Target Limb Code:", self.target_limb_code)
            print("Observation:\n", obs)
            print("Action: ", action)

        # * scale bounds the 2D grasp and release locations to the area over the mattress (action nums only in range [-1, 1])
        scale = [0.44, 1.05]*4
        action = action*scale
        actions = [action[0:4], action[4:]]

        # * get points on the blanket, initial state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        # * get rid of any nontarget points that are not covered by the initial state of the blanket (will not be considered in reward calculation at the end of the step)
        self.non_target_initially_uncovered(data)
        head_points = len(self.points_pos_nontarget_limb_world[self.human.head])
        point_counts = [self.total_target_point_count, self.total_nontarget_point_count-head_points, head_points]

        grasp_locs = []
        release_locs = []
        anchor_idxs = []
        constraint_ids = []
        clipped = [False, False]
        reward_distance_btw_grasp_release = 0
        for i in range(2):
            # * calculate distance between the 2D grasp location and every point on the blanket, anchor points are the 4 points on the blanket closest to the 2D grasp location
            dist = []
            for _, v in enumerate(data[1]):
                v = np.array(v)
                d = np.linalg.norm(v[0:2] - actions[i][0:2])
                dist.append(d)
            anchor_idxs.append(np.argpartition(np.array(dist), 4)[:4])
            # for a in anchor_idx:
                # print("anchor loc: ", data[1][a])

            # * if no points on the blanket are within 2.8 cm of the grasp location, track that it would have been clipped
            if not np.any(np.array(dist) < 0.028):
                clipped[i] = True

            # * update grasp_loc var with the location of the central anchor point on the cloth
            grasp_locs.append(np.array(data[1][anchor_idxs[i][0]][0:2]))
            release_locs.append(actions[i][2:])
            reward_distance_btw_grasp_release += -150 if np.linalg.norm(grasp_locs[i] - release_locs[i]) >= 1.5 else 0

            # * move sphere down to the anchor point on the blanket, create anchor point (central point first, then remaining points) and store constraint ids
            self.sphere_ees[i].set_base_pos_orient(data[1][anchor_idxs[i][0]], np.array([0,0,0]))
            constraint_ids.append([])
            constraint_ids[i].append(p.createSoftBodyAnchor(self.blanket, anchor_idxs[i][0], self.sphere_ees[i].body, -1, [0, 0, 0]))
            for id in anchor_idxs[i][1:]:
                pos_diff = np.array(data[1][id]) - np.array(data[1][anchor_idxs[i][0]])
                constraint_ids[i].append(p.createSoftBodyAnchor(self.blanket, id, self.sphere_ees[i].body, -1, pos_diff))
        
        # * take image after blanket grasped
        if self.take_images: self.capture_images()

        # * move sphere 40 cm from the top of the bed
        delta_z = 0.4                           # distance to move up (with respect to the top of the bed)
        bed_height = 0.58                       # height of the bed
        final_z = delta_z + bed_height          # global goal z position
        num_steps = 0
        travel_z = []
        current_pos = []
        for i in range(2):
             current_pos.append(self.sphere_ees[i].get_base_pos_orient()[0])
             travel_z.append(final_z - current_pos[i][2])
             steps = np.abs(travel_z[i]//0.005)
             num_steps = steps if steps > num_steps else num_steps
        delta_zs = np.array(travel_z)/num_steps

        for _ in range(int(num_steps)):
            for i in range(2):
                self.sphere_ees[i].set_base_pos_orient(current_pos[i] + np.array([0, 0, delta_zs[i]]), np.array([0,0,0]))
                p.stepSimulation(physicsClientId=self.id)
                current_pos[i] = self.sphere_ees[i].get_base_pos_orient()[0]
        # * take image after blanket lifted up by 40 cm
        if self.take_images: self.capture_images()


        num_steps = 0
        travel_dist = []
        for i in range(2):
             travel_dist.append(release_locs[i] - grasp_locs[i])
             steps = np.abs(travel_dist[i]//0.005).max()
             num_steps = steps if steps > num_steps else num_steps

        delta_xs = []
        delta_ys = []
        for i in range(2):
            x, y = travel_dist[i]/num_steps
            delta_xs.append(x)
            delta_ys.append(y)
        
        for _ in range(int(num_steps)):
            for i in range(2):
                self.sphere_ees[i].set_base_pos_orient(current_pos[i] + np.array([delta_xs[i], delta_ys[i], 0]), np.array([0,0,0]))
                p.stepSimulation(physicsClientId=self.id)
                current_pos[i] = self.sphere_ees[i].get_base_pos_orient()[0]

        # * continue stepping simulation to allow the cloth to settle before release
        for _ in range(20):
            p.stepSimulation(physicsClientId=self.id)
        
        # * take image after moving to grasp location, before releasing cloth
        if self.take_images: self.capture_images()

        # * release the cloth at the release point, sphere is at the same arbitrary z position in the air
        for i in range(2):
            for id in constraint_ids[i]:
                p.removeConstraint(id, physicsClientId=self.id)
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.id)

        # * take image after cloth is released and settled
        if self.take_images: self.capture_images()

        # * get points on the blanket, final state of the cloth
        data = p.getMeshData(self.blanket, -1, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=self.id)

        # * compute rewards
        reward_uncover_target, uncovered_target_count = self.uncover_target_reward(data)
        reward_uncover_nontarget, uncovered_nontarget_count = self.uncover_nontarget_reward(data)
        reward_head_kept_uncovered, covered_head_count = self.keep_head_uncovered_reward(data)
        # * sum and weight rewards from individual functions to get overall reward
        reward = self.config('uncover_target_weight')*reward_uncover_target + self.config('uncover_nontarget_weight')*reward_uncover_nontarget + self.config('grasp_release_distance_max_weight')*reward_distance_btw_grasp_release + self.config('keep_head_uncovered_weight')*reward_head_kept_uncovered
        
        if self.verbose:
            print(f"Rewards for each measure:\n\tUncover Target: {reward_uncover_target}, Uncover Nontarget: {reward_uncover_nontarget}, Cover Head: {reward_head_kept_uncovered}, Excessive Distance: {reward_distance_btw_grasp_release}")
            print("overall reward: ", reward)

        # * prepare info
        split_reward = [reward_uncover_target, reward_uncover_nontarget, reward_distance_btw_grasp_release, reward_head_kept_uncovered]
        post_action_point_counts = [uncovered_target_count, uncovered_nontarget_count, covered_head_count]
        info = {'split_reward':split_reward, 'total_point_counts':point_counts,'post_action_point_counts': post_action_point_counts, 'clipped':clipped}

        self.iteration += 1
        done = self.iteration >= 1

        # * take image after reward computed
        if self.take_images: self.capture_images()

        # return 0, 0, 1, {}
        return obs, reward, done, info

    def change_point_color(self, points_target_limb, limb, ind, rgb = [0, 1, 0.5, 1]):
        p.changeVisualShape(points_target_limb[limb][ind].body, -1, rgbaColor=rgb, flags=0, physicsClientId=self.id)

    def uncover_target_reward(self, blanket_state):
        '''
        give the robot a reward for uncovering the target body part
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
                if self.render_body_points:
                    rgb = covered_rgb if covered else uncovered_rgb
                    self.change_point_color(self.points_target_limb, limb, point, rgb = rgb)
        points_uncovered = total_points - points_covered

        if self.verbose:
            print(f"Total Target Points: {total_points}, Target Uncovered: {points_uncovered}")

        return (points_uncovered/total_points)*100, points_uncovered

    def uncover_nontarget_reward(self, blanket_state):
        '''
        discourage the robot from learning policies that uncover nontarget body parts
        '''
        points_covered = 0
        uncovered_rgb = [1, 0, 0, 1]
        covered_rgb = [0, 0, 1, 1]
        threshold = 0.028
        total_points = self.total_nontarget_point_count - len(self.points_pos_nontarget_limb_world[self.human.head])
        total_target_points = self.total_target_point_count

        # account for case where all nontarget points were initially uncovered
        if total_points == 0:
            return 0, 0

        # count number of target points covered by the blanket
        for limb, points_pos_nontarget_limb_world in self.points_pos_nontarget_limb_world.items():
            if limb != self.human.head:
                # print(limb)
                for point in range(len(points_pos_nontarget_limb_world)):
                    covered = False
                    for i, v in enumerate(blanket_state[1]):
                        if abs(np.linalg.norm(v[0:2]-points_pos_nontarget_limb_world[point][0:2])) < threshold:
                            covered = True
                            points_covered += 1
                            break
                    # print("limb", limb, "covered", covered)
                    if self.render_body_points:
                        rgb = covered_rgb if covered else uncovered_rgb
                        self.change_point_color(self.points_nontarget_limb, limb, point, rgb = rgb)
        points_uncovered = total_points - points_covered
        
        if self.verbose:
            print(f"Total Nontarget Points: {total_points}, Nontarget Uncovered: {points_uncovered}")

        # if the same number of target and nontarget points are uncovered, total reward is 0
        return -(points_uncovered/total_target_points)*100, points_uncovered
        
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

        if total_points == 0:
            return 0, 0

        # count number of target points covered by the blanket
        for point in range(len(self.points_pos_nontarget_limb_world[self.human.head])):
            covered = False
            for i, v in enumerate(blanket_state[1]):
                if abs(np.linalg.norm(v[0:2]-points_pos_head_world[point][0:2])) < threshold:
                    covered = True
                    points_covered += 1
                    break
            if self.render_body_points:
                rgb = covered_rgb if covered else uncovered_rgb
                self.change_point_color(self.points_nontarget_limb, self.human.head, point, rgb = rgb)

        if self.verbose:
            print(f"Total Head Points: {total_points}, Covered Head Points: {points_covered}")
        
        # penalize on double the percentage of head points covered (doubled to increase weight of covering the head)
        return -(points_covered/total_points)*200, points_covered

    def non_target_initially_uncovered(self, blanket_state):
        '''
        removes nontarget points on the body that are uncovered when the blanket is in it's initial state from the nontarget point set
        also handles points on the head that are initially covered
        '''
        points_covered = 0
        threshold = 0.028
        points_to_remove = {}
        points_to_remove_count = 0
        

        # * create a list of the nontarget points not covered by the blanket
        for limb, points_pos_nontarget_limb_world in self.points_pos_nontarget_limb_world.items():
            # if limb != self.human.head:
            points_to_remove[limb] = []
            for point in range(len(points_pos_nontarget_limb_world)):
                covered = False
                for i, v in enumerate(blanket_state[1]):
                    if abs(np.linalg.norm(v[0:2]-points_pos_nontarget_limb_world[point][0:2])) < threshold:
                        covered = True
                        points_covered += 1
                        break
                if limb == self.human.head:
                    if covered == True:
                        points_to_remove[limb].append(point)
                elif covered == False:
                    points_to_remove[limb].append(point)
        
        # * remove the identified points from the list of all nontarget points for each limb
        # *     points removed in reverse order so that indexs of identified points don't shift
        for limb in self.points_pos_nontarget_limb_world.keys():
            # if limb != self.human.head:
            points_to_remove_count += len(points_to_remove[limb])
            # print("count of points on nontarget initially:", len(self.points_pos_nontarget_limb_world[limb]), len(self.points_nontarget_limb[limb]))
            for point in reversed(points_to_remove[limb]):
                self.points_pos_nontarget_limb_world[limb].pop(point)
                if self.render_body_points:
                    p.removeBody(self.points_nontarget_limb[limb][point].body)
                    self.points_nontarget_limb[limb].pop(point)
            # print("count of points on nontarget now:", len(self.points_pos_nontarget_limb_world[limb]), len(self.points_nontarget_limb[limb]))

        # print(self.total_nontarget_point_count)
        self.total_nontarget_point_count -= points_to_remove_count

    def _get_obs(self, agent=None):
        pose = []
        for limb in self.human.obs_limbs:
            pos, orient = self.human.get_pos_orient(limb)
            # print("pose", limb, pos, orient)
            pos2D = pos[0:2]
            yaw = p.getEulerFromQuaternion(orient)[-1]
            pose.append(np.concatenate((pos2D, np.array([yaw])), axis=0))
        pose = np.concatenate(pose, axis=0)
        # * collect more infomation for cmaes data collect, enables you to train model with different observations if you want to
        if self.cmaes_dc:
            output = [None]*12
            all_joint_angles = self.human.get_joint_angles(self.human.all_joint_indices)
            all_pos_orient = [self.human.get_pos_orient(limb) for limb in self.human.all_body_parts]
            output[0], output[1], output[2] = pose, all_joint_angles, all_pos_orient
            return output

        return pose.astype('float32')

    def set_seed_val(self, seed = 1001):
        if seed != self.seed_val:
            self.seed_val = seed
    
    def set_target_limb_code(self, target_limb_code=None):
        if target_limb_code == None:
            self.target_limb_code = self.np_random.randint(0,12)
        else:
            self.target_limb_code = target_limb_code

    def set_pstate_file(self, filename):
        if self.pstate_file != filename:
            self.pstate_file = filename
            self.save_pstate = True

    def reset(self):
        super(BeddingManipulationParallelEnv, self).reset()
        print('\nPARALLEL\n')

        if not self.fixed_pose and not self.cmaes_dc:
            self.set_seed_val(seeding.create_seed())
        self.seed(self.seed_val)

        self.build_assistive_env(fixed_human_base=False, gender='female', human_impairment='none', furniture_type='hospital_bed', body_shape=self.body_shape)

        # * enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        # * configure directory to save captured images to
        if self.take_images:
            self.image_dir = os.path.join(self.save_image_dir, time.strftime("%Y%m%d-%H%M%S"))
            if not os.path.exists(self.image_dir):
                os.makedirs(self.image_dir)

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

        # * take image after human settles
        if self.take_images: self.capture_images()
        
        if self.use_mesh:
            # * we do not use a mesh in this work
            # Replace the capsulized human with a human mesh
            self.human = HumanMesh()
            joints_positions = [(self.human.j_right_shoulder_z, 60), (self.human.j_right_elbow_y, 90), (self.human.j_left_shoulder_z, -10), (self.human.j_left_elbow_y, -90), (self.human.j_right_hip_x, -60), (self.human.j_right_knee_x, 80), (self.human.j_left_hip_x, -90), (self.human.j_left_knee_x, 80)]
            body_shape = np.zeros((1, 10))
            gender = 'female' # 'random'
            self.human.init(self.directory, self.id, self.np_random, gender=gender, height=None, body_shape=body_shape, joint_angles=joints_positions, left_hand_pose=[[-2, 0, 0, -2, 0, 0]])

            chair_seat_position = np.array([0, 0.1, 0.55])
            self.human.set_base_pos_orient(chair_seat_position - self.human.get_vertex_positions(self.human.bottom_index), [0, 0, 0, 1])
 
        # * select a target limb to uncover (may be fixed or random), generate points along the body accordingly
        if not self.fixed_target_limb:
            self.set_target_limb_code()
        self.target_limb = self.human.all_possible_target_limbs[self.target_limb_code]
        self.generate_points_along_body()

        # * take image after points generated (only visible if --render_body_points flag is used)
        if self.take_images: self.capture_images()
       
        # * spawn blanket
        self.blanket = p.loadSoftBody(os.path.join(self.directory, 'clothing', 'blanket_2089v.obj'), scale=0.75, mass=0.15, useBendingSprings=1, useMassSpring=1, springElasticStiffness=1, springDampingStiffness=0.0005, springDampingAllDirections=1, springBendingStiffness=0, useSelfCollision=1, collisionMargin=0.006, frictionCoeff=0.5, useFaceContact=1, physicsClientId=self.id)
        # * change alpha value so that it is a little more translucent, easier to see the relationship the human
        p.changeVisualShape(self.blanket, -1, rgbaColor=[0, 0, 1, 0.75], flags=0, physicsClientId=self.id)
        p.changeVisualShape(self.blanket, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=4, numSolverIterations = 4, physicsClientId=self.id)

        if self.blanket_pose_var:
            delta_y = self.np_random.uniform(-0.25, 0.05)
            delta_x = self.np_random.uniform(-0.02, 0.02)
            deg = 45
            delta_rad = self.np_random.uniform(-np.radians(deg), np.radians(deg)) # * +/- degrees
            p.resetBasePositionAndOrientation(self.blanket, [0+delta_x, 0.2+delta_y, 1.5], self.get_quaternion([np.pi/2.0, 0, 0+delta_rad]), physicsClientId=self.id)
        else:
            p.resetBasePositionAndOrientation(self.blanket, [0, 0.2, 1.5], self.get_quaternion([np.pi/2.0, 0, 0]), physicsClientId=self.id)


        # * Drop the blanket on the person, allow to settle
        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

    
        # * Initialize enviornment variables
        # *     if using the sphere manipulator, spawn the sphere and run a modified version of init_env_variables()
        if self.robot is None:
            # * spawn sphere manipulator
            position = np.array([-0.3, -0.86, 0.8])
            self.sphere_ees = []
            self.sphere_ees.append(self.create_sphere(radius=0.025, mass=0.0, pos = position, visual=True, collision=True, rgba=[1, 0, 0, 1]))
            self.sphere_ees.append(self.create_sphere(radius=0.025, mass=0.0, pos = position+0.2, visual=True, collision=True, rgba=[1, 0, 1, 1]))
            
            # * initialize env variables
            from gym import spaces
            # * update observation and action spaces
            obs_len = len(self._get_obs())
            self.observation_space.__init__(low=-np.ones(obs_len, dtype=np.float32)*1000000000, high=np.ones(obs_len, dtype=np.float32)*1000000000, dtype=np.float32)
            action_len = 8
            self.action_space.__init__(low=-np.ones(action_len, dtype=np.float32), high=np.ones(action_len, dtype=np.float32), dtype=np.float32)
            # * Define action/obs lengths
            self.action_robot_len = 8
            self.action_human_len = len(self.human.controllable_joint_indices) if self.human.controllable else 0
            self.obs_robot_len = len(self._get_obs('robot'))    # 1
            self.obs_human_len = 0
            self.action_space_robot = spaces.Box(low=np.array([-1.0]*self.action_robot_len, dtype=np.float32), high=np.array([1.0]*self.action_robot_len, dtype=np.float32), dtype=np.float32)
            self.action_space_human = spaces.Box(low=np.array([-1.0]*self.action_human_len, dtype=np.float32), high=np.array([1.0]*self.action_human_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_robot = spaces.Box(low=np.array([-1000000000.0]*self.obs_robot_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_robot_len, dtype=np.float32), dtype=np.float32)
            self.observation_space_human = spaces.Box(low=np.array([-1000000000.0]*self.obs_human_len, dtype=np.float32), high=np.array([1000000000.0]*self.obs_human_len, dtype=np.float32), dtype=np.float32)
        else:
            self.init_env_variables()
        
        if self.save_pstate:
            p.saveBullet(self.pstate_file)
            self.save_pstate = False
        
        # * image after blanket settles, before grasp
        if self.take_images: self.capture_images()
            
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
            # *     only generate sphere bodies if self.render_body_points == True
            if limb in self.target_limb:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_target_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_target_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                if self.render_body_points:
                    self.points_target_limb[limb] = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.points_pos_on_target_limb[limb]), visual=True, collision=False, rgba=[1, 1, 1, 1])
                self.total_target_point_count += len(self.points_pos_on_target_limb[limb])
            else:
                if limb in [self.human.left_hand, self.human.right_hand]:
                    self.points_pos_on_nontarget_limb[limb] = self.util.sphere_points(radius=radius, samples = 20)
                else:
                    self.points_pos_on_nontarget_limb[limb] = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -length]), radius=radius, distance_between_points=0.03)
                if self.render_body_points:
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
                    if self.render_body_points:
                        self.points_target_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_target_limb_world[limb] = points_pos_limb_world
            else:
                for i in range(len(self.points_pos_on_nontarget_limb[limb])):
                    point_pos = np.array(p.multiplyTransforms(limb_pos, limb_orient, self.points_pos_on_nontarget_limb[limb][i], [0, 0, 0, 1], physicsClientId=self.id)[0])
                    points_pos_limb_world.append(point_pos)
                    if self.render_body_points:
                        self.points_nontarget_limb[limb][i].set_base_pos_orient(point_pos, [0, 0, 0, 1])
                self.points_pos_nontarget_limb_world[limb] = points_pos_limb_world

    def increase_pose_variation(self):
        '''
        Allow more variation in the knee and elbow angles
          can be some random position within the lower and upper limits of the joint movement (range is made a little smaller than the limits of the joint to prevent angles that are too extreme)
        
        NOT USED IN THIS WORK
        '''
        for joint in (self.human.j_left_knee, self.human.j_right_knee, self.human.j_left_elbow, self.human.j_right_elbow):
            motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states([joint])
            self.human.set_joint_angles(motor_indices, motor_positions+self.np_random.uniform(self.human.lower_limits[joint]+0.1, self.human.upper_limits[joint]-0.1, 1))


    def capture_images(self):
        # * top view
        self.setup_camera_rpy(camera_target=[0, 0, 0.305+2.101], distance=0.01, rpy=[0, -90, 180], fov=60, camera_width=468//2, camera_height=410)
        img, depth = self.get_camera_image_depth()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = f'top_view_{time.strftime("%Y%m%d-%H%M%S")}.png'
        cv2.imwrite(os.path.join(self.image_dir, filename), img)

        # * side view
        self.setup_camera_rpy(camera_target=[0, 0, 1], distance=3, rpy=[0, -20, 120], fov=60, camera_width=468//2, camera_height=410)
        img, depth = self.get_camera_image_depth()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        filename = f'side_view_{time.strftime("%Y%m%d-%H%M%S")}.png'
        cv2.imwrite(os.path.join(self.image_dir, filename), img)
