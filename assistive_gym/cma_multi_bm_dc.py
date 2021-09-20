import gym, sys, argparse, multiprocessing, time, os
from gym.utils import seeding
import numpy as np
import cma
from cma.optimization_tools import EvalParallel2
import pickle
import pathlib
# import assistive_gym


def cost_function(action):
    pid = os.getpid()
    t0 = time.time()

    observation = env.reset()
    done = False
    while not done:
        # env.render()
        observation, reward, done, info = env.step(action)
        t1 = time.time()
        cost = -reward
        elapsed_time = t1 - t0

    return [cost, observation, elapsed_time, pid, info]


# # # TEST COST FUNCTION
# def cost_function(action):
#     pid = os.getpid()
#     t0 = time.time()

#     done = False
#     cost = 0
#     while not done:
#         t1 = time.time()
#         cost = (action[0] - 3) ** 2 + (10 * (action[1] + 2)) ** 2 + (10 * (action[2] + 2)) ** 2 + (10 * (action[3] - 3)) ** 2
#         observation = 0
#         elapsed_time = t1 - t0
#         done = True

#     return [cost, observation, elapsed_time, pid]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CMA-ES sim optimization')
    parser.add_argument('--env', default='BeddingManipulationSphere-v1', help='env', required=True)
    parser.add_argument('--target-limb-code', required=True,
                            help='Code for target limb to uncover, see human.py for a list of available target codes')
    parser.add_argument('--run_id', help='id for the run (this code can be run 4 times simultaneously', required=True)
    args, unknown = parser.parse_known_args()
    current_dir = os.getcwd()
    pkl_loc = os.path.join(current_dir,'cmaes_data_collect/pickle')
    pstate_loc = os.path.join(current_dir,'cmaes_data_collect/bullet_state')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)
    pathlib.Path(pstate_loc).mkdir(parents=True, exist_ok=True)


    # * make the enviornment, set the specified target limb code and an initial seed value
    env = gym.make(args.env)
    env.set_seed_val(seeding.create_seed())
    if args.target_limb_code == 'random':
        env.set_target_limb_code()
    else:
        target_limb = int(args.target_limb_code)
        env.set_target_limb_code(target_limb)

    # * set the number of processes to 1/4 of the total number of cpus
    # *     collect data for 4 different target limbs simultaneously by running this script in 4 terminals
    num_proc = multiprocessing.cpu_count()//4

    # * set variables to initialize CMA-ES
    opts = cma.CMAOptions({'verb_disp': 1, 'popsize': num_proc}) # , 'tolfun': 10, 'maxfevals': 500
    bounds = np.array([1]*4)
    opts.set('bounds', [[-1]*4, bounds])
    opts.set('CMA_stds', bounds)
    x0 = bounds/2.0
    # x0 = np.random.uniform(-1,1,4)
    sigma0 = 0.2
    reward_threshold = 95

    pose_count = 1
    total_fevals = 0

    # * repeat optimization for x number of human poses
    for _ in range(200):
        # * open the pickle file to send optimization data to
        filename = f"targ{env.target_limb_code}_p{pose_count}_{env.seed_val}_{args.run_id}"
        f = open(os.path.join(pkl_loc, filename +".pkl"),"wb")
        env.set_pstate_file(os.path.join(pstate_loc, filename +".bullet"))


        print(f"Pose number: {pose_count}, total fevals: {total_fevals}, target limb code: {env.target_limb_code}, enviornment seed: {env.seed_val}")
        fevals = 0
        iterations = 0
        t0 = time.time()

        # * initialize CMA-ES
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

        # * evaluate cost function in parallel over num_cpus/4 processes
        with EvalParallel2(cost_function, number_of_processes=num_proc) as eval_all:
            # * continue running optimization until termination criteria is reached
            while not es.stop():
                iterations += 1
                fevals += num_proc
                total_fevals += num_proc
                
                actions = es.ask()
                output = eval_all(actions)
                t1 = time.time()
                output = [list(x) for x in zip(*output)]
                costs = output[0]
                observations = output[1]
                elapsed_time = output[2]
                pids = output[3]
                info = output[4]
                es.tell(actions, costs)
                
                rewards = [-c for c in costs]
                mean = np.mean(rewards)
                min = np.min(rewards)
                max = np.max(rewards)
                total_elapsed_time = t1-t0

                #! TESTING ONLY
                # if iterations == 1: costs = [-95]
                # iterations = 300

                print(f"Pose: {pose_count}, iteration: {iterations}, total fevals: {total_fevals}, fevals: {fevals}, elapsed time: {total_elapsed_time:.2f}, mean reward = {mean:.2f}, min/max reward = {min:.2f}/{max:.2f}")
                pickle.dump({
                    "seed": env.seed_val,
                    "target_limb": env.target_limb_code, 
                    "iteration": iterations, "pids": pids, 
                    "fevals": fevals, 
                    "total_elapsed_time":total_elapsed_time, 
                    "actions": actions,
                    "rewards": rewards, 
                    "observations":observations, #? save only the first observation (they are all the same since pose is the same)
                    "elapsed_time": elapsed_time,
                    "info":info}, f)

                # * if any of the processes reached the reward_threshold, stop optimizing
                if np.any(np.array(costs) <= -reward_threshold):
                    print("Reward threshold reached")
                    break
                if fevals >= 300:
                    print("No solution found after 300 fevals")
                    break

            es.result_pretty()
            f.close()
            print("Data saved to file:", filename+".pkl")
            print()

            env.set_seed_val(seeding.create_seed())
            if args.target_limb_code == 'random':
                env.set_target_limb_code()
            else:
                env.set_target_limb_code(target_limb)
            pose_count += 1

