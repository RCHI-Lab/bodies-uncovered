import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo, sac, ddpg, impala
from ray.rllib.agents.ppo import appo
from ray.tune.logger import pretty_print
from numpngw import write_apng
import pathlib, pickle,time
import keras
from .envs.bm_config import BM_Config



def setup_config(env, algo, coop=False, seed=0, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 32
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 2
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [50, 50]
    elif algo == 'sac':
        # NOTE: pip3 install tensorflow_probability
        config = sac.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
        config['Q_model']['fcnet_hiddens'] = [100, 100]
        config['policy_model']['fcnet_hiddens'] = [100, 100]
    elif algo == 'ddpg':
        config = ddpg.DEFAULT_CONFIG.copy()
        config['timesteps_per_iteration'] = 1
        config['learning_starts'] = 16
        config['actor_hiddens'] = [20, 20]
        config['critic_hiddens'] = [20, 20]
        config['buffer_size'] = 1000
    elif algo == 'appo':
        config = appo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 32
        config['rollout_fragment_length'] = 1
        config['num_sgd_iter'] = 50
        config['vtrace'] = False
        config['lambda'] = 0.95
        config['replay_proportion'] = 0.25
        config['replay_buffer_num_slots'] = 1000
        config['model']['fcnet_hiddens'] = [20, 20]
    elif algo == 'impala':
        config = impala.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 32
        config['rollout_fragment_length'] = 1
        config['num_sgd_iter'] = 50
        config['replay_proportion'] = 0.25
        config['replay_buffer_num_slots'] = 1000
        config['model']['fcnet_hiddens'] = [20, 20]

    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    if algo == 'sac':
        config['timesteps_per_iteration'] = 400
        config['learning_starts'] = 1000
    if coop:
        obs = env.reset()
        policies = {'robot': (None, env.observation_space_robot, env.action_space_robot, {}), 'human': (None, env.observation_space_human, env.action_space_human, {})}
        config['multiagent'] = {'policies': policies, 'policy_mapping_fn': lambda a: a}
        config['env_config'] = {'num_agents': 2}
    return {**config, **extra_configs}

def load_policy(env, algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'sac':
        agent = sac.SACTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'ddpg':
        agent = ddpg.DDPGTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'appo':
        agent = appo.APPOTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    elif algo == 'impala':
        agent = impala.ImpalaTrainer(setup_config(env, algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def train(env_name, algo, timesteps_total=1000000, save_dir='./trained_models/', load_policy_path='', coop=False, seed=0, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    env = make_env(env_name, coop)
    agent, checkpoint_path = load_policy(env, algo, env_name, load_policy_path, coop, seed, extra_configs)
    env.disconnect()

    timesteps = 0
    while timesteps < timesteps_total:
        result = agent.train()
        timesteps = result['timesteps_total']
        if coop:
            # Rewards are added in multi agent envs, so we divide by 2 since agents share the same reward in coop
            result['episode_reward_mean'] /= 2
            result['episode_reward_min'] /= 2
            result['episode_reward_max'] /= 2
        print(f"Iteration: {result['training_iteration']}, total timesteps: {result['timesteps_total']}, total time: {result['time_total_s']:.1f}, FPS: {result['timesteps_total']/result['time_total_s']:.1f}, mean reward: {result['episode_reward_mean']:.1f}, min/max reward: {result['episode_reward_min']:.1f}/{result['episode_reward_max']:.1f}")
        sys.stdout.flush()

        # Delete the old saved policy
        if checkpoint_path is not None:
            shutil.rmtree(os.path.dirname(checkpoint_path), ignore_errors=True)
        # Save the recently trained policy
        checkpoint_path = agent.save(os.path.join(save_dir, algo, env_name))
    return checkpoint_path

def render_policy(env, env_name, algo, policy_path, coop=False, colab=False, seed=0, n_episodes=1, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
    if env is None:
        env = make_env(env_name, coop, seed=seed)
        if colab:
            env.setup_camera(camera_eye=[0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=60, camera_width=1920//4, camera_height=1080//4)
    test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    if not colab:
        env.render()
    frames = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            if coop:
                # Compute the next action for the robot/human using the trained policies
                action_robot = test_agent.compute_action(obs['robot'], policy_id='robot')
                action_human = test_agent.compute_action(obs['human'], policy_id='human')
                # Step the simulation forward using the actions from our trained policies
                obs, reward, done, info = env.step({'robot': action_robot, 'human': action_human})
                done = done['__all__']
            else:
                # Compute the next action using the trained policy
                action = test_agent.compute_action(obs)
                # Step the simulation forward using the action from our trained policy
                obs, reward, done, info = env.step(action)
            if colab:
                # Capture (render) an image from the camera
                img, depth = env.get_camera_image_depth()
                frames.append(img)
    env.disconnect()
    if colab:
        filename = 'output_%s.png' % env_name
        write_apng(filename, frames, delay=100)
        return filename

#! has been significantly modified for evaluting bm policies
def evaluate_policy(env_name, algo, policy_path, n_episodes=100, coop=False, seed=0, verbose=False, extra_configs={}):
    if algo == 'cmaes':
        print('CMA-ES EVALUATION', policy_path)
        env = make_env(env_name, coop, seed=seed)
        model = keras.models.load_model(policy_path)
    else:
        ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)
        env = make_env(env_name, coop, seed=seed)
        test_agent, _ = load_policy(env, algo, env_name, policy_path, coop, seed, extra_configs)

    current_dir = os.getcwd()
    pkl_loc = os.path.join(current_dir,'bm_eval')
    pathlib.Path(pkl_loc).mkdir(parents=True, exist_ok=True)
    filename = f"evalData_{policy_path.split('/')[-1]}"
    f = open(os.path.join(pkl_loc, filename +".pkl"),"wb")
    eval_t0 = time.time()

    rewards = []
    grasp_not_over_blanket_count = 0
    uncovered_target = []
    uncovered_nontarget = []
    covered_head = []

    for episode in range(n_episodes):

        t0 = time.time()
        obs = env.reset()

        if algo == 'cmaes':
            obs = np.reshape(obs, (-1, 12))
            action = model.predict(obs)[0]
        else:
            action = test_agent.compute_action(obs)

        obs, reward, done, info = env.step(action)
        t1 = time.time()

        total_elapsed_time = t1-eval_t0
        elapsed_time = t1-t0

        if info['clipped']:
            grasp_not_over_blanket_count += 1

        uncovered_target_count, uncovered_nontarget_count, covered_head_count = info['post_action_point_counts']

        rewards.append(reward)
        uncovered_target.append(uncovered_target_count)
        uncovered_nontarget.append(uncovered_nontarget_count)
        covered_head.append(covered_head_count)
    
        pickle.dump({
            "total_elapsed_time":total_elapsed_time, 
            "action": action,
            "reward": reward, 
            "observation":obs,
            "elapsed_time": elapsed_time,
            "info":info}, f)

        if verbose:
            print(f"Episode {episode+1} | Reward total: {reward}, Total Counts: {info['total_point_counts']}, Post Action: {info['post_action_point_counts']}")
        sys.stdout.flush()

    env.disconnect()

    print('\n', '-'*50, '\n')
    # print('Rewards:', rewards)
    print('Reward Mean:', np.mean(rewards))
    print('Reward Std:', np.std(rewards))
    print('Mean Uncovered Target:', np.mean(uncovered_target))
    print('Mean Uncovered NonTarget:', np.mean(uncovered_nontarget))
    print('Mean Covered Head:', np.mean(covered_head))
    print('Number of cases where grasp not over blanket:', grasp_not_over_blanket_count)


    sys.stdout.flush()
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='ScratchItchJaco-v0',
                        help='Environment to train on (default: ScratchItchJaco-v0)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Whether to train a new policy')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Whether to render a single rollout of a trained policy')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='Whether to evaluate a trained policy over n_episodes')
    parser.add_argument('--train-timesteps', type=int, default=1000000,
                        help='Number of simulation timesteps to train a policy (default: 1000000)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='Directory to save trained policy in (default ./trained_models/)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--render-episodes', type=int, default=1,
                        help='Number of rendering episodes (default: 1)')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--colab', action='store_true', default=False,
                        help='Whether rendering should generate an animated png rather than open a window (e.g. when using Google Colab)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to output more verbose prints')
    bm_config = BM_Config()
    parser = bm_config.add_bm_args(parser)
    args = parser.parse_args()
    bm_config.change_bm_config(args)
    args, unknown = parser.parse_known_args()

    coop = ('Human' in args.env)
    checkpoint_path = None

    if args.train:
        checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, coop=coop, seed=args.seed)
    if args.render:
        render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    if args.evaluate:
        evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose)

