import gym, sys, argparse
import numpy as np
from .learn import make_env
from .envs.bm_config import BM_Config
# import assistive_gym

if sys.version_info < (3, 0):
    print('Please use Python 3')
    exit()

def sample_action(env, coop):
    if coop:
        return {'robot': env.action_space_robot.sample(), 'human': env.action_space_human.sample()}
    return env.action_space.sample()

def viewer(env_name):
    coop = 'Human' in env_name
    env = make_env(env_name, coop=True) if coop else gym.make(env_name)

    while True:
        done = False
        env.render()
        observation = env.reset()
        action = sample_action(env, coop)
        if coop:
            print('Robot observation size:', np.shape(observation['robot']), 'Human observation size:', np.shape(observation['human']), 'Robot action size:', np.shape(action['robot']), 'Human action size:', np.shape(action['human']))

        elif 'BeddingManipulationSphere-v1' in env_name:
            action = np.array([0.3, 0.5, 0, 0])
        elif 'RemoveContactSphere-v1' in env_name:
            action = np.array([0.3, 0.45])
        else:
            print('Observation size:', np.shape(observation), 'Action size:', np.shape(action))

        while not done:
            # observation, reward, done, info = env.step(sample_action(env, coop))
            observation, reward, done, info = env.step(action)
            if coop:
                done = done['__all__']

# see __main__.py file for main