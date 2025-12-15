import pyRDDLGym

from pyRDDLGym.core.visualizer.movie import MovieGenerator

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxBackpropPlanner, JaxOfflineController, JaxDeepReactivePolicy, JaxOnlineController
from pyRDDLGym_jax.core.tuning import JaxParameterTuning, Hyperparameter
from pyRDDLGym_jax.core.planner import load_config
from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator
from pathlib import Path

import pandas as pd
from IPython.display import Image
from IPython.utils import io
import matplotlib.pyplot as plt

import os
import csv
import pprint
import math

ONLINE_CONFIG = """
[Model]
comparison_kwargs={'weight': 100}
rounding_kwargs={'weight': 100}
control_kwargs={'weight': 100}
[Optimizer]
method='JaxStraightLinePlan'
pgpe=None
rollout_horizon=5
utility='cvar'
utility_kwargs={'alpha': 0.2}
[Training]
key=42
train_seconds=5
"""

OFFLINE_CONFIG = """
[Model]
comparison_kwargs={'weight': 100}
rounding_kwargs={'weight': 100}
control_kwargs={'weight': 100}
[Optimizer]
method='JaxDeepReactivePolicy'
optimizer_kwargs={'learning_rate': 0.001}
utility='cvar'
utility_kwargs={'alpha': 0.2}

rollout_horizon=100
[Training]
key=42
train_seconds=60
"""


# Set the root directory to either MDP or POMDP, and the instance name to the instance you would like to evaluate.
# Note: Running POMDP does not currently work 
# (JaxPlan does not work with replanning in POMDP environments currently, though it may be fixed soon).

root_directory = 'MDP'
instance_name = 'instance3'
run_online = True





domain = Path.cwd().joinpath(root_directory + '/domain.rddl')
instance = Path.cwd().joinpath(root_directory + '/' + instance_name + '.rddl')

def run_planner(dom, prob, online=True):
    with open("config.cfg", "w") as f:
        if online:
            f.write(ONLINE_CONFIG)
        else:
            f.write(OFFLINE_CONFIG)
    planner_args, plan_args, train_args = load_config("config.cfg")
    # os.system("rm -f logs/data_log.csv")

    # set up the environment (note the vectorized option must be True)
    env = pyRDDLGym.make(domain=dom, instance=prob, vectorized=True, backend=JaxRDDLSimulator, log_path=Path.cwd().joinpath(f"logs/{instance_name}"))
    print(env.action_space, env.state)
    recorder = MovieGenerator(f"logs", "pump-flow-ctrl", max_frames=env.horizon)
    env.set_visualizer(viz=None, movie_gen=recorder)

    if online:
        planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
        controller = JaxOnlineController(planner, **train_args)
        stats = controller.evaluate(env, episodes=1, verbose=False, render=True)
        print(env.state)

    else:
        planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
        controller = JaxOfflineController(planner, **train_args)
        stats = controller.evaluate(env, episodes=1, verbose=False, render=True)
        pprint.pprint(stats)


    # env.horizon = 30
    
    
    # total_reward = 0
    # state, _ = env.reset()
    # print(f'Initial state = {state}')
    # for step in range(10):
    #     env.render()
    #     action = controller.sample_action(state)
    #     next_state, reward, terminated, truncated, _ = env.step(action)
        
    #     total_reward += reward
    #     state = next_state
    #     print(f'action = {action}, reward = {reward}, next state = {state}')
    #     #print(terminated, truncated)
    #     if terminated or truncated:
    #         break

    rendered_image = env.render()
    env.close()

    # print(total_reward)
    # remove the first 3 lines of the csv file
    # with open('logs/data_log.csv', 'r') as f:
    #     lines = f.readlines()
    # with open('logs/data_log.csv', 'w') as f:
    #     f.writelines(lines[3:])
    # show_actions()

    return rendered_image

if __name__ == '__main__':
    run_planner(domain, instance, run_online).show()