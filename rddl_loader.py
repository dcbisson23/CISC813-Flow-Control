import pyRDDLGym

from pyRDDLGym.core.visualizer.movie import MovieGenerator

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan, JaxBackpropPlanner, JaxOfflineController, JaxDeepReactivePolicy, JaxOnlineController
from pyRDDLGym_jax.core.planner import load_config
from pyRDDLGym_jax.core.simulator import JaxRDDLSimulator
from pathlib import Path
import os
import csv
import pprint

# Two options for [Optimizer] planner
# Copy-paste desired model into the [Optimizer] section, then run it
"""
method='JaxStraightLinePlan'
method_kwargs={}

method='JaxDeepReactivePolicy'
method_kwargs={'topology': [128, 64]}
"""

CONFIG = """
[Model]
logic='FuzzyLogic'
logic_kwargs={}
tnorm='ProductTNorm'
tnorm_kwargs={}

[Optimizer]
method='JaxStraightLinePlan'
method_kwargs={}
optimizer='rmsprop'
optimizer_kwargs={'learning_rate': 0.001}
batch_size_train=32
batch_size_test=32
rollout_horizon=10

[Training]
key=42
epochs=5000
train_seconds=30
"""

with open("config.cfg", "w") as f:
    f.write(CONFIG)


domain = Path.cwd().joinpath('pump_domain.rddl')
instance = Path.cwd().joinpath('pump_instance0.rddl')

# domain = Path.cwd().joinpath('archives/Reservoir/domain.rddl')
# instance = Path.cwd().joinpath('archives/Reservoir/instance0.rddl')

def run_planner(dom, prob, online=True):

    planner_args, plan_args, train_args = load_config("config.cfg")
    # os.system("rm -f logs/data_log.csv")

    # set up the environment (note the vectorized option must be True)
    env = pyRDDLGym.make(domain=dom, instance=prob, vectorized=True, log_path=Path.cwd().joinpath("logs"))
    recorder = MovieGenerator("logs", "pump-flow-ctrl", max_frames=env.horizon)
    env.set_visualizer(viz=None, movie_gen=recorder)

    if online:
        planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
        controller = JaxOnlineController(planner, train_seconds=2)
        stats = controller.evaluate(env, episodes=1, verbose=False, render=True)

    else:
        planner = JaxBackpropPlanner(rddl=env.model, **planner_args)
        controller = JaxOfflineController(planner, **train_args)
        stats = controller.evaluate(env, episodes=1, verbose=False, render=True)
        pprint.pprint(stats)

    rendered_image = env.render()
    env.close()

    # remove the first 3 lines of the csv file
    # with open('logs/data_log.csv', 'r') as f:
    #     lines = f.readlines()
    # with open('logs/data_log.csv', 'w') as f:
    #     f.writelines(lines[3:])
    # show_actions()

    return rendered_image

def show_actions():

    actions = ['put-out', 'cut-out', 'water-drop']

    with open('logs/data_log.csv', 'r') as f:
        reader = csv.reader(f)

        # get the columns from the header row
        action_cols = []
        header = next(reader)
        for i, col in enumerate(header):
            for action in actions:
                if action in col:
                    action_cols.append(i)

        for row in reader:
            print(f"{row[1]}:", end=" ")
            for col in action_cols:
                if row[col] == "True":
                    print(f"{header[col]}", end=" ")
            print()

run_planner(domain, instance).show()