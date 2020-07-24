# construct AST env

import math
from ast_toolbox.envs.ast_env import ASTEnv
from ast_toolbox.rewards import ASTRewardS
from garage.experiment import Snapshotter
from CartPole.cartpole_simulator import CartpoleSimulator
import joblib
import tensorflow as tf
from garage.sampler.utils import rollout
from ast_toolbox.utils.simulator_rollout import rollout as sim_rollout

file = "/home/csidrane/Documents/NASA/garage/examples/jupyter/data/" #params.pkl"

#"/home/csidrane/Documents/NASA/data/local/experiment/train_cartpole_policy/params.pkl"
#"/home/csidrane/Documents/NASA/garage/examples/jupyter/data/"

# # for snapshotted policies: 
snapshotter = Snapshotter()
with tf.compat.v1.Session() as sess: # optional, only for TensorFlow
    data = snapshotter.load(file)

policy = data['algo'].policy
env0 = data['env']
rollout(env0, policy, animated=False)

# for old tf code:
#data = joblib.load(file)
#policy = data['policy']

reward_function = ASTRewardS()
simulator = CartpoleSimulator(sut=policy, max_path_length=100, use_seed=False)
env = ASTEnv(open_loop=False,
                simulator=simulator,
                fixed_init_state=True,
                s_0=[0.0, 0.0, 0.0 * math.pi / 180, 0.0],
                reward_function=reward_function,
                )

import pdb; pdb.set_trace()
sim_rollout(env, 0)